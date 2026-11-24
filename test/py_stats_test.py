#!/usr/bin/env python3
"""pl_stats 模块自测。

覆盖：
- 11 个统计方法（describe/compare/assumption/corr/regression/chi2/normality/roc/survival/
  adjust/pca）各跑一次，检查返回契约：
  title/method/summary/tables/figures/report/annotation/warnings 齐全、
  bullets 非空且带数字、表格 columns/rows 结构合法、p 值 ∈ [0,1] 或 None、
  report 非空、figure 文件真实存在、整体 JSON 可序列化（无 NaN/Inf）、单次 <10s；
- compare 的 3 组（ANOVA/Kruskal + 事后两两）与配对分支；
- 确定性（同一输入两次运行结果一致）；
- 10 条错误路径（EngineError.code 正确）。

运行：
    export MPLBACKEND=Agg MPLCONFIGDIR=$(mktemp -d)
    PYTHONPATH=lib python3 test/py_stats_test.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="plotlab_mpl_")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pl_common as C  # noqa: E402
import pl_prep as P  # noqa: E402
import pl_state as S  # noqa: E402
import pl_stats as T  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="plotlab_stats_test_")
STYLE = C.resolve_style({
    "theme": "minimal",
    "palette": "npg",
    "fig_width": 5.6,
    "fig_height": 4.2,
    "dpi": 150,
})
ALLOWED_TYPES = {"str", "num", "p"}
REQUIRED_KEYS = ("title", "method", "summary", "tables", "figures", "report",
                 "annotation", "warnings")
REQUIRED_STATS = ("describe", "compare", "assumption", "corr", "regression", "chi2",
                  "normality", "roc", "survival", "adjust", "pca")
MAX_SECONDS = 10.0

STATS_RUN = 0
ERRORS_RUN = 0


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------


def make_multigroup() -> dict:
    """3 组模拟数据（检验 ANOVA/Kruskal + 事后两两比较）。"""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "arm": ["A"] * 15 + ["B"] * 14 + ["C"] * 13,
        "value": np.concatenate([
            rng.normal(10.0, 2.0, 15),
            rng.normal(12.5, 2.0, 14),
            rng.normal(16.0, 2.0, 13),
        ]).round(3),
    })
    return S.build_dataset(frame, "三组模拟", kind="tidy")


def make_paired() -> dict:
    """配对数据：tidy 长表，两组按行顺序一一对应。"""
    rng = np.random.default_rng(5)
    n = 20
    before = rng.normal(10.0, 2.0, n)
    after = before + rng.normal(1.2, 1.0, n)
    frame = pd.DataFrame({
        "group": ["Before"] * n + ["After"] * n,
        "value": np.concatenate([before, after]).round(3),
    })
    return S.build_dataset(frame, "配对模拟", kind="tidy")


def make_single_group() -> dict:
    frame = pd.DataFrame({"group": ["A"] * 10, "value": np.arange(10.0)})
    return S.build_dataset(frame, "单组模拟", kind="tidy")


def make_single_class() -> dict:
    frame = pd.DataFrame({
        "label": ["Yes"] * 12,
        "score": np.linspace(0.05, 0.95, 12),
    })
    return S.build_dataset(frame, "单类别模拟", kind="tidy")


def warmup_matplotlib() -> None:
    """预热 matplotlib 字体缓存，避免首个出图方法的耗时被算进 <10s 断言。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2, 1.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    path = os.path.join(OUTDIR, "warmup.png")
    fig.savefig(path, dpi=50)
    plt.close(fig)
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 构建与运行
# ---------------------------------------------------------------------------


def build_data(dataset: dict, stat_id: str, params: dict, roles: dict | None) -> dict:
    meta = T.stat_meta(stat_id)
    needs = meta.get("needs") or "tidy"
    if roles is None:
        roles = P.recommend_roles(dataset, meta)
    return P.build_data(
        dataset,
        {"id": stat_id, "needs": needs, "roles": meta.get("roles") or {}},
        params or {},
        roles,
    )


def run_stat(dataset: dict, stat_id: str, params: dict | None = None,
             roles: dict | None = None, expect_figures: bool = False) -> dict:
    global STATS_RUN
    params = dict(params or {})
    data = build_data(dataset, stat_id, params, roles)
    started = time.perf_counter()
    result = T.run(stat_id, dataset, data, params, STYLE, OUTDIR)
    elapsed = time.perf_counter() - started
    check_result(stat_id, result, elapsed, expect_figures=expect_figures)
    STATS_RUN += 1
    print(f"  PASS {stat_id:11s} {elapsed * 1000.0:7.0f}ms  "
          f"tables={len(result['tables'])} figures={len(result['figures'])} "
          f"bullets={len(result['summary']['bullets'])}")
    return result


# ---------------------------------------------------------------------------
# 契约校验
# ---------------------------------------------------------------------------


def parse_p_cell(value):
    """p 单元格 → None / float / "bad"。允许 "< 1e-4" 与 em dash。"""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "—", "NA", "-"):
        return None
    if text.startswith("<"):
        text = text.lstrip("<").strip()
    try:
        number = float(text)
    except ValueError:
        return "bad"
    if not (0.0 <= number <= 1.0):
        return "bad"
    return number


def parse_num_cell(value) -> str:
    """数值单元格 → "ok" / "empty" / "bad"（用于确认格式化后的数值仍可解析）。"""
    if value is None:
        return "empty"
    if isinstance(value, (int, float)):
        return "ok" if math.isfinite(float(value)) else "bad"
    text = str(value).strip()
    if text in ("", "—", "NA", "-", "是", "否", "ns", "***", "**", "*"):
        return "empty"
    try:
        float(text.rstrip("*").strip())
        return "ok"
    except ValueError:
        return "empty"


def check_tables(stat_id: str, tables) -> None:
    assert isinstance(tables, list), f"{stat_id}: tables 必须是 list"
    assert tables, f"{stat_id}: tables 不能为空"
    for table in tables:
        assert isinstance(table, dict), f"{stat_id}: table 必须是 dict"
        for key in ("id", "title", "columns", "rows"):
            assert key in table, f"{stat_id}/{table.get('id')}: 表缺少键 {key}"
        assert isinstance(table["columns"], list) and table["columns"], \
            f"{stat_id}/{table['id']}: columns 不能为空"
        assert isinstance(table["rows"], list) and table["rows"], \
            f"{stat_id}/{table['id']}: rows 不能为空"
        keys = []
        for column in table["columns"]:
            assert isinstance(column, dict), f"{stat_id}: column 必须是 dict"
            for key in ("key", "label", "type"):
                assert key in column, f"{stat_id}/{table['id']}: 列缺少 {key}"
            assert column["type"] in ALLOWED_TYPES, \
                f"{stat_id}/{table['id']}: 非法列类型 {column['type']}"
            assert str(column["label"]).strip(), f"{stat_id}/{table['id']}: 列标签为空"
            keys.append(column["key"])
        assert len(set(keys)) == len(keys), f"{stat_id}/{table['id']}: 列键重复"
        for row in table["rows"]:
            assert isinstance(row, dict), f"{stat_id}/{table['id']}: 行必须是 dict"
            assert set(row.keys()) == set(keys), \
                f"{stat_id}/{table['id']}: 行列键不一致 {set(row) ^ set(keys)}"
        for column in table["columns"]:
            if column["type"] == "p":
                for row in table["rows"]:
                    parsed = parse_p_cell(row[column["key"]])
                    assert parsed != "bad", (
                        f"{stat_id}/{table['id']}: p 值越界或无意义 "
                        f"{column['key']}={row[column['key']]!r}"
                    )
            elif column["type"] == "num":
                for row in table["rows"]:
                    assert parse_num_cell(row[column["key"]]) != "bad", (
                        f"{stat_id}/{table['id']}: 数值单元格非有限 "
                        f"{column['key']}={row[column['key']]!r}"
                    )


def check_figures(stat_id: str, figures, expect_figures: bool) -> None:
    assert isinstance(figures, list), f"{stat_id}: figures 必须是 list"
    if expect_figures:
        assert figures, f"{stat_id}: 期望生成图但没有 figures"
    for figure in figures:
        assert isinstance(figure, dict)
        for key in ("id", "title", "file"):
            assert key in figure, f"{stat_id}: figure 缺少 {key}"
        assert os.path.isfile(figure["file"]), f"{stat_id}: 图文件不存在 {figure['file']}"
        assert int(figure.get("bytes") or 0) > 0, f"{stat_id}: 图文件为空 {figure['file']}"
        assert int(figure.get("width") or 0) > 0 and int(figure.get("height") or 0) > 0


def check_annotation(stat_id: str, annotation) -> None:
    if annotation is None:
        return
    assert isinstance(annotation, dict), f"{stat_id}: annotation 必须是 dict"
    json.dumps(C.jsonable(annotation), ensure_ascii=False, allow_nan=False)
    if stat_id == "compare":
        for key in ("groups", "pairs", "test"):
            assert key in annotation, f"compare: annotation 缺少 {key}"
        assert isinstance(annotation["groups"], list) and len(annotation["groups"]) >= 2
        assert isinstance(annotation["pairs"], list) and annotation["pairs"]
        for pair in annotation["pairs"]:
            for key in ("a", "b", "p_adj", "stars", "label"):
                assert key in pair, f"compare: pair 缺少 {key}"
            assert 0.0 <= float(pair["p_adj"]) <= 1.0, f"compare: p_adj 越界 {pair}"
            assert pair["stars"] in ("***", "**", "*", "ns"), f"compare: stars 非法 {pair}"
            assert str(pair["label"]).strip(), "compare: label 为空"
        assert isinstance(annotation["test"], str) and annotation["test"].strip()


def check_result(stat_id: str, result, elapsed: float, expect_figures: bool) -> None:
    assert isinstance(result, dict), f"{stat_id}: 返回必须是 dict"
    for key in REQUIRED_KEYS:
        assert key in result, f"{stat_id}: 返回缺少键 {key}"
    assert isinstance(result["title"], str) and result["title"].strip()
    assert isinstance(result["method"], str) and result["method"].strip()
    summary = result["summary"]
    assert isinstance(summary, dict)
    bullets = summary.get("bullets")
    assert isinstance(bullets, list) and bullets, f"{stat_id}: summary.bullets 不能为空"
    for bullet in bullets:
        assert isinstance(bullet, str) and bullet.strip(), f"{stat_id}: bullet 必须是非空字符串"
        assert re.search(r"\d", bullet), f"{stat_id}: bullet 缺少关键数字：{bullet}"
    key_numbers = summary.get("key_numbers")
    assert isinstance(key_numbers, dict), f"{stat_id}: summary.key_numbers 必须是 dict"
    json.dumps(C.jsonable(key_numbers), ensure_ascii=False, allow_nan=False)

    check_tables(stat_id, result["tables"])
    check_figures(stat_id, result["figures"], expect_figures)
    check_annotation(stat_id, result["annotation"])

    report = result["report"]
    assert isinstance(report, str) and report.strip(), f"{stat_id}: report 不能为空"
    assert report.lstrip().startswith("###"), f"{stat_id}: report 应以 ### 标题开头"
    assert "**方法**" in report, f"{stat_id}: report 缺少方法段"

    warnings = result["warnings"]
    assert isinstance(warnings, list), f"{stat_id}: warnings 必须是 list"
    for message in warnings:
        assert isinstance(message, str) and message.strip()

    payload = C.jsonable(result)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert elapsed < MAX_SECONDS, f"{stat_id}: 单次统计 {elapsed:.2f}s 超过 {MAX_SECONDS:.0f}s"


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------


def expect_error(code: str, func, label: str) -> None:
    global ERRORS_RUN
    try:
        func()
    except C.EngineError as exc:
        assert exc.code == code, f"{label}: 期望错误码 {code}，实际 {exc.code}（{exc.message}）"
        assert str(exc.message).strip(), f"{label}: 错误消息为空"
        ERRORS_RUN += 1
        print(f"  PASS error {label:24s} → {exc.code}: {exc.message[:56]}")
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{label}: 期望 EngineError({code})，实际 {type(exc).__name__}: {exc}")
    raise AssertionError(f"{label}: 未抛出任何异常（期望 EngineError({code})）")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def check_catalog() -> None:
    catalog = T.stats_catalog()
    assert isinstance(catalog, list) and len(catalog) == len(REQUIRED_STATS)
    ids = []
    for item in catalog:
        for key in ("id", "title", "group", "desc", "params", "roles"):
            assert key in item, f"catalog 缺少 {key}: {item.get('id')}"
        assert isinstance(item["params"], list)
        for param in item["params"]:
            for key in ("key", "label", "type", "default"):
                assert key in param, f"{item['id']} 参数缺少 {key}"
            assert param["type"] in ("number", "select", "boolean", "text", "columns")
        for role, kind in item["roles"].items():
            assert kind in ("required", "optional"), f"{item['id']}: 角色 {role} 声明非法"
        ids.append(item["id"])
    assert len(set(ids)) == len(ids)
    assert set(REQUIRED_STATS) <= set(ids)
    assert T.stat_meta("compare")["needs"] == "tidy"
    print(f"  PASS catalog  {len(catalog)} 个统计方法")


def main() -> int:
    started_at = time.time()
    print("pl_stats 自测开始")
    print(f"  输出目录：{OUTDIR}")
    warmup_matplotlib()
    check_catalog()

    clinical = S.create_demo("clinical")
    expression = S.create_demo("expression")
    multigroup = make_multigroup()
    paired_ds = make_paired()
    print(f"  demo：clinical={clinical['df'].shape}，expression={expression['df'].shape}")

    # 1 描述统计（多列 + 分组）
    result = run_stat(clinical, "describe", {"columns": ["score", "crp"]},
                      {"x": "group", "y": "score"})
    assert result["tables"][0]["total"] == 4, "describe 应有 2 列 × 2 组 = 4 行"

    # 2 组间比较（2 组 Welch + Mann-Whitney）
    result = run_stat(clinical, "compare", {}, {"x": "group", "y": "score"})
    assert result["annotation"]["groups"] == ["Control", "Treatment"]
    assert len(result["annotation"]["pairs"]) == 1
    assert result["annotation"]["pairs"][0]["p_adj"] < 0.05

    # 3 组间比较（>2 组：ANOVA / Kruskal + 事后两两）
    result = run_stat(multigroup, "compare", {"adjust": "holm"}, {"x": "arm", "y": "value"})
    assert len(result["annotation"]["pairs"]) == 3
    assert result["annotation"]["test"] in ("One-way ANOVA", "Kruskal-Wallis")
    assert result["summary"]["key_numbers"]["k"] == 3

    # 4 组间比较（配对）
    result = run_stat(paired_ds, "compare", {"paired": True}, {"x": "group", "y": "value"})
    assert any("配对" in bullet for bullet in result["summary"]["bullets"])

    # 5 前提检查
    result = run_stat(clinical, "assumption", {}, {"x": "group", "y": "score"})
    assert len(result["tables"]) == 2

    # 6 相关性（tidy 多列 + 热图）
    result = run_stat(clinical, "corr",
                      {"method": "spearman", "plot": True,
                       "columns": ["age", "bmi", "crp", "score"]}, {}, expect_figures=True)
    assert len(result["tables"]) >= 2

    # 7 相关性（matrix 数据按特征）
    result = run_stat(expression, "corr", {"method": "pearson"}, None)
    assert "12 个变量" in result["summary"]["bullets"][0]

    # 8 线性回归（OLS + 分类协变量 + 诊断图）
    result = run_stat(clinical, "regression",
                      {"y": "score", "x": ["age", "crp"], "covariates": ["group"],
                       "plot": True}, None, expect_figures=True)
    assert "R²" in result["summary"]["bullets"][0]

    # 9 logistic 回归（二分类 y 自动切换）
    result = run_stat(clinical, "regression",
                      {"y": "response", "x": ["crp", "age"], "positive": "Yes"}, None)
    assert "logistic" in result["summary"]["bullets"][0]
    assert result["tables"][1]["columns"][5]["key"] == "or", "logistic 表应含 OR 列"

    # 10 卡方 / Fisher（2×2）
    result = run_stat(clinical, "chi2", {"x": "group", "y": "response"}, {"x": "group"})
    assert any("Fisher" in bullet for bullet in result["summary"]["bullets"])

    # 11 正态性（含直方图/QQ 图）
    result = run_stat(clinical, "normality", {"columns": ["score", "crp"], "plot": True},
                      None, expect_figures=True)
    assert len(result["figures"]) == 2

    # 12 ROC
    result = run_stat(clinical, "roc",
                      {"label": "response", "score": "crp", "positive": "Yes",
                       "bootstrap": 300}, None, expect_figures=True)
    numbers = result["summary"]["key_numbers"]
    assert 0.0 <= numbers["auc"] <= 1.0
    assert numbers["n"] == numbers["n_pos"] + numbers["n_neg"]

    # 13 生存分析（KM + log-rank + Cox）
    result = run_stat(clinical, "survival", {"covariates": ["age", "group"]},
                      None, expect_figures=True)
    assert result["summary"]["key_numbers"]["n"] > 0
    assert len(result["tables"]) >= 3, "survival 应有分组/log-rank/Cox 三张表"

    # 14 多重检验校正
    result = run_stat(clinical, "adjust",
                      {"pvalues": [0.001, 0.02, 0.04, 0.2, 0.5], "method": "bh"}, None)
    numbers = result["summary"]["key_numbers"]
    assert numbers["n"] == 5
    assert numbers["n_sig_raw"] == 3 and numbers["n_sig_adjusted"] == 1
    assert numbers["min_p_adjusted"] >= numbers["min_p"]

    # 15 PCA（矩阵数据 + 分组着色）
    result = run_stat(expression, "pca", {"plot": True, "sample_group_col": "group"},
                      None, expect_figures=True)
    numbers = result["summary"]["key_numbers"]
    assert 0.0 < numbers["pc1_var_pct"] <= 100.0
    assert numbers["n_samples"] == 16 and numbers["n_features"] == 12

    # 16 确定性：bootstrap / 检验 p 同输入同输出
    compare_params: dict = {}
    data_first = build_data(clinical, "compare", compare_params, {"x": "group", "y": "score"})
    p_first = T.run("compare", clinical, data_first, compare_params, STYLE,
                    OUTDIR)["summary"]["key_numbers"]["p"]
    data_second = build_data(clinical, "compare", compare_params, {"x": "group", "y": "score"})
    p_second = T.run("compare", clinical, data_second, compare_params, STYLE,
                     OUTDIR)["summary"]["key_numbers"]["p"]
    assert p_first == p_second, "compare 结果不确定"

    roc_params = {"label": "response", "score": "crp", "positive": "Yes", "bootstrap": 200}
    ci_values = []
    for _ in range(2):
        data = build_data(clinical, "roc", roc_params, None)
        numbers = T.run("roc", clinical, data, roc_params, STYLE,
                        OUTDIR)["summary"]["key_numbers"]
        ci_values.append((numbers["auc_low"], numbers["auc_high"]))
    assert ci_values[0] == ci_values[1], "ROC bootstrap 置信区间不确定"
    print("  PASS 确定性      compare p 与 ROC bootstrap CI 两次运行完全一致")

    # 17 错误路径
    expect_error("not_found",
                 lambda: T.run("not-a-stat", clinical, {}, {}, STYLE, OUTDIR),
                 "未知 stat_id")
    expect_error("not_found", lambda: T.stat_meta("not-a-stat"), "stat_meta 未知")

    def missing_role() -> None:
        P.build_data(clinical,
                     {"id": "compare", "needs": "tidy",
                      "roles": {"x": "required", "y": "required"}},
                     {}, {"x": "group"})

    expect_error("bad_request", missing_role, "compare 缺 y 角色")

    def regression_missing_y() -> None:
        params = {"x": ["age"]}
        # roles 传空 dict 会被 build_data 视为“未指定”而自动推荐，这里显式置空
        data = build_data(clinical, "regression", params, {"x": None, "y": None})
        T.run("regression", clinical, data, params, STYLE, OUTDIR)

    expect_error("bad_request", regression_missing_y, "regression 缺 y")

    def regression_bad_column() -> None:
        params = {"y": "not_a_column", "x": ["age"]}
        data = build_data(clinical, "regression", params, {})
        T.run("regression", clinical, data, params, STYLE, OUTDIR)

    expect_error("bad_request", regression_bad_column, "regression 列不存在")

    def chi2_bad_column() -> None:
        params = {"x": "group", "y": "not_a_column"}
        data = build_data(clinical, "chi2", params, {"x": "group"})
        T.run("chi2", clinical, data, params, STYLE, OUTDIR)

    expect_error("bad_request", chi2_bad_column, "chi2 列不存在")

    def survival_on_matrix() -> None:
        params: dict = {}
        data = build_data(expression, "survival", params, None)
        T.run("survival", expression, data, params, STYLE, OUTDIR)

    expect_error("prerequisite", survival_on_matrix, "survival 用矩阵数据")

    def roc_single_class() -> None:
        dataset = make_single_class()
        params = {"label": "label", "score": "score"}
        data = build_data(dataset, "roc", params, None)
        T.run("roc", dataset, data, params, STYLE, OUTDIR)

    expect_error("prerequisite", roc_single_class, "roc 单类别标签")

    def adjust_out_of_range() -> None:
        params = {"pvalues": [0.5, 1.2]}
        data = build_data(clinical, "adjust", params, {})
        T.run("adjust", clinical, data, params, STYLE, OUTDIR)

    expect_error("bad_request", adjust_out_of_range, "adjust p 越界")

    def compare_single_group() -> None:
        dataset = make_single_group()
        params: dict = {}
        data = build_data(dataset, "compare", params, {"x": "group", "y": "value"})
        T.run("compare", dataset, data, params, STYLE, OUTDIR)

    expect_error("prerequisite", compare_single_group, "compare 单组")

    elapsed = time.time() - started_at
    assert STATS_RUN == 15, f"统计运行次数异常：{STATS_RUN}"
    assert ERRORS_RUN == 10, f"错误路径次数异常：{ERRORS_RUN}"
    print(f"PASS pl_stats 自测：{STATS_RUN} 次统计运行 + {ERRORS_RUN} 条错误路径，"
          f"用时 {elapsed:.1f}s，图输出 {OUTDIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL pl_stats 自测")
        sys.exit(1)
