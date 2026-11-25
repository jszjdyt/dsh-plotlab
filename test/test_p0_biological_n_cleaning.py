#!/usr/bin/env python3
"""P0 生物学 n 与清洗回归测试（正式命名）。

对应任务单 §5「P0 生物学 n 与清洗」：
  C1 矩阵融化视图必须标记；矩阵汇总图默认描述性并跳过推断；
     compare 默认拒绝矩阵数据，除非显式 independent_samples="sample"
  C2 parse_need 词触发做否定处理（「不删除重复」≠ 去重）并固定清洗优先级
  C3 自动清洗保护端点/ID/分组列（不删除/不改类型/不填补/不缩尾）
  C4 缺失按列独立处理；clean 支持 preview/dry_run 不落盘

运行：
    export MPLBACKEND=Agg MPLCONFIGDIR=$(mktemp -d)
    PYTHONPATH=lib python3 test/test_p0_biological_n_cleaning.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="plotlab_mpl_")

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pl_charts as CH  # noqa: E402
import pl_clean  # noqa: E402
import pl_common as C  # noqa: E402
import pl_prep as P  # noqa: E402
import pl_registry  # noqa: E402
import pl_state as S  # noqa: E402
import pl_stats as T  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="plotlab_bio_n_")
STYLE = C.resolve_style({"theme": "minimal", "palette": "npg", "dpi": 120})

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL {name} {detail}")
    PASSED.append(name)
    print(f"PASS {name}")


def expect_engine_error(code: str, func, label: str) -> str:
    try:
        func()
    except C.EngineError as exc:
        assert exc.code == code, f"{label}: 期望 {code}，实际 {exc.code}（{exc.message}）"
        return str(exc.message)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{label}: 期望 EngineError({code})，实际 {type(exc).__name__}: {exc}")
    raise AssertionError(f"{label}: 未抛异常（期望 EngineError({code})）")


def run_stat(dataset: dict, stat_id: str, params: dict | None = None,
             roles: dict | None = None) -> dict:
    params = dict(params or {})
    meta = T.stat_meta(stat_id)
    spec = {"id": stat_id, "needs": meta.get("needs") or "tidy",
            "roles": meta.get("roles") or {}}
    if roles is None:
        roles = P.recommend_roles(dataset, spec)
    data = P.build_data(dataset, spec, params, roles)
    return T.run(stat_id, dataset, data, params, STYLE, OUTDIR)


def bullets_text(result: dict) -> str:
    return "\n".join((result.get("summary") or {}).get("bullets") or [])


# ---------------------------------------------------------------------------
# C1：矩阵融化视图的生物学 n
# ---------------------------------------------------------------------------


def matrix_n_dataset() -> dict:
    """5 基因 × 4 样本；样本分 A/A/B/B 两组。旧实现会把同列 5 个基因当 n=5 的样本组。"""
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({
        "gene": [f"g{index}" for index in range(1, 6)],
        "s1": rng.normal(5.0, 0.4, 5).round(3),
        "s2": rng.normal(5.2, 0.4, 5).round(3),
        "s3": rng.normal(6.4, 0.4, 5).round(3),
        "s4": rng.normal(6.6, 0.4, 5).round(3),
    })
    info = pd.DataFrame({"sample": ["s1", "s2", "s3", "s4"], "group": ["A", "A", "B", "B"]})
    dataset = S.build_dataset(frame, "matrix-n", kind="matrix", sample_info=info)
    dataset["group_col"] = "group"
    return dataset


def test_matrix_melted_view() -> None:
    dataset = matrix_n_dataset()
    meta = CH.CHART_INDEX["box"]
    data = P.build_data(dataset, meta, {}, P.recommend_roles(dataset, meta))
    check("C1 build_data marks melted", data.get("melted") is True, str(data.get("melted")))
    check("C1 build_data view label", data.get("view") == "matrix_melted", str(data.get("view")))
    notes = " ".join(str(note) for note in data.get("notes") or [])
    check("C1 note says technical replicates", "技术重复" in notes, notes)

    # compare 默认拒绝：不能把同一生物样本的多个特征值当独立重复
    message = expect_engine_error(
        "prerequisite",
        lambda: run_stat(dataset, "compare", {}, None),
        "C1 compare on melted matrix",
    )
    check("C1 error explains biological n",
          any(token in message for token in ("生物学重复", "独立", "技术重复")), message)
    check("C1 error offers opt-in", "independent_samples" in message, message)

    # 显式声明独立样本才允许，并且必须留痕
    declared = run_stat(dataset, "compare", {"independent_samples": "sample"}, None)
    numbers = declared["summary"]["key_numbers"]
    check("C1 declared sample_unit recorded", numbers.get("sample_unit") == "sample", str(numbers))
    # 声明后按样本聚合：每组 n=2（样本数），绝不是 5（特征数/技术重复）
    check("C1 declared n is sample count", numbers.get("n_a") == 2 and numbers.get("n_b") == 2,
          str(numbers))
    check("C1 declared total n is sample count", numbers.get("n") == 4, str(numbers))
    check("C1 declared keeps technical-n warning",
          any(token in str(w) for w in declared.get("warnings") or []
              for token in ("技术", "生物学", "独立")),
          str(declared.get("warnings")))

    # 描述统计可用，但 n 必须标成技术重复/特征数
    described = run_stat(dataset, "describe", {}, None)
    text = bullets_text(described) + " " + " ".join(str(w) for w in described.get("warnings") or [])
    check("C1 describe works", bool(described.get("tables")), str(described.keys()))
    check("C1 describe labels technical n",
          any(token in text for token in ("技术重复", "特征数", "生物学重复")), text)

    # 汇总图默认描述性：跳过推断并明确提示
    rendered = CH.render("box", data, {"stat_annot": True}, STYLE, OUTDIR)
    check("C1 chart skips inference", rendered.get("stats") in (None, {}), str(rendered.get("stats")))
    check("C1 chart warns descriptive only",
          any(token in str(w) for w in rendered.get("warnings") or []
              for token in ("描述", "技术重复", "生物学")),
          str(rendered.get("warnings")))

    # 显式声明后才允许在图上做推断
    rendered_optin = CH.render("box", data, {"stat_annot": True, "independent_samples": "sample"},
                               STYLE, OUTDIR)
    check("C1 opt-in chart runs inference", bool(rendered_optin.get("stats")),
          str(rendered_optin.get("stats")))


# ---------------------------------------------------------------------------
# C2：parse_need 否定处理与优先级
# ---------------------------------------------------------------------------


def clinical_dataset() -> dict:
    return S.create_demo("clinical")


def clean_ops(plan: dict) -> list[str]:
    return [str(step.get("op")) for step in plan.get("cleaning") or []]


def test_parse_need_negation() -> None:
    dataset = clinical_dataset()

    negated = pl_registry.parse_need("不删除重复，按 group 分组画箱线图比较 score", dataset)
    check("C2 negated dedupe not planned", "drop_duplicates" not in clean_ops(negated),
          str(clean_ops(negated)))
    check("C2 negation recorded in rationale",
          any("否定" in str(item) for item in negated.get("rationale") or []),
          str(negated.get("rationale")))

    negated2 = pl_registry.parse_need("不要删除重复", dataset)
    check("C2 不要 dedupe not planned", "drop_duplicates" not in clean_ops(negated2),
          str(clean_ops(negated2)))

    positive = pl_registry.parse_need("删除重复行，按 group 分组画箱线图", dataset)
    check("C2 positive dedupe planned", "drop_duplicates" in clean_ops(positive),
          str(clean_ops(positive)))

    dirty = S.build_dataset(pd.DataFrame({
        "group": ["A", "A", "B", "B", "A", "B"],
        "score": [1.0, 2.0, np.nan, 4.0, 1.0, 6.0],
        "crp": [1.0, 2.0, 3.0, 4.0, 1.0, 6.0],
    }), "dirty", kind="tidy")
    raw = pl_registry.parse_need("去掉缺失值并删除重复", dirty)
    ops = clean_ops(raw)
    check("C2 去掉 is positive (missing planned)", any(op in ("impute", "dropna") for op in ops), str(ops))
    check("C2 去掉 is positive (dedupe planned)", "drop_duplicates" in ops, str(ops))

    negated_chart = pl_registry.parse_need("不画箱线图，画柱状图", dataset)
    check("C2 negated chart keyword skipped", negated_chart["chart"] == "bar",
          str(negated_chart["chart"]))

    negated_stat = pl_registry.parse_need("不做显著性检验，画 score 直方图", dataset)
    check("C2 negated stat keyword skipped", "compare" not in (negated_stat.get("stats") or []),
          str(negated_stat.get("stats")))

    # 固定优先级：duplicate 在 missing 之前，与文本顺序无关
    priorities = pl_registry.parse_need("先处理缺失，再删除重复", dataset)
    ops = clean_ops(priorities)
    if "drop_duplicates" in ops and any(op in ("impute", "dropna") for op in ops):
        first_missing = min(index for index, op in enumerate(ops) if op in ("impute", "dropna"))
        check("C2 fixed priority dedupe before missing",
              ops.index("drop_duplicates") < first_missing, str(ops))
    else:
        check("C2 fixed priority plan built", bool(ops), str(ops))


# ---------------------------------------------------------------------------
# C3：自动清洗保护端点 / ID / 分组列
# ---------------------------------------------------------------------------


def protected_dataset() -> dict:
    frame = pd.DataFrame({
        "sample_id": [f"S{index:02d}" for index in range(1, 13)],
        "group": ["A"] * 6 + ["B"] * 6,
        "response": ["yes", "no"] * 6,
        "score": list(np.linspace(1, 5, 11)) + [50.0],
        "crp": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "batch": ["b1"] * 12,
        "survival_months": [10.0] * 11 + [400.0],
    })
    return S.build_dataset(frame, "protect", kind="tidy")


def _step_columns(step: dict) -> set[str]:
    params = step.get("params") or {}
    columns = params.get("columns")
    if isinstance(columns, dict):
        return {str(key) for key in columns}
    if isinstance(columns, str):
        return {columns}
    return {str(item) for item in columns or []}


def test_clean_protection() -> None:
    dataset = protected_dataset()
    protected = pl_clean.protected_columns(dataset)
    for column in ("sample_id", "group", "response", "survival_months"):
        check(f"C3 {column} protected", column in protected, str(sorted(protected)))

    before = dataset["df"].copy()
    plan = pl_clean.auto_plan(dataset)
    dropped = [step for step in plan if str(step.get("op")) in ("drop_columns", "drop_empty")]
    for step in dropped:
        check(f"C3 protected cols not dropped ({step.get('op')})",
              not (_step_columns(step) & protected), str(step))
    for step in plan:
        if str(step.get("op")) in ("impute", "outlier", "transform"):
            check(f"C3 protected cols not modified ({step.get('op')})",
                  not (_step_columns(step) & protected), str(step))

    report = pl_clean.apply_clean(dataset, plan=plan, auto=False)
    check("C3 clean applied", "applied" in report and "n_rows" in report, str(report.keys()))
    after = dataset["df"]
    for column in ("sample_id", "group", "response", "survival_months"):
        check(f"C3 {column} still exists", column in after.columns, str(after.columns))
        check(f"C3 {column} values untouched",
              before[column].reset_index(drop=True).astype(str).tolist()
              == after[column].reset_index(drop=True).astype(str).tolist(),
              column)
    # 非保护列的清洗效果仍然生效
    check("C3 outlier column winsorized", float(after["score"].max()) < 50.0,
          str(after["score"].max()))
    check("C3 crp imputed", int(after["crp"].isna().sum()) == 0, str(after["crp"].tolist()))

    # 常量分组列也不能被自动删除
    frame = pd.DataFrame({
        "group": ["A"] * 8,
        "score": np.linspace(1, 2, 8),
        "const_metric": [1.0] * 8,
    })
    constant_group = S.build_dataset(frame, "const-group", kind="tidy")
    constants = [step for step in pl_clean.auto_plan(constant_group)
                 if str(step.get("op")) == "drop_columns"]
    assert constants, "应当有常量列删除步骤"
    check("C3 constant group column not dropped", "group" not in _step_columns(constants[0]),
          str(constants[0]))
    check("C3 constant metric dropped", "const_metric" in _step_columns(constants[0]),
          str(constants[0]))


# ---------------------------------------------------------------------------
# C4：dry_run 不落盘 + 缺失按列独立处理
# ---------------------------------------------------------------------------


def test_clean_dry_run() -> None:
    state = pl_registry.WorkerState(tempfile.mkdtemp(prefix="plotlab-clean-dry-"))
    state.op_demo({"name": "clinical"})
    dataset = state.get()
    df_before = dataset["df"].copy()
    cleaning_before = len(dataset.get("cleaning") or [])
    plan = [{"op": "drop_duplicates", "params": {}}]

    preview = state.op_clean({"plan": plan, "preview": True})
    check("C4 preview flags dry_run", preview.get("dry_run") is True, str(preview.get("dry_run")))
    check("C4 preview computes plan", bool(preview.get("applied")), str(preview.get("applied")))
    check("C4 preview keeps data", dataset["df"].equals(df_before))
    check("C4 preview keeps cleaning log",
          len(dataset.get("cleaning") or []) == cleaning_before, str(len(dataset.get("cleaning") or [])))
    check("C4 preview reports rows before/after",
          int(preview.get("n_rows_before")) == int(len(df_before)), str(preview.get("n_rows_before")))

    auto_preview = state.op_clean({"preview": True, "auto": True})
    check("C4 auto preview returns plan", isinstance(auto_preview.get("plan"), list),
          str(type(auto_preview.get("plan"))))
    check("C4 auto preview keeps data", dataset["df"].equals(df_before))

    applied = state.op_clean({"plan": plan})
    check("C4 real clean flags not dry_run", applied.get("dry_run") is False, str(applied.get("dry_run")))
    check("C4 real clean logs step", len(dataset.get("cleaning") or []) == cleaning_before + 1,
          str(len(dataset.get("cleaning") or [])))
    check("C4 real clean updates data", len(dataset["df"]) == int(applied["n_rows"]))


def missing_dataset() -> dict:
    frame = pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 5,
        "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, None],
        "sex": ["F", "M", "F", "M", "F", None, "F", "M", None, "M"],
        "response": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", None],
        "event": [1] * 10,
    })
    return S.build_dataset(frame, "missing-per-column", kind="tidy")


def test_missing_per_column() -> None:
    dataset = missing_dataset()
    plan = pl_clean.auto_plan(dataset)
    impute = next((step for step in plan if str(step.get("op")) == "impute"), None)
    dropna = next((step for step in plan if str(step.get("op")) == "dropna"), None)
    check("C4 numeric missing imputed", impute is not None, str(plan))
    check("C4 numeric impute targets score", _step_columns(impute) == {"score"}, str(impute))
    check("C4 categorical missing dropped per column", dropna is not None, str(plan))
    check("C4 categorical dropna targets sex", "sex" in _step_columns(dropna), str(dropna))
    protected = pl_clean.protected_columns(dataset)
    check("C4 endpoint protected", "response" in protected, str(sorted(protected)))
    for step in plan:
        if str(step.get("op")) in ("impute", "outlier", "transform"):
            check("C4 endpoint never filled/transformed by auto",
                  "response" not in _step_columns(step), str(step))
    endpoint_step = next((step for step in plan
                          if str(step.get("op")) == "dropna"
                          and "response" in _step_columns(step)), None)
    check("C4 endpoint missing handled by dropping rows (not filling)",
          endpoint_step is not None and "不填补" in str(endpoint_step.get("reason")),
          str(endpoint_step))

    before = len(dataset["df"])
    pl_clean.apply_clean(dataset, plan=plan, auto=False)
    after = dataset["df"]
    check("C4 rows dropped for categorical/missing columns", len(after) < before,
          f"{before} -> {len(after)}")
    check("C4 score fully imputed", int(after["score"].isna().sum()) == 0, str(after["score"].tolist()))
    check("C4 sex fully present", int(after["sex"].isna().sum()) == 0, str(after["sex"].tolist()))
    check("C4 endpoint not fabricated (rows dropped, no fill)",
          int(after["response"].isna().sum()) == 0 and len(after) < before,
          f"{after['response'].tolist()} rows={len(after)}")


TESTS = [
    ("C1 matrix melted view", test_matrix_melted_view),
    ("C2 parse_need negation", test_parse_need_negation),
    ("C3 clean protection", test_clean_protection),
    ("C4 clean dry_run/missing", test_clean_dry_run),
    ("C4 missing per column", test_missing_per_column),
]


def main() -> int:
    print(f"生物学 n 与清洗回归测试（输出 {OUTDIR}）")
    failures = 0
    for name, func in TESTS:
        try:
            func()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {name}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
            traceback.print_exc(limit=2)
    if failures:
        print(f"\nFAILED：{failures}/{len(TESTS)} 组失败；通过 {len(PASSED)} 条断言")
        return 1
    print(f"\nALL PASS（{len(PASSED)} 条断言）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
