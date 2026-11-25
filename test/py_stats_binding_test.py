#!/usr/bin/env python3
"""统计绑定回归测试（P0 反面用例）。

每个用例对应任务单 §5 里的一条 P0，全部先在旧实现上复现为 FAIL：

  B1  method=student 时主结果必须用 Student 的 p/t/df/CI（不能用 Welch 的）
  B2  Kendall / Spearman 的 P 必须按方法分派到 scipy，不能用 Pearson t 近似
  B3  配对检验必须按显式 ID 对齐；无 ID 必须报错（不再静默按行序）
  B4  配对按 ID 对齐后，乱序输入与有序输入结果完全一致（行序配对会错）
  B5  p 值家族 >1000 时不拒绝、不截断，按整个家族一次性校正并提示
  B6  多组显著性 annotation 必须绑定确切 contrast（能看出比的是哪两组）
  B7  矩阵数据（特征×样本）不能把同列多个特征当生物学重复：默认描述性，
      正式检验必须显式声明独立样本，否则拒绝

运行：
    export MPLBACKEND=Agg MPLCONFIGDIR=$(mktemp -d)
    PYTHONPATH=lib python3 test/py_stats_binding_test.py
"""
from __future__ import annotations

import json
import math
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
from scipy import stats as sps  # noqa: E402

import pl_charts as CH  # noqa: E402
import pl_common as C  # noqa: E402
import pl_prep as P  # noqa: E402
import pl_state as S  # noqa: E402
import pl_stats as T  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="plotlab_binding_")
STYLE = C.resolve_style({"theme": "minimal", "palette": "npg", "dpi": 120})

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL {name} {detail}")
    PASSED.append(name)
    print(f"PASS {name}")


def tmpdir() -> str:
    return tempfile.mkdtemp(prefix="plotlab-binding-")


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


def expect_engine_error(code: str, func, label: str) -> str:
    try:
        func()
    except C.EngineError as exc:
        assert exc.code == code, f"{label}: 期望 {code}，实际 {exc.code}（{exc.message}）"
        return str(exc.message)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{label}: 期望 EngineError({code})，实际 {type(exc).__name__}: {exc}")
    raise AssertionError(f"{label}: 未抛异常（期望 EngineError({code})）")


def rows_of(result: dict, table_id: str) -> list[dict]:
    for table in result.get("tables") or []:
        if table.get("id") == table_id:
            return list(table.get("rows") or [])
    return []


def bullets_text(result: dict) -> str:
    return "\n".join((result.get("summary") or {}).get("bullets") or [])


def two_group_dataset() -> dict:
    """a=[1,2,3,4,5] b=[2,3.5,4,6,7.5]（任务单给的复现输入）。"""
    frame = pd.DataFrame({
        "grp": ["A"] * 5 + ["B"] * 5,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.5, 4.0, 6.0, 7.5],
    })
    return S.build_dataset(frame, "two-group", kind="tidy")


# ---------------------------------------------------------------------------
# B1：Student t 主结果不能串 Welch
# ---------------------------------------------------------------------------


def test_student_primary() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 3.5, 4.0, 6.0, 7.5])
    student = sps.ttest_ind(a, b, equal_var=True)
    welch = sps.ttest_ind(a, b, equal_var=False)
    assert abs(float(student.pvalue) - 0.21841182354726743) < 1e-12

    dataset = two_group_dataset()
    result = run_stat(dataset, "compare", {"method": "student", "bootstrap": 200},
                      {"x": "grp", "y": "value"})
    numbers = result["summary"]["key_numbers"]
    annotation = result["annotation"]

    check("B1 annotation names Student", annotation.get("test") == "Student t-test",
          str(annotation.get("test")))
    check("B1 key p is Student p", abs(float(numbers["p"]) - float(student.pvalue)) < 1e-9,
          f"key p={numbers['p']} student={student.pvalue} welch={welch.pvalue}")
    check("B1 annotation p is Student p",
          abs(float(annotation["pairs"][0]["p_adj"]) - float(student.pvalue)) < 1e-9,
          str(annotation["pairs"][0]))

    # 检验表里主检验行 = Student 的统计量
    primary = [row for row in rows_of(result, "compare_tests")
               if "主检验" in str(row.get("test"))]
    check("B1 primary test row exists", len(primary) == 1, str(primary))
    row = primary[0]
    check("B1 primary row is Student", "Student" in str(row["test"]), str(row["test"]))
    check("B1 primary row statistic", abs(float(row["statistic"]) - float(student.statistic)) < 1e-3,
          f"{row['statistic']} vs {student.statistic}")
    check("B1 primary row p", abs(float(row["p"]) - float(student.pvalue)) < 1e-4,
          f"{row['p']} vs {student.pvalue}")
    check("B1 primary row df = n-2", abs(float(row["df"]) - 8.0) < 1e-9, str(row["df"]))

    # 效应量 / CI 也必须来自 Student（等方差 pooled 版本）
    effects = {str(row.get("metric")): row.get("value") for row in rows_of(result, "compare_effects")}
    na, nb = a.size, b.size
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    diff = float(np.mean(a) - np.mean(b))
    se = pooled * math.sqrt(1.0 / na + 1.0 / nb)
    tcrit = float(sps.t.ppf(0.975, na + nb - 2))
    ci_low, ci_high = diff - tcrit * se, diff + tcrit * se
    d_value = diff / pooled
    low_row = next((v for k, v in effects.items() if k.startswith("95%CI 下限")), None)
    high_row = next((v for k, v in effects.items() if k.startswith("95%CI 上限")), None)
    check("B1 CI rows present", low_row is not None and high_row is not None, str(effects))
    check("B1 CI lower is Student", abs(float(low_row) - ci_low) < 1e-2,
          f"{low_row} vs {ci_low}")
    check("B1 CI upper is Student", abs(float(high_row) - ci_high) < 1e-2,
          f"{high_row} vs {ci_high}")
    d_row = effects.get("Cohen's d")
    check("B1 Cohen d is Student", abs(float(d_row) - d_value) < 1e-2, f"{d_row} vs {d_value}")

    text = bullets_text(result)
    check("B1 bullets mention Student t", "Student t=" in text, text)
    check("B1 bullets do not claim Welch as primary", "Welch t=" not in text, text)
    check("B1 bullets carry Student p", "p=0.2184" in text or "0.218" in text, text)
    check("B1 method text is Student", "Student" in str(result["method"]), str(result["method"]))

    # Welch 仍作为参考行存在
    tests = rows_of(result, "compare_tests")
    check("B1 Welch kept as reference",
          any("Welch" in str(r["test"]) and "参考" in str(r["test"]) for r in tests),
          str([r["test"] for r in tests]))


# ---------------------------------------------------------------------------
# B2：Kendall / Spearman 的 p 必须按方法分派
# ---------------------------------------------------------------------------


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> float:
    """n<=6 的精确 Spearman 置换 p（双侧 |rho|）。"""
    from itertools import permutations

    rx, ry = sps.rankdata(x), sps.rankdata(y)

    def rho(left: np.ndarray, right: np.ndarray) -> float:
        a = left - left.mean()
        b = right - right.mean()
        return float((a * b).sum() / math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum())))

    observed = abs(rho(rx, ry))
    hits = 0
    total = 0
    for order in permutations(range(rx.size)):
        total += 1
        if abs(rho(rx, ry[list(order)])) >= observed - 1e-12:
            hits += 1
    return hits / total


def corr_matrix_dataset() -> dict:
    """3 个特征 × 4 个样本，四个样本上完全同序（Kendall tau = 1）。"""
    frame = pd.DataFrame({
        "gene": ["g1", "g2", "g3"],
        "s1": [1.0, 10.0, 100.0],
        "s2": [2.0, 20.0, 200.0],
        "s3": [3.0, 30.0, 300.0],
        "s4": [4.0, 40.0, 400.0],
    })
    return S.build_dataset(frame, "corr-matrix", kind="matrix")


def render_corr(dataset: dict, method: str) -> dict:
    meta = CH.CHART_INDEX["corr"]
    params = {"method": method}
    data = P.build_data(dataset, meta, params, P.recommend_roles(dataset, meta))
    return CH.render("corr", data, params, STYLE, OUTDIR)


def test_corr_pvalue() -> None:
    dataset = corr_matrix_dataset()
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    kendall_p = float(sps.kendalltau(x, y).pvalue)
    # scipy>=1.15 的 spearmanr 对小样本 rho=±1 用 t 渐近近似会给 p=0（零分布被压扁）；
    # 参考值改为精确置换 p（枚举 4! 个秩排列，双侧 |rho|），与正式用例一致。
    spearman_p = exact_spearman_p(x, y)
    pearson_p = float(sps.pearsonr(x, y).pvalue)
    check("B2 reference kendall p ~0.08333", abs(kendall_p - 0.0833333) < 1e-4, str(kendall_p))

    for method, expected in (("kendall", kendall_p), ("spearman", spearman_p)):
        result = render_corr(dataset, method)
        rows = [row for row in rows_of(result, "corr_pairs")
                if {row.get("var1"), row.get("var2")} == {"g1", "g2"}]
        check(f"B2 {method} pair row", len(rows) == 1, str(rows))
        p_value = float(rows[0]["p"])
        check(f"B2 {method} p matches scipy", abs(p_value - expected) < 5e-3,
              f"{method} p={p_value} scipy={expected}")
        check(f"B2 {method} p not zero", p_value > 1e-6, str(p_value))

    result = render_corr(dataset, "pearson")
    rows = [row for row in rows_of(result, "corr_pairs")
            if {row.get("var1"), row.get("var2")} == {"g1", "g2"}]
    check("B2 pearson kept", len(rows) == 1, str(rows))
    check("B2 pearson p matches scipy",
          abs(float(rows[0]["p"]) - pearson_p) < 5e-3,
          f"{rows[0]['p']} vs {pearson_p}")


# ---------------------------------------------------------------------------
# B3/B4：配对必须按 ID
# ---------------------------------------------------------------------------


def paired_dataset(shuffle: bool = False) -> dict:
    ids = [f"S{index:02d}" for index in range(1, 11)]
    before = np.arange(1.0, 11.0)
    after = before + np.linspace(0.5, 2.0, 10)
    frame = pd.DataFrame({
        "group": ["Before"] * 10 + ["After"] * 10,
        "value": np.concatenate([before, after]),
        "subject": ids + ids,
    })
    if shuffle:
        order = np.random.default_rng(7).permutation(len(frame))
        frame = frame.iloc[order].reset_index(drop=True)
    return S.build_dataset(frame, "paired", kind="tidy")


def paired_dataset_no_id() -> dict:
    frame = pd.DataFrame({
        "group": ["Before"] * 8 + ["After"] * 8,
        "value": list(np.arange(1.0, 9.0)) + list(np.arange(1.0, 9.0) + 1.0),
    })
    return S.build_dataset(frame, "paired-no-id", kind="tidy")


def test_pairing() -> None:
    # 无 ID 却要求配对 → 必须报错，不能静默按行序
    message = expect_engine_error(
        "bad_request",
        lambda: run_stat(paired_dataset_no_id(), "compare", {"paired": True},
                         {"x": "group", "y": "value"}),
        "B3 paired without id",
    )
    check("B3 error mentions id", "ID" in message or "id" in message.lower(), message)

    ordered = run_stat(paired_dataset(False), "compare",
                       {"paired": True, "paired_id": "subject"}, {"x": "group", "y": "value"})
    numbers = ordered["summary"]["key_numbers"]
    check("B3 pairing_basis id", numbers.get("pairing_basis") == "id", str(numbers))
    check("B3 annotation pairing_basis id",
          (ordered.get("annotation") or {}).get("pairing_basis") == "id",
          str(ordered.get("annotation")))
    check("B3 bullets say id pairing", "ID" in bullets_text(ordered), bullets_text(ordered))
    check("B3 n_pairs recorded", int(numbers.get("n_pairs") or 0) == 10, str(numbers))

    # 行序打乱后按 ID 配对：结果必须完全一致（行序配对会错）
    shuffled = run_stat(paired_dataset(True), "compare",
                        {"paired": True, "paired_id": "subject"}, {"x": "group", "y": "value"})
    shuffled_numbers = shuffled["summary"]["key_numbers"]
    check("B4 shuffled p identical",
          abs(float(shuffled_numbers["p"]) - float(numbers["p"])) < 1e-12,
          f"{shuffled_numbers['p']} vs {numbers['p']}")
    effects_ordered = {str(r["metric"]): r["value"] for r in rows_of(ordered, "compare_effects")}
    effects_shuffled = {str(r["metric"]): r["value"] for r in rows_of(shuffled, "compare_effects")}
    diff_key = next(k for k in effects_ordered if "配对差" in k)
    check("B4 shuffled mean_diff identical",
          abs(float(effects_ordered[diff_key]) - float(effects_shuffled[diff_key])) < 1e-12,
          f"{effects_ordered[diff_key]} vs {effects_shuffled[diff_key]}")

    # 显式逃生门：row_order 必须三处留痕
    escape = run_stat(paired_dataset_no_id(), "compare",
                      {"paired": True, "pairing": "row_order"}, {"x": "group", "y": "value"})
    escape_numbers = escape["summary"]["key_numbers"]
    check("B4 row_order recorded in key_numbers",
          escape_numbers.get("pairing_basis") == "row_order", str(escape_numbers))
    check("B4 row_order recorded in annotation",
          (escape.get("annotation") or {}).get("pairing_basis") == "row_order",
          str(escape.get("annotation")))
    check("B4 row_order warns",
          any("行序" in str(w) for w in escape.get("warnings") or []),
          str(escape.get("warnings")))

    # ID 集合不匹配 / 同侧 ID 重复 → 报错
    broken = paired_dataset(False)
    frame = broken["df"].copy()
    frame.loc[frame.index[-1], "subject"] = "SXX"
    broken_sets = S.build_dataset(frame, "paired-broken", kind="tidy")
    expect_engine_error(
        "bad_request",
        lambda: run_stat(broken_sets, "compare", {"paired": True, "paired_id": "subject"},
                         {"x": "group", "y": "value"}),
        "B4 id set mismatch",
    )
    dup = paired_dataset(False)
    frame_dup = dup["df"].copy()
    frame_dup.loc[frame_dup.index[1], "subject"] = frame_dup.loc[frame_dup.index[0], "subject"]
    dup_ds = S.build_dataset(frame_dup, "paired-dup", kind="tidy")
    expect_engine_error(
        "bad_request",
        lambda: run_stat(dup_ds, "compare", {"paired": True, "paired_id": "subject"},
                         {"x": "group", "y": "value"}),
        "B4 duplicate id",
    )


# ---------------------------------------------------------------------------
# B5：p 值家族 >1000 不拒绝、不截断
# ---------------------------------------------------------------------------


def test_large_family() -> None:
    values = [min(1.0, 1e-4 * (index + 1)) for index in range(1500)]
    result = run_stat(S.create_demo("clinical"), "adjust",
                      {"pvalues": values, "method": "bh"}, None)
    numbers = result["summary"]["key_numbers"]
    check("B5 all p values kept", int(numbers["n"]) == 1500, str(numbers.get("n")))
    expected = C.adjust_pvalues(values, "bh")
    low = float(numbers["min_p_adjusted"])
    check("B5 adjusted on full family", abs(low - min(expected)) < 1e-12,
          f"{low} vs {min(expected)}")
    text = " ".join(str(w) for w in result.get("warnings") or [])
    check("B5 warns about large family", ("1000" in text or "家族" in text or "分批" in text), text)
    table = result.get("tables")[0]
    check("B5 table total recorded", int(table.get("total") or 0) == 1500, str(table.get("total")))
    check("B5 table truncated explicitly", bool(table.get("truncated")), str(table))


# ---------------------------------------------------------------------------
# B6：多组显著性必须绑定确切 contrast
# ---------------------------------------------------------------------------


def multigroup_dataset() -> dict:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "arm": ["A"] * 12 + ["B"] * 12 + ["C"] * 12,
        "value": np.concatenate([rng.normal(10, 1.5, 12), rng.normal(13, 1.5, 12),
                                 rng.normal(16, 1.5, 12)]).round(3),
    })
    return S.build_dataset(frame, "multi", kind="tidy")


def test_contrast_binding() -> None:
    dataset = multigroup_dataset()
    result = run_stat(dataset, "compare", {"method": "anova"}, {"x": "arm", "y": "value"})
    annotation = result["annotation"]
    check("B6 annotation has factor", annotation.get("factor") in ("x", "__x", "arm"),
          str(annotation.get("factor")))
    pairs = annotation.get("pairs") or []
    check("B6 three contrasts", len(pairs) == 3, str(pairs))
    for pair in pairs:
        check("B6 pair has a/b", bool(pair.get("a")) and bool(pair.get("b")), str(pair))
        check("B6 pair has contrast text",
              str(pair.get("contrast") or "") == f"{pair['a']} vs {pair['b']}",
              str(pair.get("contrast")))
        check("B6 pair has test name", bool(pair.get("test")), str(pair))

    # 图上的 stats 载荷同样绑定 contrast（多面板/多组时能看出比的是哪两组）
    meta = CH.CHART_INDEX["box"]
    data = P.build_data(dataset, meta, {"stat_annot": True},
                        P.recommend_roles(dataset, meta))
    rendered = CH.render("box", data, {"stat_annot": True}, STYLE, OUTDIR)
    stats_payload = rendered.get("stats") or {}
    check("B6 chart stats has factor", bool(stats_payload.get("factor")), str(stats_payload))
    contrasts = stats_payload.get("contrasts") or []
    check("B6 chart contrasts bound", len(contrasts) == 3, str(contrasts))
    for item in contrasts:
        check("B6 chart contrast id", bool(item.get("contrast_id")), str(item))
        check("B6 chart contrast a/b", bool(item.get("a")) and bool(item.get("b")), str(item))
    annotations = rendered.get("annotations") or {}
    check("B6 chart annotations contrasts",
          len(annotations.get("contrasts") or []) == 3, str(annotations))


# ---------------------------------------------------------------------------
# B7：矩阵数据的生物学 n
# ---------------------------------------------------------------------------


def matrix_n_dataset() -> dict:
    """5 基因 × 4 样本；样本分 A/A/B/B 两组。旧实现会把同列 5 个基因当 n=5。"""
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


def test_matrix_biological_n() -> None:
    dataset = matrix_n_dataset()
    message = expect_engine_error(
        "prerequisite",
        lambda: run_stat(dataset, "compare", {}, None),
        "B7 matrix compare without independent samples",
    )
    check("B7 error mentions biological replicates",
          ("生物学重复" in message or "独立" in message or "技术重复" in message), message)

    # 描述统计仍可用，但必须标注 n 是特征数（技术重复）
    described = run_stat(dataset, "describe", {}, None)
    text = bullets_text(described) + " " + " ".join(str(w) for w in described.get("warnings") or [])
    check("B7 describe works", bool(described.get("tables")), str(described.keys()))
    check("B7 describe labels technical n",
          ("技术重复" in text or "特征数" in text or "生物学重复" in text), text)

    # 显式声明独立样本后才允许正式检验，并必须留痕
    declared = run_stat(dataset, "compare",
                        {"independent_samples": "sample"}, None)
    numbers = declared["summary"]["key_numbers"]
    check("B7 declared independent samples recorded",
          numbers.get("sample_unit") in ("sample", "independent"), str(numbers))
    check("B7 declared warns about technical n",
          any("技术" in str(w) or "生物学" in str(w) or "独立" in str(w)
              for w in declared.get("warnings") or []),
          str(declared.get("warnings")))

    # 汇总图（box/violin）默认只做描述性：跳过显著性检验并明确提示
    meta = CH.CHART_INDEX["box"]
    data = P.build_data(dataset, meta, {}, P.recommend_roles(dataset, meta))
    check("B7 build_data marks melted matrix", data.get("melted") is True, str(data.get("melted")))
    rendered = CH.render("box", data, {"stat_annot": True}, STYLE, OUTDIR)
    check("B7 chart skips inference", rendered.get("stats") in (None, {}),
          str(rendered.get("stats")))
    check("B7 chart warns descriptive only",
          any("描述" in str(w) or "技术重复" in str(w) or "生物学" in str(w)
              for w in rendered.get("warnings") or []),
          str(rendered.get("warnings")))


# ---------------------------------------------------------------------------


TESTS = [
    ("B1 student primary", test_student_primary),
    ("B2 corr pvalue", test_corr_pvalue),
    ("B3/B4 pairing", test_pairing),
    ("B5 large p family", test_large_family),
    ("B6 contrast binding", test_contrast_binding),
    ("B7 matrix biological n", test_matrix_biological_n),
]


def main() -> int:
    print(f"统计绑定回归测试（输出 {OUTDIR}）")
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