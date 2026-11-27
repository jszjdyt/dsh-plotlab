#!/usr/bin/env python3
"""P0 回归：Kendall P 值口径 + 导出绑定 figure_run_id（两处统计/溯源缺陷）。

对应任务单两条缺陷，全部先在旧实现上复现为 FAIL：

K  Kendall 双侧 P 不能套 Pearson 的 t 近似：
     `_corr_pvalue(tau, n, method="kendall")` 必须「小 n 精确置换 / 大 n 正态渐近」。
     旧实现 t=r*sqrt((n-2)/(1-r^2)) 在 tau=1 时分母为 0 → nan；
     正确值 tau=1、n=4 时 p = 2/24 = 0.08333（scipy.stats.kendalltau 精确口径）。
K1 tau=1,n=4 → 0.08333（有限值，不是 nan）
K2 n=3..8 与 scipy 精确置换口径逐位一致
K3 n>8 与 scipy 渐近口径一致
K4 退化输入（tau=±1、n<3、nan）也返回有限值
K5 向量接口向后兼容：Pearson/Spearman/Kendall 行为不变

E  导出必须绑定并校验 figure_run_id，不能串图：
     「渲染 A → 请求导出 B」要么按请求重画成 B，要么报「图形与导出请求不一致」，
     绝不能静默把 A 的 PNG 导出。
E1 直接调用导出层：A=box、请求 B=bar → 重画成 bar（字节不同、run id 不同）
E2 无法重画（无数据/无渲染器）→ 抛 EngineError，消息含「图形与导出请求不一致」，不落盘
E3 请求与 A 一致 → 复用 A（字节相同、figure_run_id 相同）
E4 worker 路径：plot box → export protocol(bar) → 协议里的 chart/图/run id 都是 bar
E5 清洗数据后同请求导出 → 数据指纹失效检测，必须重画新 run
E6 显式按旧 figure_run_id 归档回查 → 固定旧版本，不得被重画回最新
E7 导出 artifacts（payload / files / protocol.json）记录 figure_run_id

运行：
    export MPLBACKEND=Agg MPLCONFIGDIR=$(mktemp -d)
    PYTHONPATH=lib python3 test/test_p0_kendall_export_binding.py
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
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="plotlab_p0_kx_mpl_")

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

import pl_charts as CH  # noqa: E402
import pl_codegen  # noqa: E402
import pl_common as C  # noqa: E402
import pl_export  # noqa: E402
import pl_prep as P  # noqa: E402
import pl_registry  # noqa: E402
import pl_state as S  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="plotlab_p0_kx_out_")
STYLE = C.resolve_style({"theme": "minimal", "palette": "npg", "dpi": 110,
                         "fig_width": 3.2, "fig_height": 2.4})
PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL {name} {detail}")
    PASSED.append(name)
    print(f"PASS {name}")


def tmpdir() -> str:
    return tempfile.mkdtemp(prefix="plotlab-p0-kx-")


# ---------------------------------------------------------------------------
# K：Kendall P 值口径
# ---------------------------------------------------------------------------


def test_kendall_exact_acceptance() -> None:
    """任务单给的最小反例：tau=1、n=4 → p=2/24=0.08333，不能再是 nan。"""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    tau = float(sps.kendalltau(x, x).statistic)
    scipy_p = float(sps.kendalltau(x, x).pvalue)
    got = float(CH._corr_pvalue(tau, 4.0, method="kendall"))
    check("K1 tau=1,n=4 p finite (not nan)", math.isfinite(got), repr(got))
    check("K1 tau=1,n=4 p=2/24", abs(got - 2.0 / 24.0) < 1e-12, f"{got!r}")
    check("K1 matches scipy exact p", abs(got - scipy_p) < 1e-12, f"{got!r} vs {scipy_p!r}")
    # 反向完全同序同样是 2/24（双侧）
    tau_rev = float(sps.kendalltau(x, -x).statistic)
    got_rev = float(CH._corr_pvalue(tau_rev, 4.0, method="kendall"))
    check("K1 tau=-1,n=4 p=2/24", abs(got_rev - 2.0 / 24.0) < 1e-12, f"{got_rev!r}")


def test_kendall_exact_matches_scipy() -> None:
    """n<=8：精确置换口径必须与 scipy 的精确 p 完全一致（无并列）。"""
    rng = np.random.default_rng(20261127)
    for n in range(3, 9):
        x = np.arange(n, dtype=float)
        for _ in range(12):
            y = x.copy()
            rng.shuffle(y)
            tau = float(sps.kendalltau(x, y).statistic)
            reference = float(sps.kendalltau(x, y).pvalue)
            got = float(CH._corr_pvalue(tau, float(n), method="kendall"))
            check(f"K2 n={n} exact p matches scipy", abs(got - reference) < 1e-12,
                  f"{got!r} vs {reference!r}")


def test_kendall_asymptotic_matches_scipy() -> None:
    """n>8：标准正态渐近口径必须与 scipy 的 asymptotic p 一致。"""
    rng = np.random.default_rng(7)
    for n in (9, 10, 12, 20, 40):
        x = np.arange(n, dtype=float)
        y = x.copy()
        rng.shuffle(y)
        tau = float(sps.kendalltau(x, y).statistic)
        reference = float(sps.kendalltau(x, y, method="asymptotic").pvalue)
        got = float(CH._corr_pvalue(tau, float(n), method="kendall"))
        check(f"K3 n={n} asymptotic p matches scipy", abs(got - reference) < 1e-12,
              f"{got!r} vs {reference!r}")
    # n>8 时也覆盖极端 tau：仍然有限（渐近式不会发散）
    for n in (9, 30, 200):
        value = float(CH._corr_pvalue(1.0, float(n), method="kendall"))
        check(f"K3 n={n} tau=1 asymptotic finite", math.isfinite(value) and 0.0 <= value <= 1.0,
              repr(value))


def test_kendall_finite_degenerate() -> None:
    """退化输入也不得返回 nan：n<3 保守给 p=1；nan 统计量给 p=1。"""
    cases = [(1.0, 2.0), (-1.0, 2.0), (1.0, 0.0), (float("nan"), 4.0),
             (0.0, 5.0), (1.0, 3.0), (0.9999999999, 8.0)]
    for tau, n in cases:
        value = float(CH._corr_pvalue(tau, n, method="kendall"))
        check(f"K4 degenerate tau={tau} n={n} finite", math.isfinite(value) and 0.0 <= value <= 1.0,
              repr(value))


def test_corr_pvalue_vector_interface_backcompat() -> None:
    """向量接口（corr 热图走这条）保持既有行为：Kendall=scipy，Pearson=scipy。"""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = np.array([4.0, 1.0, 0.0, 3.0, 2.0, 5.0])
    check("K5 vector kendall == scipy",
          abs(CH._corr_pvalue(x, y, "kendall") - float(sps.kendalltau(x, y).pvalue)) < 1e-12)
    check("K5 vector pearson == scipy",
          abs(CH._corr_pvalue(x, y, "pearson") - float(sps.pearsonr(x, y).pvalue)) < 1e-12)
    check("K5 vector spearman finite",
          math.isfinite(float(CH._corr_pvalue(x, y, "spearman"))))
    # 旧标量契约（统计量, n）的 Pearson 行为不变：|r|>=1 → 0.0
    check("K5 legacy scalar pearson |r|=1 -> 0", float(CH._corr_pvalue(1.0, 4.0)) == 0.0)
    # 两个长度 1 的入参按 (statistic, n) 解释，而不是当成 1 个样本的向量
    check("K5 scalar form dispatch",
          abs(float(CH._corr_pvalue(np.array([1.0]), np.array([4.0]), method="kendall"))
              - 2.0 / 24.0) < 1e-12)


# ---------------------------------------------------------------------------
# E：导出绑定 figure_run_id
# ---------------------------------------------------------------------------


def tidy_dataset() -> dict:
    frame = pd.DataFrame({
        "grp": ["Control"] * 6 + ["Treatment"] * 6,
        "score": [61.2, 55.4, 70.1, 64.8, 58.9, 67.3, 80.3, 75.6, 88.2, 72.4, 83.1, 79.0],
    })
    return S.build_dataset(frame, "export-binding", kind="tidy")


def worker() -> pl_registry.WorkerState:
    state = pl_registry.WorkerState(tmpdir())
    state.op_demo({"name": "clinical"})
    return state


def read_protocol(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def render_record(dataset: dict, chart: str, roles: dict, params: dict) -> tuple[dict, dict]:
    """渲染 A 并构造与 registry figure run 同构的记录（用于直接调用导出层）。"""
    meta = CH.CHART_INDEX[chart]
    data = P.build_data(dataset, meta, params, roles)
    result = CH.render(chart, data, params, STYLE, OUTDIR)
    figures = C.jsonable(result.get("figures") or [])
    assert figures, f"渲染 {chart} 没有产出图片"
    record = {
        "figure_run_id": pl_registry._figure_run_id(
            pl_registry._figure_run_key(dataset, chart, params, STYLE, roles)),
        "chart": chart,
        "chart_title": meta["title"],
        "params": C.jsonable(dict(params)),
        "style": C.jsonable(dict(STYLE)),
        "roles": {k: v for k, v in roles.items() if v},
        "data_digest": pl_registry.data_digest(dataset),
        "figures": figures,
        "figure": figures[0],
        "summary": C.jsonable(result.get("summary") or {}),
        "tables": C.jsonable(result.get("tables") or []),
        "warnings": C.jsonable(result.get("warnings") or []),
        "code": {},
    }
    return record, data


def test_export_mismatched_chart_rerenders() -> None:
    """E1：渲染 A=box → 请求导出 B=bar，必须重画成 bar，不能导出 box 的 PNG。"""
    dataset = tidy_dataset()
    roles = {"x": "grp", "y": "score", "group": "grp"}
    record_a, _data_a = render_record(dataset, "box", roles, {})
    outdir = tmpdir()
    meta_b = CH.CHART_INDEX["bar"]
    data_b = P.build_data(dataset, meta_b, {}, roles)
    exported = pl_export.export(
        what="figure", dataset=dataset, chart_id="bar", meta=meta_b, data=data_b,
        params={}, style=STYLE, roles=roles, charts_module=CH, codegen=pl_codegen,
        stats_result=None, last_plot=record_a, outdir=outdir,
    )
    with open(record_a["figure"]["file"], "rb") as handle:
        old_bytes = handle.read()
    with open(exported["file"], "rb") as handle:
        new_bytes = handle.read()
    check("E1 exported figure is not A", new_bytes != old_bytes,
          f"sizes {len(new_bytes)} vs {len(old_bytes)}")
    check("E1 binding status rerendered",
          (exported.get("figure_binding") or {}).get("status") == "rerendered",
          str(exported.get("figure_binding")))
    check("E1 exported run id differs from A",
          str(exported.get("figure_run_id")) not in ("", "None", str(record_a["figure_run_id"])),
          f"{exported.get('figure_run_id')} vs {record_a['figure_run_id']}")
    check("E1 artifact records figure_run_id",
          str(exported["files"][0].get("figure_run_id")) == str(exported.get("figure_run_id")),
          str(exported["files"][0]))
    check("E1 file on disk", os.path.isfile(exported["file"]), str(exported["file"]))


def test_export_mismatch_without_rerender_raises() -> None:
    """E2：无法重画（无数据、无渲染器）时必须报「不一致」，且不落盘旧图。"""
    dataset = tidy_dataset()
    roles = {"x": "grp", "y": "score", "group": "grp"}
    record_a, _data_a = render_record(dataset, "box", roles, {})
    outdir = tmpdir()
    try:
        pl_export.export(
            what="figure", dataset=dataset, chart_id="bar", meta=CH.CHART_INDEX["bar"],
            data=None, params={}, style=STYLE, roles=roles, charts_module=None,
            codegen=pl_codegen, stats_result=None, last_plot=record_a, outdir=outdir,
        )
    except C.EngineError as exc:
        check("E2 error names mismatch", "图形与导出请求不一致" in str(exc.message), str(exc.message))
        check("E2 error gives fix advice", "修复建议" in str(exc.message), str(exc.message))
    else:
        raise AssertionError("FAIL E2 未重画却静默导出了旧图（串图）")
    leftovers = sorted(os.listdir(outdir))
    check("E2 nothing written to disk", leftovers == [], str(leftovers))


def test_export_same_request_reuses() -> None:
    """E3：请求与 A 完全一致 → 复用 A（字节相同、figure_run_id 相同）。"""
    dataset = tidy_dataset()
    roles = {"x": "grp", "y": "score", "group": "grp"}
    record_a, _ = render_record(dataset, "box", roles, {})
    outdir = tmpdir()
    exported = pl_export.export(
        what="figure", dataset=dataset, chart_id="box", meta=CH.CHART_INDEX["box"],
        data=None, params={}, style=STYLE, roles=roles, charts_module=CH,
        codegen=pl_codegen, stats_result=None, last_plot=record_a, outdir=outdir,
    )
    with open(record_a["figure"]["file"], "rb") as handle:
        old_bytes = handle.read()
    with open(exported["file"], "rb") as handle:
        new_bytes = handle.read()
    check("E3 same request reuses A", new_bytes == old_bytes)
    check("E3 reused run id", str(exported.get("figure_run_id")) == str(record_a["figure_run_id"]),
          f"{exported.get('figure_run_id')} vs {record_a['figure_run_id']}")
    check("E3 binding status reused",
          (exported.get("figure_binding") or {}).get("status") == "reused",
          str(exported.get("figure_binding")))


def test_worker_export_redraws_not_crosses() -> None:
    """E4：worker 真实链路——先画 box，再 export(chart=bar)，协议/图/run id 必须同源。"""
    state = worker()
    roles = {"x": "group", "y": "score", "group": "group"}
    box = state.op_plot({"chart": "box", "params": {}, "roles": roles,
                         "style": {"palette": "npg"}})
    box_figure = str(box["figure"]["file"])
    box_run = str(box["figure_run_id"])
    try:
        exported = state.op_export({"what": "protocol", "chart": "bar", "params": {},
                                     "roles": roles})
    except C.EngineError as exc:
        # 允许的实现之一：明确拒绝而不是串图；但错误必须点明原因
        check("E4 mismatch reported rather than crossed",
              "图形与导出请求不一致" in str(exc.message), str(exc.message))
        return
    payload = read_protocol(exported["file"])
    figure = payload["figure"]
    figure_file = str((figure.get("figures") or [{}])[0].get("file"))
    check("E4 protocol chart is the requested bar", figure.get("chart") == "bar",
          str(figure.get("chart")))
    check("E4 exported figure is not the box file",
          os.path.abspath(figure_file) != os.path.abspath(box_figure),
          f"{figure_file} vs {box_figure}")
    check("E4 exported figure exists", os.path.isfile(figure_file), figure_file)
    check("E4 exported run id is not the box run",
          str(figure.get("figure_run_id")) not in ("", "None", box_run),
          f"{figure.get('figure_run_id')} vs {box_run}")
    check("E4 binding status rerendered",
          (figure.get("binding") or {}).get("status") == "rerendered",
          str(figure.get("binding")))
    check("E4 last_plot untouched", state.last_plot["chart"] == "box" and
          str(state.last_plot["figure_run_id"]) == box_run,
          f"{state.last_plot['chart']} {state.last_plot.get('figure_run_id')}")


def test_worker_clean_invalidates_figure() -> None:
    """E5：清洗后同请求导出 → 数据指纹失效，必须重画（不能导出清洗前的旧图）。"""
    state = worker()
    roles = {"x": "group", "y": "score", "group": "group"}
    style = {"palette": "npg"}
    before = state.op_plot({"chart": "box", "params": {}, "roles": roles, "style": style})
    before_file = str(before["figure"]["file"])
    before_run = str(before["figure_run_id"])
    state.op_clean({"plan": [{"op": "outlier", "params": {"columns": ["score"], "method": "iqr",
                                                           "action": "winsorize"}}]})
    after = state.op_export({"what": "protocol", "chart": "box", "params": {},
                             "roles": roles, "style": style})
    payload = read_protocol(after["file"])
    figure = payload["figure"]
    after_file = str((figure.get("figures") or [{}])[0].get("file"))
    check("E5 clean triggers rerender",
          (figure.get("binding") or {}).get("status") == "rerendered",
          str(figure.get("binding")))
    check("E5 new run id after clean", str(figure.get("figure_run_id")) != before_run,
          f"{figure.get('figure_run_id')} vs {before_run}")
    check("E5 new figure file after clean",
          os.path.abspath(after_file) != os.path.abspath(before_file),
          f"{after_file} vs {before_file}")
    check("E5 stale digest detected",
          any("数据指纹" in str(item) for item in (figure.get("binding") or {}).get("diffs") or []),
          str((figure.get("binding") or {}).get("diffs")))


def test_worker_archived_run_stays_pinned() -> None:
    """E6：按旧 figure_run_id 归档回查 → 固定旧版本，不能被重画回最新。"""
    state = worker()
    roles = {"x": "group", "y": "score", "group": "group"}
    style = {"palette": "npg"}
    old = state.op_plot({"chart": "box", "params": {}, "roles": roles, "style": style})
    old_run = str(old["figure_run_id"])
    old_file = str(old["figure"]["file"])
    state.op_clean({"plan": [{"op": "outlier", "params": {"columns": ["score"], "method": "iqr",
                                                           "action": "winsorize"}}]})
    archived = state.op_export({"what": "protocol", "figure_run_id": old_run})
    payload = read_protocol(archived["file"])
    figure = payload["figure"]
    check("E6 archived run id preserved", str(figure.get("figure_run_id")) == old_run,
          str(figure.get("figure_run_id")))
    check("E6 archived figure file preserved",
          str((figure.get("figures") or [{}])[0].get("file")) == old_file,
          str(figure.get("figures")))
    check("E6 archived export reused (not redrawn)",
          (figure.get("binding") or {}).get("status") == "reused",
          str(figure.get("binding")))
    check("E6 archived figure still on disk", os.path.isfile(old_file), old_file)


def test_artifacts_bundle_record_run_id() -> None:
    """E7：bundle 的 payload、每个 file 条目、protocol.json 都记录 figure_run_id。"""
    state = worker()
    roles = {"x": "group", "y": "score", "group": "group"}
    box = state.op_plot({"chart": "box", "params": {}, "roles": roles,
                         "style": {"palette": "npg"}})
    run_id = str(box["figure_run_id"])
    bundle = state.op_export({"what": "bundle", "chart": "box", "params": {},
                              "roles": roles, "style": {"palette": "npg"}})
    check("E7 payload records figure_run_id", str(bundle.get("figure_run_id")) == run_id,
          str(bundle.get("figure_run_id")))
    check("E7 every artifact records figure_run_id",
          all(str(item.get("figure_run_id")) == run_id for item in bundle.get("files") or []),
          str([(item.get("kind"), item.get("figure_run_id")) for item in bundle.get("files") or []]))
    import zipfile
    with zipfile.ZipFile(bundle["file"]) as archive:
        names = archive.namelist()
        protocol_name = next((name for name in names if name.endswith("protocol.json")), None)
        check("E7 bundle contains protocol.json", protocol_name is not None, str(names))
        payload = json.loads(archive.read(protocol_name).decode("utf-8"))
    check("E7 protocol.json figure_run_id", str(payload["figure"].get("figure_run_id")) == run_id,
          str(payload["figure"].get("figure_run_id")))
    check("E7 protocol.json binding status reused",
          (payload["figure"].get("binding") or {}).get("status") == "reused",
          str(payload["figure"].get("binding")))


TESTS = [
    ("K1 exact acceptance", test_kendall_exact_acceptance),
    ("K2 exact vs scipy", test_kendall_exact_matches_scipy),
    ("K3 asymptotic vs scipy", test_kendall_asymptotic_matches_scipy),
    ("K4 degenerate finite", test_kendall_finite_degenerate),
    ("K5 vector interface backcompat", test_corr_pvalue_vector_interface_backcompat),
    ("E1 mismatched chart rerenders", test_export_mismatched_chart_rerenders),
    ("E2 mismatch without rerender raises", test_export_mismatch_without_rerender_raises),
    ("E3 same request reuses", test_export_same_request_reuses),
    ("E4 worker export redraws not crosses", test_worker_export_redraws_not_crosses),
    ("E5 clean invalidates figure", test_worker_clean_invalidates_figure),
    ("E6 archived run stays pinned", test_worker_archived_run_stays_pinned),
    ("E7 artifacts record run id", test_artifacts_bundle_record_run_id),
]


def main() -> int:
    print("Kendall P 值 + 导出绑定 回归测试")
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
