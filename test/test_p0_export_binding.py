#!/usr/bin/env python3
"""P0 导出串图回归测试（正式命名；由 py_export_binding_test 反例整理）。

对应任务单 §5 的「导出串图」：
  E1  先画箱线图、再 export(chart=bar)：协议里 chart 与 figures 必须同源，
      请求的图型与 last_plot 不一致时必须重新出图，而不是复用旧 PNG
  E2  params / style / roles 任一与 last_plot 不一致也必须重画
  E3  完全一致的请求可以复用旧图（仍带 figure_run_id），且旧版本仍能按 id 取回
  E4  没有出图历史但显式给了 chart + 参数时，导出应自己出图而不是报「还没有出图」
  E5  figure_run_id 固化了源数据 + 图型 + 参数 + 样式 + 角色 + 统计表 + 图层

运行：
    export MPLBACKEND=Agg MPLCONFIGDIR=$(mktemp -d)
    PYTHONPATH=lib python3 test/test_p0_export_binding.py
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

import pl_common as C  # noqa: E402
import pl_registry  # noqa: E402
import pl_state as S  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL {name} {detail}")
    PASSED.append(name)
    print(f"PASS {name}")


def worker() -> pl_registry.WorkerState:
    state = pl_registry.WorkerState(tempfile.mkdtemp(prefix="plotlab-export-binding-"))
    state.op_demo({"name": "clinical"})
    return state


def read_protocol(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_chart_mismatch_redraws() -> None:
    state = worker()
    box = state.op_plot({"chart": "box", "params": {},
                         "roles": {"x": "group", "y": "score", "group": "group"},
                         "style": {"palette": "npg"}})
    box_figure = box["figure"]["file"]
    box_run = box.get("figure_run_id")
    check("E1 plot carries figure_run_id", bool(box_run), str(box.get("figure_run_id")))

    exported = state.op_export({"what": "protocol", "chart": "bar", "params": {},
                                "roles": {"x": "group", "y": "score", "group": "group"}})
    payload = read_protocol(exported["file"])
    figure = payload["figure"]
    check("E1 protocol chart is bar", figure["chart"] == "bar", str(figure["chart"]))
    figures = figure["figures"] or []
    check("E1 protocol has figures", bool(figures), str(figures))
    exported_file = str(figures[0]["file"])
    check("E1 exported figure is new file", os.path.abspath(exported_file) != os.path.abspath(box_figure),
          f"{exported_file} vs {box_figure}")
    check("E1 exported file exists", os.path.isfile(exported_file), exported_file)
    check("E1 protocol figure_run_id differs", str(figure.get("figure_run_id")) not in ("", "None"),
          str(figure.get("figure_run_id")))
    check("E1 protocol figure_run_id is not box run", str(figure.get("figure_run_id")) != str(box_run),
          f"{figure.get('figure_run_id')} vs {box_run}")
    check("E1 protocol records chart title", "Bar" in str(figure.get("title") or "")
          or bool(figure.get("title")), str(figure.get("title")))

    # what=figure 同样必须重画，且实际像素与箱线图不同
    fig_export = state.op_export({"what": "figure", "chart": "bar", "params": {},
                                  "roles": {"x": "group", "y": "score", "group": "group"}})
    check("E1 figure export file differs", os.path.abspath(fig_export["file"]) != os.path.abspath(box_figure),
          f"{fig_export['file']} vs {box_figure}")
    with open(fig_export["file"], "rb") as handle:
        new_bytes = handle.read()
    with open(box_figure, "rb") as handle:
        old_bytes = handle.read()
    check("E1 figure export is not the box PNG", new_bytes != old_bytes,
          f"sizes {len(new_bytes)} vs {len(old_bytes)}")

    # last_plot 仍是箱线图；导出不能污染它
    check("E1 last_plot untouched", state.last_plot["chart"] == "box", str(state.last_plot["chart"]))


def test_params_style_roles_redraw() -> None:
    state = worker()
    base = {"chart": "box", "roles": {"x": "group", "y": "score", "group": "group"},
            "style": {"palette": "npg"}, "params": {}}
    first = state.op_plot(base)
    first_file = first["figure"]["file"]
    first_run = first["figure_run_id"]

    # params 变化 → 重画
    changed_params = state.op_export({"what": "protocol", "chart": "box",
                                      "roles": base["roles"], "style": base["style"],
                                      "params": {"show_points": False, "agg": "mean"}})
    payload = read_protocol(changed_params["file"])
    check("E2 params change redraws",
          str(payload["figure"].get("figure_run_id")) != str(first_run),
          f"{payload['figure'].get('figure_run_id')} vs {first_run}")

    # style 变化 → 重画
    changed_style = state.op_export({"what": "protocol", "chart": "box",
                                     "roles": base["roles"], "params": {},
                                     "style": {"palette": "npg", "title_text": "Custom Title"}})
    payload_style = read_protocol(changed_style["file"])
    check("E2 style change redraws",
          str(payload_style["figure"].get("figure_run_id")) != str(first_run),
          str(payload_style["figure"].get("figure_run_id")))
    check("E2 style recorded in protocol",
          payload_style["figure"]["style"].get("title_text") == "Custom Title",
          str(payload_style["figure"]["style"].get("title_text")))

    # roles 变化 → 重画
    changed_roles = state.op_export({"what": "protocol", "chart": "box",
                                     "roles": {"x": "group", "y": "crp", "group": "group"},
                                     "params": {}, "style": base["style"]})
    payload_roles = read_protocol(changed_roles["file"])
    check("E2 roles change redraws",
          str(payload_roles["figure"].get("figure_run_id")) != str(first_run),
          str(payload_roles["figure"].get("figure_run_id")))
    check("E2 roles recorded in protocol",
          payload_roles["figure"]["roles"].get("y") == "crp",
          str(payload_roles["figure"]["roles"]))
    check("E2 first figure still on disk", os.path.isfile(first_file), first_file)


def test_same_request_reuses() -> None:
    state = worker()
    plot = state.op_plot({"chart": "box", "params": {},
                          "roles": {"x": "group", "y": "score", "group": "group"},
                          "style": {"palette": "npg"}})
    run_id = plot["figure_run_id"]
    again = state.op_export({"what": "protocol", "chart": "box", "params": {},
                             "roles": {"x": "group", "y": "score", "group": "group"},
                             "style": {"palette": "npg"}})
    payload = read_protocol(again["file"])
    check("E3 identical request keeps run id",
          str(payload["figure"].get("figure_run_id")) == str(run_id),
          f"{payload['figure'].get('figure_run_id')} vs {run_id}")
    history = state.op_records({})["history"]
    check("E3 history keeps old version", len(history) >= 1, str(len(history)))
    ids = {item.get("figure", {}).get("id") for item in history}
    check("E3 old figure id retrievable", str(plot["figure"]["id"]) in ids, str(ids))
    check("E3 history carries figure_run_id",
          all(item.get("figure_run_id") for item in history), str(history[:1]))


def test_export_without_prior_plot() -> None:
    state = worker()
    exported = state.op_export({"what": "figure", "chart": "bar", "params": {},
                                "roles": {"x": "group", "y": "score", "group": "group"}})
    check("E4 renders on demand", bool(exported.get("file")) and os.path.isfile(exported["file"]),
          str(exported.get("file")))


def test_run_fingerprint_contents() -> None:
    state = worker()
    plot = state.op_plot({"chart": "box", "params": {},
                          "roles": {"x": "group", "y": "score", "group": "group"},
                          "style": {"palette": "npg"}})
    fingerprint = plot.get("figure_run") or {}
    check("E5 run has chart", fingerprint.get("chart") == "box", str(fingerprint.get("chart")))
    check("E5 run has params", isinstance(fingerprint.get("params"), dict), str(fingerprint.get("params")))
    check("E5 run has style", isinstance(fingerprint.get("style"), dict), str(fingerprint.get("style")))
    check("E5 run has roles", isinstance(fingerprint.get("roles"), dict), str(fingerprint.get("roles")))
    check("E5 run has tables", isinstance(fingerprint.get("tables"), list), str(fingerprint.get("tables")))
    check("E5 run has layers", isinstance(fingerprint.get("layers"), list), str(fingerprint.get("layers")))
    check("E5 run has dataset id", bool(fingerprint.get("dataset_id")), str(fingerprint.get("dataset_id")))
    check("E5 run has source digest", bool(fingerprint.get("data_digest")), str(fingerprint.get("data_digest")))

    # 数据集清洗后 digest 变化，同参数也必须重画
    before_digest = fingerprint.get("data_digest")
    state.op_clean({"plan": [{"op": "outlier", "params": {"columns": ["score"], "method": "iqr",
                                                          "action": "winsorize"}}]})
    after = state.op_export({"what": "protocol", "chart": "box", "params": {},
                             "roles": {"x": "group", "y": "score", "group": "group"},
                             "style": {"palette": "npg"}})
    payload = read_protocol(after["file"])
    check("E5 clean invalidates run",
          str(payload["figure"].get("data_digest")) != str(before_digest),
          f"{payload['figure'].get('data_digest')} vs {before_digest}")
    check("E5 run id changes after clean",
          str(payload["figure"].get("figure_run_id")) != str(plot["figure_run_id"]),
          str(payload["figure"].get("figure_run_id")))


def multigroup_worker() -> pl_registry.WorkerState:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "arm": ["A"] * 12 + ["B"] * 12 + ["C"] * 12,
        "value": np.concatenate([rng.normal(10, 1.5, 12), rng.normal(13, 1.5, 12),
                                 rng.normal(16, 1.5, 12)]).round(3),
    })
    state = pl_registry.WorkerState(tempfile.mkdtemp(prefix="plotlab-export-multi-"))
    state.store_dataset(S.build_dataset(frame, "multi", kind="tidy"))
    return state


def test_archive_retrievable_after_clean() -> None:
    """E6 清洗后当前图必须重画，但旧 version 仍能按 figure_run_id 取回。"""
    state = worker()
    roles = {"x": "group", "y": "score", "group": "group"}
    plot = state.op_plot({"chart": "box", "params": {}, "roles": roles,
                          "style": {"palette": "npg"}})
    old_run = plot["figure_run_id"]
    old_file = plot["figure"]["file"]
    state.op_clean({"plan": [{"op": "outlier", "params": {
        "columns": ["score"], "method": "iqr", "action": "winsorize"}}]})
    fresh = state.op_export({"what": "protocol", "chart": "box", "params": {},
                             "roles": roles, "style": {"palette": "npg"}})
    fresh_payload = read_protocol(fresh["file"])
    check("E6 clean changes current run",
          str(fresh_payload["figure"]["figure_run_id"]) != str(old_run),
          str(fresh_payload["figure"]["figure_run_id"]))
    archived = state.op_export({"what": "protocol", "figure_run_id": old_run})
    archived_payload = read_protocol(archived["file"])
    check("E6 archived run id", str(archived_payload["figure"]["figure_run_id"]) == str(old_run),
          str(archived_payload["figure"]["figure_run_id"]))
    check("E6 archived figure still on disk", os.path.isfile(old_file), old_file)
    check("E6 archived figures exist",
          all(os.path.isfile(str(item["file"])) for item in archived_payload["figure"]["figures"]),
          str(archived_payload["figure"]["figures"]))
    records = state.op_records({})
    ids = {str(item.get("figure_run_id")) for item in records.get("figure_runs") or []}
    check("E6 records list runs", str(old_run) in ids, str(ids))


def test_multigroup_multifigure_isolation() -> None:
    """E7 多组别/多图依次导出互不串图：不同 chart 各自 run，重取原请求仍复用原 run。"""
    state = multigroup_worker()
    roles = {"x": "arm", "y": "value", "group": "arm"}
    style = {"palette": "npg"}
    box = state.op_plot({"chart": "box", "params": {}, "roles": roles, "style": style})
    check("E7 multigroup annotation contrasts", len((box.get("annotations") or {}).get("contrasts") or []) == 3,
          str(box.get("annotations")))
    check("E7 multigroup stats factor", (box.get("stats") or {}).get("factor") == "arm",
          str(box.get("stats")))
    bar_protocol = state.op_export({"what": "protocol", "chart": "bar", "params": {},
                                    "roles": roles, "style": style})
    bar_payload = read_protocol(bar_protocol["file"])
    bar_run = str(bar_payload["figure"]["figure_run_id"])
    bar_file = str(bar_payload["figure"]["figures"][0]["file"])
    violin_protocol = state.op_export({"what": "protocol", "chart": "violin", "params": {},
                                       "roles": roles, "style": style})
    violin_payload = read_protocol(violin_protocol["file"])
    violin_run = str(violin_payload["figure"]["figure_run_id"])
    violin_file = str(violin_payload["figure"]["figures"][0]["file"])
    check("E7 chart runs differ", bar_run != violin_run, f"{bar_run} vs {violin_run}")
    check("E7 chart files differ", os.path.abspath(bar_file) != os.path.abspath(violin_file),
          f"{bar_file} vs {violin_file}")
    check("E7 both files on disk", os.path.isfile(bar_file) and os.path.isfile(violin_file))
    bar_again = state.op_export({"what": "protocol", "chart": "bar", "params": {},
                                 "roles": roles, "style": style})
    bar_again_payload = read_protocol(bar_again["file"])
    check("E7 same request reuses bar run",
          str(bar_again_payload["figure"]["figure_run_id"]) == bar_run,
          str(bar_again_payload["figure"]["figure_run_id"]))
    check("E7 box figure untouched", os.path.isfile(str(box["figure"]["file"])))


TESTS = [
    ("E1 chart mismatch redraws", test_chart_mismatch_redraws),
    ("E2 params/style/roles redraw", test_params_style_roles_redraw),
    ("E3 same request reuses", test_same_request_reuses),
    ("E4 export without prior plot", test_export_without_prior_plot),
    ("E5 run fingerprint", test_run_fingerprint_contents),
    ("E6 archived run retrievable", test_archive_retrievable_after_clean),
    ("E7 multigroup/multifigure isolation", test_multigroup_multifigure_isolation),
]


def main() -> int:
    print("导出串图回归测试")
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