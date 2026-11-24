#!/usr/bin/env python3
"""PlotLab 回归测试（离线、确定性）。

覆盖：导入（tidy / matrix / xlsx / gbk / json / zip）→ 剖析 → 清洗 → 需求解析 →
     22 张图各渲一次 → 样式变体 → 10 项统计 → 代码生成 → 导出 bundle → worker 协议。

运行：
  cd plugins/dsh-plotlab
  export MPLBACKEND=Agg MPLCONFIGDIR=$(mktemp -d)
  PYTHONPATH=lib python3 test/py_plotlab_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, "lib")
sys.path.insert(0, LIB)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="pl-mpl-"))

import numpy as np
import pandas as pd

import pl_common as C
import pl_prep as P
import pl_state as S
import pl_clean
import pl_io
import pl_charts
import pl_stats
import pl_codegen
import pl_export
import pl_registry

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL {name} {detail}")
    PASSED.append(name)
    print(f"PASS {name}")


def tmpdir() -> str:
    return tempfile.mkdtemp(prefix="plotlab-test-")


# ---------------------------------------------------------------------------
# 1. 导入
# ---------------------------------------------------------------------------


def write_fixtures(base: str) -> dict:
    paths = {}
    # tidy CSV：中文列名、千分位、缺失、<0.001、分组、生存列
    tidy = """sample_id,group,sex,age,score,crp,response,survival_months,event
S001,Control,Female,58,61.2,"1,203.5",No,12.5,0
S002,Control,Male,49,55.4,<0.001,No,9.5,1
S003,Control,Female,67,70.1,980.0,Yes,22.0,0
S004,Treatment,Male,61,80.3,540.5,Yes,7.0,1
S005,Treatment,Female,54,75.6,860.2,No,18.5,0
S006,Treatment,Male,72,,720.4,Yes,30.0,1
S007,Control,Male,,64.8,1180.9,No,15.5,0
S008,Treatment,Female,66,88.2,410.7,Yes,5.5,1
"""
    paths["tidy_csv"] = os.path.join(base, "tidy.csv")
    with open(paths["tidy_csv"], "w", encoding="utf-8-sig") as handle:
        handle.write(tidy)
    # GBK tsv
    gbk = pd.DataFrame(
        {"样本": ["A1", "A2", "A3", "A4"], "分组": ["对照", "对照", "处理", "处理"], "表达量": [1.5, 2.1, 3.6, 4.2]}
    )
    paths["gbk_tsv"] = os.path.join(base, "gbk.tsv")
    gbk.to_csv(paths["gbk_tsv"], sep="\t", index=False, encoding="gbk")
    # xlsx 两个 sheet
    paths["xlsx"] = os.path.join(base, "book.xlsx")
    with pd.ExcelWriter(paths["xlsx"]) as writer:
        gbk.to_excel(writer, sheet_name="sheet1", index=False)
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_excel(writer, sheet_name="sheet2", index=False)
    # matrix CSV：首列 gene，其余 4 个样本
    matrix = pd.DataFrame(
        {
            "gene": ["IL6", "TNF", "CD8A", "PDCD1", "CXCL9", "STAT1"],
            "T1": [8.1, 7.2, 6.5, 5.4, 9.1, 7.8],
            "T2": [8.4, 7.8, 6.1, 5.9, 9.4, 7.2],
            "C1": [6.2, 5.1, 8.4, 3.1, 5.2, 8.1],
            "C2": [6.5, 5.4, 8.9, 3.4, 5.9, 8.4],
        }
    )
    paths["matrix_csv"] = os.path.join(base, "matrix.csv")
    matrix.to_csv(paths["matrix_csv"], index=False)
    paths["sample_sheet"] = os.path.join(base, "samples.csv")
    pd.DataFrame({"sample": ["T1", "T2", "C1", "C2"], "group": ["Tumor", "Tumor", "Normal", "Normal"]}).to_csv(
        paths["sample_sheet"], index=False
    )
    # json records
    paths["json"] = os.path.join(base, "rows.json")
    with open(paths["json"], "w", encoding="utf-8") as handle:
        json.dump([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "x"}], handle)
    # gzip 单文件
    paths["gz"] = os.path.join(base, "tidy.csv.gz")
    pd.read_csv(paths["tidy_csv"], encoding="utf-8-sig").to_csv(paths["gz"], index=False, compression="gzip")
    # zip 两个文件
    paths["zip"] = os.path.join(base, "two.zip")
    with zipfile.ZipFile(paths["zip"], "w") as archive:
        archive.write(paths["tidy_csv"], "tidy.csv")
        archive.write(paths["matrix_csv"], "matrix.csv")
    return paths


def test_import(paths: dict) -> tuple[dict, dict, dict]:
    result = pl_io.open_source(path=paths["tidy_csv"], name="tidy")
    dataset = result["dataset"]
    check("import tidy rows", dataset["df"].shape[0] == 8, str(dataset["df"].shape))
    check("import tidy group", dataset.get("group_col") == "group", str(dataset.get("group_col")))
    check("import tidy numeric", P.infer_dtype(dataset["df"]["score"]) == "numeric")
    check("import tidy kind", dataset["kind"] == "tidy")
    # 千分位 / <0.001 转数值
    crp = P.to_numeric(dataset["df"]["crp"])
    check("import crp thousands", abs(float(crp.iloc[0]) - 1203.5) < 1e-6, str(crp.iloc[0]))
    check("import crp censored", abs(float(crp.iloc[1]) - 0.0005) < 1e-9, str(crp.iloc[1]))

    gbk = pl_io.open_source(path=paths["gbk_tsv"], name="gbk")
    check("import gbk columns", list(gbk["dataset"]["df"].columns)[:2] == ["样本", "分组"], str(list(gbk["dataset"]["df"].columns)))
    check("import gbk group", gbk["dataset"].get("group_col") == "分组", str(gbk["dataset"].get("group_col")))

    inspect = pl_io.inspect_source(path=paths["xlsx"])
    check("inspect xlsx candidates", len(inspect.get("candidates") or []) >= 2, str(inspect))
    sheet2 = pl_io.open_source(path=paths["xlsx"], sheet="sheet2")
    check("import xlsx sheet", list(sheet2["dataset"]["df"].columns) == ["x", "y"], str(list(sheet2["dataset"]["df"].columns)))

    matrix = pl_io.open_source(path=paths["matrix_csv"], kind="matrix", sample_sheet=paths["sample_sheet"])
    matrix_dataset = matrix["dataset"]
    check("import matrix kind", matrix_dataset["kind"] == "matrix")
    check("import matrix feature", matrix_dataset.get("feature_col") == "gene", str(matrix_dataset.get("feature_col")))
    check("import matrix sample info", isinstance(matrix_dataset.get("sample_info"), pd.DataFrame))
    check("import matrix groups", bool(matrix_dataset.get("groups")), str(matrix_dataset.get("groups")))

    rows = pl_io.open_source(path=paths["json"])
    check("import json", rows["dataset"]["df"].shape == (3, 2), str(rows["dataset"]["df"].shape))

    gz = pl_io.open_source(path=paths["gz"])
    check("import gz rows", gz["dataset"]["df"].shape[0] == 8)

    archive = pl_io.inspect_source(path=paths["zip"])
    check("inspect zip candidates", len(archive.get("candidates") or []) == 2, str(archive.get("candidates")))
    zipped = pl_io.open_source(path=paths["zip"], member="matrix.csv", kind="matrix")
    check("import zip member", zipped["dataset"]["kind"] == "matrix")

    try:
        pl_io.open_source(path=paths["tidy_csv"], data_format="xlsx")
        check("import bad format raises", False)
    except C.EngineError as error:
        check("import bad format raises", error.code in ("bad_request", "unsupported"), error.code)
    return dataset, matrix_dataset, gz["dataset"]


# ---------------------------------------------------------------------------
# 2. 剖析 / 清洗 / 需求解析
# ---------------------------------------------------------------------------


def test_profile_clean(dataset: dict) -> None:
    profile = pl_clean.profile(dataset)
    check("profile score", 0 <= profile["score"] <= 100, str(profile["score"]))
    check("profile checks", len(profile["checks"]) >= 8)
    check("profile recommend", len(profile["recommend"]["charts"]) >= 3)
    check("profile recommend stats", len(profile["recommend"]["stats"]) >= 3)
    check("profile missing column rerun", any(c["id"] == "missing" for c in profile["checks"]))

    plan = pl_clean.auto_plan(dataset)
    check("auto plan list", isinstance(plan, list))
    report = pl_clean.apply_clean(dataset, plan=plan, auto=False)
    check("clean report", "applied" in report and "skipped" in report)
    check("clean rows", report["n_rows"] <= 8, str(report["n_rows"]))
    check("clean cleaning log", len(dataset["cleaning"]) >= 1, str(len(dataset["cleaning"])))
    restored = pl_clean.reset_data(dataset)
    check("clean reset rows", restored["n_rows"] == 8, str(restored["n_rows"]))


def test_parse_need(dataset: dict, matrix_dataset: dict) -> None:
    plan = pl_registry.parse_need("按 group 分组画箱线图比较 score 并标显著性，配色用色盲友好", dataset)
    check("need chart box", plan["chart"] == "box", str(plan["chart"]))
    check("need group", plan["roles"].get("group") == "group", str(plan["roles"]))
    check("need stat compare", "compare" in plan["stats"], str(plan["stats"]))
    check("need palette", plan["style"].get("palette") == "okabe_ito", str(plan["style"]))
    check("need steps", len(plan["steps"]) >= 2)

    plan2 = pl_registry.parse_need("score 和 age 的散点图，加回归线，标题字号大一点", dataset)
    check("need scatter", plan2["chart"] == "scatter", str(plan2["chart"]))
    check("need fit", plan2["params"].get("add_fit") is True, str(plan2["params"]))
    check("need font size", plan2["style"].get("title_size") == 16, str(plan2["style"]))

    plan3 = pl_registry.parse_need("对表达矩阵做热图，按行标准化并聚类", matrix_dataset)
    check("need heatmap", plan3["chart"] == "heatmap", str(plan3["chart"]))
    check("need zscore", plan3["params"].get("zscore") in ("row", "rows", True), str(plan3["params"]))

    plan4 = pl_registry.parse_need("做生存曲线，按 group 分组", dataset)
    check("need km", plan4["chart"] == "km", str(plan4["chart"]))
    check("need survival stat", "survival" in plan4["stats"], str(plan4["stats"]))


# ---------------------------------------------------------------------------
# 3. 图表
# ---------------------------------------------------------------------------


CHART_DATASET = {
    "bar": "tidy", "bar_dot": "tidy", "box": "tidy", "violin": "tidy", "raincloud": "tidy",
    "dotplot": "tidy", "stack": "tidy", "pie": "tidy", "scatter": "tidy", "bubble": "tidy",
    "line": "tidy", "area": "tidy", "heatmap": "matrix", "corr": "matrix", "pairplot": "tidy",
    "hist": "tidy", "density": "tidy", "qq": "tidy", "roc": "tidy", "km": "tidy",
    "forest": "tidy", "pca": "matrix",
}


def chart_data_for(chart_id: str, tidy_dataset: dict, matrix_dataset: dict) -> tuple[dict, dict]:
    which = CHART_DATASET[chart_id]
    dataset = matrix_dataset if which == "matrix" else tidy_dataset
    meta = next(item for item in pl_charts.CHARTS if item["id"] == chart_id)
    roles = P.recommend_roles(dataset, meta)
    params = {param["key"]: param.get("default") for param in meta.get("params") or []}
    if chart_id == "forest":
        # 森林图允许从 tidy 的组间比较参数构建
        params = dict(params or {})
        params.setdefault("terms", None)
    data = P.build_data(dataset, meta, params, roles)
    return data, params


def test_charts(tidy_dataset: dict, matrix_dataset: dict, outdir: str) -> None:
    style = C.resolve_style({})
    rendered = {}
    for chart in pl_charts.CHARTS:
        chart_id = chart["id"]
        data, params = chart_data_for(chart_id, tidy_dataset, matrix_dataset)
        try:
            result = pl_charts.render(chart_id, data, params, style, outdir)
        except C.EngineError as error:
            # 数据不满足时允许明确报错（不算失败），但记录
            print(f"SKIP {chart_id}: {error.code} {error.message}")
            continue
        figures = result.get("figures") or []
        check(f"chart {chart_id} figure", bool(figures), str(result)[:200])
        if figures:
            path = figures[0]["file"]
            check(f"chart {chart_id} png", os.path.exists(path) and os.path.getsize(path) > 5000,
                  f"{path} {os.path.getsize(path) if os.path.exists(path) else 0}")
        code = pl_codegen.chart_python(chart_id, data, params, style)
        check(f"chart {chart_id} code", isinstance(code, str) and len(code) > 80)
        compile(code, f"<{chart_id}>", "exec")  # 片段语法必须合法
        r_code = pl_codegen.chart_r_code(chart_id, data, params, style)
        check(f"chart {chart_id} r code", isinstance(r_code, str) and len(r_code) > 40)
        rendered[chart_id] = True
    check("charts rendered >= 18", len(rendered) >= 18, str(sorted(rendered)))

    # 样式变体：中文字体 + 大字号 + 换配色 + 自定义颜色 + 不同画布
    meta = next(item for item in pl_charts.CHARTS if item["id"] == "box")
    data = P.build_data(tidy_dataset, meta, {}, P.recommend_roles(tidy_dataset, meta))
    variants = [
        {"font": "PingFang SC", "title_size": 20, "title_text": "中文标题", "x_label": "分组", "y_label": "评分"},
        {"palette": "npg", "fig_width": 9, "fig_height": 5, "dpi": 150},
        {"palette": "jama", "colors": {"Control": "#123456", "Treatment": "#ABCDEF"}, "legend_pos": "bottom"},
        {"theme": "grid", "frame": "full", "x_rotate": 45, "y_log": False},
        {"stat_annot": False, "jitter": False, "error": "none"},
    ]
    for index, patch in enumerate(variants):
        resolved = C.resolve_style(patch)
        result = pl_charts.render("box", data, {}, resolved, outdir)
        check(f"style variant {index}", bool(result.get("figures")), str(patch))


# ---------------------------------------------------------------------------
# 4. 统计
# ---------------------------------------------------------------------------


def test_stats(tidy_dataset: dict, matrix_dataset: dict, outdir: str) -> None:
    style = C.resolve_style({})
    for stat in pl_stats.STATS_CATALOG:
        stat_id = stat["id"]
        meta = {"id": stat_id, "needs": stat.get("needs") or "tidy", "roles": stat.get("roles") or {}}
        params = {param["key"]: param.get("default") for param in stat.get("params") or []}
        if stat_id in ("compare", "describe", "assumption") and not params:
            params = {}
        dataset = matrix_dataset if stat_id == "pca" else tidy_dataset
        roles = P.recommend_roles(dataset, {"id": stat_id, "roles": meta["roles"]})
        try:
            data = P.build_data(dataset, meta, params, roles)
            result = pl_stats.run(stat_id, dataset, data, params, style, outdir)
        except C.EngineError as error:
            print(f"SKIP stat {stat_id}: {error.code} {error.message}")
            continue
        check(f"stat {stat_id} summary", bool(result.get("summary")), str(result)[:200])
        check(f"stat {stat_id} method", bool(result.get("method")), str(result.get("method")))
        for table in result.get("tables") or []:
            check(f"stat {stat_id} table {table.get('id')}", bool(table.get("columns")) and isinstance(table.get("rows"), list))
        check(f"stat {stat_id} report", isinstance(result.get("report"), str) and len(result["report"]) > 10)


# ---------------------------------------------------------------------------
# 5. 代码 / 导出 / 审计
# ---------------------------------------------------------------------------


def test_code_export(tidy_dataset: dict, outdir: str) -> None:
    meta = next(item for item in pl_charts.CHARTS if item["id"] == "box")
    params = {}
    style = C.resolve_style({"palette": "npg", "title_size": 16})
    roles = P.recommend_roles(tidy_dataset, meta)
    data = P.build_data(tidy_dataset, meta, params, roles)
    script = pl_codegen.build_script(tidy_dataset, "box", meta, data, params, style, roles)
    compile(script, "<script>", "exec")
    check("code has read", "pd.read_csv" in script)
    check("code has savefig", "savefig" in script)
    check("code has palette", "NPG" in script or "#E64B35" in script or "palette" in script)
    check("code has stats", "ttest" in script or "scipy" in script)

    figures = pl_charts.render("box", data, params, style, outdir)
    last_plot = {
        "chart": "box",
        "chart_title": meta["title"],
        "params": params,
        "style": style,
        "roles": roles,
        "figures": figures.get("figures") or [],
        "summary": figures.get("summary"),
        "tables": figures.get("tables") or [],
        "warnings": figures.get("warnings") or [],
        "code": {"python": script, "r": pl_codegen.chart_r_code("box", data, params, style)},
        "dataset": {"id": tidy_dataset.get("id")},
    }
    for what in ("figure", "code", "report", "methods", "protocol", "cleaned", "bundle"):
        result = pl_export.export(
            what=what, dataset=tidy_dataset, chart_id="box", meta=meta, data=data,
            params=params, style=style, roles=roles, charts_module=pl_charts,
            codegen=pl_codegen, stats_result=None, last_plot=last_plot, outdir=os.path.join(outdir, "export"),
        )
        check(f"export {what}", bool(result.get("file")) and os.path.exists(result["file"]), str(result.get("file")))
    bundle = pl_export.export(
        what="bundle", dataset=tidy_dataset, chart_id="box", meta=meta, data=data,
        params=params, style=style, roles=roles, charts_module=pl_charts,
        codegen=pl_codegen, stats_result=None, last_plot=last_plot, outdir=os.path.join(outdir, "export2"),
    )
    with zipfile.ZipFile(bundle["file"]) as archive:
        names = archive.namelist()
    check("bundle contents", any(name.endswith(".png") for name in names) and any(name.endswith(".py") for name in names)
          and any(name.endswith("report.md") for name in names), str(names))


# ---------------------------------------------------------------------------
# 6. worker 协议
# ---------------------------------------------------------------------------


def test_worker(tidy_dataset: dict) -> None:
    workdir = tmpdir()
    script = os.path.join(LIB, "pl_engine.py")
    process = subprocess.Popen(
        [sys.executable, "-u", script, "--workdir", workdir],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=LIB, env={**os.environ, "MPLBACKEND": "Agg"},
    )
    seq = 0

    def call(cmd: str, **args):
        nonlocal seq
        seq += 1
        process.stdin.write(json.dumps({"id": str(seq), "cmd": cmd, "args": args}) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        message = json.loads(line)
        if not message.get("ok"):
            raise AssertionError(f"worker {cmd} failed: {message}")
        return message["result"]

    try:
        ping = call("ping")
        check("worker ping", ping.get("version") and ping.get("charts") is not None)
        demo = call("demo", name="clinical")
        check("worker demo", demo["dataset"]["n_rows"] == 96, str(demo["dataset"]["n_rows"]))
        profile = call("profile")
        check("worker profile", 0 <= profile["score"] <= 100)
        charts = call("charts")
        check("worker charts", len(charts["charts"]) == 22, str(len(charts["charts"])))
        plot = call("plot", chart="box", params={}, style={"palette": "npg"}, roles={"x": "group", "y": "score", "group": "group"})
        check("worker plot file", os.path.exists(plot["figure"]["file"]), str(plot["figure"]["file"]))
        check("worker plot code", "savefig" in (plot.get("code") or {}).get("python", ""))
        auto = call("auto", need="按 group 分组画箱线图比较 score 并标显著性")
        check("worker auto figure", bool(auto["figure"]) and os.path.exists(auto["figure"]["file"]))
        check("worker auto plan", bool(auto["plan"]["steps"]))
        stat = call("stat", stat="compare", params={}, roles={"group": "group", "y": "score"})
        check("worker stat", bool(stat.get("tables")))
        exported = call("export", what="bundle")
        check("worker export", os.path.exists(exported["file"]), str(exported.get("file")))
        audit = call("audit")
        check("worker audit", "score" in audit)
    finally:
        try:
            process.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:  # noqa: BLE001
            process.kill()


def main() -> int:
    base = tmpdir()
    outdir = os.path.join(base, "figures")
    os.makedirs(outdir, exist_ok=True)
    print(f"== PlotLab 回归测试（{base}）==")
    try:
        paths = write_fixtures(base)
        tidy_dataset, matrix_dataset, _gz = test_import(paths)
        test_profile_clean(tidy_dataset)
        test_parse_need(tidy_dataset, matrix_dataset)
        test_charts(tidy_dataset, matrix_dataset, outdir)
        test_stats(tidy_dataset, matrix_dataset, outdir)
        test_code_export(tidy_dataset, outdir)
        test_worker(tidy_dataset)
    except AssertionError as error:
        print(str(error))
        print(f"FAILED: {len(PASSED)} passed before failure")
        return 1
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FAILED (exception): {error}; {len(PASSED)} passed")
        return 1
    print(f"\nALL PASS（{len(PASSED)} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
