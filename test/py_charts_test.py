#!/usr/bin/env python3
"""dsh-plotlab 图表模块自测：22 张图渲染 + 样式变体 + 显著性开关 + 代码生成。

运行：
    export MPLBACKEND=Agg MPLCONFIGDIR=$(mktemp -d)
    PYTHONPATH=lib python3 test/py_charts_test.py

覆盖：
1. 22 张图各渲染一次（matrix 图用 expression demo；roc/km/forest 用 clinical；
   直方/密度/QQ 用显式数值角色），断言 PNG > 5KB、summary.bullets 非空、
   python_code 含 savefig 且可编译、r_code 非空。
2. 3 组样式变体（PingFang SC 中文标题 / palette=npg / fig_width=10+dpi=150）全图重渲。
3. stat_annot 开/关都能跑，且关掉时 stats 为 None。
4. 非法 style 兜底、22 张图的 python_code 片段全部实际执行一次（独立进程内 exec）。
全部通过打印 PASS 并 exit 0。
"""
import os
import sys
import tempfile
import time
import traceback

import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import pl_charts as CH  # noqa: E402
import pl_prep as P  # noqa: E402
import pl_state as S  # noqa: E402

OUTDIR = tempfile.mkdtemp(prefix="plotlab_charts_test_")

# chart -> (demo, params, roles)；roles=None 用 pl_prep.recommend_roles
CASES = {
    "bar": ("clinical", {}, None),
    "bar_dot": ("clinical", {}, None),
    "box": ("clinical", {}, None),
    "violin": ("clinical", {}, None),
    "raincloud": ("clinical", {}, None),
    "dotplot": ("clinical", {}, None),
    "stack": ("clinical", {}, None),
    "pie": ("clinical", {}, None),
    "scatter": ("clinical", {}, {"x": "bmi", "y": "crp", "color": "group"}),
    "bubble": ("clinical", {}, {"x": "bmi", "y": "crp", "size": "score", "color": "group"}),
    "line": ("dose", {}, {"x": "dose", "y": "viability", "group": "genotype"}),
    "area": ("dose", {}, {"x": "dose", "y": "viability", "group": "genotype"}),
    "heatmap": ("expression", {"sample_group_col": "group"}, None),
    "corr": ("expression", {}, None),
    "pairplot": ("clinical", {}, None),
    "hist": ("clinical", {}, {"x": "score", "color": "group"}),
    "density": ("clinical", {}, {"x": "crp", "color": "group"}),
    "qq": ("clinical", {}, {"x": "score"}),
    "roc": ("clinical", {}, None),
    "km": ("clinical", {"risk_table": True}, None),
    "forest": ("clinical", {"terms": "age,crp,score,bmi"},
               {"time": "survival_months", "event": "event"}),
    "pca": ("expression", {"sample_group_col": "group"}, None),
}

STYLE_VARIANTS = [
    {"dpi": 110, "font": "PingFang SC", "title_text": "中文标题样式测试", "title_size": 18},
    {"dpi": 110, "palette": "npg", "legend_pos": "bottom", "grid": "y"},
    {"dpi": 150, "fig_width": 10, "fig_height": 6, "palette": "lancet",
     "frame": "full", "x_rotate": 30},
]

DEFAULT_STYLE = {"dpi": 110}
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build(demo, chart, params, roles):
    meta = CH.CHART_INDEX[chart]
    resolved_roles = roles if roles is not None else P.recommend_roles(demo, meta)
    return P.build_data(demo, meta, dict(params), resolved_roles)


def check_render_result(result, chart, style, elapsed):
    check(isinstance(result, dict), f"{chart}: render 返回类型错误")
    figures = result.get("figures") or []
    check(len(figures) >= 1, f"{chart}: figures 为空")
    figure = figures[0]
    check(os.path.exists(figure["file"]), f"{chart}: PNG 不存在 {figure.get('file')}")
    size = os.path.getsize(figure["file"])
    check(size > 5000, f"{chart}: PNG 太小（{size} bytes）")
    check(figure.get("width") and figure.get("height"), f"{chart}: figure 缺 width/height")
    check(str(figure["id"]), f"{chart}: figure id 为空")
    bullets = (result.get("summary") or {}).get("bullets") or []
    check(len(bullets) > 0, f"{chart}: summary.bullets 为空")
    check(all(isinstance(item, str) and item.strip() for item in bullets),
          f"{chart}: summary.bullets 含空字符串")
    check(elapsed < 3.0, f"{chart}: 单图耗时 {elapsed:.2f}s 超预期（>3s）")
    return figure, size


def run_render(demo, chart, params, roles, style):
    data = build(demo, chart, params, roles)
    start = time.perf_counter()
    result = CH.render(chart, data, params, style, OUTDIR)
    elapsed = time.perf_counter() - start
    return data, result, elapsed


def validate_codes(data, chart, params, style, execute=False):
    python_source = CH.python_code(chart, data, params, style)
    check(isinstance(python_source, str) and python_source.strip(), f"{chart}: python_code 为空")
    check("savefig" in python_source, f"{chart}: python_code 缺少 savefig")
    check('plt.savefig("plotlab_figure.png"' in python_source, f"{chart}: python_code savefig 行不规范")
    check("fig, ax" in python_source or "plt.subplots" in python_source,
          f"{chart}: python_code 缺少显式画布")
    compile(python_source, f"<{chart}_python>", "exec")
    r_source = CH.r_code(chart, data, params, style)
    check(isinstance(r_source, str) and len(r_source) > 100, f"{chart}: r_code 为空或过短")
    check("ggsave" in r_source, f"{chart}: r_code 缺少 ggsave")
    if execute:
        workdir = tempfile.mkdtemp(prefix=f"plotlab_code_{chart}_")
        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            namespace = {"__name__": "__main__"}
            exec(compile(python_source, f"{chart}_snippet.py", "exec"), namespace)
        finally:
            os.chdir(cwd)
        snippet_png = os.path.join(workdir, "plotlab_figure.png")
        check(os.path.exists(snippet_png) and os.path.getsize(snippet_png) > 5000,
              f"{chart}: 生成的 Python 片段执行后没有产出 PNG")


def main() -> int:
    print(f"PlotLab 图表自测：{len(CH.CHARTS)} 个图表，输出目录 {OUTDIR}")
    check(len(CH.CHARTS) == 22, f"CHARTS 数量应为 22，实际 {len(CH.CHARTS)}")
    chart_ids = [chart["id"] for chart in CH.CHARTS]
    check(len(set(chart_ids)) == 22, "CHARTS id 有重复")
    check(set(chart_ids) == set(CASES), "CASES 与 CHARTS 不一致")
    for chart in CH.CHARTS:
        for field in ("id", "title", "group", "desc", "needs", "roles", "params", "stat"):
            check(field in chart, f"{chart.get('id')}: CHARTS 缺少字段 {field}")
        check(chart["needs"] in ("tidy", "wide"), f"{chart['id']}: needs 非法")
        for param in chart["params"]:
            for field in ("key", "label", "type", "default"):
                check(field in param, f"{chart['id']}: param 缺少字段 {field}")
            check(param["type"] in ("number", "select", "boolean", "text"),
                  f"{chart['id']}: param 类型非法")

    demos = {name: S.create_demo(name) for name in ("clinical", "expression", "dose")}

    # ---- 1/4 22 张图默认渲染 + 代码生成 ----
    print("== [1/4] 22 张图各渲染一次 ==")
    figures_seen = {}
    timings = []
    for chart, (demo_name, params, roles) in CASES.items():
        demo = demos[demo_name]
        try:
            data, result, elapsed = run_render(demo, chart, params, roles, DEFAULT_STYLE)
            figure, size = check_render_result(result, chart, DEFAULT_STYLE, elapsed)
            timings.append((chart, elapsed, size))
            check(figure["id"] not in figures_seen,
                  f"{chart}: figure id 与其他图重复（{figure['id']}）")
            figures_seen[figure["id"]] = chart
            validate_codes(data, chart, params, DEFAULT_STYLE)
            print(f"  {chart:10s} PASS  {elapsed * 1000:6.0f} ms  PNG {size // 1024:4d} KB  "
                  f"bullets={len(result['summary']['bullets'])} "
                  f"tables={len(result.get('tables') or [])}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{chart}: {type(exc).__name__}: {exc}")
            print(f"  {chart:10s} FAIL  {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
    if timings:
        slowest = max(timings, key=lambda item: item[1])
        print(f"  渲染耗时：最慢 {slowest[0]} {slowest[1] * 1000:.0f} ms；"
              f"平均 {sum(item[1] for item in timings) / len(timings) * 1000:.0f} ms")

    # ---- 2/4 三组样式变体 ----
    print("== [2/4] 3 组样式变体（字体/配色/画布）==")
    for index, style in enumerate(STYLE_VARIANTS, start=1):
        failed_here = 0
        for chart, (demo_name, params, roles) in CASES.items():
            if failures and chart in {item.split(":")[0] for item in failures}:
                continue
            try:
                demo = demos[demo_name]
                _data, result, elapsed = run_render(demo, chart, params, roles, style)
                check_render_result(result, chart, style, elapsed)
            except Exception as exc:  # noqa: BLE001
                failed_here += 1
                failures.append(f"{chart}@style{index}: {type(exc).__name__}: {exc}")
                print(f"  style{index} {chart:10s} FAIL {type(exc).__name__}: {exc}")
        if not failed_here:
            print(f"  style{index} 全 22 图 PASS  {style}")

    # ---- 3/4 stat_annot 开 / 关 ----
    print("== [3/4] stat_annot 开 / 关 ==")
    stat_charts = [chart["id"] for chart in CH.CHARTS if chart.get("stat")]
    check(len(stat_charts) >= 5, "带组间显著性的图表少于 5 个")
    for chart in stat_charts:
        demo_name, params, roles = CASES[chart]
        demo = demos[demo_name]
        for annot in (True, False):
            try:
                style = {"dpi": 100, "stat_annot": annot}
                data, result, elapsed = run_render(demo, chart, params, roles, style)
                check_render_result(result, chart, style, elapsed)
                if annot:
                    check(result.get("stats") is not None,
                          f"{chart}: stat_annot=True 时 stats 为空")
                    check(result.get("tables"), f"{chart}: stat_annot=True 时没有比较表")
                else:
                    check(result.get("stats") is None,
                          f"{chart}: stat_annot=False 时不应返回 stats")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{chart}@stat_annot={annot}: {type(exc).__name__}: {exc}")
                print(f"  {chart:10s} stat_annot={annot} FAIL {type(exc).__name__}: {exc}")
        if not any(item.startswith(f"{chart}@stat_annot") for item in failures):
            print(f"  {chart:10s} stat_annot True/False PASS")

    # ---- 4/4 兜底与片段可执行性 ----
    print("== [4/4] 非法 style 兜底 + 生成片段实际执行 ==")
    demo = demos["clinical"]
    try:
        garbage_style = {"dpi": "not-a-number", "palette": "不存在的配色", "font": "NoSuchFont",
                         "fig_width": 999, "stat_method": "??", "p_adjust": "??"}
        _data, result, elapsed = run_render(demo, "box", {}, None, garbage_style)
        check_render_result(result, "box", garbage_style, elapsed)
        print("  非法 style 兜底 PASS")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"garbage_style: {type(exc).__name__}: {exc}")
        print(f"  非法 style 兜底 FAIL {type(exc).__name__}: {exc}")

    for chart, (demo_name, params, roles) in CASES.items():
        try:
            data = build(demos[demo_name], chart, params, roles)
            validate_codes(data, chart, params, {"dpi": 100}, execute=True)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{chart}@exec: {type(exc).__name__}: {exc}")
            print(f"  {chart:10s} 片段执行 FAIL {type(exc).__name__}: {exc}")
    print(f"  生成 Python 片段执行校验：{22 - sum(1 for item in failures if '@exec' in item)}/22 PASS")

    # ---- 汇总 ----
    print()
    if failures:
        print(f"FAIL：{len(failures)} 项")
        for item in failures:
            print("  -", item)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
