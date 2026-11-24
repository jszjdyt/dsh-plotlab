"""PlotLab 导出：图片 / 代码 / 报告 / Methods / protocol / 清洗数据 / 统计表 / bundle.zip。"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import zipfile
from typing import Any, Mapping, Sequence

import pandas as pd

import pl_common as C
import pl_clean
import pl_prep as P

VERSION = "0.1.0"

EXPORT_KINDS = [
    {"id": "figure", "label": "图片 PNG（300 dpi）", "ext": "png"},
    {"id": "code", "label": "可复现代码（Python + R）", "ext": "py"},
    {"id": "report", "label": "分析报告 report.md", "ext": "md"},
    {"id": "methods", "label": "英文 Methods（论文可用）", "ext": "md"},
    {"id": "protocol", "label": "溯源 protocol.json", "ext": "json"},
    {"id": "cleaned", "label": "清洗后数据 CSV", "ext": "csv"},
    {"id": "stats", "label": "统计结果表 CSV", "ext": "csv"},
    {"id": "bundle", "label": "打包（图+代码+报告+数据），zip", "ext": "zip"},
]


def _stamp() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _safe(name: str) -> str:
    return C.slug(name, "plotlab")


def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_text(path: str, text: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _figures_from_last_plot(last_plot: Mapping[str, Any] | None) -> list[dict]:
    if not last_plot:
        return []
    figures = list(last_plot.get("figures") or [])
    if not figures and last_plot.get("figure"):
        figures = [last_plot["figure"]]
    return figures


def _table_markdown(table: Mapping[str, Any]) -> str:
    columns = table.get("columns") or []
    rows = table.get("rows") or []
    if not columns:
        return ""
    header = "| " + " | ".join(str(c.get("label") or c.get("key")) for c in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows[:60]:
        cells = []
        for column in columns:
            value = row.get(column.get("key"))
            cells.append(C.fmt_p(value) if column.get("type") == "p" else C.fmt_num(value) if column.get("type") == "num" else ("—" if value is None else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > 60:
        lines.append(f"\n*（仅显示前 60 / {len(rows)} 行）*")
    note = table.get("note")
    if note:
        lines.append(f"\n{note}")
    return "\n".join(lines)


def _clean_markdown(dataset: Mapping[str, Any]) -> str:
    steps = dataset.get("cleaning") or []
    if not steps:
        return "未应用清洗步骤（数据按导入原样使用）。"
    lines = ["| # | 操作 | 说明 | 行数 |", "| --- | --- | --- | --- |"]
    for index, step in enumerate(steps, 1):
        lines.append(
            f"| {index} | {step.get('op')} | {step.get('reason') or '—'} | "
            f"{step.get('rows_before', '?')} → {step.get('rows_after', '?')} |"
        )
    return "\n".join(lines)


def _style_summary(style: Mapping[str, Any]) -> str:
    return (
        f"字体 {style.get('font')}，标题 {style.get('title_size')}pt，坐标轴标题 {style.get('label_size')}pt，"
        f"刻度 {style.get('tick_size')}pt，图例 {style.get('legend_size')}pt；配色 {style.get('palette')}，"
        f"画布 {style.get('fig_width')}×{style.get('fig_height')} in，{style.get('dpi')} dpi。"
    )


def _methods_text(dataset: Mapping[str, Any], chart_meta: Mapping[str, Any] | None,
                  stats_result: Mapping[str, Any] | None, style: Mapping[str, Any],
                  profile: Mapping[str, Any]) -> str:
    n_rows = int(dataset["df"].shape[0]) if dataset.get("df") is not None else 0
    n_cols = int(dataset["df"].shape[1]) if dataset.get("df") is not None else 0
    kind = "a feature-by-sample matrix" if dataset.get("kind") == "matrix" else "a tidy observation table"
    cleaning = dataset.get("cleaning") or []
    clean_text = "The dataset was used as imported." if not cleaning else (
        "Pre-processing comprised " + "; ".join(f"{step.get('op')}" for step in cleaning) + "."
    )
    chart = chart_meta.get("title") if chart_meta else "the requested chart"
    chart_id = chart_meta.get("id") if chart_meta else "plot"
    stat_sentence = ""
    if stats_result:
        method = stats_result.get("method") or "the corresponding statistical test"
        stat_sentence = (
            f" Statistical analysis was performed using {method}; "
            "p values were adjusted for multiple comparisons where applicable."
        )
    return (
        "# Methods (auto-generated)\n\n"
        f"Data analysis and visualization were performed with the PlotLab workbench (v{VERSION}) "
        f"running Python {__import__('sys').version.split()[0]} and matplotlib {__import__('matplotlib').__version__}. "
        f"The input comprised {n_rows} rows and {n_cols} columns representing {kind}. "
        f"{clean_text} Data quality was profiled with deterministic checks (quality score {profile.get('score')}/100). "
        f"Figure '{chart}' ({chart_id}) was generated with the following style: {_style_summary(style)}."
        f"{stat_sentence}\n"
    )


def _protocol(dataset: Mapping[str, Any], chart_id: str | None, chart_meta: Mapping[str, Any] | None,
              params: Mapping[str, Any], style: Mapping[str, Any], roles: Mapping[str, Any],
              last_plot: Mapping[str, Any] | None, stats_result: Mapping[str, Any] | None,
              profile: Mapping[str, Any]) -> dict:
    import platform
    import sys

    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "statsmodels", "sklearn", "lifelines"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "?")
        except Exception:  # noqa: BLE001
            packages[name] = None
    return C.jsonable(
        {
            "tool": "PlotLab 绘图工作台",
            "version": VERSION,
            "generated_at": _stamp(),
            "engine": {"python": sys.executable, "version": sys.version.split()[0], "platform": platform.platform(),
                       "packages": packages},
            "dataset": {
                "id": dataset.get("id"),
                "name": dataset.get("name"),
                "source": dataset.get("source"),
                "kind": dataset.get("kind"),
                "n_rows": int(dataset["df"].shape[0]) if dataset.get("df") is not None else 0,
                "n_cols": int(dataset["df"].shape[1]) if dataset.get("df") is not None else 0,
                "columns": [c.get("key") for c in dataset.get("columns") or []],
                "group_col": dataset.get("group_col"),
                "groups": dataset.get("groups"),
                "need": dataset.get("need"),
            },
            "quality": {"score": profile.get("score"), "level": profile.get("level"),
                        "checks": [{"id": c["id"], "level": c["level"], "title": c["title"]} for c in profile.get("checks") or []]},
            "cleaning": dataset.get("cleaning") or [],
            "figure": {
                "chart": chart_id,
                "title": (chart_meta or {}).get("title"),
                "params": params,
                "roles": roles,
                "style": style,
                "figures": C.jsonable(_figures_from_last_plot(last_plot)),
            },
            "stats": {
                "stat": (stats_result or {}).get("stat"),
                "method": (stats_result or {}).get("method"),
                "summary": (stats_result or {}).get("summary"),
                "tables": [t.get("title") for t in (stats_result or {}).get("tables") or []],
            },
            "reproduce": "运行导出的 plotlab_*.py 可逐参数复现同一张图。",
        }
    )


def _report(dataset: Mapping[str, Any], chart_id: str | None, chart_meta: Mapping[str, Any] | None,
            last_plot: Mapping[str, Any] | None, code_python: str, stats_result: Mapping[str, Any] | None,
            profile: Mapping[str, Any], figure_paths: Sequence[str]) -> str:
    name = dataset.get("name") or "data"
    n_rows = int(dataset["df"].shape[0]) if dataset.get("df") is not None else 0
    n_cols = int(dataset["df"].shape[1]) if dataset.get("df") is not None else 0
    lines = [
        f"# PlotLab 分析报告：{name}",
        "",
        f"- 生成时间：{_stamp()}",
        f"- 数据：{n_rows} 行 × {n_cols} 列（{dataset.get('kind')}），来源：{dataset.get('source') or '—'}",
        f"- 数据质量：{profile.get('score')}/100（{profile.get('level')}）",
        f"- 图表：{(chart_meta or {}).get('title') or chart_id or '—'}",
        "",
        "## 1. 数据概览",
        "",
    ]
    for bullet in (profile.get("summary") or {}).get("bullets") or []:
        lines.append(f"- {bullet}")
    problems = [c for c in profile.get("checks") or [] if c.get("level") != "ok"]
    if problems:
        lines += ["", "**待改进项**", ""]
        for item in problems:
            lines.append(f"- **{item['title']}**：{item['detail']}" + (f"（建议：{item['fix']}）" if item.get("fix") else ""))
    lines += ["", "## 2. 数据清洗", "", _clean_markdown(dataset), ""]
    if figure_paths:
        lines += ["## 3. 图表", ""]
        for index, path in enumerate(figure_paths, 1):
            # bundle 里图片在 figures/ 下：优先用相对路径，保证 markdown 链接在 zip 内可点
            rel = path if not os.path.isabs(path) else os.path.basename(path)
            lines.append(f"![图 {index}]({rel})")
        summary = (last_plot or {}).get("summary") or {}
        for bullet in summary.get("bullets") or []:
            lines.append(f"- {bullet}")
        warnings = (last_plot or {}).get("warnings") or []
        if warnings:
            lines += ["", "**提示**", ""]
            for warning in warnings:
                lines.append(f"- {warning}")
        lines.append("")
    if stats_result:
        lines += ["## 4. 统计分析", "", f"方法：{stats_result.get('method') or '—'}", ""]
        for bullet in ((stats_result.get("summary") or {}).get("bullets") or []):
            lines.append(f"- {bullet}")
        for table in stats_result.get("tables") or []:
            lines += ["", f"### {table.get('title')}", "", _table_markdown(table)]
        stats_figs = stats_result.get("figures") or []
        if stats_figs:
            lines.append("")
            for index, figure in enumerate(stats_figs, 1):
                lines.append(f"![统计图 {index}]({os.path.basename(str(figure.get('file')))})")
        lines.append("")
    if code_python:
        lines += ["## 5. 关键步骤代码", "", "```python", code_python.strip(), "```", ""]
    lines += [
        "## 6. 复现方式",
        "",
        "1. 安装依赖：`pip install pandas numpy matplotlib seaborn scipy`（统计可能还需 statsmodels scikit-learn lifelines）",
        "2. 运行导出的 `plotlab_*.py`；脚本里已包含读取数据、重放清洗步骤、绘图与统计。",
        "3. 图片默认 300 dpi PNG。",
        "",
        "---",
        "",
        f"*由 PlotLab 绘图工作台 v{VERSION} 生成；如需修改字体/字号/颜色，在插件页面调整后重新导出即可。*",
    ]
    return "\n".join(lines)


def _stats_tables(stats_result: Mapping[str, Any] | None, last_plot: Mapping[str, Any] | None) -> list[dict]:
    tables: list[dict] = []
    if stats_result:
        tables.extend(stats_result.get("tables") or [])
    if last_plot:
        tables.extend(last_plot.get("tables") or [])
    return tables


def export(what: str, dataset: Mapping[str, Any], chart_id: str | None,
           meta: Mapping[str, Any] | None, data: Mapping[str, Any] | None,
           params: Mapping[str, Any], style: Mapping[str, Any], roles: Mapping[str, Any],
           charts_module: Any, codegen: Any, stats_result: Mapping[str, Any] | None,
           last_plot: Mapping[str, Any] | None, outdir: str,
           out_path: str | None = None, fmt: str = "png") -> dict:
    what = str(what or "bundle").strip().lower()
    valid = {item["id"] for item in EXPORT_KINDS}
    if what not in valid:
        raise C.bad_request(f"未知导出类型 {what}，可选：{', '.join(sorted(valid))}")
    _mkdir(outdir)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = _safe(str(dataset.get("name") or "plotlab"))
    tag = f"{base}_{stamp}"
    subdir = outdir if what in ("figure",) else os.path.join(outdir, tag)
    if what != "figure":
        _mkdir(subdir)
    exported: list[dict] = []
    profile = pl_clean.profile(dataset)

    figures = _figures_from_last_plot(last_plot)
    figure_paths = [str(item.get("file")) for item in figures if item.get("file")]

    python_code = ""
    r_code = ""
    try:
        if last_plot and last_plot.get("code"):
            python_code = last_plot["code"].get("python") or ""
            r_code = last_plot["code"].get("r") or ""
        elif chart_id and meta and data is not None:
            python_code = codegen.build_script(dataset, chart_id, meta, data, params, style, roles)
            r_code = codegen.chart_r_code(chart_id, data, params, style)
    except Exception as error:  # noqa: BLE001 - 代码生成失败不阻断导出
        python_code = f"# 代码生成失败：{error}"

    def add(kind: str, path: str) -> None:
        exported.append({"kind": kind, "path": path, "label": os.path.basename(path),
                         "bytes": os.path.getsize(path) if os.path.exists(path) else 0})

    if what in ("figure", "bundle"):
        if not figure_paths:
            raise C.prerequisite("还没有出图：先用 action=plot / auto 生成一张图再导出")
        target_dir = outdir if what == "figure" else _mkdir(os.path.join(subdir, "figures"))
        for index, source in enumerate(figure_paths, 1):
            ext = str(fmt or "png").lower()
            if ext not in ("png", "pdf", "jpg", "jpeg"):
                raise C.unsupported(f"图片格式 {ext} 暂不支持（可选 png/pdf/jpg）；SVG 需要矢量重绘，暂不提供")
            name = f"figure{'' if index == 1 else f'_{index}'}.{ext}"
            destination = os.path.join(target_dir, name)
            try:
                if ext == "png":
                    shutil.copyfile(source, destination)
                else:
                    from PIL import Image
                    with Image.open(source) as image:
                        image.convert("RGB").save(destination, "PDF" if ext == "pdf" else "JPEG", resolution=300)
            except Exception as error:  # noqa: BLE001
                raise C.EngineError("internal", f"图片导出失败：{error}")
            if what == "figure":
                return {"what": what, "file": destination, "dir": target_dir,
                        "files": [{"kind": "figure", "path": destination, "label": name, "bytes": os.path.getsize(destination)}],
                        "bytes": os.path.getsize(destination)}
            add("figure", destination)

    if what in ("code", "bundle"):
        if not python_code:
            raise C.prerequisite("没有可导出的代码：先出一张图")
        py_path = _write_text(os.path.join(subdir if what == "bundle" else outdir, f"plotlab_{chart_id or 'plot'}.py"), python_code)
        add("code", py_path)
        if r_code:
            r_path = _write_text(os.path.join(subdir if what == "bundle" else outdir, f"plotlab_{chart_id or 'plot'}.R"), r_code)
            add("code", r_path)
        if what == "code":
            return _finish(what, outdir, exported)

    if what in ("report", "bundle", "methods", "protocol", "cleaned", "stats"):
        pass

    if what in ("report", "bundle"):
        # 用导出后（实际写盘）的图片路径生成 markdown，避免报告里引用不存在的文件名
        report_figs = [os.path.relpath(item["path"], subdir) if what == "bundle" else item["path"]
                       for item in exported if item.get("kind") == "figure"]
        report = _report(dataset, chart_id, meta, last_plot, python_code, stats_result, profile,
                         report_figs if (what == "bundle" or report_figs) else [])
        path = _write_text(os.path.join(subdir, "report.md"), report)
        add("report", path)

    if what in ("methods", "bundle"):
        path = _write_text(os.path.join(subdir, "methods.md"),
                           _methods_text(dataset, meta, stats_result, style, profile))
        add("methods", path)

    if what in ("protocol", "bundle"):
        payload = _protocol(dataset, chart_id, meta, params, style, roles, last_plot, stats_result, profile)
        path = _write_text(os.path.join(subdir, "protocol.json"),
                           json.dumps(payload, ensure_ascii=False, indent=2))
        add("protocol", path)

    if what in ("cleaned", "bundle"):
        df = dataset.get("df")
        if df is None:
            raise C.prerequisite("没有数据可导出")
        path = os.path.join(subdir, "cleaned_data.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        add("cleaned", path)

    if what in ("stats", "bundle"):
        tables = _stats_tables(stats_result, last_plot)
        if tables:
            for table in tables:
                columns = [c.get("key") for c in table.get("columns") or []]
                frame = pd.DataFrame(table.get("rows") or [], columns=columns)
                path = os.path.join(subdir, f"stats_{_safe(str(table.get('id') or table.get('title')))}.csv")
                frame.to_csv(path, index=False, encoding="utf-8-sig")
                add("stats", path)
        elif what == "stats":
            raise C.prerequisite("没有统计结果可导出：先跑 action=stat")

    if what == "bundle":
        zip_path = os.path.join(outdir, f"plotlab_{tag}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in exported:
                archive.write(item["path"], os.path.relpath(item["path"], subdir))
            for figure in figure_paths:
                if os.path.exists(figure) and not os.path.exists(os.path.join(subdir, "figures", os.path.basename(figure))):
                    pass
            archive.writestr(
                "README.txt",
                "PlotLab 导出包\n"
                f"数据：{dataset.get('name')}（{int(dataset['df'].shape[0])}×{int(dataset['df'].shape[1])}）\n"
                "包含：图（figures/）、可复现代码（plotlab_*.py/.R）、报告（report.md）、英文 Methods、"
                "溯源（protocol.json）、清洗后数据、统计表。\n"
                "复现：python plotlab_*.py\n",
            )
        size = os.path.getsize(zip_path)
        return {"what": what, "file": zip_path, "dir": subdir,
                "files": [{"kind": "bundle", "path": zip_path, "label": os.path.basename(zip_path), "bytes": size}]
                + exported,
                "bytes": size}

    return _finish(what, outdir, exported)


def _finish(what: str, outdir: str, exported: list[dict]) -> dict:
    total = sum(item.get("bytes") or 0 for item in exported)
    return {"what": what, "file": exported[0]["path"] if exported else None,
            "dir": outdir, "files": exported, "bytes": total}
