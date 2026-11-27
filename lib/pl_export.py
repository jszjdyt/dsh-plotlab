"""PlotLab 导出：图片 / 代码 / 报告 / Methods / protocol / 清洗数据 / 统计表 / bundle.zip。"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import zipfile
from typing import Any, Callable, Mapping, Sequence

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
              profile: Mapping[str, Any], binding: Mapping[str, Any] | None = None) -> dict:
    import platform
    import sys

    binding = binding or {}
    figure_run_id = (last_plot or {}).get("figure_run_id") or binding.get("figure_run_id")
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
                "data_digest": (last_plot or {}).get("data_digest"),
            },
            "quality": {"score": profile.get("score"), "level": profile.get("level"),
                        "checks": [{"id": c["id"], "level": c["level"], "title": c["title"]} for c in profile.get("checks") or []]},
            "cleaning": dataset.get("cleaning") or [],
            "figure": {
                "chart": chart_id,
                "title": (chart_meta or {}).get("title"),
                "figure_run_id": figure_run_id,
                "data_digest": (last_plot or {}).get("data_digest") or binding.get("data_digest"),
                # 导出绑定：记录本次导出请求与 figure run 的比对结论（一致/重画/旧记录），
                # 证据链可凭 figure_run_id 回查到具体是哪一张图，而不是靠「最近一次」猜。
                "binding": C.jsonable({
                    "version": FIGURE_BINDING_VERSION,
                    "status": binding.get("status"),
                    "chart": binding.get("chart"),
                    "request": binding.get("request"),
                    "diffs": binding.get("diffs") or [],
                    "notes": binding.get("notes") or [],
                }),
                "params": params,
                "roles": roles,
                "style": style,
                # 一个 run 固化以下全部内容，保证「协议里的图」和「这次导出的图」同源
                "tables": C.jsonable((last_plot or {}).get("tables") or []),
                "layers": C.jsonable((last_plot or {}).get("layers") or []),
                "confidence_intervals": C.jsonable((last_plot or {}).get("confidence_intervals") or []),
                "stats": C.jsonable((last_plot or {}).get("stats")),
                "annotations": C.jsonable((last_plot or {}).get("annotations")),
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
            profile: Mapping[str, Any], figure_paths: Sequence[str],
            binding: Mapping[str, Any] | None = None) -> str:
    binding = binding or {}
    binding_labels = {
        "reused": "与导出请求一致（复用同一 figure run）",
        "rerendered": "与上一张图不一致 → 已按导出请求重画（rerender）",
        "rendered": "按导出请求现场出图",
        "legacy": "旧记录（无 figure_run_id，已按请求 + 数据指纹派生）",
        "absent": "本次导出不包含图",
    }
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
    ]
    report_run_id = str((last_plot or {}).get("figure_run_id") or binding.get("figure_run_id") or "")
    if report_run_id:
        lines.append(
            f"- 图形绑定：figure_run_id={report_run_id}；校验："
            + binding_labels.get(str(binding.get("status") or ""), "已绑定")
        )
    lines += ["", "## 1. 数据概览", ""]
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


# ---------------------------------------------------------------------------
# 导出绑定：把「这次导出的图」钉死到同一个 figure_run_id
#
# 旧 bug（串图）：先画箱线图 → 再请求导出 bar 图。导出实现只读 last_plot，
# 于是把上一张箱线图原样复制出去，而报告/protocol 里写的是 bar —— 导出的图和
# 导出的文字不是同一张图，证据链断裂。修复原则：导出前先拿 figure run 记录
# （figure_run_id + chart + 关键 params/style/roles 快照 + 数据指纹）与本次导出
# 请求逐项比对，只有三种结局：
#   ① 一致        → 复用该 run，并让所有导出 artifacts 记录 figure_run_id；
#   ② 不一致      → 必须按请求重画（rerender），绝不复用旧图；
#   ③ 无法重画    → 抛含「图形与导出请求不一致」的明确错误 + 修复建议，
#                    宁可失败也不静默导出旧图。
# 下面这些函数只做「取证 + 分派重画」，真正的渲染仍由 registry 的 figure run
# 或 charts.render 完成（保证同一套参数/样式语义）。
# ---------------------------------------------------------------------------

FIGURE_BINDING_VERSION = 1


def _canonical(value: Any) -> str:
    """稳定序列化：比较 figure run 快照与导出请求是否指向同一份内容。"""
    return json.dumps(C.jsonable(value), sort_keys=True, ensure_ascii=False, default=str)


def _figure_data_digest(dataset: Mapping[str, Any]) -> str:
    """当前数据集的数据指纹；与 registry 里 figure run 的 data_digest 同口径。"""
    try:
        import pl_registry

        return str(pl_registry.data_digest(dataset))
    except Exception:  # noqa: BLE001 - registry 不可用时退化：不做数据指纹比对
        return ""


def _figure_run_id_for(dataset: Mapping[str, Any], chart_id: str, params: Mapping[str, Any],
                       style: Mapping[str, Any], roles: Mapping[str, Any] | None) -> str:
    """按 registry 的同一套 key 派生 figure_run_id（直接调用 export 的兼容路径用）。"""
    clean_roles = {str(k): v for k, v in (roles or {}).items() if v}
    try:
        import pl_registry

        key = pl_registry._figure_run_key(dataset, str(chart_id or ""), dict(params or {}),
                                          dict(style or {}), clean_roles)
        return str(pl_registry._figure_run_id(key))
    except Exception:  # noqa: BLE001 - 兜底：同一确定性 hash，保证本进程内可回查
        payload = _canonical({
            "dataset": dataset.get("id"), "chart": str(chart_id or ""),
            "params": dict(params or {}), "style": dict(style or {}),
            "roles": clean_roles, "data_digest": _figure_data_digest(dataset),
        })
        return "fr_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _figure_snapshot(record: Mapping[str, Any] | None) -> dict:
    """从 figure run 记录抽出用于比对的快照（run id / 图型 / 参数 / 样式 / 角色 / 指纹 / 图文件）。"""
    if not isinstance(record, Mapping):
        return {}
    figures = list(record.get("figures") or [])
    if not figures and record.get("figure"):
        figures = [record.get("figure")]
    return {
        "figure_run_id": str(record.get("figure_run_id") or ""),
        "chart": str(record.get("chart") or ""),
        "params": C.jsonable(dict(record.get("params") or {})),
        "style": C.jsonable(dict(record.get("style") or {})),
        "roles": {str(k): v for k, v in (record.get("roles") or {}).items() if v},
        "data_digest": str(record.get("data_digest") or ""),
        "figures": [str(item.get("file")) for item in figures
                    if isinstance(item, Mapping) and item.get("file")],
    }


def _figure_binding_request(dataset: Mapping[str, Any], chart_id: str | None,
                            params: Mapping[str, Any], style: Mapping[str, Any],
                            roles: Mapping[str, Any] | None) -> dict:
    """本次导出请求的快照（与 _figure_snapshot 同格式，便于逐项比对）。"""
    return {
        "chart": str(chart_id or ""),
        "params": C.jsonable(dict(params or {})),
        "style": C.jsonable(dict(style or {})),
        "roles": {str(k): v for k, v in (roles or {}).items() if v},
        "data_digest": _figure_data_digest(dataset),
    }


def _figure_binding_diffs(snapshot: Mapping[str, Any], request: Mapping[str, Any]) -> list[str]:
    """比对 figure run 快照与导出请求，返回中文差异说明（空列表 = 同一张图）。"""
    diffs: list[str] = []
    request_chart = str(request.get("chart") or "")
    record_chart = str(snapshot.get("chart") or "")
    if request_chart and record_chart and request_chart != record_chart:
        diffs.append(f"chart 不一致（请求 {request_chart} ≠ 记录 {record_chart}）")
    if request.get("params") != snapshot.get("params"):
        diffs.append("params 快照不一致（图表参数与记录不同）")
    if request.get("style") != snapshot.get("style"):
        diffs.append("style 快照不一致（样式与记录不同）")
    if request.get("roles") != snapshot.get("roles"):
        diffs.append("roles 快照不一致（列角色与记录不同）")
    record_digest = str(snapshot.get("data_digest") or "")
    request_digest = str(request.get("data_digest") or "")
    if record_digest and request_digest and record_digest != request_digest:
        diffs.append(f"数据指纹不一致（记录 {record_digest} ≠ 当前 {request_digest}）")
    return diffs


def _render_figure_record(*, dataset: Mapping[str, Any], chart_id: str,
                          meta: Mapping[str, Any] | None, params: Mapping[str, Any],
                          style: Mapping[str, Any], roles: Mapping[str, Any] | None,
                          data: Mapping[str, Any], charts_module: Any, outdir: str) -> dict:
    """按导出请求现场重画一张图，返回与 registry figure run 同构的记录。"""
    if not chart_id:
        raise C.bad_request("按导出请求重画图形需要 chart（图表 id）")
    try:
        result = charts_module.render(str(chart_id), data, params, style, outdir)
    except C.EngineError:
        raise
    except Exception as error:  # noqa: BLE001 - 渲染失败要变成明确错误，不能回退旧图
        raise C.EngineError("internal", f"按导出请求重画图形失败：{error}", chart=chart_id) from error
    figures = C.jsonable(result.get("figures") or [])
    if not figures:
        raise C.EngineError("internal", f"按导出请求重画 {chart_id} 没有产出图片", chart=chart_id)
    record = {
        "chart": str(chart_id),
        "chart_title": (meta or {}).get("title") or str(chart_id),
        "params": C.jsonable(dict(params or {})),
        "style": C.jsonable(dict(style or {})),
        "roles": {str(k): v for k, v in (roles or {}).items() if v},
        "data_digest": _figure_data_digest(dataset),
        "figures": figures,
        "figure": figures[0],
        "summary": C.jsonable(result.get("summary") or {}),
        "tables": C.jsonable(result.get("tables") or []),
        "warnings": C.jsonable(result.get("warnings") or []),
        "code": {},
        "source": "export_rerender",
    }
    record["figure_run_id"] = _figure_run_id_for(
        dataset, record["chart"], record["params"], record["style"], record["roles"])
    return record


def _ensure_figure_current(*, what: str, dataset: Mapping[str, Any], chart_id: str | None,
                           params: Mapping[str, Any], style: Mapping[str, Any],
                           roles: Mapping[str, Any] | None, meta: Mapping[str, Any] | None,
                           last_plot: Mapping[str, Any] | None,
                           figure_run: Mapping[str, Any] | None = None,
                           data: Mapping[str, Any] | None = None,
                           charts_module: Any = None, outdir: str = "",
                           rerender: Callable[[], Mapping[str, Any] | None] | None = None,
                           frozen: bool = False) -> dict:
    """导出前校验「要导出的图」就是「本次请求的那张图」。

    last_plot  = 「最近一次真正渲染出来的图」记录（比对基线，来自 registry 的 last_plot/run）；
    figure_run = 导出实现已经解析好的目标 run（可选；复用分支优先用它，保证带全图层/统计/CI）；
    rerender   = registry 提供的「按请求重新取/建 figure run」回调（有它才能分支②）。

    返回绑定结果 dict：
      run / figure_run_id / status（reused|rerendered|rendered|legacy|absent）/
      chart / request（导出请求快照）/ diffs（不一致项）/ notes / data_digest

    三种分支（对应旧 bug：先画箱线、再请求导出 bar，导出仍是箱线图）：
      ① 一致     → 复用该 figure run，导出 artifacts 记录 figure_run_id；
      ② 不一致   → 必须重画（rerender）成请求要求的图，绝不复用旧 PNG；
      ③ 不可重画 → 抛 prerequisite，消息含「图形与导出请求不一致」+ 修复建议，
                   宁可报错也不静默导出旧图。
    """
    request = _figure_binding_request(dataset, chart_id, params, style, roles)
    needs_figure = str(what or "").strip().lower() in ("figure", "bundle")
    record = last_plot if isinstance(last_plot, Mapping) else None
    baseline = _figure_snapshot(record) if record else {}
    target = figure_run if isinstance(figure_run, Mapping) else None
    target_snapshot = _figure_snapshot(target) if target else {}
    target_chart = (request["chart"] or str(baseline.get("chart") or "")
                    or str(target_snapshot.get("chart") or ""))
    notes: list[str] = []
    if frozen and baseline:
        # 导出的是固定历史版本：当前数据集可能已被清洗/修改，不能拿「当前数据指纹」
        # 去判它 stale——以该版本记录的指纹为准，否则会把归档回查误判成不一致而重画。
        request["data_digest"] = str(baseline.get("data_digest") or "")

    def make_run() -> dict | None:
        """按导出请求取/建一个 figure run；做不到时返回 None（交给调用方报错）。"""
        if callable(rerender):
            produced = rerender()
            return dict(produced) if isinstance(produced, Mapping) else None
        if data is None or charts_module is None:
            return None
        return _render_figure_record(
            dataset=dataset, chart_id=target_chart, meta=meta, params=params,
            style=style, roles=roles, data=data, charts_module=charts_module, outdir=outdir,
        )

    def payload(run: dict | None, status: str, diffs: list[str],
                digest: str = "") -> dict:
        return {
            "run": run,
            "figure_run_id": str((run or {}).get("figure_run_id") or ""),
            "status": status,
            "chart": target_chart,
            "request": request,
            "diffs": diffs,
            "notes": notes,
            "data_digest": digest or request.get("data_digest") or "",
        }

    def verify(produced: Mapping[str, Any]) -> tuple[dict, dict]:
        """重画后必须与导出请求逐项一致；否则说明重画链路本身串了，直接报错。"""
        fresh = _figure_snapshot(produced)
        remaining = _figure_binding_diffs(fresh, request)
        if remaining:
            raise C.EngineError(
                "internal",
                "重画后的图形仍与导出请求不一致：" + "；".join(remaining)
                + "。请用 action=plot 显式出图后重试",
                diffs=remaining,
            )
        run = dict(produced)
        figure_run_id = fresh.get("figure_run_id") or _figure_run_id_for(
            dataset, target_chart, params, style, roles)
        run["figure_run_id"] = figure_run_id
        return run, fresh

    # 没有基线也没有目标 run：只有「必须带图」的导出才现场画一张
    if record is None and target is None:
        if not needs_figure:
            return payload(None, "absent", [])
        produced = make_run()
        if produced is None:
            raise C.prerequisite(
                "还没有出图：先用 action=plot / auto 生成一张图再导出，"
                "或在导出时同时给出 chart + params（引擎会现场作图）"
            )
        run, fresh = verify(produced)
        return payload(run, "rendered", [], str(fresh.get("data_digest") or ""))

    # 基线 = 最近一次渲染记录；没有记录时（例如首次导出就显式给 chart）用目标 run 比对
    if record is not None:
        diffs = _figure_binding_diffs(baseline, request)
        if baseline.get("figures"):
            missing = [path for path in baseline["figures"] if not os.path.isfile(path)]
            if missing and frozen:
                # 固定历史版本的 PNG 被清理/移动：既不能重画回旧数据，也不该拿别的图顶替
                raise C.prerequisite(
                    "固定历史版本的图形文件缺失（stale artifact），无法按 figure_run_id 导出："
                    + "、".join(missing) + "。修复建议：重新 action=plot 出图后再导出，"
                    "或从备份恢复该 figure run 的图片文件。"
                )
            if missing:
                diffs.append("图形文件缺失（stale artifact：记录指向的 PNG 已不存在或被清理）")
        else:
            diffs.append("图记录没有可导出的图片")
    else:
        diffs = _figure_binding_diffs(target_snapshot, request)

    if not diffs and target is not None:
        # 已解析的目标 run 也必须与请求一致（防御 registry 解析与导出参数脱节）
        target_diffs = _figure_binding_diffs(target_snapshot, request)
        if target_diffs:
            diffs = target_diffs

    if not diffs:
        # ① 一致：复用；有已解析目标 run 时用它（图层/统计/CI 更全）
        chosen = target if target is not None else record
        chosen_snapshot = target_snapshot if target is not None else baseline
        figure_run_id = str(chosen_snapshot.get("figure_run_id") or "") or _figure_run_id_for(
            dataset, target_chart, params, style, roles)
        run = dict(chosen or {})
        run["figure_run_id"] = figure_run_id
        status = "reused" if chosen_snapshot.get("figure_run_id") else "legacy"
        if status == "legacy":
            notes.append(
                "该图记录来自旧协议（没有 figure_run_id），已按本次请求 + 数据指纹派生 "
                f"{figure_run_id}；如需严格回查，请用 op_plot 重新出图后再导出"
            )
        return payload(run, status, [],
                       str(chosen_snapshot.get("data_digest") or request.get("data_digest") or ""))

    # ② 不一致：必须重画成请求要求的图，不能复用旧 PNG
    produced = make_run()
    if produced is None:
        # ③ 不可重画：明确报错（含修复建议），绝不静默导出旧图
        raise C.prerequisite(
            "图形与导出请求不一致，且当前无法重画：" + "；".join(diffs) + "。"
            "修复建议：1) 先 action=plot 按目标 chart/params 重新出图，再导出；"
            "2) 或导出时同时给出 chart + params 并保留当前数据集；"
            "3) 或用 action=records 查到目标 figure_run_id 后按 id 导出（不会串图）。",
            figure_run_id=baseline.get("figure_run_id") or None,
            diffs=diffs,
        )
    run, fresh = verify(produced)
    notes.append(
        "导出请求与最近一次渲染不一致（" + "；".join(diffs)
        + "），已按请求重画（rerender），没有复用旧图"
    )
    return payload(run, "rerendered", diffs, str(fresh.get("data_digest") or ""))


def export(what: str, dataset: Mapping[str, Any], chart_id: str | None,
           meta: Mapping[str, Any] | None, data: Mapping[str, Any] | None,
           params: Mapping[str, Any], style: Mapping[str, Any], roles: Mapping[str, Any],
           charts_module: Any, codegen: Any, stats_result: Mapping[str, Any] | None,
           last_plot: Mapping[str, Any] | None, outdir: str,
           out_path: str | None = None, fmt: str = "png",
           rerender: Callable[[], Mapping[str, Any] | None] | None = None,
           figure_run: Mapping[str, Any] | None = None, frozen: bool = False) -> dict:
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

    # 导出前先做 figure 绑定校验：要导出的图必须就是本次请求的那张图（见 _ensure_figure_current）。
    binding = _ensure_figure_current(
        what=what, dataset=dataset, chart_id=chart_id, params=params, style=style,
        roles=roles, meta=meta, last_plot=last_plot, figure_run=figure_run,
        data=data, charts_module=charts_module, outdir=outdir, rerender=rerender,
        frozen=frozen,
    )
    if binding.get("run") is not None:
        last_plot = binding["run"]
        chart_id = str(last_plot.get("chart") or chart_id or "")
        if meta is None and chart_id:
            meta = {"id": chart_id, "title": last_plot.get("chart_title") or chart_id}
    binding_id = str(binding.get("figure_run_id") or "")

    def tag_artifacts(payload: dict) -> dict:
        """导出 artifacts 统一带上 figure_run_id，供 evidence 链按 id 回查。"""
        if binding_id:
            payload["figure_run_id"] = binding_id
            payload["data_digest"] = binding.get("data_digest")
            payload["figure_binding"] = C.jsonable({
                "version": FIGURE_BINDING_VERSION,
                "status": binding.get("status"),
                "verified": str(binding.get("status") or "") in
                            ("reused", "legacy", "rerendered", "rendered"),
                "chart": binding.get("chart"),
                "diffs": binding.get("diffs") or [],
                "notes": binding.get("notes") or [],
            })
        return payload

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
        item = {"kind": kind, "path": path, "label": os.path.basename(path),
                "bytes": os.path.getsize(path) if os.path.exists(path) else 0}
        if binding_id:
            item["figure_run_id"] = binding_id
        exported.append(item)

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
                figure_artifact = {"kind": "figure", "path": destination, "label": name,
                                   "bytes": os.path.getsize(destination)}
                if binding_id:
                    figure_artifact["figure_run_id"] = binding_id
                return tag_artifacts({"what": what, "file": destination, "dir": target_dir,
                                      "files": [figure_artifact],
                                      "bytes": os.path.getsize(destination)})
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
            return tag_artifacts(_finish(what, outdir, exported))

    if what in ("report", "bundle", "methods", "protocol", "cleaned", "stats"):
        pass

    if what in ("report", "bundle"):
        # 用导出后（实际写盘）的图片路径生成 markdown，避免报告里引用不存在的文件名
        report_figs = [os.path.relpath(item["path"], subdir) if what == "bundle" else item["path"]
                       for item in exported if item.get("kind") == "figure"]
        report = _report(dataset, chart_id, meta, last_plot, python_code, stats_result, profile,
                         report_figs if (what == "bundle" or report_figs) else [],
                         binding=binding)
        path = _write_text(os.path.join(subdir, "report.md"), report)
        add("report", path)

    if what in ("methods", "bundle"):
        path = _write_text(os.path.join(subdir, "methods.md"),
                           _methods_text(dataset, meta, stats_result, style, profile))
        add("methods", path)

    if what in ("protocol", "bundle"):
        payload = _protocol(dataset, chart_id, meta, params, style, roles, last_plot, stats_result,
                            profile, binding=binding)
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
        bundle_artifact = {"kind": "bundle", "path": zip_path,
                           "label": os.path.basename(zip_path), "bytes": size}
        if binding_id:
            bundle_artifact["figure_run_id"] = binding_id
        return tag_artifacts({"what": what, "file": zip_path, "dir": subdir,
                              "files": [bundle_artifact] + exported,
                              "bytes": size})

    return tag_artifacts(_finish(what, outdir, exported))


def _finish(what: str, outdir: str, exported: list[dict]) -> dict:
    total = sum(item.get("bytes") or 0 for item in exported)
    return {"what": what, "file": exported[0]["path"] if exported else None,
            "dir": outdir, "files": exported, "bytes": total}
