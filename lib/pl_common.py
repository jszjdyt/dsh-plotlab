"""PlotLab 绘图工作台 —— 公共基础设施。

本模块只依赖标准库 + numpy/matplotlib 的少量能力，供 worker 内所有模块使用：
- 错误类型与快捷构造（码表与 dsh-geobulk 一致）
- 结果信封：result / table / figure / data_file
- 风格系统：STYLE_DEFAULT / resolve_style / palette_colors / finalize_figure / save_fig
- 数值格式与多重检验校正

约定：任何对外返回的值都必须先经 jsonable()，不允许 NaN/Inf。
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------


class EngineError(Exception):
    """带码的引擎错误；code 参与 HTTP/工具错误映射。"""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = details or {}


def prerequisite(message: str, **details: Any) -> EngineError:
    return EngineError("prerequisite", message, **details)


def bad_request(message: str, **details: Any) -> EngineError:
    return EngineError("bad_request", message, **details)


def not_found(message: str, **details: Any) -> EngineError:
    return EngineError("not_found", message, **details)


def unsupported(message: str, **details: Any) -> EngineError:
    return EngineError("unsupported", message, **details)


# ---------------------------------------------------------------------------
# JSON 安全
# ---------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """递归把 numpy / NaN / Inf / Timestamp 等转成 JSON 可序列化的原生值。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    # pandas / numpy 标量与时间
    for attr in ("item", "to_pydatetime", "isoformat"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return jsonable(method())
            except Exception:  # noqa: BLE001 - 落到字符串
                break
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 文件 / 文本
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def slug(text: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", str(text or "")).strip("_")
    return cleaned[:80] or fallback


def unique_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# 结果信封
# ---------------------------------------------------------------------------


def table(
    tid: str,
    title: str,
    columns: Sequence[Mapping[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    note: str | None = None,
    max_rows: int = 500,
) -> dict:
    """构造结果表。columns=[{"key","label","type"}]，type ∈ str/num/p。>max_rows 自动截断。"""
    cols = []
    for column in columns:
        key = str(column.get("key"))
        cols.append(
            {
                "key": key,
                "label": str(column.get("label") or key),
                "type": str(column.get("type") or "str"),
            }
        )
    keys = [c["key"] for c in cols]
    out_rows = []
    total = 0
    for row in rows:
        total += 1
        if len(out_rows) >= max_rows:
            continue
        item = {}
        for key in keys:
            value = row.get(key) if isinstance(row, Mapping) else None
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                value = None
            item[key] = jsonable(value)
        out_rows.append(item)
    payload = {"id": tid, "title": title, "columns": cols, "rows": out_rows, "total": total}
    if total > len(out_rows):
        payload["truncated"] = True
        payload["note"] = f"仅显示前 {len(out_rows)} / {total} 行"
    if note:
        payload["note"] = note
    return payload


def figure(fid: str, title: str, file: str, width: int | None = None, height: int | None = None) -> dict:
    return {
        "id": str(fid),
        "title": str(title),
        "file": str(file),
        "url": f"/plotlab/figure?id={fid}",
        "width": width,
        "height": height,
    }


def result(summary: Any = None, tables: Any = None, figures: Any = None, **extra: Any) -> dict:
    payload: dict = {"summary": summary, "tables": list(tables or []), "figures": list(figures or [])}
    payload.update(extra)
    return payload


def data_file(kind: str, path: str, label: str | None = None) -> dict:
    return {"kind": str(kind), "path": str(path), "label": label or os.path.basename(path)}


# ---------------------------------------------------------------------------
# 风格系统
# ---------------------------------------------------------------------------

FONTS: list[dict] = [
    {"id": "auto", "label": "自动（Arial + 中文回退）", "family": "Arial"},
    {"id": "Arial", "label": "Arial", "family": "Arial"},
    {"id": "Helvetica", "label": "Helvetica", "family": "Helvetica"},
    {"id": "Times New Roman", "label": "Times New Roman（衬线）", "family": "Times New Roman"},
    {"id": "Georgia", "label": "Georgia（衬线）", "family": "Georgia"},
    {"id": "DejaVu Sans", "label": "DejaVu Sans", "family": "DejaVu Sans"},
    {"id": "Arial Unicode MS", "label": "Arial Unicode MS（中文可用）", "family": "Arial Unicode MS"},
    {"id": "PingFang SC", "label": "苹方 PingFang SC（中文）", "family": "PingFang SC"},
    {"id": "Songti SC", "label": "宋体 Songti SC（中文）", "family": "Songti SC"},
    {"id": "Hiragino Sans GB", "label": "冬青黑体 Hiragino Sans GB（中文）", "family": "Hiragino Sans GB"},
]

# 中文回退链：matplotlib>=3.6 支持 font.family 列表回退
_CJK_FALLBACK = ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]

PALETTES: dict[str, list[str]] = {
    "okabe_ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#7F7F7F"],
    "npg": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
    "lancet": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6", "#1B1919"],
    "jama": ["#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599", "#80796B"],
    "set1": ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"],
    "set2": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
    "tableau": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"],
    "pastel": ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9"],
    "grey": ["#4D4D4D", "#737373", "#999999", "#BDBDBD", "#D9D9D9", "#262626", "#8C8C8C"],
}

# 连续色系（色盲友好 / 发散）
PALETTE_CMAPS: dict[str, str] = {
    "viridis": "viridis",
    "blue_red": "RdBu_r",
    "red_blue": "coolwarm",
    "magma": "magma",
    "sequential": "Blues",
}

THEMES: list[dict] = [
    {"id": "classic", "label": "经典（全框）"},
    {"id": "minimal", "label": "极简（左/下轴）"},
    {"id": "grid", "label": "网格"},
    {"id": "void", "label": "无框"},
    {"id": "nature", "label": "Nature 风格"},
    {"id": "lancet", "label": "Lancet 风格"},
    {"id": "jama", "label": "JAMA 风格"},
]

STYLE_DEFAULT: dict[str, Any] = {
    "font": "auto",
    "title_size": 14,
    "label_size": 12,
    "tick_size": 10,
    "legend_size": 10,
    "text_size": 10,
    "title_bold": True,
    "fig_width": 6.2,
    "fig_height": 4.4,
    "dpi": 300,
    "theme": "classic",
    "palette": "okabe_ito",
    "colors": {},
    "alpha": 0.85,
    "line_width": 1.4,
    "point_size": 28,
    "bar_width": 0.72,
    "edge_width": 0.8,
    "grid": "none",
    "frame": "left_bottom",
    "tick_dir": "out",
    "legend_pos": "right",
    "legend_title": "",
    "title_text": "",
    "x_label": "",
    "y_label": "",
    "x_rotate": 0,
    "y_rotate": 0,
    "x_log": False,
    "y_log": False,
    "x_lim": [None, None],
    "y_lim": [None, None],
    "error": "se",
    "jitter": True,
    "jitter_size": 2.2,
    "jitter_alpha": 0.55,
    "stat_annot": True,
    "stat_method": "auto",
    "p_adjust": "bh",
    "annot_style": "stars",
    "show_n": False,
    "legend_inside": False,
}

STYLE_GROUPS: list[dict] = [
    {"id": "font", "label": "字体与字号"},
    {"id": "color", "label": "颜色"},
    {"id": "canvas", "label": "画布与导出"},
    {"id": "axis", "label": "坐标轴"},
    {"id": "legend", "label": "图例与标注"},
    {"id": "stat", "label": "统计标注"},
]

_NUMERIC_STYLE = {
    "title_size": (6, 40),
    "label_size": (6, 40),
    "tick_size": (5, 40),
    "legend_size": (5, 40),
    "text_size": (5, 40),
    "fig_width": (2.0, 20.0),
    "fig_height": (2.0, 20.0),
    "dpi": (72, 600),
    "alpha": (0.05, 1.0),
    "line_width": (0.2, 8.0),
    "point_size": (4, 400),
    "bar_width": (0.1, 1.0),
    "edge_width": (0.0, 4.0),
    "x_rotate": (0, 90),
    "y_rotate": (0, 90),
    "jitter_size": (0.5, 12.0),
    "jitter_alpha": (0.05, 1.0),
}

_ENUMS = {
    "theme": {t["id"] for t in THEMES},
    "palette": set(PALETTES) | set(PALETTE_CMAPS),
    "grid": {"none", "x", "y", "both"},
    "frame": {"left_bottom", "none", "full"},
    "tick_dir": {"out", "in"},
    "legend_pos": {"right", "top", "bottom", "none"},
    "error": {"sd", "se", "ci95", "none"},
    "stat_method": {"auto", "welch", "student", "wilcox", "anova", "kruskal"},
    "p_adjust": {"none", "bh", "bonferroni", "holm"},
    "annot_style": {"stars", "p", "both"},
    "font": {f["id"] for f in FONTS},
}


def resolve_style(style: Mapping[str, Any] | None) -> dict:
    """校验并补全风格参数；非法值一律 bad_request（含具体键名）。"""
    merged = dict(STYLE_DEFAULT)
    if isinstance(style, dict):
        for key, value in style.items():
            if key not in STYLE_DEFAULT:
                continue
            merged[key] = value
    for key, bounds in _NUMERIC_STYLE.items():
        try:
            value = float(merged[key])
        except (TypeError, ValueError):
            value = float(STYLE_DEFAULT[key])
        lo, hi = bounds
        merged[key] = int(value) if isinstance(STYLE_DEFAULT[key], int) else max(lo, min(hi, value))
        if isinstance(STYLE_DEFAULT[key], int):
            merged[key] = int(max(lo, min(hi, value)))
    for key, allowed in _ENUMS.items():
        if merged.get(key) not in allowed:
            merged[key] = STYLE_DEFAULT[key]
    for key in ("title_text", "x_label", "y_label", "legend_title"):
        merged[key] = "" if merged.get(key) is None else str(merged[key])
    for key in ("title_bold", "x_log", "y_log", "jitter", "stat_annot", "show_n", "legend_inside"):
        merged[key] = bool(merged.get(key))
    colors = merged.get("colors")
    if not isinstance(colors, dict):
        colors = {}
    clean_colors: dict[str, str] = {}
    for name, value in colors.items():
        text = str(value or "").strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
            clean_colors[str(name)] = text.upper()
    merged["colors"] = clean_colors
    for key in ("x_lim", "y_lim"):
        value = merged.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            merged[key] = [None if v in (None, "") else float(v) for v in value]
        else:
            merged[key] = [None, None]
    return merged


def font_family(style: Mapping[str, Any]) -> str:
    key = str(style.get("font") or "auto")
    for item in FONTS:
        if item["id"] == key:
            return item["family"]
    return "Arial"


def _is_serif(family: str) -> bool:
    return family in ("Times New Roman", "Georgia")


def style_rc(style: Mapping[str, Any]) -> dict:
    """返回 matplotlib rcParams 覆盖字典。"""
    family = font_family(style)
    generic = "serif" if _is_serif(family) else "sans-serif"
    fallback = [family] + [f for f in _CJK_FALLBACK if f != family]
    rc = {
        "font.family": generic,
        "font.serif": fallback if generic == "serif" else ["Times New Roman", "DejaVu Serif"],
        "font.sans-serif": fallback if generic == "sans-serif" else ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": float(style.get("text_size", 10)),
        "axes.titlesize": float(style.get("title_size", 14)),
        "axes.labelsize": float(style.get("label_size", 12)),
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1A1A1A",
        "xtick.color": "#1A1A1A",
        "ytick.color": "#1A1A1A",
        "text.color": "#1A1A1A",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "figure.autolayout": False,
    }
    return rc


def style_context(style: Mapping[str, Any]):
    """with 语句用的 rc_context 包装，图表统一使用。"""
    import matplotlib

    return matplotlib.rc_context(style_rc(style))


def palette_colors(style_or_name: Any, levels: Sequence[Any]) -> dict:
    """把 levels 映射成颜色。参数可传 style dict 或 palette 名。"""
    if isinstance(style_or_name, Mapping):
        name = str(style_or_name.get("palette") or "okabe_ito")
        overrides = style_or_name.get("colors") or {}
    else:
        name = str(style_or_name or "okabe_ito")
        overrides = {}
    base = PALETTES.get(name)
    if base is None:
        base = PALETTES["okabe_ito"]
    out = {}
    for index, level in enumerate(levels):
        out[level] = base[index % len(base)]
    if isinstance(overrides, Mapping):
        for level, color in overrides.items():
            out[level] = color
    return out


def color_list(style: Mapping[str, Any], n: int) -> list[str]:
    """连续取 n 个离散颜色（用于无分组的多系列）。"""
    name = str(style.get("palette") or "okabe_ito")
    base = PALETTES.get(name) or PALETTES["okabe_ito"]
    return [base[i % len(base)] for i in range(max(1, n))]


def palette_cmap(style: Mapping[str, Any]):
    """返回连续色的 matplotlib colormap。"""
    import matplotlib

    name = str(style.get("palette") or "viridis")
    cmap_name = PALETTE_CMAPS.get(name, "viridis")
    try:
        return matplotlib.colormaps[cmap_name]
    except Exception:  # noqa: BLE001
        return matplotlib.colormaps["viridis"]


# ---------------------------------------------------------------------------
# Figure 生命周期
# ---------------------------------------------------------------------------


def new_fig(style: Mapping[str, Any], nrows: int = 1, ncols: int = 1, **kwargs):
    import matplotlib.pyplot as plt

    width = float(style.get("fig_width", 6.2))
    height = float(style.get("fig_height", 4.4))
    kwargs.setdefault("layout", "constrained")
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
    return fig, axes


def apply_axes_style(ax, style: Mapping[str, Any], grid_axis: str | None = None) -> None:
    """单轴：spines / ticks / grid。"""
    theme = str(style.get("theme") or "classic")
    frame = str(style.get("frame") or "left_bottom")
    if theme in ("void",):
        frame = "none"
    elif theme in ("minimal", "nature", "lancet", "jama"):
        frame = "left_bottom" if frame == "left_bottom" else frame
    spines = ax.spines
    for side in ("top", "right", "left", "bottom"):
        if frame == "full":
            spines[side].set_visible(True)
            spines[side].set_linewidth(float(style.get("edge_width", 0.8)) or 0.8)
        elif frame == "none":
            spines[side].set_visible(False)
        elif frame == "left_bottom":
            spines[side].set_visible(side in ("left", "bottom"))
            spines[side].set_linewidth(float(style.get("edge_width", 0.8)) or 0.8)
    direction = str(style.get("tick_dir") or "out")
    ax.tick_params(axis="both", which="major", direction=direction, length=3.5,
                   labelsize=float(style.get("tick_size", 10)))
    ax.tick_params(axis="both", which="minor", direction=direction, length=2)
    grid = str(style.get("grid") or "none")
    if theme == "grid" and grid == "none":
        grid = "both"
    if grid in ("both", "x"):
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#B0B0B0")
        ax.set_axisbelow(True)
    else:
        ax.xaxis.grid(False)
    if grid in ("both", "y"):
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#B0B0B0")
        ax.set_axisbelow(True)
    else:
        ax.yaxis.grid(False)
    if style.get("x_rotate"):
        ax.tick_params(axis="x", rotation=float(style["x_rotate"]))
        for label in ax.get_xticklabels():
            label.set_ha("right" if float(style["x_rotate"]) >= 30 else "center")
    if style.get("y_rotate"):
        ax.tick_params(axis="y", rotation=float(style["y_rotate"]))
    for axis_name in ("x", "y"):
        if style.get(f"{axis_name}_log"):
            getattr(ax, f"set_{axis_name}scale")("log")
        lim = style.get(f"{axis_name}_lim")
        if isinstance(lim, (list, tuple)) and len(lim) == 2 and any(v is not None for v in lim):
            getter = getattr(ax, f"get_{axis_name}lim")
            lo, hi = lim
            cur = getter()
            setter = getattr(ax, f"set_{axis_name}lim")
            setter(lo if lo is not None else cur[0], hi if hi is not None else cur[1])
    label_size = float(style.get("label_size", 12))
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_size)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_size)


def finalize_figure(fig, style: Mapping[str, Any], title: str | None = None,
                    legend: bool = True, note: str | None = None) -> None:
    """统一收尾：轴样式、标题、图例位置、脚注。

    图表内部只管 label=... 建图例；这里收集所有 axes 图例合并成一个 fig 级图例，
    按 style.legend_pos 摆放，避免每张图重复实现图例逻辑。
    """
    axes = [ax for ax in fig.axes if ax.get_visible()]
    for ax in axes:
        apply_axes_style(ax, style)

    title_text = str(style.get("title_text") or "").strip() or (title or "").strip()
    if title_text:
        fig.suptitle(
            title_text,
            fontsize=float(style.get("title_size", 14)),
            fontweight="bold" if style.get("title_bold") else "normal",
        )

    legend_pos = str(style.get("legend_pos") or "right")
    handles: list = []
    labels: list[str] = []
    for ax in axes:
        handle, label = ax.get_legend_handles_labels()
        for h, l in zip(handle, label):
            if l and l not in labels and not str(l).startswith("_"):
                handles.append(h)
                labels.append(l)
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            legend_obj.remove()
    if legend and handles and legend_pos != "none":
        legend_title = str(style.get("legend_title") or "").strip() or None
        kwargs = dict(
            fontsize=float(style.get("legend_size", 10)),
            title=legend_title,
        )
        if legend_title:
            kwargs["title_fontsize"] = float(style.get("legend_size", 10))
        if legend_pos == "right":
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), **kwargs)
        elif legend_pos == "top":
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                       ncol=min(4, max(1, len(labels))), **kwargs)
        elif legend_pos == "bottom":
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03),
                       ncol=min(4, max(1, len(labels))), **kwargs)
    if note:
        fig.text(0.01, -0.01, note, fontsize=max(6.0, float(style.get("text_size", 10)) - 2),
                 color="#666666", ha="left", va="top")


def save_fig(fig, outdir: str, name: str, title: str | None = None,
             dpi: int | None = None, fmt: str = "png") -> dict:
    """保存图并返回 figure dict（id 由文件名派生）。"""
    import matplotlib.pyplot as plt

    ensure_dir(outdir)
    fid = unique_id("f_")
    filename = f"{fid}_{slug(name, 'figure')}.{fmt}"
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=int(dpi or 300), bbox_inches="tight", facecolor="white")
    width = int(round(fig.get_figwidth() * float(dpi or 300)))
    height = int(round(fig.get_figheight() * float(dpi or 300)))
    plt.close(fig)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    payload = figure(fid, title or name, path, width, height)
    payload["bytes"] = size
    payload["dpi"] = int(dpi or 300)
    payload["filename"] = filename
    return payload


# ---------------------------------------------------------------------------
# 数值格式 / 统计小工具
# ---------------------------------------------------------------------------


def fmt_num(value: Any, digits: int = 3) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(num) or math.isinf(num):
        return "—"
    if num == 0:
        return "0"
    abs_value = abs(num)
    if abs_value >= 1e5 or abs_value < 1e-3:
        return f"{num:.2e}"
    return f"{num:.{digits}g}"


def fmt_p(p: Any) -> str:
    try:
        value = float(p)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(value):
        return "—"
    if value < 1e-4:
        return "< 1e-4"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def p_stars(p: Any) -> str:
    try:
        value = float(p)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def bh_fdr(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg 校正。"""
    values = np.asarray([float(p) for p in pvalues], dtype=float)
    n = values.size
    if n == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return [float(v) for v in out]


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray([float(p) for p in pvalues], dtype=float)
    n = values.size
    if n == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for i in range(n):
        candidate = (n - i) * ranked[i]
        running = max(running, candidate)
        adjusted[i] = min(1.0, running)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return [float(v) for v in out]


def adjust_pvalues(pvalues: Sequence[float], method: str = "bh") -> list[float]:
    method = str(method or "bh").lower()
    if method in ("none", "raw", ""):
        return [float(p) for p in pvalues]
    if method == "bonferroni":
        n = max(1, len(pvalues))
        return [min(1.0, float(p) * n) for p in pvalues]
    if method == "holm":
        return holm_adjust(pvalues)
    return bh_fdr(pvalues)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    def reset(self) -> None:
        self.t0 = time.perf_counter()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str, text: str) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path
