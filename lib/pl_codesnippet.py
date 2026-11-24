"""PlotLab 关键步骤代码生成（兜底实现）。

优先使用 pl_charts.python_code / pl_charts.r_code；当图表模块尚未提供这两个函数时，
这里按「图表家族」生成同样可独立运行的 Python / ggplot2 片段：
- 数据准备与工作台一致（角色映射、分组、清洗后的 df）
- 图谱参数与 style 一致（字体/字号/配色/画布/误差线/显著性标注）
- 输出到 plotlab_figure.png（300 dpi）

注意：片段以工作台清洗后的 DataFrame `df` 为输入（完整脚本由 pl_codegen.build_script 拼装）。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import pl_common as C

# 图表 -> 代码家族
FAMILY = {
    "bar": "bar", "bar_dot": "bar_dot", "stack": "stack", "pie": "pie",
    "box": "box", "violin": "violin", "raincloud": "raincloud", "dotplot": "dotplot",
    "scatter": "scatter", "bubble": "bubble", "line": "line", "area": "area",
    "heatmap": "heatmap", "corr": "corr", "pairplot": "pairplot",
    "hist": "hist", "density": "density", "qq": "qq",
    "roc": "roc", "km": "km", "forest": "forest", "pca": "pca",
}

ROLE_ORDER = ("x", "y", "group", "color", "size", "time", "event")


def _colors(style: Mapping[str, Any], levels: Sequence[Any]) -> dict:
    return C.palette_colors(style, [str(level) for level in levels])


def _tidy_setup(data: Mapping[str, Any], roles: Sequence[str], style: Mapping[str, Any]) -> list[str]:
    """生成从 df 构造 plot_df 的代码。"""
    labels = data.get("labels") or {}
    lines = ["# 与工作台一致的角色映射（列名来自你的数据）"]
    mapping = {}
    for role in roles:
        column = labels.get(role) or (data.get("roles") or {}).get(role)
        if column:
            mapping[role] = str(column)
    if "x" in mapping:
        lines.append(f'X = [{mapping["x"]!r}][0]')
    if "y" in mapping:
        lines.append(f'Y = [{mapping["y"]!r}][0]')
    if "group" in mapping:
        lines.append(f'GROUP = [{mapping["group"]!r}][0]')
    if "color" in mapping:
        lines.append(f'COLOR = [{mapping["color"]!r}][0]')
    if "size" in mapping:
        lines.append(f'SIZE = [{mapping["size"]!r}][0]')
    if "time" in mapping:
        lines.append(f'TIME = [{mapping["time"]!r}][0]')
    if "event" in mapping:
        lines.append(f'EVENT = [{mapping["event"]!r}][0]')
    lines.append("")
    lines.append("plot_df = pd.DataFrame({")
    if "x" in mapping:
        lines.append('    "x": df[X],')
    if "y" in mapping:
        lines.append('    "y": pd.to_numeric(df[Y], errors="coerce"),')
    if "group" in mapping:
        lines.append('    "group": df[GROUP].astype(str),')
    if "color" in mapping:
        lines.append('    "color": df[COLOR],')
    if "size" in mapping:
        lines.append('    "size": pd.to_numeric(df[SIZE], errors="coerce"),')
    if "time" in mapping:
        lines.append('    "time": pd.to_numeric(df[TIME], errors="coerce"),')
    if "event" in mapping:
        lines.append('    "event": pd.to_numeric(df[EVENT], errors="coerce"),')
    lines.append("}).dropna(subset=[c for c in (['x'] if 'x' in X else [])])" if False else "})")
    drop_cols = [c for c in ("x", "y") if c in mapping]
    if drop_cols:
        lines.append(f'plot_df = plot_df.dropna(subset={drop_cols!r})')
    lines.append("")
    return lines


def _style_setup(style: Mapping[str, Any], levels: Sequence[Any], title: str, xlabel: str, ylabel: str) -> list[str]:
    family = C.font_family(style)
    colors = _colors(style, levels) if levels else {}
    lines = [
        "# ---- 样式（与工作台一致）----",
        "STYLE = {",
        f'    "font": {family!r},',
        f'    "title_size": {style.get("title_size")},',
        f'    "label_size": {style.get("label_size")},',
        f'    "tick_size": {style.get("tick_size")},',
        f'    "legend_size": {style.get("legend_size")},',
        f'    "fig_width": {style.get("fig_width")},',
        f'    "fig_height": {style.get("fig_height")},',
        f'    "dpi": {style.get("dpi")},',
        f'    "alpha": {style.get("alpha")},',
        f'    "line_width": {style.get("line_width")},',
        f'    "point_size": {style.get("point_size")},',
        f'    "bar_width": {style.get("bar_width")},',
        f'    "error": {str(style.get("error") or "se")!r},',
        f'    "grid": {str(style.get("grid") or "none")!r},',
        f'    "frame": {str(style.get("frame") or "left_bottom")!r},',
        f'    "legend_pos": {str(style.get("legend_pos") or "right")!r},',
        f'    "title": {str(style.get("title_text") or title)!r},',
        f'    "x_label": {str(style.get("x_label") or xlabel)!r},',
        f'    "y_label": {str(style.get("y_label") or ylabel)!r},',
        "}",
        f'GROUP_COLORS = {colors!r}',
        "",
        'plt.rcParams.update({' ,
        '    "font.family": "sans-serif",',
        '    "font.sans-serif": [STYLE["font"], "Arial Unicode MS", "PingFang SC", "DejaVu Sans"],',
        '    "axes.unicode_minus": False,',
        '    "font.size": STYLE["tick_size"],',
        '})',
        'fig, ax = plt.subplots(figsize=(STYLE["fig_width"], STYLE["fig_height"]), layout="constrained")',
        'fig.patch.set_facecolor("white")',
        "",
    ]
    return lines


def _finish(style: Mapping[str, Any], levels: Sequence[Any], rotate: bool = True) -> list[str]:
    lines = [
        "# ---- 坐标轴 / 图例 / 保存 ----",
        'for side in ("top", "right"):',
        '    ax.spines[side].set_visible(STYLE["frame"] == "full")',
        'if STYLE["frame"] == "left_bottom":',
        '    pass',
        f'ax.tick_params(labelsize=STYLE["tick_size"], direction="out", length=3.5)',
        'ax.tick_params(axis="x", labelrotation=%d)' % int(style.get("x_rotate") or 0),
        'if STYLE["grid"] in ("both", "y"):',
        '    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#B0B0B0")',
        '    ax.set_axisbelow(True)',
        'if STYLE["grid"] in ("both", "x"):',
        '    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#B0B0B0")',
        '    ax.set_axisbelow(True)',
        'ax.set_title(STYLE["title"], fontsize=STYLE["title_size"], fontweight="bold")',
        'ax.set_xlabel(STYLE["x_label"], fontsize=STYLE["label_size"])',
        'ax.set_ylabel(STYLE["y_label"], fontsize=STYLE["label_size"])',
        'handles, labels = ax.get_legend_handles_labels()',
        'if handles and STYLE["legend_pos"] != "none":',
        '    if STYLE["legend_pos"] == "right":',
        '        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=STYLE["legend_size"], title=STYLE.get("legend_title") or None)',
        '    elif STYLE["legend_pos"] == "top":',
        '        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(labels)), frameon=False, fontsize=STYLE["legend_size"])',
        '    elif STYLE["legend_pos"] == "bottom":',
        '        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=min(4, len(labels)), frameon=False, fontsize=STYLE["legend_size"])',
        'fig.savefig("plotlab_figure.png", dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")',
        'print("已保存：plotlab_figure.png")',
    ]
    return lines


def _stat_snippet(style: Mapping[str, Any]) -> list[str]:
    """组间显著性：重算 pairwise 检验并标注星号。"""
    return [
        "# ---- 组间统计与显著性标注（与工作台同样的检验选择）----",
        "from scipy import stats",
        "import itertools",
        "_levels = [g for g in plot_df['group'].dropna().unique()]",
        "_pairs = list(itertools.combinations(_levels, 2))",
        "_tests = []",
        "for a, b in _pairs:",
        "    va = plot_df.loc[plot_df['group'] == a, 'y'].dropna().to_numpy()",
        "    vb = plot_df.loc[plot_df['group'] == b, 'y'].dropna().to_numpy()",
        "    if len(va) < 2 or len(vb) < 2:",
        "        continue",
        "    if len(_levels) == 2:",
        "        _t, _p = stats.ttest_ind(va, vb, equal_var=False)",
        "    else:",
        "        _t, _p = stats.ttest_ind(va, vb, equal_var=False)",
        "    _tests.append((a, b, _p))",
        "from statsmodels.stats.multitest import multipletests",
        "_raw = [p for _, _, p in _tests]",
        "_adj = multipletests(_raw, method='fdr_bh')[1] if _raw else []",
        "def _stars(p):",
        "    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'",
        "_ymax = plot_df['y'].max()",
        "_span = (plot_df['y'].max() - plot_df['y'].min()) or 1",
        "for (_a, _b, _p), _padj in zip(_tests, _adj):",
        "    _x1 = list(_levels).index(_a)",
        "    _x2 = list(_levels).index(_b)",
        "    _y = _ymax + _span * (0.06 + 0.06 * list(_pairs).index((_a, _b)))",
        "    ax.plot([_x1, _x1, _x2, _x2], [_y - _span * 0.02, _y, _y, _y - _span * 0.02], color='#333333', linewidth=0.9)",
        "    ax.text((_x1 + _x2) / 2, _y, _stars(_padj), ha='center', va='bottom', fontsize=STYLE['tick_size'])",
        "    print(f'{_a} vs {_b}: p={_p:.4g} (adj {_padj:.4g}) {_stars(_padj)}')",
        "",
    ]


def python_code(chart_id: str, data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> str:
    """生成「数据准备 + 作图」的 Python 片段（约定输入变量名 df）。

    使用 seaborn 高阶接口 + pandas 透视，保证生成的代码短、可读、缩进正确，
    同时把 style 的字体/字号/配色/画布/误差线/显著性全部落进代码。
    """
    family = FAMILY.get(chart_id, "box")
    params = dict(params or {})
    style = C.resolve_style(style)
    labels = data.get("labels") or {}
    levels = (data.get("levels") or {}).get("group") or []
    title_map = {
        "bar": "Bar plot", "bar_dot": "Bar plot with points", "box": "Box plot",
        "violin": "Violin plot", "raincloud": "Raincloud plot", "dotplot": "Dot plot",
        "stack": "Stacked bar", "pie": "Pie chart", "scatter": "Scatter plot",
        "bubble": "Bubble plot", "line": "Line plot", "area": "Area plot",
        "heatmap": "Heatmap", "corr": "Correlation matrix", "pairplot": "Pair plot",
        "hist": "Histogram", "density": "Density plot", "qq": "Q-Q plot",
        "roc": "ROC curve", "km": "Kaplan-Meier curve", "forest": "Forest plot",
        "pca": "PCA",
    }
    title = f"{title_map.get(chart_id, 'Plot')}: {labels.get('y', '')} by {labels.get('x', '')}".strip(" by")
    x_label = labels.get("x", "")
    y_label = labels.get("y", "")
    header = _style_setup(style, levels, title, x_label, y_label)
    header = [line for line in header if line.strip() != ""][:0] + header  # 保留顺序，便于阅读

    lines: list[str] = list(header)
    lines.append('import seaborn as sns  # 仅用其绘图接口；样式由上面的 rcParams/收尾统一控制')
    lines.append("")

    err = {"sd": "sd", "se": "se", "ci95": "ci", "none": None}.get(str(style.get("error") or "se"), "se")
    alpha = float(style.get("alpha") or 0.85)
    lw = float(style.get("line_width") or 1.4)
    ps = float(style.get("point_size") or 28)
    bw = float(style.get("bar_width") or 0.72)
    jitter = bool(style.get("jitter"))
    stat = bool(style.get("stat_annot"))
    announce = "# ==== 数据准备（与工作台完全一致的角色映射）===="

    def tidy_setup(roles=("x", "y", "group")) -> list[str]:
        return ["# ==== 数据准备（与工作台完全一致的角色映射）===="] + _tidy_setup(data, roles, style)

    if family in ("bar", "bar_dot"):
        agg = str(params.get("agg") or "mean")
        lines += tidy_setup(("x", "y", "group"))
        lines += [
            "# ==== 作图：分组柱状图（均值/中位数/求和 + 误差线）====",
            "agg_map = {'mean': 'mean', 'median': 'median', 'sum': 'sum'}",
            f"estimator = {agg!r}",
            f"errorbar = {err!r}",
            'hue = "group" if "group" in plot_df.columns else None',
            "sns.barplot(data=plot_df, x='x', y='y', hue=hue, estimator=estimator, errorbar=errorbar,",
            f"            palette=GROUP_COLORS or None, alpha={alpha}, width={bw},",
            "            edgecolor='black', linewidth=0.6, ax=ax)",
        ]
        if family == "bar_dot":
            lines += [
                "# 叠加原始散点",
                "sns.stripplot(data=plot_df, x='x', y='y', hue=hue, dodge=True, size=3,",
                f"              alpha=0.5, color='#333333', jitter={0.18 if jitter else 0.0}, ax=ax)",
                "handles, labels_ = ax.get_legend_handles_labels()",
                "if handles and len(handles) > 1:",
                "    ax.legend(handles[: len(handles) // 2], labels_[: len(labels_) // 2], title=None, frameon=False)",
            ]
        lines += ["ax.set_xlabel(STYLE['x_label'] or '')", "ax.set_ylabel(STYLE['y_label'] or '')"]
        if stat:
            lines += ["", *_stat_snippet(style)]
        lines += _finish(style, levels)
    elif family == "box":
        lines += tidy_setup(("x", "y", "group"))
        lines += [
            "# ==== 作图：箱线图 + 抖动散点 ====",
            'hue = "group" if "group" in plot_df.columns else None',
            "sns.boxplot(data=plot_df, x='x', y='y', hue=hue, width=0.6, palette=GROUP_COLORS or None,",
            f"            linewidth={lw}, fliersize=0, ax=ax)",
            "sns.stripplot(data=plot_df, x='x', y='y', hue=hue, dodge=True, size=3,",
            f"              alpha={0.45 if jitter else 0.0}, color='#333333', jitter={0.15 if jitter else 0.0}, ax=ax)",
        ]
        if stat:
            lines += ["", *_stat_snippet(style)]
        lines += _finish(style, levels)
    elif family in ("violin", "raincloud"):
        lines += tidy_setup(("x", "y", "group"))
        lines += [
            f"# ==== 作图：{'小提琴图' if family == 'violin' else '雨云图（半小提琴+箱+散点）'} ====",
            'hue = "group" if "group" in plot_df.columns else None',
            "sns.violinplot(data=plot_df, x='x', y='y', hue=hue, split=False, inner=None,",
            f"               palette=GROUP_COLORS or None, linewidth={lw}, cut=0, ax=ax)",
        ]
        if family == "raincloud":
            lines += ["# 半小提琴效果：把左半边隐藏（matplotlib 层面可用 clip 实现，这里用箱+散点近似）"]
        lines += [
            "sns.boxplot(data=plot_df, x='x', y='y', hue=hue, width=0.12, showfliers=False,",
            "            boxprops={'facecolor': 'white'}, ax=ax)",
            "sns.stripplot(data=plot_df, x='x', y='y', hue=hue, dodge=True, size=2.5,",
            f"              alpha={0.4 if jitter else 0.0}, color='#333333', jitter={0.1 if jitter else 0.0}, ax=ax)",
        ]
        if stat:
            lines += ["", *_stat_snippet(style)]
        lines += _finish(style, levels)
    elif family == "stack":
        lines += tidy_setup(("x", "y", "group"))
        lines += [
            "# ==== 作图：堆叠柱 ====",
            "pivot = plot_df.pivot_table(index='x', columns='group', values='y', aggfunc='sum', fill_value=0)",
            "if %r:" % bool(params.get("normalize")),
            "    pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100",
            "colors = [GROUP_COLORS.get(str(c)) or f'C{i}' for i, c in enumerate(pivot.columns)]",
            "pivot.plot(kind='bar', stacked=True, color=colors, alpha=STYLE['alpha'], ax=ax, width=STYLE['bar_width'])",
            "ax.set_xlabel(STYLE['x_label'] or '')",
        ]
        lines += _finish(style, levels)
    elif family == "pie":
        lines += tidy_setup(("x", "y"))
        lines += [
            "# ==== 作图：饼/环图 ====",
            "agg = plot_df.groupby('x', observed=True)['y'].sum()",
            "colors = [GROUP_COLORS.get(str(v)) or f'C{i}' for i, v in enumerate(agg.index)]",
            "ax.pie(agg.to_numpy(), labels=[str(v) for v in agg.index], autopct='%.1f%%', colors=colors,",
            "       startangle=90, counterclock=False,",
            "       wedgeprops=dict(width=0.62 if %r else 1.0, edgecolor='white'))" % bool(params.get("donut")),
            "ax.set_aspect('equal')",
        ]
        lines += _finish(style, levels, rotate=False)
    elif family == "dotplot":
        lines += tidy_setup(("x", "y", "group"))
        lines += [
            "# ==== 作图：点图（x 分类 × group 分类，颜色=均值，大小=|均值|）====",
            "pivot = plot_df.pivot_table(index='group', columns='x', values='y', aggfunc='mean')",
            "xs = [str(v) for v in pivot.columns]",
            "ys = [str(v) for v in pivot.index]",
            "xi, yi = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)))",
            "values = pivot.to_numpy(dtype=float)",
            "import matplotlib.cm as cm",
            "cmap = plt.get_cmap('viridis')",
            "norm = plt.Normalize(np.nanmin(values), np.nanmax(values) if np.nanmax(values) > np.nanmin(values) else np.nanmin(values) + 1)",
            "sizes = 20 + 300 * (np.abs(values) - np.nanmin(np.abs(values))) / ((np.nanmax(np.abs(values)) - np.nanmin(np.abs(values))) or 1)",
            "ax.scatter(xi.ravel(), yi.ravel(), s=sizes.ravel(), c=cmap(norm(values.ravel())),",
            "           edgecolors='#333333', linewidths=0.6)",
            "ax.set_xticks(np.arange(len(xs))); ax.set_xticklabels(xs)",
            "ax.set_yticks(np.arange(len(ys))); ax.set_yticklabels(ys)",
            "ax.set_xlabel(STYLE['x_label'] or ''); ax.set_ylabel(STYLE['y_label'] or '')",
        ]
        lines += _finish(style, levels)
    elif family in ("scatter", "bubble"):
        lines += tidy_setup(("x", "y", "group", "size"))
        lines += [
            f"# ==== 作图：{'气泡图' if family == 'bubble' else '散点图'}{'（+回归线/相关系数）' if params.get('add_fit', True) else ''} ====",
            "plot_df['x_num'] = pd.to_numeric(plot_df['x'], errors='coerce')",
            "plot_df['y_num'] = pd.to_numeric(plot_df['y'], errors='coerce')",
            "plot_df = plot_df.dropna(subset=['x_num', 'y_num'])",
            'hue = "group" if "group" in plot_df.columns else None',
        ]
        if family == "bubble":
            lines += [
                "size_values = plot_df['size'] if 'size' in plot_df.columns else pd.Series([1.0] * len(plot_df))",
                "sizes = 20 + 300 * (size_values - size_values.min()) / ((size_values.max() - size_values.min()) or 1)",
                "ax.scatter(plot_df['x_num'], plot_df['y_num'], s=sizes, c='#4E79A7', alpha=STYLE['alpha'], edgecolors='white', linewidths=0.5)",
            ]
        else:
            lines += [
                "sns.scatterplot(data=plot_df, x='x_num', y='y_num', hue=hue,",
                f"                palette=GROUP_COLORS or None, s={ps}, alpha={alpha}, linewidth=0.5, edgecolor='white', ax=ax)",
            ]
        if params.get("add_fit", True):
            lines += [
                "# 线性拟合 + Pearson 相关",
                "coef = np.polyfit(plot_df['x_num'], plot_df['y_num'], 1)",
                "xs_ = np.linspace(plot_df['x_num'].min(), plot_df['x_num'].max(), 100)",
                "ax.plot(xs_, np.polyval(coef, xs_), color='#222222', linewidth=STYLE['line_width'], linestyle='--', label='linear fit')",
                "from scipy import stats as _st",
                "r_value, p_value = _st.pearsonr(plot_df['x_num'], plot_df['y_num'])",
                "ax.text(0.02, 0.98, f'Pearson r = {r_value:.3f}\\np = {p_value:.4g}', transform=ax.transAxes, va='top', fontsize=STYLE['tick_size'])",
            ]
        lines += _finish(style, levels)
    elif family in ("line", "area"):
        lines += tidy_setup(("x", "y", "group"))
        lines += [
            f"# ==== 作图：{'折线图' if family == 'line' else '面积图'}（均值 ± 标准误）====",
            "plot_df['x'] = pd.to_numeric(plot_df['x'], errors='coerce')",
            "plot_df = plot_df.dropna(subset=['x'])",
        ]
        if family == "line":
            lines += [
                "sns.lineplot(data=plot_df, x='x', y='y', hue='group' if 'group' in plot_df.columns else None,",
                f"             errorbar={err!r}, marker='o', linewidth={lw}, palette=GROUP_COLORS or None, ax=ax)",
            ]
        else:
            lines += [
                "summary = plot_df.groupby(['x', 'group'], observed=True)['y'].sum().reset_index()",
                "for name, sub in summary.groupby('group', observed=True):",
                "    sub = sub.sort_values('x')",
                "    ax.fill_between(sub['x'], 0, sub['y'], alpha=STYLE['alpha'] * 0.5,",
                "                    color=GROUP_COLORS.get(str(name)) or None, label=str(name))",
                "ax.legend(frameon=False, fontsize=STYLE['legend_size'])",
            ]
        lines += _finish(style, levels)
    elif family == "heatmap":
        lines += [
            "# ==== 作图：热图（首列=特征，其余列=样本）====",
            "value_cols = df.select_dtypes(include=[np.number]).columns.tolist()",
            "feature_col = df.columns[0]",
            "mat = df.set_index(feature_col)[value_cols].to_numpy(dtype=float)",
        ]
        zscore = str(params.get("zscore") or "").lower() in ("row", "rows", "true", "zscore")
        if zscore:
            lines += [
                "row_mean = np.nanmean(mat, axis=1, keepdims=True)",
                "row_std = np.nanstd(mat, axis=1, keepdims=True)",
                "mat = (mat - row_mean) / np.where(row_std == 0, 1, row_std)",
            ]
        lines += [
            "cmap = plt.get_cmap(" + repr("RdBu_r" if str(style.get("palette")) == "blue_red" else "viridis") + ")",
            "im = ax.imshow(mat, aspect='auto', cmap=cmap)",
            "fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)",
            "ax.set_xticks(np.arange(len(value_cols)))",
            "ax.set_xticklabels(list(value_cols), rotation=45, ha='right')",
            "ax.set_yticks(np.arange(len(df)))",
            "ax.set_yticklabels(df[feature_col].astype(str).tolist())",
            "ax.set_title(STYLE['title'], fontsize=STYLE['title_size'], fontweight='bold')",
            'fig.savefig("plotlab_figure.png", dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")',
            'print("已保存：plotlab_figure.png")',
        ]
    elif family == "corr":
        lines += [
            "# ==== 相关矩阵（Pearson + 显著性星号）====",
            "from scipy import stats as _st",
            "numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')",
            "corr = numeric.corr(method='pearson')",
            "n = len(numeric)",
            "im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap=plt.get_cmap('RdBu_r'))",
            "fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)",
            "ax.set_xticks(np.arange(len(corr.columns))); ax.set_xticklabels(list(corr.columns), rotation=45, ha='right')",
            "ax.set_yticks(np.arange(len(corr.columns))); ax.set_yticklabels(list(corr.columns))",
            "for i in range(len(corr.columns)):",
            "    for j in range(len(corr.columns)):",
            "        r_ = corr.iloc[i, j]",
            "        t_ = r_ * np.sqrt((n - 2) / max(1e-9, 1 - r_ * r_)) if n > 2 and abs(r_) < 1 else 0",
            "        p_ = 2 * _st.t.sf(abs(t_), df=max(1, n - 2)) if n > 2 else 1",
            "        stars_ = '***' if p_ < 0.001 else '**' if p_ < 0.01 else '*' if p_ < 0.05 else ''",
            "        ax.text(j, i, f'{r_:.2f}{stars_}', ha='center', va='center',",
            "                fontsize=STYLE['tick_size'], color='white' if abs(r_) > 0.6 else '#222222')",
            "ax.set_title(STYLE['title'], fontsize=STYLE['title_size'], fontweight='bold')",
            'fig.savefig("plotlab_figure.png", dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")',
            'print("已保存：plotlab_figure.png")',
        ]
    elif family == "pairplot":
        lines += [
            "# ==== 散点矩阵（最多 6 个数值列，按 group 着色）====",
            "from pandas.plotting import scatter_matrix",
            "numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all').iloc[:, :6]",
            "axes = scatter_matrix(numeric, figsize=(STYLE['fig_width'], STYLE['fig_height']), diagonal='hist', alpha=STYLE['alpha'], color='#4E79A7')",
            "for row_ in axes:",
            "    for item_ in row_:",
            "        item_.tick_params(labelsize=max(6, STYLE['tick_size'] - 2))",
            'fig.savefig("plotlab_figure.png", dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")',
            'print("已保存：plotlab_figure.png")',
        ]
    elif family in ("hist", "density", "qq"):
        lines += tidy_setup(("x", "y", "group"))
        if family == "hist":
            lines += [
                "# ==== 作图：直方图（可分组叠加）====",
                "sns.histplot(data=plot_df, x='x', hue='group' if 'group' in plot_df.columns else None,",
                f"             bins={int(params.get('bins') or 30)}, stat={'density' if params.get('density') else 'count'},",
                f"             alpha={alpha * 0.75}, palette=GROUP_COLORS or None, ax=ax)",
            ]
        elif family == "density":
            lines += [
                "# ==== 作图：分组密度曲线 ====",
                "sns.kdeplot(data=plot_df, x='x', hue='group' if 'group' in plot_df.columns else None,",
                f"            fill={bool(params.get('fill', True))}, alpha={alpha * 0.45}, linewidth={lw},",
                "            palette=GROUP_COLORS or None, ax=ax)",
            ]
        else:
            lines += [
                "# ==== 作图：Q-Q 图 ====",
                "from scipy import stats as _st",
                "groups_ = list(pd.unique(plot_df['group'])) if 'group' in plot_df.columns else ['all']",
                "for i_, name_ in enumerate(groups_):",
                "    values_ = pd.to_numeric(plot_df.loc[plot_df['group'] == name_, 'x'], errors='coerce').dropna() if 'group' in plot_df.columns else pd.to_numeric(plot_df['x'], errors='coerce').dropna()",
                "    if len(values_) < 3:",
                "        continue",
                "    qq_ = _st.probplot(values_, dist='norm')",
                "    color_ = GROUP_COLORS.get(str(name_)) or f'C{i_}'",
                "    ax.scatter(qq_[0][0], qq_[0][1], s=STYLE['point_size'], color=color_, alpha=STYLE['alpha'], label=str(name_))",
                "    ax.plot(qq_[0][0], qq_[0][0] * qq_[1][0] + qq_[1][1], color='#222222', linestyle='--', linewidth=1)",
                "ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Sample quantiles')",
            ]
        lines += _finish(style, levels)
    elif family == "roc":
        lines += tidy_setup(("x", "y", "group"))
        lines += [
            "# ==== 作图：ROC 曲线（需要 0/1 标签与分数列）====",
            "from sklearn.metrics import roc_curve, auc",
            f"positive = {str(params.get('positive') or '')!r}",
            "label_series = plot_df['group'] if 'group' in plot_df.columns else plot_df['x']",
            "label_series = label_series.astype(str)",
            "if positive and positive in set(label_series.dropna()):",
            "    y_true = (label_series == positive).astype(int)",
            "elif label_series.nunique() == 2:",
            "    levels_ = list(pd.unique(label_series.dropna()))",
            "    y_true = (label_series == levels_[1]).astype(int)",
            "else:",
            "    raise ValueError('ROC 需要二分类标签列（group 或 x）')",
            "scores = pd.to_numeric(plot_df['y'], errors='coerce')",
            "mask = scores.notna()",
            "fpr, tpr, thresholds = roc_curve(y_true[mask], scores[mask])",
            "auc_value = auc(fpr, tpr)",
            "ax.plot(fpr, tpr, color='#D55E00', linewidth=STYLE['line_width'] + 0.4, label=f'AUC = {auc_value:.3f}')",
            "ax.plot([0, 1], [0, 1], color='#999999', linestyle='--', linewidth=1)",
            "ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')",
            "ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)",
            "ax.legend(frameon=False, fontsize=STYLE['legend_size'])",
            "print(f'AUC = {auc_value:.4f}')",
            "ax.set_title(STYLE['title'], fontsize=STYLE['title_size'], fontweight='bold')",
            'fig.savefig("plotlab_figure.png", dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")',
            'print("已保存：plotlab_figure.png")',
        ]
    elif family == "km":
        lines += tidy_setup(("time", "event", "group"))
        lines += [
            "# ==== 作图：Kaplan-Meier 生存曲线 + log-rank（lifelines）====",
            "from lifelines import KaplanMeierFitter",
            "from lifelines.statistics import logrank_test",
            "plot_df['event'] = pd.to_numeric(plot_df['event'], errors='coerce').fillna(0).astype(int)",
            "groups_ = list(pd.unique(plot_df['group'])) if 'group' in plot_df.columns else ['all']",
            "for i_, name_ in enumerate(groups_):",
            "    sub_ = plot_df[plot_df['group'] == name_] if 'group' in plot_df.columns else plot_df",
            "    km_ = KaplanMeierFitter()",
            "    km_.fit(sub_['time'], sub_['event'], label=str(name_))",
            "    km_.plot_survival_function(ax=ax, ci_show=%r, linewidth=STYLE['line_width']," % bool(params.get("ci", True)),
            "                             color=GROUP_COLORS.get(str(name_)) or f'C{i_}')",
            "if len(groups_) == 2:",
            "    a_ = plot_df[plot_df['group'] == groups_[0]]",
            "    b_ = plot_df[plot_df['group'] == groups_[1]]",
            "    result_ = logrank_test(a_['time'], b_['time'], a_['event'], b_['event'])",
            "    ax.text(0.02, 0.04, f'log-rank p = {result_.p_value:.4g}', transform=ax.transAxes, fontsize=STYLE['tick_size'])",
            "    print(f'log-rank p = {result_.p_value:.4g}')",
            "ax.set_xlabel(STYLE['x_label'] or 'Time'); ax.set_ylabel(STYLE['y_label'] or 'Survival probability')",
            "ax.set_title(STYLE['title'], fontsize=STYLE['title_size'], fontweight='bold')",
            "ax.legend(frameon=False, fontsize=STYLE['legend_size'])",
            'fig.savefig("plotlab_figure.png", dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")',
            'print("已保存：plotlab_figure.png")',
        ]
    elif family == "forest":
        estimates = params.get("estimates")
        lines += ["# ==== 作图：森林图（效应量 + 95% CI）===="]
        if isinstance(estimates, list) and estimates:
            lines += ["estimates = " + repr(estimates)]
        else:
            lines += [
                "# 工作台没有拿到 estimates 参数时，可在这里手填：",
                "# estimates = [{'label': 'age', 'estimate': 1.42, 'lower': 1.10, 'upper': 1.83, 'p': 0.01}]",
                "estimates = []",
            ]
        lines += [
            "if not estimates:",
            "    raise ValueError('森林图需要 estimates 参数')",
            "labels_ = [str(item.get('label') or item.get('term') or '') for item in estimates]",
            "values_ = np.array([float(item['estimate']) for item in estimates])",
            "lowers_ = np.array([float(item.get('lower', item['estimate'])) for item in estimates])",
            "uppers_ = np.array([float(item.get('upper', item['estimate'])) for item in estimates])",
            "positions_ = np.arange(len(labels_))",
            "c_ = ['#D55E00' if low > 1 else '#0072B2' if high < 1 else '#666666' for low, high in zip(lowers_, uppers_)]",
            "ax.errorbar(values_, positions_, xerr=[values_ - lowers_, uppers_ - values_], fmt='o',",
            "            color='#222222', ecolor=c_, elinewidth=1.4, capsize=3, markersize=5)",
            "ax.axvline(1.0, color='#999999', linestyle='--', linewidth=1)",
            "ax.set_yticks(positions_); ax.set_yticklabels(labels_)",
            "ax.set_xlabel(STYLE['x_label'] or 'Hazard ratio / Effect size')",
            "ax.set_title(STYLE['title'], fontsize=STYLE['title_size'], fontweight='bold')",
            'fig.savefig("plotlab_figure.png", dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")',
            'print("已保存：plotlab_figure.png")',
        ]
    elif family == "pca":
        lines += [
            "# ==== 作图：PCA（数值列标准化后降维，按分组着色）====",
            "from sklearn.decomposition import PCA",
            "from sklearn.preprocessing import StandardScaler",
            "numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')",
            "X = StandardScaler().fit_transform(numeric.fillna(numeric.median()))",
            "pca = PCA(n_components=2).fit(X)",
            "scores = pca.transform(X)",
            "group_values = df[df.columns[0]].astype(str)",
            "sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=group_values,",
            "                palette=GROUP_COLORS or None, s=STYLE['point_size'] * 2, alpha=STYLE['alpha'], ax=ax)",
            "ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')",
            "ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')",
            "ax.set_title(STYLE['title'], fontsize=STYLE['title_size'], fontweight='bold')",
            'fig.savefig("plotlab_figure.png", dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")',
            'print("已保存：plotlab_figure.png")',
        ]
    else:
        lines += tidy_setup(("x", "y", "group"))
        lines += ["sns.boxplot(data=plot_df, x='x', y='y', ax=ax)"] + _finish(style, levels)
    return "\n".join(lines) + "\n"


def r_code(chart_id: str, data: Mapping[str, Any], params: Mapping[str, Any],
           style: Mapping[str, Any]) -> str:
    """生成等价的 ggplot2 片段（与工作台图近似；显著性标注可用 ggpubr）。"""
    family = FAMILY.get(chart_id, "box")
    style = C.resolve_style(style)
    labels = data.get("labels") or {}
    levels = (data.get("levels") or {}).get("group") or []
    colors = _colors(style, levels) if levels else {}
    x = labels.get("x") or "x"
    y = labels.get("y") or "y"
    group = labels.get("group") or labels.get("color")
    fills = ", ".join(f'"{name}" = "{color}"' for name, color in colors.items())
    size = int(round(float(style.get("tick_size", 10)) * 1.4))
    title_size = int(style.get("title_size", 14))
    label_size = int(style.get("label_size", 12))
    legend_pos = "none" if style.get("legend_pos") == "none" else "right"
    width = float(style.get("fig_width", 6.2))
    height = float(style.get("fig_height", 4.4))
    dpi = int(style.get("dpi", 300))
    title = str(style.get("title_text") or "")
    title_line = f"  labs(title = {title!r})" if title else "  labs(title = NULL)"

    # 各家族的 aes() 与图层
    mapping = f'aes(x = .data[["{x}"]]'
    if family in ("bar", "bar_dot", "box", "violin", "raincloud", "line", "area",
                  "scatter", "bubble", "hist", "density"):
        mapping += f', y = .data[["{y}"]]'
    if group and family in ("bar", "bar_dot", "box", "violin", "raincloud", "line", "area", "density"):
        mapping += f', fill = .data[["{group}"]]'
    if group and family in ("scatter", "bubble"):
        mapping += f', color = .data[["{group}"]]'
    mapping += ")"

    layers = {
        "bar": [
            f'  stat_summary(aes(fill = .data[["{group}"]]), fun = mean, geom = "bar", position = position_dodge(0.8)) +' if group else
            '  stat_summary(fun = mean, geom = "bar", fill = "#4E79A7") +',
            '  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2, position = position_dodge(0.8))',
        ],
        "bar_dot": [
            f'  stat_summary(aes(fill = .data[["{group}"]]), fun = mean, geom = "bar", position = position_dodge(0.8)) +' if group else
            '  stat_summary(fun = mean, geom = "bar", fill = "#4E79A7") +',
            '  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2, position = position_dodge(0.8)) +',
            '  geom_jitter(width = 0.12, alpha = 0.4, size = 0.8)',
        ],
        "box": [
            '  geom_boxplot(width = 0.6, outlier.shape = NA) +',
            '  geom_jitter(width = 0.12, alpha = 0.45, size = 0.8)',
        ],
        "violin": [
            '  geom_violin(alpha = 0.6, width = 0.8, color = NA) +',
            '  geom_boxplot(width = 0.12, outlier.shape = NA) +',
            '  geom_jitter(width = 0.08, alpha = 0.35, size = 0.7)',
        ],
        "raincloud": [
            '  geom_violin(alpha = 0.55, width = 0.8, color = NA) +',
            '  geom_boxplot(width = 0.1, outlier.shape = NA) +',
            '  geom_jitter(width = 0.06, alpha = 0.35, size = 0.7)',
        ],
        "scatter": [
            '  geom_point(size = 2, alpha = 0.8) +',
            '  geom_smooth(method = "lm", se = TRUE, color = "#222222")',
        ],
        "bubble": [
            '  geom_point(aes(size = .data[["size"]]), alpha = 0.8)',
        ],
        "line": [
            f'  stat_summary(fun = mean, geom = "line", aes(color = .data[["{group}"]])) +' if group else
            '  stat_summary(fun = mean, geom = "line") +',
            f'  stat_summary(fun.data = mean_se, geom = "ribbon", aes(fill = .data[["{group}"]]), alpha = 0.2)',
        ],
        "area": [
            '  stat_summary(fun = mean, geom = "area", alpha = 0.5)',
        ],
        "stack": [
            '  stat_summary(fun = sum, geom = "bar")',
        ],
        "dotplot": [
            '  geom_point(aes(size = .data[["y"]], color = .data[["y"]]))',
        ],
        "hist": [
            '  geom_histogram(bins = 30, fill = "#4E79A7", color = "white")',
        ],
        "density": [
            '  geom_density(alpha = 0.35)',
        ],
    }
    if family in layers:
        body = "\n".join(["p <- ggplot(df, " + mapping + ")", *layers[family]])
    elif family == "pie":
        body = "p <- ggplot(df, " + f'aes(x = "", fill = .data[["{x}"]])' + ") +\n  stat_summary(fun = sum, geom = \"bar\") +\n  coord_polar(theta = \"y\")"
    elif family == "heatmap":
        body = ("# 热图建议用 pheatmap：\n"
                "# library(pheatmap)\n"
                "# mat <- as.matrix(df[, -1]); rownames(mat) <- df[[1]]\n"
                "# pheatmap(mat, scale = \"row\", color = colorRampPalette(c(\"#3B4CC0\", \"white\", \"#B40426\"))(100))")
    elif family == "corr":
        body = ("# 相关矩阵建议用 corrplot：\n"
                "# library(corrplot)\n"
                "# corrplot(cor(df[sapply(df, is.numeric)]), method = \"color\", addCoef.col = \"black\", tl.col = \"black\")")
    elif family == "pairplot":
        body = ("# 散点矩阵建议用 GGally：\n"
                "# library(GGally)\n"
                "# ggpairs(df[sapply(df, is.numeric)][, 1:6])")
    elif family == "roc":
        body = ("# ROC 建议用 pROC：\n"
                "# library(pROC)\n"
                "# roc_obj <- roc(df$label, df$score)\n"
                "# plot(roc_obj, print.auc = TRUE, col = \"#D55E00\")")
    elif family == "km":
        body = ("# 生存分析建议用 survminer：\n"
                "# library(survival); library(survminer)\n"
                "# fit <- survfit(Surv(time, event) ~ group, data = df)\n"
                "# ggsurvplot(fit, data = df, pval = TRUE, risk.table = TRUE)")
    elif family == "forest":
        body = ("# 森林图（先算 HR/OR 与 CI）：\n"
                "# ggplot(res, aes(x = estimate, y = term, xmin = lower, xmax = upper)) +\n"
                "#   geom_vline(xintercept = 1, linetype = 2, color = \"grey60\") +\n"
                "#   geom_pointrange() + theme_bw()")
    elif family == "qq":
        body = ("p <- ggplot(df, " + mapping + ") +\n"
                "  stat_qq(alpha = 0.8) +\n"
                "  stat_qq_line(color = \"#222222\", linetype = \"dashed\")")
    elif family == "pca":
        body = ("# PCA 建议用 prcomp：\n"
                "# pca <- prcomp(df[sapply(df, is.numeric)], scale. = TRUE)\n"
                "# ggplot(as.data.frame(pca$x), aes(PC1, PC2)) + geom_point() + theme_bw()")
    else:
        body = "# 暂不支持该图表的 R 版本"
    if body.startswith("#"):
        return "# R / ggplot2 版本（该图建议用专门的 R 包，见下）\n" + body + "\n"
    lines = [
        "# R / ggplot2 版本（与工作台图近似；显著性标注可用 ggpubr::stat_compare_means）",
        "library(ggplot2)",
        body,
        f"p <- p + theme_bw(base_size = {size}) +",
        f"  theme(plot.title = element_text(size = {title_size}, face = \"bold\"),",
        f"        axis.title = element_text(size = {label_size}),",
        f"        legend.position = \"{legend_pos}\") +",
        title_line,
    ]
    if colors:
        lines.append(f'p <- p + scale_fill_manual(values = c({fills})) + scale_color_manual(values = c({fills}))')
    lines.append(f'ggsave("plotlab_figure.png", p, width = {width:.2f}, height = {height:.2f}, dpi = {dpi})')
    return "\n".join(lines) + "\n"
