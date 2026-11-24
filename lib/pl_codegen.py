"""PlotLab 代码生成：把一次出图/统计翻译成可独立运行的 Python（另附 R 片段）。

要点：
- 脚本先读数据，再按 dataset["cleaning"] 重放清洗步骤，再画图，最后统计，保证「图 = 代码产物」。
- demo 数据（没有真实源文件）直接内嵌 DataFrame 字面量，脚本拷走就能跑。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import pl_common as C
import pl_prep as P


HEADER = '''"""
PlotLab 绘图工作台 —— 自动生成的可复现脚本
生成时间：{stamp}
数据：{dataset_name}（{n_rows} 行 × {n_cols} 列，{kind}）
图表：{chart_title}（{chart_id}）
说明：本脚本与工作台里的图**逐参数一致**；路径/列名如需修改，改最上面的 CONFIG 即可。
依赖：pandas numpy matplotlib seaborn scipy  （统计部分可能用到 statsmodels scikit-learn lifelines）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {{
    "figure_width": {fig_width},
    "figure_height": {fig_height},
    "dpi": {dpi},
    "out_png": "plotlab_figure.png",
}}
'''


def _stamp() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _data_literal(frame: pd.DataFrame, limit: int = 2000) -> str:
    head = frame.head(limit)
    records = []
    for _, row in head.iterrows():
        item = {}
        for key in head.columns:
            value = row[key]
            if isinstance(value, (np.floating, float)):
                item[str(key)] = None if pd.isna(value) else float(value)
            elif isinstance(value, (np.integer, int)):
                item[str(key)] = int(value)
            elif isinstance(value, (np.bool_, bool)):
                item[str(key)] = bool(value)
            elif value is None or (isinstance(value, float) and pd.isna(value)):
                item[str(key)] = None
            else:
                item[str(key)] = str(value)
        records.append(item)
    text = json.dumps(records, ensure_ascii=False, indent=None)
    lines = [
        "# 内置数据（demo/无本地源文件时内嵌；真实项目请改成 pd.read_csv('你的文件.csv')）",
        "import io as _io",
        "_DATA_JSON = r'''" + text + "'''",
        "df = pd.DataFrame.from_records(__import__('json').loads(_DATA_JSON))",
    ]
    if len(frame) > limit:
        lines.append(f"# 注意：原数据 {len(frame)} 行，脚本内只内嵌前 {limit} 行")
    return "\n".join(lines)


def _read_block(dataset: Mapping[str, Any]) -> str:
    """按 dataset["import"]（pl_io 记录的真实格式/编码/分隔符/sheet/member）生成读取代码。"""
    source = str(dataset.get("source") or "")
    info = dataset.get("import") or {}
    fmt = str(info.get("format") or "").lower()
    encoding = str(info.get("encoding") or "utf-8")
    delimiter = info.get("delimiter")
    sheet = info.get("sheet") or dataset.get("sheet")
    member = info.get("member")
    archive = info.get("archive")
    gzip_flag = bool(info.get("gzip"))
    transpose = bool(info.get("transpose"))
    header = "# ==== 1. 读取原始数据 ===="
    lines = [header]
    if source and not source.startswith("demo:") and os.path.exists(source):
        sep_repr = repr(delimiter) if delimiter else '","'
        if fmt in ("xlsx", "excel") or source.lower().endswith((".xlsx", ".xlsm")):
            lines += [
                f"SOURCE = r\"{source}\"",
                f"df = pd.read_excel(SOURCE, sheet_name={sheet!r})" if sheet not in (None, "") else "df = pd.read_excel(SOURCE)",
            ]
        elif fmt == "json" or source.lower().endswith(".json"):
            lines += [f"SOURCE = r\"{source}\"", "df = pd.read_json(SOURCE)"]
        elif archive and member:
            lines += [
                f"ARCHIVE = r\"{source}\"",
                f"MEMBER = r\"{member}\"",
                "import zipfile",
                "with zipfile.ZipFile(ARCHIVE) as _z:",
                "    with _z.open(MEMBER) as _f:",
                f"        df = pd.read_csv(_f, sep={sep_repr}, encoding={encoding!r})",
            ]
        elif gzip_flag or source.lower().endswith(".gz"):
            lines += [
                f"SOURCE = r\"{source}\"",
                f"df = pd.read_csv(SOURCE, sep={sep_repr}, encoding={encoding!r}, compression='gzip')",
            ]
        else:
            lines += [
                f"SOURCE = r\"{source}\"",
                f"df = pd.read_csv(SOURCE, sep={sep_repr}, encoding={encoding!r})",
            ]
    elif isinstance(dataset.get("raw_df"), pd.DataFrame):
        lines.append(_data_literal(dataset["raw_df"]))
        return "\n".join(lines)
    else:
        lines.append("# 未找到源文件，请在下面手动指定数据：")
        lines.append("# df = pd.read_csv('your_data.csv')")
        return "\n".join(lines)
    if transpose:
        lines.append("# 注意：工作台导入时做过转置，这里保持同样方向：")
        lines.append("df = df.set_index(df.columns[0]).T.reset_index().rename(columns={'index': df.columns[0]})")
    lines.append("print('原始数据：', df.shape)")
    return "\n".join(lines)


def _repr_list(values: Sequence[Any]) -> str:
    return "[" + ", ".join(repr(v) for v in values) + "]"


def _clean_block(dataset: Mapping[str, Any]) -> str:
    steps = list(dataset.get("cleaning") or [])
    lines = ["# ==== 2. 数据清洗（与工作台一致）===="]
    if not steps:
        lines.append("# 工作台里没有应用清洗步骤")
        return "\n".join(lines)
    for index, step in enumerate(steps, 1):
        op = str(step.get("op") or "")
        params = dict(step.get("params") or {})
        reason = step.get("reason")
        lines.append(f"# [{index}] {op}" + (f" —— {reason}" if reason else ""))
        if op == "drop_empty":
            columns = params.get("columns") or []
            lines.append(f"df = df.drop(columns={_repr_list(columns)}, errors='ignore')")
        elif op == "drop_columns":
            columns = params.get("columns") or []
            lines.append(f"df = df.drop(columns={_repr_list(columns)}, errors='ignore')")
        elif op == "drop_duplicates":
            subset = params.get("subset") or None
            if subset:
                lines.append(f"df = df.drop_duplicates(subset={_repr_list(subset)})")
            else:
                lines.append("df = df.drop_duplicates()")
        elif op == "rename":
            mapping = params.get("mapping") or {}
            lines.append(f"df = df.rename(columns={mapping!r})")
        elif op == "coerce":
            targets = params.get("columns") or {}
            if isinstance(targets, str):
                targets = {targets: "numeric"}
            lines.append("def _to_numeric(s):")
            lines.append("    # 宽松转数值：去千分位/百分号，'<0.001' 取 0.0005")
            lines.append("    text = s.astype(str).str.strip()")
            lines.append("    halve = text.str.startswith('<')")
            lines.append("    text = text.str.lstrip('<≥>≤').str.replace(r'[,\\s]', '', regex=True).str.rstrip('%')")
            lines.append("    values = pd.to_numeric(text, errors='coerce')")
            lines.append("    values = values.where(~halve, values.abs() / 2.0)")
            lines.append("    return values")
            for column, target in targets.items():
                if target == "numeric":
                    lines.append(f"df[{column!r}] = _to_numeric(df[{column!r}])")
                elif target == "datetime":
                    lines.append(f"df[{column!r}] = pd.to_datetime(df[{column!r}], errors='coerce')")
                else:
                    lines.append(f"df[{column!r}] = df[{column!r}].astype(str).where(df[{column!r}].notna())")
            lines.append("ddof_note = None")
        elif op == "dropna":
            columns = params.get("columns") or []
            how = params.get("how") or "any"
            lines.append(f"df = df.dropna(subset={_repr_list(columns)}, how={how!r})")
        elif op == "impute":
            columns = params.get("columns") or []
            method = str(params.get("method") or "median")
            for column in columns:
                if method == "mean":
                    lines.append(f"df[{column!r}] = df[{column!r}].fillna(df[{column!r}].mean())")
                elif method == "median":
                    lines.append(f"df[{column!r}] = pd.to_numeric(df[{column!r}], errors='coerce')")
                    lines.append(f"df[{column!r}] = df[{column!r}].fillna(df[{column!r}].median())")
                elif method == "mode":
                    lines.append(f"df[{column!r}] = df[{column!r}].fillna(df[{column!r}].mode().iloc[0])")
                else:
                    lines.append(f"df[{column!r}] = df[{column!r}].fillna(0)")
        elif op == "outlier":
            columns = params.get("columns") or []
            method = str(params.get("method") or "iqr")
            action = str(params.get("action") or "flag")
            k = float(params.get("k") or (1.5 if method == "iqr" else 3.0))
            if action == "remove":
                lines.append("_keep = pd.Series(True, index=df.index)")
                for column in columns:
                    lines.append(f"_v = pd.to_numeric(df[{column!r}], errors='coerce')")
                    if method == "zscore":
                        lines.append(f"_z = (_v - _v.mean()) / (_v.std(ddof=1) or 1)")
                        lines.append(f"_keep &= _z.abs() <= {k}")
                    elif method == "mad":
                        lines.append(f"_med = _v.median(); _mad = (_v - _med).abs().median() * 1.4826 or 1")
                        lines.append(f"_keep &= ((_v - _med) / _mad).abs() <= {k}")
                    else:
                        lines.append(f"_q1, _q3 = _v.quantile([0.25, 0.75]); _iqr = _q3 - _q1 or 1")
                        lines.append(f"_keep &= (_v >= _q1 - {k} * _iqr) & (_v <= _q3 + {k} * _iqr)")
                lines.append("df = df[_keep].reset_index(drop=True)")
            elif action == "winsorize":
                for column in columns:
                    lines.append(f"_v = pd.to_numeric(df[{column!r}], errors='coerce')")
                    if method == "iqr":
                        lines.append(f"_q1, _q3 = _v.quantile([0.25, 0.75]); _iqr = _q3 - _q1")
                        lines.append(f"df[{column!r}] = _v.clip(_q1 - {k} * _iqr, _q3 + {k} * _iqr)")
                    else:
                        lines.append(f"df[{column!r}] = _v.clip(_v.mean() - {k} * _v.std(ddof=1), _v.mean() + {k} * _v.std(ddof=1))")
            else:
                for column in columns:
                    lines.append(f"_v = pd.to_numeric(df[{column!r}], errors='coerce')")
                    lines.append(f"_q1, _q3 = _v.quantile([0.25, 0.75]); _iqr = _q3 - _q1 or 1")
                    lines.append(f"df[{column!r} + '_outlier'] = (_v < _q1 - {k} * _iqr) | (_v > _q3 + {k} * _iqr)")
        elif op == "filter":
            column = params.get("column")
            operator = str(params.get("op") or "==")
            value = params.get("value")
            if column:
                if operator in ("==", "="):
                    lines.append(f"df = df[df[{column!r}].astype(str) == {str(value)!r}]")
                elif operator == "!=":
                    lines.append(f"df = df[df[{column!r}].astype(str) != {str(value)!r}]")
                elif operator == "in":
                    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
                    lines.append(f"df = df[df[{column!r}].astype(str).isin({_repr_list([str(v) for v in items])})]")
                elif operator == "contains":
                    lines.append(f"df = df[df[{column!r}].astype(str).str.contains({str(value)!r}, na=False)]")
                elif operator == "notnull":
                    lines.append(f"df = df[df[{column!r}].notna()]")
                else:
                    lines.append(f"df = df[pd.to_numeric(df[{column!r}], errors='coerce') {operator} {float(value)}]")
        elif op == "transform":
            columns = params.get("columns") or []
            method = str(params.get("method") or "log2")
            offsets = params.get("_offsets") or {}
            for column in columns:
                offset = offsets.get(column)
                prefix = f"df[{column!r}]"
                lines.append(f"_v = pd.to_numeric({prefix}, errors='coerce').astype(float)")
                if method in ("log2", "log10", "ln", "log"):
                    if offset:
                        lines.append(f"_v = _v + {float(offset)}  # 含非正值，整体平移")
                    func = {"log2": "np.log2", "log10": "np.log10", "ln": "np.log", "log": "np.log"}[method]
                    lines.append(f"{prefix} = {func}(_v)")
                elif method == "log1p":
                    lines.append(f"{prefix} = np.log1p(_v)")
                elif method == "sqrt":
                    lines.append(f"{prefix} = np.sqrt(_v)")
                elif method == "zscore":
                    lines.append(f"{prefix} = (_v - _v.mean()) / _v.std(ddof=1)")
                elif method == "minmax":
                    lines.append(f"{prefix} = (_v - _v.min()) / (_v.max() - _v.min())")
                elif method == "percent":
                    lines.append(f"{prefix} = _v / _v.sum() * 100")
                elif method == "rank":
                    lines.append(f"{prefix} = _v.rank(pct=True) * 100")
                else:
                    lines.append(f"# 未知变换 {method}（跳过）")
        elif op == "aggregate":
            group_by = params.get("group_by")
            value_cols = params.get("value_cols") or []
            how = params.get("how") or "mean"
            if group_by:
                lines.append(f"df = df.groupby({group_by!r}, as_index=False).agg({{{', '.join(repr(c) + ': ' + repr(how) for c in value_cols)}}})")
            lines.append("_note = '聚合后行数变化'")
        elif op == "reset":
            lines.append("# reset：恢复原始数据（见读取部分）")
        else:
            lines.append(f"# 未支持的清洗操作：{op}（{params}）")
        if step.get("rows_before") is not None and step.get("rows_after") is not None:
            lines.append(f"# 行数：{step['rows_before']} → {step['rows_after']}")
    lines.append("print('清洗后数据：', df.shape)")
    return "\n".join(lines)


def stats_snippet(stat_id: str, roles: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> str:
    """生成统计检验片段；未知 stat 返回空。"""
    params = dict(params or {})
    group = roles.get("group") or roles.get("x")
    y = roles.get("y")
    lines: list[str] = ["# ==== 4. 统计分析 ===="]
    if stat_id == "compare" and group and y:
        lines += [
            "from scipy import stats",
            f"_groups = [g for g in df[{group!r}].dropna().unique()]",
            f"_samples = [pd.to_numeric(df.loc[df[{group!r}] == g, {y!r}], errors='coerce').dropna() for g in _groups]",
            "_samples = [s for s in _samples if len(s) >= 2]",
            "if len(_samples) == 2:",
            "    _t, _p = stats.ttest_ind(*_samples, equal_var=False)   # Welch t",
            "    _u, _p2 = stats.mannwhitneyu(*_samples, alternative='two-sided')",
            "    print(f'Welch t={_t:.3f}, p={_p:.4g}; Mann-Whitney p={_p2:.4g}')",
            "elif len(_samples) > 2:",
            "    _f, _p = stats.f_oneway(*_samples)                      # one-way ANOVA",
            "    _h, _p2 = stats.kruskal(*_samples)                      # Kruskal-Wallis",
            "    print(f'ANOVA F={_f:.3f}, p={_p:.4g}; Kruskal-Wallis p={_p2:.4g}')",
            "else:",
            "    print('组数不足，无法做组间检验')",
        ]
    elif stat_id == "corr":
        x = roles.get("x")
        if x and y:
            lines += [
                "from scipy import stats",
                f"_r, _p = stats.pearsonr(df[{x!r}], df[{y!r}])",
                f"_rho, _p2 = stats.spearmanr(df[{x!r}], df[{y!r}])",
                "print(f'Pearson r={_r:.3f}, p={_p:.4g}; Spearman rho={_rho:.3f}, p={_p2:.4g}')",
            ]
    elif stat_id == "regression":
        x = roles.get("x")
        if x and y:
            lines += [
                "import statsmodels.formula.api as smf",
                f"_model = smf.ols('Q(\"{y}\") ~ Q(\"{x}\")', data=df.dropna(subset=[{y!r}, {x!r}])).fit()",
                "print(_model.summary())",
            ]
    elif stat_id == "chi2":
        if group and y:
            lines += [
                "from scipy import stats",
                f"_table = pd.crosstab(df[{group!r}], df[{y!r}])",
                "_chi2, _p, _dof, _exp = stats.chi2_contingency(_table)",
                "print(_table); print(f'chi2={_chi2:.3f}, dof={_dof}, p={_p:.4g}')",
            ]
    elif stat_id == "normality":
        if y:
            lines += [
                "from scipy import stats",
                f"_x = pd.to_numeric(df[{y!r}], errors='coerce').dropna()",
                "_w, _p = stats.shapiro(_x) if len(_x) <= 5000 else (float('nan'), float('nan'))",
                "print(f'Shapiro-Wilk W={_w:.3f}, p={_p:.4g}; skew={_x.skew():.3f}, kurt={_x.kurt():.3f}')",
            ]
    elif stat_id == "roc":
        if group and y:
            lines += [
                "from sklearn.metrics import roc_curve, auc",
                f"_y = pd.to_numeric(df[{group!r}], errors='coerce')",
                "_y = (_y > 0).astype(int) if _y.nunique() == 2 else _y.astype(int)",
                f"_score = pd.to_numeric(df[{y!r}], errors='coerce')",
                "_mask = _y.notna() & _score.notna()",
                "_fpr, _tpr, _thr = roc_curve(_y[_mask], _score[_mask])",
                "print(f'AUC={auc(_fpr, _tpr):.3f}')",
            ]
    elif stat_id == "survival":
        time_col = roles.get("time")
        event = roles.get("event")
        if time_col and event and group:
            lines += [
                "from lifelines import KaplanMeierFitter",
                "from lifelines.statistics import logrank_test",
                f"_km = KaplanMeierFitter()",
                f"_groups = df[{group!r}].dropna().unique().tolist()",
                "for g in _groups:",
                f"    _m = df[{group!r}] == g",
                f"    _km.fit(df.loc[_m, {time_col!r}], df.loc[_m, {event!r}], label=str(g))",
                "    _km.plot_survival_function(ci_show=True)",
                "# log-rank 两两检验",
                "import itertools",
                f"_sub = df.dropna(subset=[{time_col!r}, {event!r}, {group!r}])",
                "_groups = _sub[" + repr(group) + "].unique().tolist()",
                "for a, b in itertools.combinations(_groups, 2):",
                "    _r = logrank_test(",
                f"        _sub.loc[_sub[{group!r}] == a, {time_col!r}], _sub.loc[_sub[{group!r}] == b, {time_col!r}],",
                f"        _sub.loc[_sub[{group!r}] == a, {event!r}], _sub.loc[_sub[{group!r}] == b, {event!r}])",
                "    print(f'{a} vs {b}: log-rank p={_r.p_value:.4g}')",
                "plt.savefig('plotlab_survival.png', dpi=CONFIG['dpi'], bbox_inches='tight')",
            ]
    elif stat_id == "pca":
        lines += [
            "from sklearn.decomposition import PCA",
            "from sklearn.preprocessing import StandardScaler",
            "_num = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')",
            "_X = StandardScaler().fit_transform(_num.fillna(_num.median()))",
            "_pca = PCA(n_components=min(5, _X.shape[1])).fit(_X)",
            "print('解释方差比：', np.round(_pca.explained_variance_ratio_, 3))",
        ]
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def chart_python(chart_id: str, data: Mapping[str, Any], params: Mapping[str, Any],
                 style: Mapping[str, Any]) -> str:
    """取「只画图」的 Python 片段：优先图表模块自带实现，缺失时用 pl_codesnippet 兜底。"""
    try:
        import pl_charts
        fn = getattr(pl_charts, "python_code", None)
        if callable(fn):
            snippet = str(fn(chart_id, data, params, style))
            compile(snippet, f"<{chart_id}>", "exec")  # 语法不合法就退回兜底
            return snippet
    except Exception:  # noqa: BLE001
        pass
    import pl_codesnippet
    return pl_codesnippet.python_code(chart_id, data, params, style)


def chart_r_code(chart_id: str, data: Mapping[str, Any], params: Mapping[str, Any],
                 style: Mapping[str, Any]) -> str:
    """取 ggplot2 片段：优先图表模块，缺失时用 pl_codesnippet 兜底。"""
    try:
        import pl_charts
        fn = getattr(pl_charts, "r_code", None)
        if callable(fn):
            return str(fn(chart_id, data, params, style))
    except Exception:  # noqa: BLE001
        pass
    import pl_codesnippet
    return pl_codesnippet.r_code(chart_id, data, params, style)


def build_script(dataset: Mapping[str, Any], chart_id: str, meta: Mapping[str, Any],
                 data: Mapping[str, Any], params: Mapping[str, Any], style: Mapping[str, Any],
                 roles: Mapping[str, Any]) -> str:
    """完整可运行脚本：读取 → 清洗 → 作图 → 统计检验。"""
    import pl_charts

    df = dataset.get("df")
    n_rows = int(df.shape[0]) if df is not None else 0
    n_cols = int(df.shape[1]) if df is not None else 0
    head = HEADER.format(
        stamp=_stamp(),
        dataset_name=dataset.get("name") or "data",
        n_rows=n_rows,
        n_cols=n_cols,
        kind=dataset.get("kind") or "tidy",
        chart_title=meta.get("title") or chart_id,
        chart_id=chart_id,
        fig_width=float(style.get("fig_width", 6.2)),
        fig_height=float(style.get("fig_height", 4.4)),
        dpi=int(style.get("dpi", 300)),
    )
    parts = [head, _read_block(dataset), "", _clean_block(dataset), ""]
    try:
        snippet = chart_python(chart_id, data, params, style)
    except Exception as error:  # noqa: BLE001
        snippet = f"# 作图代码生成失败：{error}\n# 你可以用工作台导出的图片作为参考"
    parts.append("# ==== 3. 作图（与工作台逐参数一致）====")
    parts.append(str(snippet or "").strip())
    parts.append("")
    roles_map = {k: v for k, v in (roles or {}).items() if v}
    stat_ids = list(params.get("_stats") or []) if isinstance(params.get("_stats"), list) else []
    for stat_id in stat_ids:
        fragment = stats_snippet(str(stat_id), roles_map)
        if fragment:
            parts.append(fragment)
            parts.append("")
    parts.append("# ==== 5. 保存 ====")
    parts.append('os.makedirs(os.path.dirname(CONFIG["out_png"]) or ".", exist_ok=True)')
    parts.append('plt.savefig(CONFIG["out_png"], dpi=CONFIG["dpi"], bbox_inches="tight")')
    parts.append('print("已保存：", os.path.abspath(CONFIG["out_png"]))')
    return "\n".join(parts) + "\n"


def session_code(dataset: Mapping[str, Any], charts: Sequence[Mapping[str, Any]],
                 style: Mapping[str, Any] | None = None) -> dict:
    """整个会话的复现脚本：按最后一次出图；没有出图则用推荐图表。"""
    import pl_charts

    history = dataset.get("history") or []
    if history:
        last = history[-1]
        chart_id = str(last.get("chart") or "")
        params = dict(last.get("params") or {})
        style = C.resolve_style(last.get("style") or style or {})
        roles = dict(last.get("roles") or {})
    else:
        recommendations = []
        try:
            import pl_clean
            recommendations = pl_clean.profile(dataset)["recommend"]["charts"]
        except Exception:  # noqa: BLE001
            recommendations = []
        chart_id = recommendations[0]["id"] if recommendations else "box"
        meta = next((c for c in charts if c["id"] == chart_id), charts[0])
        style = C.resolve_style(style or {})
        params = {}
        roles = P.recommend_roles(dataset, meta)
    meta = next((c for c in charts if c["id"] == chart_id), None)
    if meta is None:
        raise C.bad_request(f"未知图表 {chart_id}")
    data = P.build_data(dataset, meta, params, roles)
    script = build_script(dataset, chart_id, meta, data, params, style, roles)
    return {
        "chart": chart_id,
        "python": script,
        "filename": "plotlab_pipeline.py",
        "cleaning_steps": len(dataset.get("cleaning") or []),
    }
