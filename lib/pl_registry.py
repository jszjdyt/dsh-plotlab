"""PlotLab worker 状态与调度。

一个 WorkerState 实例持有：数据集表、活动数据集、图注册表、历史记录。
命令分发 handle(cmd, args) 与 DESIGN §7 一致；HTTP/Agent 工具共用这一套实现。
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io as _io
import json
import os
import re
import time
import traceback
from collections import OrderedDict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import pl_common as C
import pl_clean
import pl_prep as P
import pl_state as S

MAX_DATASETS = 12
MAX_HISTORY = 40
MAX_FIGURES = 200
MAX_FIGURE_RUNS = 200
FIGURE_RUN_VERSION = 1

# 图表/统计模块可能由并行开发者维护，导入放在函数内（缺模块时给可读错误而不是启动失败）
def _charts():
    try:
        import pl_charts
        return pl_charts
    except Exception as error:  # noqa: BLE001
        raise C.EngineError("internal", f"图表模块不可用：{error}")


def _stats():
    try:
        import pl_stats
        return pl_stats
    except Exception as error:  # noqa: BLE001
        raise C.EngineError("internal", f"统计模块不可用：{error}")


def _io():
    try:
        import pl_io
        return pl_io
    except Exception as error:  # noqa: BLE001
        raise C.EngineError("internal", f"导入模块不可用：{error}")


def _codegen():
    import pl_codegen
    return pl_codegen


def _export():
    import pl_export
    return pl_export


# ---------------------------------------------------------------------------
# 需求解析（把一句中文/英文需求翻译成可执行计划）
# ---------------------------------------------------------------------------

CHART_KEYWORDS: list[tuple[str, str]] = [
    ("散点矩阵", "pairplot"), ("pairplot", "pairplot"), ("pairs", "pairplot"),
    ("箱线", "box"), ("箱型", "box"), ("boxplot", "box"), ("box plot", "box"), ("box", "box"),
    ("小提琴", "violin"), ("violin", "violin"),
    ("雨云", "raincloud"), ("raincloud", "raincloud"),
    ("柱状", "bar"), ("柱形", "bar"), ("条形", "bar"), ("barchart", "bar"), ("bar chart", "bar"),
    ("点图", "dotplot"), ("dotplot", "dotplot"), ("dot plot", "dotplot"),
    ("饼图", "pie"), ("环形图", "pie"), ("pie", "pie"), ("donut", "pie"),
    ("堆叠", "stack"), ("stack", "stack"),
    ("气泡", "bubble"), ("bubble", "bubble"),
    ("散点", "scatter"), ("scatter", "scatter"),
    ("折线", "line"), ("趋势图", "line"), ("line plot", "line"), ("line chart", "line"), ("line", "line"),
    ("面积图", "area"), ("area", "area"),
    ("热图", "heatmap"), ("热力", "heatmap"), ("heatmap", "heatmap"), ("heat map", "heatmap"),
    ("相关矩阵", "corr"), ("相关性图", "corr"), ("correlation", "corr"), ("corrplot", "corr"),
    ("直方", "hist"), ("histogram", "hist"), ("hist", "hist"),
    ("密度", "density"), ("density", "density"), ("kde", "density"),
    ("qq", "qq"), ("分位数", "qq"),
    ("roc", "roc"), ("auc", "roc"),
    ("生存", "km"), ("kaplan", "km"), ("km 曲线", "km"), ("survival", "km"),
    ("森林", "forest"), ("forest", "forest"),
    ("主成分", "pca"), ("pca", "pca"),
]

STAT_KEYWORDS: list[tuple[str, str]] = [
    ("描述统计", "describe"), ("describe", "describe"), ("汇总统计", "describe"),
    ("前提", "assumption"), ("假设检验", "assumption"), ("assumption", "assumption"),
    ("比较", "compare"), ("差异", "compare"), ("显著性", "compare"), ("t检验", "compare"),
    ("t 检验", "compare"), ("anova", "compare"), ("方差分析", "compare"), ("秩和", "compare"),
    ("相关", "corr"), ("correlation", "corr"), ("pearson", "corr"), ("spearman", "corr"),
    ("回归", "regression"), ("regression", "regression"), ("线性模型", "regression"),
    ("卡方", "chi2"), ("列联", "chi2"), ("chi", "chi2"),
    ("正态", "normality"), ("normality", "normality"), ("shapiro", "normality"),
    ("cox", "survival"), ("生存分析", "survival"), ("log-rank", "survival"),
    ("auc", "roc"), ("roc", "roc"),
    ("主成分", "pca"), ("pca", "pca"),
    ("校正", "adjust"), ("多重检验", "adjust"),
]

STYLE_HINTS = [
    (r"色盲友好|colorblind|okabe", {"palette": "okabe_ito"}),
    (r"nature", {"theme": "nature", "palette": "npg"}),
    (r"lancet|柳叶刀", {"theme": "lancet", "palette": "lancet"}),
    (r"jama", {"theme": "jama", "palette": "jama"}),
    (r"set ?1", {"palette": "set1"}),
    (r"set ?2", {"palette": "set2"}),
    (r"马卡龙|pastel|柔和", {"palette": "pastel"}),
    (r"灰色|灰度|grayscale|grey", {"palette": "grey"}),
    (r"viridis", {"palette": "viridis"}),
    (r"字号?大|字体大|大一点", {"title_size": 16, "label_size": 14, "tick_size": 12, "legend_size": 12}),
    (r"字号?小|字体小|小一点", {"title_size": 12, "label_size": 10, "tick_size": 9, "legend_size": 9}),
    (r"times|衬线", {"font": "Times New Roman"}),
    (r"arial", {"font": "Arial"}),
    (r"中文标题|中文标签", {"font": "Arial Unicode MS"}),
    (r"网格|grid", {"grid": "both"}),
    (r"不要图例|去掉图例|无图例", {"legend_pos": "none"}),
    (r"图例在?上面|图例上", {"legend_pos": "top"}),
    (r"图例在?下面|图例下", {"legend_pos": "bottom"}),
    (r"横向|横版|宽一点", {"fig_width": 8.4, "fig_height": 4.6}),
    (r"竖版|竖向|窄一点", {"fig_width": 5.0, "fig_height": 6.4}),
    (r"正方形|方形", {"fig_width": 5.6, "fig_height": 5.6}),
    (r"高分辨率|600 ?dpi|印刷", {"dpi": 600}),
    (r"标准差|error bar|误差线", {"error": "sd"}),
    (r"标准误|se\b", {"error": "se"}),
    (r"95%?ci|置信区间", {"error": "ci95"}),
    (r"不要标注|去掉显著性", {"stat_annot": False}),
    (r"标星|加星|显著性标注", {"stat_annot": True}),
    (r"对数轴|log 轴|log轴", {"y_log": True}),
]

CLEAN_HINTS = [
    (r"缺失|缺测|na\b|空值", "missing"),
    (r"异常值|离群|outlier", "outlier"),
    (r"去重|重复", "duplicate"),
    (r"log ?变换|对数变换|log2 变换", "log"),
    (r"标准化|z-?score|归一化", "zscore"),
    (r"删除空列|去掉空列|空列", "empty"),
]


# 否定词（含常见“不要/无需”变体）：命中窗口内出现即视为该触发词被否定
# 注意：「去掉/删除」对清洗是肯定动词（去掉重复=去重），不是否定词，不能列入
_NEGATION_TOKENS = ("不", "别", "勿", "无需", "无须", "不用", "不要", "禁止", "取消",
                    "没有", "未")
_NEGATION_WINDOW = 8
_CLAUSE_DELIMITERS = "，,。;；、！？!?\n"


def _negated_at(text: str, start: int) -> bool:
    """同一分句内、触发词前 8 个字符里是否出现否定词。

    分句边界（逗号/句号/分号/顿号…）会截断否定作用域：
    “不画箱线图，画柱状图”只否定箱线图，“不删除重复”否定去重。
    """
    clause_start = 0
    for delimiter in _CLAUSE_DELIMITERS:
        position = text.rfind(delimiter, 0, start)
        if position + 1 > clause_start:
            clause_start = position + 1
    prefix = text[max(clause_start, start - _NEGATION_WINDOW):start]
    return any(token in prefix for token in _NEGATION_TOKENS)


def _first_active_hit(text: str, needle: str) -> int:
    """返回第一个未被否定的 needle 位置；没有则 -1。"""
    lowered = text.lower()
    key = str(needle).lower()
    if not key:
        return -1
    index = lowered.find(key)
    while index >= 0:
        if not _negated_at(lowered, index):
            return index
        index = lowered.find(key, index + len(key))
    return -1


def _first_active_regex(text: str, pattern: str) -> int:
    """正则版：返回第一个未被否定的匹配起点；没有则 -1。"""
    for match in re.finditer(pattern, text):
        if not _negated_at(text, match.start()):
            return match.start()
    return -1


def _match_keyword(text: str, table: Sequence[tuple[str, str]]) -> str | None:
    best: tuple[int, str] | None = None
    for keyword, value in table:
        index = _first_active_hit(text, keyword)
        if index >= 0 and (best is None or index < best[0]):
            best = (index, value)
    return best[1] if best else None


def _match_columns_in_text(text: str, dataset: Mapping[str, Any]) -> list[str]:
    """找出需求文本里提到的列名（长的优先，避免 group_text 命中 group）。"""
    names = sorted({str(c["key"]) for c in dataset.get("columns") or []}, key=lambda n: (-len(n), n))
    found: list[str] = []
    lowered = text.lower()
    for name in names:
        if name.lower() in lowered and name not in found:
            found.append(name)
    return found


def _strip_quotes(text: str) -> str:
    return text.strip(" 　「」『』\"'`：:；;，,。.")


def parse_need(need: str, dataset: Mapping[str, Any]) -> dict:
    """把自然语言需求解析成执行计划（不抛异常，无法识别时按数据推荐）。"""
    need = str(need or "").strip()
    lowered = need.lower()
    notes: list[str] = []
    style: dict[str, Any] = {}
    params: dict[str, Any] = {}
    stat_ids: list[str] = []
    cleaning_request: list[str] = []
    chart = _match_keyword(need, CHART_KEYWORDS)
    mentioned = _match_columns_in_text(need, dataset)
    numeric = P.numeric_columns(dataset)
    categorical = P.categorical_columns(dataset)
    kind = dataset.get("kind")
    roles = {k: None for k in P.ROLE_KEYS}
    saved_roles = dataset.get("roles") or {}
    for key in P.ROLE_KEYS:
        if saved_roles.get(key):
            roles[key] = saved_roles[key]

    # ---- 统计关键词（否定词保护：“不做显著性”不触发 compare）----
    for keyword, stat_id in STAT_KEYWORDS:
        if _first_active_hit(lowered, keyword) >= 0 and stat_id not in stat_ids:
            stat_ids.append(stat_id)

    # ---- 清洗关键词（否定词保护 + 固定优先级）----
    # 优先级：空列 → 重复 → 缺失 → 异常值 → 变换；与文本出现顺序无关，保证计划可复现。
    negated_hints: list[str] = []
    for pattern, name in CLEAN_HINTS:
        if _first_active_regex(lowered, pattern) >= 0:
            if name not in cleaning_request:
                cleaning_request.append(name)
        elif re.search(pattern, lowered):
            negated_hints.append(name)
    if negated_hints:
        notes.append("已识别否定清洗指令并跳过：" + "、".join(negated_hints))

    # ---- 分组列：按 X 分组 / group by X / color by X ----
    group_col = None
    match = re.search(r"(?:按|以|按照|根据|用|分组于|group\s*by|grouped\s*by|by)\s*[「『\"']?([^\s,，。;；、]{1,40})",
                      need, flags=re.IGNORECASE)
    if match:
        token = _strip_quotes(match.group(1))
        token = re.sub(r"(分组|着色|上色|画|作图|颜色)$", "", token)
        for column in {str(c["key"]) for c in dataset.get("columns") or []}:
            if token and (token == column or token.lower() == column.lower()):
                group_col = column
                break
    # 「X 分组」「X 着色」也支持
    if not group_col:
        match = re.search(r"([^\s,，。;；、]{1,40})\s*(?:分组|着色|上色)", need)
        if match:
            token = _strip_quotes(match.group(1))
            for column in {str(c["key"]) for c in dataset.get("columns") or []}:
                if token == column:
                    group_col = column
                    break
    if not group_col:
        for column in mentioned:
            if column in categorical:
                group_col = column
                break
    if group_col:
        roles["group"] = group_col
        if group_col in categorical:
            roles["x"] = group_col
        notes.append(f"分组列：{group_col}")

    # ---- 列角色：X vs Y / 横轴纵轴 / 散点两个变量 ----
    x_col = y_col = None
    match = re.search(r"([\w\u4e00-\u9fa5.%-]+)\s*(?:vs\.?|versus|对|比|随|against|和|与)\s*([\w\u4e00-\u9fa5.%-]+)",
                      need, flags=re.IGNORECASE)
    if match:
        left = _strip_quotes(match.group(1))
        right = _strip_quotes(match.group(2))
        columns = {str(c["key"]) for c in dataset.get("columns") or []}
        if left in columns:
            x_col = left
        if right in columns:
            y_col = right
    if not x_col:
        match = re.search(r"(?:横轴|横坐标|x\s*轴)[^\w\u4e00-\u9fa5]{0,4}([\w\u4e00-\u9fa5.%-]+)", need, flags=re.IGNORECASE)
        if match:
            token = _strip_quotes(match.group(1))
            if token in {str(c["key"]) for c in dataset.get("columns") or []}:
                x_col = token
    if not y_col:
        match = re.search(r"(?:纵轴|纵坐标|y\s*轴)[^\w\u4e00-\u9fa5]{0,4}([\w\u4e00-\u9fa5.%-]+)", need, flags=re.IGNORECASE)
        if match:
            token = _strip_quotes(match.group(1))
            if token in {str(c["key"]) for c in dataset.get("columns") or []}:
                y_col = token
    mentioned_numeric = [c for c in mentioned if c in numeric]
    if not x_col and not y_col and len(mentioned_numeric) >= 2 and chart in ("scatter", "bubble", "line", "area", "bar", None):
        x_col, y_col = mentioned_numeric[0], mentioned_numeric[1]
    if not x_col and roles.get("x") and roles["x"] in numeric and chart in ("scatter", "bubble", "line", "area", "qq", "hist", "density"):
        x_col = roles["x"]
    if not y_col and mentioned_numeric:
        y_col = [c for c in mentioned_numeric if c != x_col][0] if any(c != x_col for c in mentioned_numeric) else mentioned_numeric[0]

    chart_meta = None
    chart_id = chart
    if chart_id:
        chart_meta = next((item for item in _charts().CHARTS if item["id"] == chart_id), None)
        if chart_meta is None:
            notes.append(f"图表 {chart_id} 不存在，改用数据推荐图表")
            chart_id = None

    # ---- 没有明确图表时按数据推荐 ----
    if chart_id is None:
        chart_id = _pick_chart_by_data(dataset, group_col, x_col, y_col, stat_ids)
        chart_meta = next((item for item in _charts().CHARTS if item["id"] == chart_id), None)
        notes.append(f"按数据结构推荐图表：{chart_meta['title'] if chart_meta else chart_id}")

    # ---- 角色补全（用图表推荐 + 需求里提到的列覆盖）----
    rec = P.recommend_roles(dataset, chart_meta or {"id": chart_id})
    for key, value in rec.items():
        if not roles.get(key):
            roles[key] = value
    if x_col:
        roles["x"] = x_col
    if y_col:
        roles["y"] = y_col
    if group_col:
        roles["group"] = group_col
        if chart_id in ("bar", "bar_dot", "box", "violin", "raincloud", "stack", "pie", "dotplot", "hist"):
            if roles.get("x") in (None, group_col) or chart_meta is None:
                roles["x"] = group_col
    if chart_id in ("km", "survival"):
        for column in mentioned:
            if re.search(r"time|survival|os|dfs|follow", column, re.I):
                roles["time"] = column
            if re.search(r"event|status|death", column, re.I):
                roles["event"] = column

    # ---- 统计选择（显式关键词；否则按图表自动）----
    if not stat_ids:
        if chart_id in ("bar", "bar_dot", "box", "violin", "raincloud") and roles.get("group"):
            stat_ids = ["compare"]
        elif chart_id in ("scatter", "bubble"):
            stat_ids = ["corr"]
        elif chart_id in ("line", "area"):
            stat_ids = []
        elif chart_id in ("heatmap", "corr"):
            stat_ids = ["corr"]
        elif chart_id == "km":
            stat_ids = ["survival"]
        elif chart_id == "roc":
            stat_ids = ["roc"]
        elif chart_id == "pca":
            stat_ids = ["pca"]
    stat_ids = [sid for sid in stat_ids if sid not in ("compare",) or roles.get("group") or roles.get("x")]

    # ---- 参数微调 ----
    if chart_id == "bar":
        if "均值" in need or "mean" in lowered:
            params["agg"] = "mean"
        if "中位数" in need or "median" in lowered:
            params["agg"] = "median"
        if "求和" in need or "total" in lowered:
            params["agg"] = "sum"
    if chart_id == "scatter":
        if "回归线" in need or "拟合" in lowered or "trend" in lowered:
            params["add_fit"] = True
            if "regression" not in stat_ids:
                stat_ids.append("regression")
        if "相关" in need:
            params["add_corr"] = True
    if chart_id == "heatmap":
        if "聚类" in need or "cluster" in lowered:
            params["cluster_rows"] = True
            params["cluster_cols"] = True
        if "标准化" in need or "z-score" in lowered:
            params["zscore"] = "row"
    if chart_id == "line":
        if "误差" in need or "置信" in lowered:
            params["error_band"] = True
    if "横向" in need and chart_id in ("bar", "box", "violin", "dotplot"):
        params["horizontal"] = True

    # ---- 风格提示 ----
    for pattern, override in STYLE_HINTS:
        if re.search(pattern, need, flags=re.IGNORECASE):
            style.update(override)
    if "百分比" in need or "proportion" in lowered:
        if chart_id == "stack":
            params["normalize"] = True

    # ---- 清洗计划（固定优先级；保护 ID/分组/端点列）----
    cleaning_steps = []
    for name in ("empty", "duplicate", "missing", "outlier", "log", "zscore"):
        if name in cleaning_request:
            cleaning_steps.extend(_cleaning_from_hints(dataset, [name]))
    if cleaning_request:
        notes.append("清洗优先级：" + " → ".join(name for name in
                                            ("empty", "duplicate", "missing", "outlier", "log", "zscore")
                                            if name in cleaning_request))

    steps = []
    for item in cleaning_steps:
        steps.append({"kind": "clean", "label": item.get("reason") or item["op"]})
    steps.append({"kind": "chart", "label": f"绘制{next((c['title'] for c in _charts().CHARTS if c['id'] == chart_id), chart_id)}"})
    for stat_id in stat_ids:
        steps.append({"kind": "stat", "label": f"统计分析：{stat_id}"})

    return {
        "need": need,
        "chart": chart_id,
        "chart_title": next((c["title"] for c in _charts().CHARTS if c["id"] == chart_id), chart_id),
        "params": params,
        "style": style,
        "roles": {k: v for k, v in roles.items() if v},
        "stats": stat_ids,
        "cleaning": cleaning_steps,
        "steps": steps,
        "rationale": notes or ["按需求执行"],
        "mentioned_columns": mentioned,
    }


def _pick_chart_by_data(dataset: Mapping[str, Any], group_col: str | None,
                        x_col: str | None, y_col: str | None, stat_ids: Sequence[str]) -> str:
    numeric = P.numeric_columns(dataset)
    categorical = P.categorical_columns(dataset)
    kind = dataset.get("kind")
    if "survival" in stat_ids or any(re.search(r"time|survival", c, re.I) for c in numeric):
        if any(re.search(r"event|status|death", c, re.I) for c in numeric + categorical):
            return "km"
    if "roc" in stat_ids:
        return "roc"
    if kind == "matrix":
        return "heatmap"
    if "corr" in stat_ids and len(numeric) >= 2:
        return "scatter"
    if "compare" in stat_ids and group_col:
        return "box"
    if group_col and numeric:
        return "box"
    if len(numeric) >= 2:
        return "scatter"
    if numeric:
        return "hist"
    return "bar"


def _cleaning_from_hints(dataset: Mapping[str, Any], hints: Sequence[str]) -> list[dict]:
    df = dataset.get("df")
    if df is None:
        return []
    columns = dataset.get("columns") or P.column_summaries(df)
    protected = pl_clean.protected_columns(dataset)
    numeric = [c["key"] for c in columns
               if c.get("dtype") == "numeric" and str(c["key"]) not in protected]
    steps: list[dict] = []
    if "empty" in hints:
        empty = [c["key"] for c in columns
                 if int(c.get("missing") or 0) >= len(df) and str(c["key"]) not in protected]
        if empty:
            steps.append({"op": "drop_empty", "params": {"columns": empty}, "reason": "删除全空列"})
        kept = [c["key"] for c in columns
                if int(c.get("missing") or 0) >= len(df) and str(c["key"]) in protected]
        if kept:
            steps.append({"op": "none", "params": {},
                          "reason": f"全空列 {'、'.join(str(c) for c in kept[:5])} 属受保护列，跳过删除"})
    if "duplicate" in hints:
        steps.append({"op": "drop_duplicates", "params": {}, "reason": "删除完全重复的行"})
    if "missing" in hints:
        cols = [c["key"] for c in columns
                if 0 < int(c.get("missing") or 0) < len(df) and str(c["key"]) not in protected]
        target_numeric = [c for c in cols if c in numeric]
        if target_numeric:
            steps.append({"op": "impute", "params": {"columns": target_numeric, "method": "median"},
                          "reason": "数值列按列独立用中位数填补"})
        others = [c for c in cols if c not in numeric]
        if others:
            steps.append({"op": "dropna", "params": {"columns": others, "how": "any"},
                          "reason": "分类列按列独立删除缺失行"})
        kept = [c["key"] for c in columns
                if 0 < int(c.get("missing") or 0) < len(df) and str(c["key"]) in protected]
        endpoints = pl_clean.endpoint_columns(dataset)
        fillable = [c for c in kept if c not in endpoints]
        endpoint_missing = [c for c in kept if c in endpoints]
        if fillable:
            steps.append({"op": "dropna", "params": {"columns": fillable, "how": "any"},
                          "reason": "受保护列（ID/分组）有缺失：不填补，按列删除缺失行"})
        if endpoint_missing:
            steps.append({"op": "dropna", "params": {"columns": endpoint_missing, "how": "any"},
                          "reason": "端点列有缺失：不填补（避免伪造结局），按列删除缺失行"})
    if "outlier" in hints:
        if numeric:
            steps.append({"op": "outlier", "params": {"columns": numeric[:8], "method": "iqr",
                                                      "action": "winsorize"},
                          "reason": "异常值缩尾到 1.5×IQR（仅数值列，保护 ID/分组/端点）"})
        else:
            steps.append({"op": "none", "params": {}, "reason": "没有可处理的数值列（或全部受保护）"})
    if "log" in hints and numeric:
        steps.append({"op": "transform", "params": {"columns": numeric[:8], "method": "log2"},
                      "reason": "对数值列做 log2 变换（保护 ID/分组/端点）"})
    if "zscore" in hints and numeric:
        steps.append({"op": "transform", "params": {"columns": numeric[:8], "method": "zscore"},
                      "reason": "对数值列做 z-score 标准化（保护 ID/分组/端点）"})
    return steps



def _dataset_groups(dataset: Mapping[str, Any]) -> list[str] | None:
    """分组水平：tidy 用 roles.group；matrix 用 sample_info 的 group 列。"""
    groups = dataset.get("groups")
    if groups:
        return [str(g) for g in groups]
    info = dataset.get("sample_info")
    if isinstance(info, pd.DataFrame) and not info.empty:
        column = dataset.get("group_col") or ("group" if "group" in info.columns else None)
        if column and column in info.columns:
            return [str(v) for v in pd.unique(info[column].dropna().astype(str))][:60]
    return None


# ---------------------------------------------------------------------------
# 统计口径（口径卡）数据源：只搬运真实状态字段，取不到给 None，不猜
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _column_kind_counts(dataset: Mapping[str, Any]) -> dict:
    """列类型计数：numeric/categorical/time 来自列摘要 dtype；event 只认显式角色映射。

    event 不是 dtype，没有角色映射时写 None（取不到），不写 0 冒充「没有事件列」。
    """
    counts = {"numeric": 0, "categorical": 0, "time": 0, "other": 0, "event": None}
    for column in dataset.get("columns") or []:
        dtype = str(column.get("dtype") or "")
        if dtype == "numeric":
            counts["numeric"] += 1
        elif dtype == "datetime":
            counts["time"] += 1
        elif dtype in ("category", "text"):
            counts["categorical"] += 1
        else:
            counts["other"] += 1
    # 事件列不是 dtype：只认用户/角色推断显式映射的 event 列（不按列名猜）
    roles = dataset.get("roles") or {}
    if roles.get("event"):
        counts["event"] = 1
    return counts


def _missing_summary(dataset: Mapping[str, Any]) -> dict:
    """缺失率：优先用列摘要里的 missing（O(列数)），列摘要缺失时才扫 DataFrame。"""
    df = dataset.get("df")
    n_rows = int(df.shape[0]) if df is not None else None
    n_cols = int(df.shape[1]) if df is not None else None
    columns = list(dataset.get("columns") or [])
    missing_cells: int | None = None
    columns_with_missing: int | None = None
    if columns:
        missing_cells = int(sum(int(column.get("missing") or 0) for column in columns))
        columns_with_missing = int(sum(1 for column in columns if int(column.get("missing") or 0) > 0))
    elif df is not None:
        missing_cells = int(df.isna().sum().sum())
        columns_with_missing = int(df.isna().any().sum())
    cells = (n_rows * n_cols) if (n_rows is not None and n_cols is not None) else None
    rate = (missing_cells / cells) if (missing_cells is not None and cells) else None
    return {
        "cells": cells,
        "missing": missing_cells,
        "columns_with_missing": columns_with_missing,
        "rate": round(rate, 6) if rate is not None else None,
    }


def _clean_summary(dataset: Mapping[str, Any]) -> dict:
    """清洗记录与行数变化：raw_df 是导入快照，逐条 rows_before/rows_after 是清洗留痕。"""
    cleaning = list(dataset.get("cleaning") or [])
    df = dataset.get("df")
    raw = dataset.get("raw_df")
    rows_current = int(df.shape[0]) if df is not None else None
    rows_import = int(raw.shape[0]) if raw is not None else None
    cols_current = int(df.shape[1]) if df is not None else None
    cols_import = int(raw.shape[1]) if raw is not None else None
    deltas: list[int] = []
    complete = True
    for step in cleaning:
        before, after = step.get("rows_before"), step.get("rows_after")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            deltas.append(int(after) - int(before))
        else:
            complete = False
    if not cleaning:
        logged_delta: int | None = 0
    else:
        logged_delta = sum(deltas) if complete else None
    changed_rows: bool | None = None
    trace_complete: bool | None = None
    if rows_import is not None and rows_current is not None:
        changed_rows = rows_import != rows_current
        if logged_delta is not None:
            trace_complete = logged_delta == (rows_current - rows_import)
    return {
        "steps": len(cleaning),
        "ops": [str(step.get("op") or "") for step in cleaning],
        "rows_import": rows_import,
        "rows_current": rows_current,
        "rows_delta": (rows_current - rows_import) if (rows_import is not None and rows_current is not None) else None,
        "columns_import": cols_import,
        "columns_current": cols_current,
        "logged_rows_delta": logged_delta,
        "changed_rows": changed_rows,
        "trace_complete": trace_complete,
    }


def _group_counts(dataset: Mapping[str, Any]) -> dict:
    """每个分组水平的行/样本数（判断「每组 n<2」的依据）；无分组返回空 dict。"""
    group_col = str(dataset.get("group_col") or (dataset.get("roles") or {}).get("group") or "")
    df = dataset.get("df")
    series = None
    if df is not None and group_col and group_col in df.columns:
        series = df[group_col]
    else:
        info = dataset.get("sample_info")
        if isinstance(info, pd.DataFrame) and not info.empty:
            column = group_col if group_col in info.columns else ("group" if "group" in info.columns else "")
            if column:
                series = info[column]
    if series is None:
        return {}
    counts: dict[str, int] = {}
    for value, count in series.astype(str).value_counts().items():
        if value in ("nan", "None", ""):
            continue
        counts[str(value)] = int(count)
    return counts


def _style_overrides(style: Mapping[str, Any] | None) -> list[str]:
    """图上样式相对默认值覆盖了哪些键（run.style 已由 resolve_style 补全，直接对比）。"""
    merged = dict(style or {})
    overrides: list[str] = []
    for key, default in C.STYLE_DEFAULT.items():
        if merged.get(key) != default:
            overrides.append(str(key))
    return sorted(overrides)


def _inference_kind(stats: Mapping[str, Any] | None) -> str | None:
    """从 figure run 的 stats 载荷判断推断类型（只按真实存在的键分类）。"""
    if not isinstance(stats, Mapping) or not stats:
        return None
    if stats.get("pairs") or stats.get("contrasts") or stats.get("global"):
        return "group_comparison"
    if stats.get("correlation") is not None:
        return "correlation"
    if "logrank_p" in stats:
        return "survival"
    if stats.get("fit") is not None:
        return "regression"
    if stats.get("describe") or stats.get("dist"):
        return "descriptive"
    return None


_STAT_CONTRACT_PARAM_KEYS = (
    "method", "adjust", "alpha", "paired", "pairing", "independent_samples",
    "label", "score", "positive", "time", "event", "group", "covariates",
    "columns", "x", "y",
)


def _stat_params_for_contract(params: Mapping[str, Any] | None) -> dict:
    """最近一次统计真正用到的参数子集（校正方法/alpha/配对/独立样本声明等）。"""
    out: dict[str, Any] = {}
    for key in _STAT_CONTRACT_PARAM_KEYS:
        value = (params or {}).get(key)
        if value is None or value == "" or value == [] or value == {}:
            continue
        if isinstance(value, (list, tuple)):
            out[key] = C.jsonable(list(value)[:10])
        elif isinstance(value, (str, int, float, bool)):
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# 出图 run：figure_run_id 固化的图版本（源数据 digest + 参数 + 样式 + 角色 + 图层 + 统计表）
# ---------------------------------------------------------------------------

def data_digest(dataset: Mapping[str, Any]) -> str:
    """当前分析数据的稳定指纹：数据内容 + 清洗记录（清洗后必须视为新源数据）。"""
    digest = hashlib.sha1()
    df = dataset.get("df")
    digest.update(str(dataset.get("kind") or "").encode("utf-8"))
    if df is not None:
        digest.update("|".join(str(c) for c in df.columns).encode("utf-8"))
        digest.update(str(df.shape).encode("utf-8"))
        try:
            hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
            digest.update(hashes.tobytes())
        except Exception:  # noqa: BLE001 - 混合类型兜底
            digest.update(df.astype(str).to_csv(index=False).encode("utf-8"))
    cleaning = [
        {"op": step.get("op"), "params": step.get("params"), "reason": step.get("reason")}
        for step in (dataset.get("cleaning") or [])
    ]
    digest.update(json.dumps(cleaning, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return digest.hexdigest()[:16]


def _canonical(value: Any) -> str:
    return json.dumps(C.jsonable(value), sort_keys=True, ensure_ascii=False, default=str)


def _figure_run_key(dataset: Mapping[str, Any], chart_id: str, params: Mapping[str, Any],
                    style: Mapping[str, Any], roles: Mapping[str, Any]) -> dict:
    return {
        "dataset_id": dataset.get("id"),
        "data_digest": data_digest(dataset),
        "kind": dataset.get("kind"),
        "feature_col": dataset.get("feature_col"),
        "chart": chart_id,
        "params": C.jsonable(dict(params or {})),
        "style": C.jsonable(dict(style or {})),
        "roles": {str(k): v for k, v in (roles or {}).items() if v},
    }


def _figure_run_id(key: Mapping[str, Any]) -> str:
    return "fr_" + hashlib.sha1(_canonical(key).encode("utf-8")).hexdigest()[:16]


def _figure_layers(data: Mapping[str, Any], chart_id: str, params: Mapping[str, Any],
                   style: Mapping[str, Any], stats: Any, annotations: Any) -> list[dict]:
    """描述这张图实际由哪些层组成（数据层/图元层/推断层/标注层/样式层）。"""
    tidy = data.get("tidy")
    layers: list[dict] = [{
        "id": "data",
        "kind": data.get("kind"),
        "view": data.get("view") or data.get("kind"),
        "melted": bool(data.get("melted")),
        "n_rows": int(len(tidy)) if tidy is not None else None,
        "n_dropped": data.get("n_dropped"),
    }]
    layers.append({"id": "mark", "chart": chart_id, "params": C.jsonable(dict(params or {}))})
    if stats:
        layers.append({
            "id": "inference",
            "method": (stats or {}).get("method"),
            "p_adjust": (stats or {}).get("p_adjust"),
            "factor": (stats or {}).get("factor"),
            "contrasts": len((stats or {}).get("contrasts") or (stats or {}).get("pairs") or []),
        })
    else:
        layers.append({"id": "inference", "method": None, "note": "no test drawn"})
    if annotations:
        layers.append({"id": "annotation", "payload": C.jsonable(annotations)})
    layers.append({
        "id": "style",
        "palette": style.get("palette"), "theme": style.get("theme"),
        "dpi": style.get("dpi"),
        "size": [style.get("fig_width"), style.get("fig_height")],
    })
    return layers


def _figure_intervals(tables: Sequence[Mapping[str, Any]]) -> list[dict]:
    """把图上实际使用的误差/置信区间记录进 run（有则记录，无则空表）。"""
    intervals: list[dict] = []
    for table in tables or []:
        if str(table.get("id")) != "aggregate":
            continue
        keys = {str(c.get("key")) for c in table.get("columns") or []}
        if "error" not in keys:
            continue
        for row in table.get("rows") or []:
            intervals.append({
                "x": row.get("x"), "group": row.get("group"),
                "n": row.get("n"), "value": row.get("value"), "error": row.get("error"),
            })
    return C.jsonable(intervals)


# ---------------------------------------------------------------------------
# WorkerState
# ---------------------------------------------------------------------------


class WorkerState:
    def __init__(self, workdir: str) -> None:
        self.workdir = os.path.abspath(workdir)
        self.dirs = {
            "root": self.workdir,
            "figures": os.path.join(self.workdir, "figures"),
            "uploads": os.path.join(self.workdir, "uploads"),
            "exports": os.path.join(self.workdir, "exports"),
            "mpl": os.path.join(self.workdir, "mpl-cache"),
        }
        for key in ("figures", "uploads", "exports", "mpl"):
            os.makedirs(self.dirs[key], exist_ok=True)
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", self.dirs["mpl"])
        self.datasets: "OrderedDict[str, dict]" = OrderedDict()
        self.active: str | None = None
        self.seq = 0
        self.figures: OrderedDict[str, dict] = OrderedDict()
        self.figure_runs: OrderedDict[str, dict] = OrderedDict()
        self.started_at = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        self.last_plot: dict | None = None
        self.progress_cb = None

    # ---- 基础 ----

    def _next_dataset_id(self) -> str:
        self.seq += 1
        return f"ds{self.seq}"

    def store_dataset(self, dataset: dict, name: str | None = None) -> dict:
        if len(self.datasets) >= MAX_DATASETS:
            oldest = next(iter(self.datasets))
            if oldest != self.active:
                self.datasets.pop(oldest, None)
        dataset["id"] = self._next_dataset_id()
        if name:
            dataset["name"] = str(name)
        self.datasets[dataset["id"]] = dataset
        self.active = dataset["id"]
        return dataset

    def get(self, dataset_id: str | None = None) -> dict:
        key = str(dataset_id or self.active or "")
        if not key or key not in self.datasets:
            raise C.prerequisite("还没有打开数据集：请先拖入数据或载入示例（action=open/demo）")
        return self.datasets[key]

    def _last_figure_summary(self, dataset: Mapping[str, Any]) -> dict | None:
        """最近一次真正用过的 figure run（按数据集 history 的最后一条回查 registry）。

        口径卡要回答「这张图是怎么来的」，所以优先 history 的最后一条——它记录的是
        「最近一次真正渲染/复用」而不是「最近一次新建」。记录被清理时给
        artifact_available=False，让 UI/Agent 知道只剩历史条目，不假装完整。
        """
        history = list(dataset.get("history") or [])
        entry: Mapping[str, Any] | None = None
        run: Mapping[str, Any] | None = None
        for candidate in reversed(history):
            run_id = str(candidate.get("figure_run_id") or "")
            if not run_id:
                continue
            entry = candidate
            run = self.figure_runs.get(run_id)
            break
        if run is None and entry is None:
            for candidate in reversed(list(self.figure_runs.values())):
                if candidate.get("dataset_id") == dataset.get("id"):
                    run = candidate
                    break
        if run is None and entry is None:
            return None
        if run is None:
            # 只剩 history 条目（run 已被 MAX_FIGURE_RUNS 挤掉/进程重启）：字段能写多少写多少
            record = dict(entry or {})
            return {
                "figure_run_id": str(record.get("figure_run_id") or "") or None,
                "chart": record.get("chart"),
                "chart_title": record.get("chart_title"),
                "at": record.get("at"),
                "data_digest": record.get("data_digest"),
                "roles": {},
                "params": {},
                "style_overrides": [],
                "inference": None,
                "n_figures": None,
                "artifact_available": False,
            }
        stats = run.get("stats") if isinstance(run.get("stats"), Mapping) else {}
        layers = [layer for layer in (run.get("layers") or []) if isinstance(layer, Mapping)]
        inference_layer = next((layer for layer in layers if layer.get("id") == "inference"), {})
        method = inference_layer.get("method") if inference_layer else None
        p_adjust = inference_layer.get("p_adjust") if inference_layer else None
        factor = inference_layer.get("factor") if inference_layer else None
        n_contrasts = inference_layer.get("contrasts") if inference_layer else None
        pairs = stats.get("pairs") or stats.get("contrasts") or []
        if not isinstance(n_contrasts, int):
            n_contrasts = len(pairs) if pairs else None
        n_per_group = stats.get("n") if isinstance(stats.get("n"), Mapping) else None
        if n_per_group:
            n_per_group = {
                str(name): int(count) for name, count in n_per_group.items()
                if isinstance(count, (int, float))
            }
        has_inference = bool(stats) or bool(method) or bool(p_adjust)
        inference = None
        if has_inference:
            inference = {
                "method": method or stats.get("method"),
                "p_adjust": p_adjust or stats.get("p_adjust"),
                # 图上星号阈值是固定的 0.001/0.01/0.05，引擎不记录可调 α → 不编默认值
                "alpha": None,
                "factor": factor or stats.get("factor"),
                "n_contrasts": n_contrasts,
                "n_per_group": n_per_group or None,
                "kind": _inference_kind(stats),
            }
        return {
            "figure_run_id": str(run.get("figure_run_id") or "") or None,
            "chart": run.get("chart"),
            "chart_title": run.get("chart_title"),
            "at": run.get("at"),
            "data_digest": run.get("data_digest"),
            "roles": {k: v for k, v in (run.get("roles") or {}).items() if v},
            "params": C.jsonable(dict(run.get("params") or {})),
            "style_overrides": _style_overrides(run.get("style") or {}),
            "inference": inference,
            "n_figures": len(run.get("figures") or []),
            "artifact_available": True,
        }

    def dataset_summary(self, dataset: Mapping[str, Any], with_columns: bool = True,
                        with_digest: bool = False) -> dict:
        df = dataset.get("df")
        n_rows, n_cols = (int(df.shape[0]), int(df.shape[1])) if df is not None else (0, 0)
        roles = {k: v for k, v in (dataset.get("roles") or {}).items() if v}
        payload = {
            "id": dataset.get("id"),
            "name": dataset.get("name"),
            "kind": dataset.get("kind"),
            "source": dataset.get("source"),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "roles": roles,
            "group_col": dataset.get("group_col"),
            "groups": _dataset_groups(dataset),
            "cleaning": [
                {"op": item.get("op"), "reason": item.get("reason"),
                 "rows_before": item.get("rows_before"), "rows_after": item.get("rows_after")}
                for item in dataset.get("cleaning") or []
            ],
            # 统计口径卡的数据源：列类型/缺失/清洗行数/每组 n/最近图·统计·导出
            "column_kinds": _column_kind_counts(dataset),
            "missing": _missing_summary(dataset),
            "clean_summary": _clean_summary(dataset),
            "group_counts": _group_counts(dataset),
            "time_col": roles.get("time"),
            "event_col": roles.get("event"),
            "last_figure": self._last_figure_summary(dataset),
            "last_stat": C.jsonable(dataset.get("last_stat") or None),
            "last_export": C.jsonable(dataset.get("last_export") or None),
            "need": dataset.get("need") or "",
            "history_count": len(dataset.get("history") or []),
            "notes": dataset.get("notes") or [],
            "warnings": dataset.get("warnings") or [],
            "feature_col": dataset.get("feature_col"),
            "created_at": dataset.get("created_at"),
            "has_sample_info": dataset.get("sample_info") is not None,
            "columns_count": len(dataset.get("columns") or []),
        }
        if with_columns:
            payload["columns"] = C.jsonable(dataset.get("columns") or [])
            payload["numeric_cols"] = P.numeric_columns(dataset)
            payload["categorical_cols"] = P.categorical_columns(dataset)
        if with_digest:
            # 当前数据指纹：口径卡用它判断「最近一次图是不是在当前数据版本上画的」。
            # 只在 status/records 里算（全量哈希有成本），其他操作回传的 summary 不带。
            payload["data_digest"] = data_digest(dataset)
        return payload

    def _register_figures(self, figures: Sequence[Mapping[str, Any]] | None) -> list[dict]:
        out = []
        for item in figures or []:
            entry = dict(item)
            fid = str(entry.get("id") or C.unique_id("f_"))
            entry["id"] = fid
            entry.setdefault("url", f"/plotlab/figure?id={fid}")
            self.figures[fid] = entry
            out.append(entry)
        while len(self.figures) > MAX_FIGURES:
            self.figures.popitem(last=False)
        return out

    def _chart_meta(self, chart_id: str) -> dict:
        for item in _charts().CHARTS:
            if item["id"] == chart_id:
                return item
        raise C.bad_request(f"未知图表 {chart_id}，可用：{', '.join(c['id'] for c in _charts().CHARTS)}")

    def _stat_meta(self, stat_id: str) -> dict:
        for item in _stats().STATS_CATALOG:
            if item["id"] == stat_id:
                return item
        raise C.bad_request(f"未知统计 {stat_id}，可用：{', '.join(s['id'] for s in _stats().STATS_CATALOG)}")

    def _style(self, raw: Mapping[str, Any] | None) -> dict:
        return C.resolve_style(raw or {})

    def _build_data(self, dataset: Mapping[str, Any], meta: Mapping[str, Any],
                    params: Mapping[str, Any], roles: Mapping[str, Any] | None) -> dict:
        return P.build_data(dataset, meta, dict(params or {}), dict(roles or {}))

    # ---- 目录 ----

    def catalog(self, dataset: Mapping[str, Any] | None = None) -> dict:
        charts = _charts()
        stats = _stats()
        items = []
        for chart in charts.CHARTS:
            entry = {k: v for k, v in chart.items() if k not in ("defaults",)}
            if dataset is not None:
                try:
                    entry["recommended_roles"] = P.recommend_roles(dataset, chart)
                except Exception:  # noqa: BLE001
                    entry["recommended_roles"] = {}
            items.append(entry)
        return {
            "charts": items,
            "stats": list(stats.STATS_CATALOG),
            "styles": {
                "groups": C.STYLE_GROUPS,
                "defaults": C.STYLE_DEFAULT,
                "fonts": C.FONTS,
                "palettes": [{"id": key, "label": key} for key in list(C.PALETTES) + list(C.PALETTE_CMAPS)],
                "themes": C.THEMES,
            },
            "roles": [
                {"key": "x", "label": "X 轴 / 分类", "hint": "柱状图的组、散点图的横轴"},
                {"key": "y", "label": "Y 轴 / 数值", "hint": "要展示的数值"},
                {"key": "group", "label": "分组", "hint": "颜色/比较分组"},
                {"key": "color", "label": "颜色", "hint": "第二分组或连续着色"},
                {"key": "size", "label": "大小", "hint": "气泡大小"},
                {"key": "facet", "label": "分面", "hint": "按列拆子图"},
                {"key": "time", "label": "时间", "hint": "生存时间"},
                {"key": "event", "label": "事件", "hint": "生存事件 0/1"},
            ],
        }

    # ---- 命令分发 ----

    def handle(self, cmd: str, args: Mapping[str, Any] | None = None) -> dict:
        args = dict(args or {})
        handler = getattr(self, f"op_{cmd}", None)
        if handler is None:
            raise C.bad_request(f"未知命令 {cmd}")
        return handler(args)

    # 状态
    def op_ping(self, args: Mapping[str, Any]) -> dict:
        import platform
        import sys

        packages = {}
        for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn", "statsmodels", "lifelines", "openpyxl"):
            try:
                module = __import__(name)
                packages[name] = getattr(module, "__version__", "?")
            except Exception:  # noqa: BLE001
                packages[name] = None
        charts_catalog = []
        try:
            charts_catalog = [{"id": c["id"], "title": c["title"], "group": c["group"]} for c in _charts().CHARTS]
        except Exception:  # noqa: BLE001
            charts_catalog = []
        return {
            "python": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
            "workdir": self.workdir,
            "mode": "builtin",
            "packages": packages,
            "fonts": [f["family"] for f in C.FONTS],
            "charts": charts_catalog,
            "stats_count": len(_stats().STATS_CATALOG) if _has_stats() else 0,
            "formats": _io().formats() if _has_io() else [],
            "capabilities": {"charts": len(charts_catalog), "formats": 8, "stats": True},
            "started_at": self.started_at,
        }

    def op_state(self, args: Mapping[str, Any]) -> dict:
        return {
            # status 是口径卡的主入口：带 data_digest 才能判断「图是否在当前数据版本上」
            "datasets": [self.dataset_summary(ds, with_columns=False, with_digest=True)
                         for ds in self.datasets.values()],
            "active": self.active,
            "engine": {"python": _python_exe(), "mode": "builtin", "workdir": self.workdir},
            "started_at": self.started_at,
        }

    def op_list(self, args: Mapping[str, Any]) -> dict:
        return self.op_state(args)

    def op_select(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset") or args.get("id"))
        self.active = dataset["id"]
        return {"dataset": self.dataset_summary(dataset), "preview": self._preview_payload(dataset)}

    def op_close(self, args: Mapping[str, Any]) -> dict:
        key = str(args.get("dataset") or args.get("id") or "")
        if key and key in self.datasets:
            self.datasets.pop(key, None)
            if self.active == key:
                self.active = next(iter(self.datasets), None)
        return self.op_state(args)

    def op_reset(self, args: Mapping[str, Any]) -> dict:
        self.datasets.clear()
        self.active = None
        self.figures.clear()
        self.figure_runs.clear()
        self.last_plot = None
        return {"ok": True, "datasets": [], "active": None}

    def op_shutdown(self, args: Mapping[str, Any]) -> dict:
        return {"bye": True}

    # 数据
    def op_preview(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        payload = self._preview_payload(dataset)
        payload["dataset"] = self.dataset_summary(dataset)
        return payload

    def _preview_payload(self, dataset: Mapping[str, Any], limit: int = 30) -> dict:
        return {
            "rows": P.sample_rows(dataset, limit=limit),
            "columns": C.jsonable(dataset.get("columns") or []),
            "n_rows": int(dataset["df"].shape[0]) if dataset.get("df") is not None else 0,
            "n_cols": int(dataset["df"].shape[1]) if dataset.get("df") is not None else 0,
            "kind": dataset.get("kind"),
            "roles": {k: v for k, v in (dataset.get("roles") or {}).items() if v},
            "group_col": dataset.get("group_col"),
            "groups": dataset.get("groups"),
            "cleaning": C.jsonable(dataset.get("cleaning") or []),
            "need": dataset.get("need") or "",
            "history": self._history_view(dataset),
            "sample_info": self._sample_info_rows(dataset),
        }

    def _history_view(self, dataset: Mapping[str, Any]) -> list[dict]:
        items = []
        for entry in (dataset.get("history") or [])[-MAX_HISTORY:]:
            items.append(
                {
                    "id": entry.get("id"),
                    "chart": entry.get("chart"),
                    "chart_title": entry.get("chart_title"),
                    "figure": entry.get("figure"),
                    "figure_run_id": entry.get("figure_run_id"),
                    "data_digest": entry.get("data_digest"),
                    "at": entry.get("at"),
                    "need": entry.get("need"),
                }
            )
        return items

    def _sample_info_rows(self, dataset: Mapping[str, Any]) -> list[dict]:
        info = dataset.get("sample_info")
        if not isinstance(info, pd.DataFrame) or info.empty:
            return []
        head = info.head(60)
        rows = []
        for _, row in head.iterrows():
            item = {}
            for key in head.columns:
                value = row[key]
                if isinstance(value, (np.generic,)):
                    value = value.item()
                item[str(key)] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
            rows.append(item)
        return C.jsonable(rows)

    def op_inspect(self, args: Mapping[str, Any]) -> dict:
        kwargs = {k: v for k, v in args.items() if k in (
            "path", "url", "name", "data_format", "delimiter", "encoding", "sheet", "header", "max_rows",
            "kind", "member", "transpose")}
        return _io().inspect_source(**kwargs)

    def op_open(self, args: Mapping[str, Any]) -> dict:
        kwargs = {k: v for k, v in args.items() if k in (
            "path", "url", "name", "data_format", "delimiter", "encoding", "sheet", "header",
            "index_col", "kind", "feature_col", "transpose", "sample_sheet", "max_rows", "member")}
        if not kwargs.get("path") and not kwargs.get("url"):
            raise C.bad_request("open 需要 path 或 url")
        kwargs["workdir"] = self.dirs["uploads"]
        payload = _io().open_source(**kwargs)
        dataset = payload.get("dataset")
        if not dataset:
            raise C.EngineError("internal", "导入返回为空")
        for note in payload.get("notes") or []:
            notes = dataset.setdefault("notes", [])
            if str(note) not in notes:
                notes.append(str(note))
        for warning in payload.get("warnings") or []:
            warnings = dataset.setdefault("warnings", [])
            if str(warning) not in warnings:
                warnings.append(str(warning))
        self.store_dataset(dataset)
        return {
            "dataset": self.dataset_summary(dataset),
            "preview": self._preview_payload(dataset),
            "format": payload.get("format"),
            "candidates": payload.get("candidates") or [],
            "notes": payload.get("notes") or [],
            "warnings": payload.get("warnings") or [],
        }

    def op_upload(self, args: Mapping[str, Any]) -> dict:
        return self.op_open(args)

    def op_demo(self, args: Mapping[str, Any]) -> dict:
        dataset = S.create_demo(str(args.get("name") or "clinical"))
        self.store_dataset(dataset)
        return {"dataset": self.dataset_summary(dataset), "preview": self._preview_payload(dataset),
                "notes": dataset.get("notes") or []}

    def op_samples(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        operation = str(args.get("op") or "list")
        if operation == "list":
            return {"sample_info": self._sample_info_rows(dataset),
                    "group_col": dataset.get("group_col"), "groups": dataset.get("groups")}
        if operation in ("join", "import"):
            path = args.get("path")
            if not path:
                raise C.bad_request("samples op=join 需要 path（样本表 CSV/TSV/XLSX）")
            info = _read_sample_table(path)
            dataset["sample_info"] = info
            dataset.setdefault("notes", []).append(f"已挂载样本表 {os.path.basename(str(path))}（{len(info)} 行）")
            return {"sample_info": self._sample_info_rows(dataset),
                    "group_col": dataset.get("group_col"), "groups": dataset.get("groups")}
        if operation == "set_group":
            column = str(args.get("column") or "").strip()
            info = dataset.get("sample_info")
            if column and isinstance(info, pd.DataFrame) and column in info.columns:
                dataset["group_col"] = column
                dataset["groups"] = [str(v) for v in pd.unique(info[column].dropna().astype(str))][:60]
                return {"sample_info": self._sample_info_rows(dataset),
                        "group_col": column, "groups": dataset["groups"]}
            # tidy 数据：直接设 roles.group
            if column in (dataset.get("df").columns if dataset.get("df") is not None else []):
                dataset["roles"]["group"] = column
                S.refresh_columns(dataset)
                return {"sample_info": self._sample_info_rows(dataset),
                        "group_col": dataset.get("group_col"), "groups": dataset.get("groups")}
            raise C.bad_request(f"样本表里没有列 {column}")
        if operation == "update":
            info = dataset.get("sample_info")
            if not isinstance(info, pd.DataFrame):
                raise C.bad_request("当前数据集没有样本表")
            samples = args.get("samples") or []
            if not isinstance(samples, list):
                raise C.bad_request("samples 需要列表")
            for item in samples:
                if not isinstance(item, dict):
                    continue
                sample = item.get("sample") or item.get("id")
                if sample is None or "sample" not in info.columns:
                    continue
                mask = info["sample"].astype(str) == str(sample)
                if item.get("group"):
                    info.loc[mask, "group"] = str(item["group"])
            dataset["sample_info"] = info
            return {"sample_info": self._sample_info_rows(dataset),
                    "group_col": dataset.get("group_col"), "groups": dataset.get("groups")}
        raise C.bad_request(f"未知 samples op {operation}")

    # 剖析 / 清洗
    def op_profile(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        profile = pl_clean.profile(dataset)
        profile["dataset"] = self.dataset_summary(dataset, with_columns=False)
        return profile

    def op_clean(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        plan = args.get("plan")
        if plan is not None and not isinstance(plan, list):
            raise C.bad_request("plan 需要是步骤列表")
        auto = bool(args.get("auto")) or not plan
        dry_run = bool(args.get("preview") or args.get("dry_run"))
        report = pl_clean.apply_clean(dataset, plan=plan, auto=auto, dry_run=dry_run)
        report["dataset"] = self.dataset_summary(dataset, with_columns=False)
        return report

    def op_reset_data(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        result = pl_clean.reset_data(dataset)
        result["dataset"] = self.dataset_summary(dataset, with_columns=False)
        return result

    # 图表 / 统计
    def op_roles(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        roles = args.get("roles")
        if isinstance(roles, dict):
            existing = {str(c["key"]) for c in dataset.get("columns") or []}
            cleaned = {}
            for key in P.ROLE_KEYS:
                value = roles.get(key)
                cleaned[key] = value if value in existing else None
            dataset["roles"] = cleaned
            S.refresh_columns(dataset)
        recommend = {}
        for chart in _charts().CHARTS:
            rec = P.recommend_roles(dataset, chart)
            recommend[chart["id"]] = {k: v for k, v in rec.items() if v}
        return {"roles": {k: v for k, v in (dataset.get("roles") or {}).items() if v},
                "recommend": recommend}

    def op_charts(self, args: Mapping[str, Any]) -> dict:
        dataset = None
        try:
            dataset = self.get(args.get("dataset"))
        except C.EngineError:
            dataset = None
        payload = self.catalog(dataset)
        if dataset is not None:
            payload["dataset"] = self.dataset_summary(dataset, with_columns=False)
            payload["roles_current"] = {k: v for k, v in (dataset.get("roles") or {}).items() if v}
        return payload

    # ---- 出图 run：渲染 + 固化 ----

    def _figure_run(self, dataset: Mapping[str, Any], chart_id: str, meta: Mapping[str, Any],
                    params: Mapping[str, Any], style: Mapping[str, Any],
                    roles: Mapping[str, Any]) -> dict:
        """按 (源数据 digest, chart, params, style, roles) 取或建一个 figure run。

        同一请求复用已固化的 run（含 PNG 路径）；任一要素变化都会新建 run 并重画。
        """
        key = _figure_run_key(dataset, chart_id, params, style, roles)
        run_id = _figure_run_id(key)
        cached = self.figure_runs.get(run_id)
        if cached is not None:
            files = [str(item.get("file")) for item in cached.get("figures") or []]
            if files and all(os.path.isfile(path) for path in files):
                return cached
        started = time.perf_counter()
        data = self._build_data(dataset, meta, params, roles)
        result = _charts().render(chart_id, data, params, style, self.dirs["figures"])
        figures = self._register_figures(result.get("figures"))
        if not figures:
            raise C.EngineError("internal", "图表渲染没有产出图片")
        python_code = ""
        r_code = ""
        try:
            python_code = _codegen().build_script(dataset, chart_id, meta, data, params, style, roles)
            r_code = _codegen().chart_r_code(chart_id, data, params, style)
        except Exception as error:  # noqa: BLE001 - 代码生成失败不影响出图
            result.setdefault("warnings", []).append(f"代码生成失败：{error}")
        warnings = list(dict.fromkeys(
            [str(item) for item in (result.get("warnings") or [])]
            + [str(item) for item in (data.get("notes") or [])]
        ))
        tables = C.jsonable(result.get("tables") or [])
        stats = C.jsonable(result.get("stats"))
        annotations = C.jsonable(result.get("annotations"))
        tidy = data.get("tidy")
        run = {
            "figure_run_id": run_id,
            "version": FIGURE_RUN_VERSION,
            "key": C.jsonable(key),
            "dataset_id": dataset.get("id"),
            "dataset_name": dataset.get("name"),
            "data_digest": key["data_digest"],
            "kind": dataset.get("kind"),
            "chart": chart_id,
            "chart_title": meta.get("title") or chart_id,
            "params": C.jsonable(dict(params or {})),
            "style": C.jsonable(dict(style or {})),
            "roles": {k: v for k, v in (roles or {}).items() if v},
            "labels": C.jsonable(data.get("labels") or {}),
            "figures": figures,
            "figure": figures[0],
            "summary": C.jsonable(result.get("summary") or {}),
            "tables": tables,
            "layers": _figure_layers(data, chart_id, params, style, stats, annotations),
            "confidence_intervals": _figure_intervals(tables),
            "stats": stats,
            "annotations": annotations,
            "warnings": warnings,
            "code": {"python": python_code, "r": r_code},
            "data_meta": {"kind": data.get("kind"), "view": data.get("view"),
                          "melted": bool(data.get("melted")),
                          "n_dropped": data.get("n_dropped"),
                          "n_rows": int(len(tidy)) if tidy is not None else None},
            "at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        self.figure_runs[run_id] = run
        while len(self.figure_runs) > MAX_FIGURE_RUNS:
            self.figure_runs.popitem(last=False)
        return run

    def _figure_run_view(self, run: Mapping[str, Any]) -> dict:
        keys = ("figure_run_id", "version", "chart", "chart_title", "params", "style", "roles",
                "tables", "layers", "confidence_intervals", "stats", "annotations", "summary",
                "dataset_id", "data_digest", "at", "data_meta", "figure", "figures")
        return C.jsonable({key: run.get(key) for key in keys if key in run})

    def get_figure_run(self, run_id: str) -> dict:
        run = self.figure_runs.get(str(run_id or ""))
        if run is None:
            raise C.not_found(f"没有这个 figure run：{run_id}（可用 op_records 查看历史 run）")
        return run

    def op_plot(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        chart_id = str(args.get("chart") or args.get("kind") or "").strip()
        if not chart_id:
            raise C.bad_request("plot 需要 chart（图表 id）")
        meta = self._chart_meta(chart_id)
        params = dict(args.get("params") or {})
        style = self._style(args.get("style"))
        roles = args.get("roles") if isinstance(args.get("roles"), dict) else None
        if roles is None:
            roles = P.recommend_roles(dataset, meta)
        roles = {k: v for k, v in (roles or {}).items() if v}
        started = time.perf_counter()
        run = self._figure_run(dataset, chart_id, meta, params, style, roles)
        elapsed = int((time.perf_counter() - started) * 1000)
        payload = {
            "figure": run.get("figure"),
            "figures": run.get("figures") or [],
            "figure_run_id": run.get("figure_run_id"),
            # 导出绑定基线要能校验「导出的图」和「最近一次渲染」是否同一份源数据
            "data_digest": run.get("data_digest"),
            "figure_run": self._figure_run_view(run),
            "chart": chart_id,
            "chart_title": run.get("chart_title") or meta.get("title") or chart_id,
            "params": run.get("params"),
            "style": run.get("style"),
            "roles": run.get("roles"),
            "labels": run.get("labels") or {},
            "summary": run.get("summary") or {},
            "tables": run.get("tables") or [],
            "warnings": run.get("warnings") or [],
            "stats": run.get("stats"),
            "annotations": run.get("annotations"),
            "code": run.get("code") or {"python": "", "r": ""},
            "data_meta": run.get("data_meta"),
            "elapsed_ms": elapsed,
            "dataset": self.dataset_summary(dataset, with_columns=False),
        }
        # 历史 & 最近一次（导出/取回都按 figure_run_id 绑定，不再靠"最后一次"猜）
        dataset.setdefault("history", []).append(
            {
                "id": C.unique_id("h_"),
                "chart": chart_id,
                "chart_title": payload["chart_title"],
                "params": payload["params"],
                "style": payload["style"],
                "roles": payload["roles"],
                "figure": run.get("figure"),
                "figure_run_id": run.get("figure_run_id"),
                "data_digest": run.get("data_digest"),
                "at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "need": dataset.get("need") or "",
            }
        )
        dataset["history"] = dataset["history"][-MAX_HISTORY:]
        self.last_plot = payload
        return payload

    def op_stat(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        stat_id = str(args.get("stat") or "").strip()
        if not stat_id:
            raise C.bad_request("stat 需要 stat（统计 id）")
        meta = self._stat_meta(stat_id)
        params = dict(args.get("params") or {})
        style = self._style(args.get("style"))
        roles = dict(args.get("roles") or {}) if isinstance(args.get("roles"), dict) else {}
        stat_meta = {"id": stat_id, "needs": str(meta.get("needs") or "tidy"),
                     "roles": meta.get("roles") or {}}
        # 组间比较/前提检查：用户给的是图表角色（group）时，自动映射到统计的 x（分组）
        if stat_id in ("compare", "assumption") and not roles.get("x") and roles.get("group"):
            roles["x"] = roles["group"]
        # 缺失的必需角色用统计语义推荐补全（避免图表角色带偏）
        recommended = P.recommend_roles(dataset, stat_meta)
        for key, value in recommended.items():
            if not roles.get(key) and value:
                roles[key] = value
        data = P.build_data(dataset, stat_meta, params, roles)
        started = time.perf_counter()
        stat_record = {
            "stat": stat_id,
            "title": stat_id,
            "method": "",
            "params": C.jsonable(_stat_params_for_contract(params)),
            "alpha": params.get("alpha") if isinstance(params.get("alpha"), (int, float)) else None,
            # 组间比较/多重校正的默认校正方法来自 style.p_adjust（引擎真实生效值）；
            # 其他统计没有多重校正，不记录，避免把不适用的方法写进口径。
            "p_adjust": (params.get("adjust") if isinstance(params.get("adjust"), str) else None)
            or (style.get("p_adjust") if stat_id in ("compare", "adjust") else None),
            "at": now_iso(),
        }
        try:
            result = _stats().run(stat_id, dataset, data, params, style, self.dirs["figures"])
        except C.EngineError as error:
            # 失败也要留痕：口径卡据此提示「最近一次统计失败，结论不可用」
            dataset["last_stat"] = {**stat_record, "ok": False, "error": str(error.message)}
            raise
        figures = self._register_figures(result.get("figures"))
        result["figures"] = figures
        result["stat"] = stat_id
        result["dataset"] = self.dataset_summary(dataset, with_columns=False)
        result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        dataset["last_stat"] = {
            **stat_record,
            "title": str(result.get("title") or stat_id),
            "method": str(result.get("method") or ""),
            "ok": True,
            "error": None,
        }
        dataset.setdefault("steps", []).append(f"stat:{stat_id}")
        return result

    def op_auto(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        need = str(args.get("need") or "").strip()
        if not need:
            raise C.bad_request("auto 需要 need（一句中文需求，例如「按分组画箱线图并标显著性」）")
        plan = parse_need(need, dataset)
        dataset["need"] = need

        cleaning_report = None
        if plan.get("cleaning"):
            cleaning_report = pl_clean.apply_clean(dataset, plan=plan["cleaning"], auto=False)

        style = self._style({**C.STYLE_DEFAULT, **plan.get("style", {})})
        roles = plan.get("roles") or P.recommend_roles(dataset, self._chart_meta(plan["chart"]))
        params = dict(plan.get("params") or {})
        # 重新计算角色（清洗后列可能变化）
        meta = self._chart_meta(plan["chart"])
        roles = {k: v for k, v in roles.items() if v in (dataset.get("df").columns if dataset.get("df") is not None else [])}
        roles = P.recommend_roles(dataset, meta) | {k: v for k, v in roles.items() if v}
        chart_result = self.op_plot(
            {"dataset": dataset["id"], "chart": plan["chart"], "params": params,
             "style": style, "roles": roles}
        )
        stat_results = []
        for stat_id in plan.get("stats") or []:
            try:
                stat_params = {}
                if stat_id in ("compare", "describe", "assumption"):
                    stat_params = {"group": roles.get("group"), "y": roles.get("y"), "x": roles.get("x")}
                stat_result = self.op_stat(
                    {"dataset": dataset["id"], "stat": stat_id, "params": stat_params,
                     "style": style, "roles": roles}
                )
                stat_results.append(stat_result)
            except C.EngineError as error:
                chart_result["warnings"].append(f"统计 {stat_id} 跳过：{error.message}")
        dataset["steps"] = (dataset.get("steps") or []) + ["auto:" + need[:40]]
        return {
            "need": need,
            "plan": {k: C.jsonable(v) for k, v in plan.items()},
            "cleaning": cleaning_report,
            "figure": chart_result.get("figure"),
            "figures": chart_result.get("figures"),
            "figure_run_id": chart_result.get("figure_run_id"),
            "figure_run": chart_result.get("figure_run"),
            "chart": chart_result.get("chart"),
            "chart_title": chart_result.get("chart_title"),
            "params": chart_result.get("params"),
            "style": chart_result.get("style"),
            "roles": chart_result.get("roles"),
            "summary": chart_result.get("summary"),
            "tables": chart_result.get("tables"),
            "warnings": chart_result.get("warnings"),
            "code": chart_result.get("code"),
            "stats": [
                {"stat": item.get("stat"), "title": item.get("title"), "method": item.get("method"),
                 "summary": item.get("summary"), "tables": item.get("tables"),
                 "figures": item.get("figures"), "report": item.get("report")}
                for item in stat_results
            ],
            "dataset": self.dataset_summary(dataset, with_columns=False),
        }

    def op_need(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        need = str(args.get("need") or "").strip()
        dataset["need"] = need
        return {"need": need, "dataset": self.dataset_summary(dataset, with_columns=False)}

    def op_code(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        what = str(args.get("what") or "plot")
        if what == "session":
            return _codegen().session_code(dataset, _charts().CHARTS, self._style(args.get("style")))
        last = self.last_plot if self.last_plot and self.last_plot.get("dataset", {}).get("id") == dataset["id"] else None
        chart_id = str(args.get("chart") or (last or {}).get("chart") or "")
        if not chart_id:
            history = dataset.get("history") or []
            if history:
                chart_id = history[-1].get("chart") or ""
        if not chart_id:
            raise C.bad_request("没有可用的图表；先 plot 或传 chart")
        meta = self._chart_meta(chart_id)
        params = dict(args.get("params") or (last or {}).get("params") or {})
        style = self._style(args.get("style") or (last or {}).get("style"))
        roles = args.get("roles") or (last or {}).get("roles") or P.recommend_roles(dataset, meta)
        data = self._build_data(dataset, meta, params, roles)
        python_code = _codegen().build_script(dataset, chart_id, meta, data, params, style, roles)
        r_code = _codegen().chart_r_code(chart_id, data, params, style)
        return {"chart": chart_id, "python": python_code, "r": r_code,
                "filename": f"plotlab_{chart_id}.py"}

    def _resolve_export_run(self, dataset: Mapping[str, Any], args: Mapping[str, Any],
                            last: Mapping[str, Any] | None) -> dict:
        """把导出请求解析成一个 figure run：同请求复用，参数/样式/角色/数据变化则重画。"""
        archived_id = str(args.get("figure_run_id") or "").strip()
        if archived_id:
            overridden = any(args.get(key) is not None for key in ("chart", "params", "style", "roles"))
            if not overridden:
                run = self.get_figure_run(archived_id)
                if run.get("dataset_id") != dataset.get("id"):
                    raise C.bad_request(
                        f"figure run {archived_id} 属于数据集 {run.get('dataset_id')}，不是当前数据集"
                    )
                return run
        chart_id = str(args.get("chart") or args.get("kind") or "").strip()
        if not chart_id and last:
            chart_id = str(last.get("chart") or "")
        if not chart_id:
            history = dataset.get("history") or []
            if history:
                chart_id = str(history[-1].get("chart") or "")
        if not chart_id:
            # 没有 last_plot/history 时回退到本数据集最近一个 figure run（例如先 export what=figure 再导 bundle）
            for run_id in reversed(list(self.figure_runs)):
                candidate = self.figure_runs.get(run_id) or {}
                if candidate.get("dataset_id") == dataset.get("id"):
                    chart_id = str(candidate.get("chart") or "")
                    break
        if not chart_id:
            raise C.bad_request(
                "没有可导出的图：请传 chart（图表 id），或先 action=plot / auto 出图再导出"
            )
        meta = self._chart_meta(chart_id)
        same_chart = bool(last) and str(last.get("chart") or "") == chart_id
        if args.get("params") is not None:
            params = dict(args.get("params") or {})
        elif same_chart:
            params = dict(last.get("params") or {})
        else:
            params = {}
        if args.get("style") is not None:
            style = self._style(args.get("style"))
        elif same_chart:
            style = self._style(last.get("style"))
        else:
            style = self._style(None)
        if isinstance(args.get("roles"), dict):
            roles = {k: v for k, v in (args.get("roles") or {}).items() if v}
        elif same_chart and last.get("roles"):
            roles = {k: v for k, v in (last.get("roles") or {}).items() if v}
        else:
            roles = P.recommend_roles(dataset, meta)
        return self._figure_run(dataset, chart_id, meta, params, style, roles)

    def op_export(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        last = self.last_plot if (self.last_plot or {}).get("dataset", {}).get("id") == dataset["id"] else None
        run = self._resolve_export_run(dataset, args, last)
        chart_id = str(run.get("chart") or "")
        meta = self._chart_meta(chart_id) if chart_id else None
        style = C.resolve_style(run.get("style"))
        roles = dict(run.get("roles") or {})
        params = dict(run.get("params") or {})
        # 导出绑定基线：默认拿「最近一次真正渲染出来的图」和本次请求比对，这样
        # 「先画箱线 → 请求导出 bar」会在导出层触发重画分支（旧 bug：直接复制旧 PNG）。
        # 例外：显式按 figure_run_id 取某个历史版本且不覆盖参数（E6 归档回查）时，
        # 基线就是这个固定版本本身——用户要的就是旧图，不能重画回最新。
        pinned = bool(str(args.get("figure_run_id") or "").strip()) and not any(
            args.get(key) is not None for key in ("chart", "params", "style", "roles"))
        baseline = run if (pinned or last is None) else last

        def _rerender_export_figure() -> dict:
            """按本次导出请求重新取/建 figure run：同请求命中缓存，不同则真正重画。"""
            return self._figure_run(dataset, chart_id, meta, params, style, roles)

        stats_result = None
        if args.get("stat"):
            try:
                stats_result = self.op_stat({"dataset": dataset["id"], "stat": args["stat"],
                                             "params": args.get("stat_params") or {},
                                             "style": style, "roles": roles})
            except C.EngineError as error:
                stats_result = {"title": "统计", "summary": {"bullets": [error.message]}, "tables": []}
        exported = _export().export(
            what=str(args.get("what") or "bundle"),
            dataset=dataset,
            chart_id=chart_id,
            meta=meta,
            data=None,
            params=params,
            style=style,
            roles=roles,
            charts_module=_charts(),
            codegen=_codegen(),
            stats_result=stats_result,
            last_plot=baseline,
            figure_run=run,
            rerender=_rerender_export_figure,
            frozen=pinned,
            outdir=self.dirs["exports"],
            out_path=args.get("out_path"),
            fmt=str(args.get("format") or "png"),
        )
        # 导出层已经校验并记录了本次真正导出的 figure_run_id，这里只补缺（不覆盖）
        exported.setdefault("figure_run_id", run.get("figure_run_id"))
        exported.setdefault("data_digest", run.get("data_digest"))
        exported.setdefault("chart", chart_id)
        if stats_result is not None:
            exported["stats"] = stats_result
        # 口径卡的导出绑定：记录最近一次成功导出的 binding.status / figure_run_id / diffs，
        # 让「导出的图」与「当前预览的图」是否同一张可被回查（而不是靠最近一次导出猜）。
        binding = exported.get("figure_binding") if isinstance(exported.get("figure_binding"), Mapping) else {}
        dataset["last_export"] = {
            "what": str(exported.get("what") or args.get("what") or "bundle"),
            "file": exported.get("file"),
            "at": now_iso(),
            "status": binding.get("status"),
            "verified": binding.get("verified"),
            "chart": exported.get("chart") or binding.get("chart") or chart_id or None,
            "figure_run_id": exported.get("figure_run_id") or None,
            "data_digest": exported.get("data_digest") or None,
            "diffs": list(binding.get("diffs") or []),
            "notes": list(binding.get("notes") or []),
            "n_files": len(exported.get("files") or []),
        }
        return exported

    def op_records(self, args: Mapping[str, Any]) -> dict:
        dataset = self.get(args.get("dataset"))
        summary = self.dataset_summary(dataset, with_columns=False, with_digest=True)
        requested = str(args.get("figure_run_id") or "").strip()
        if requested:
            run = self.get_figure_run(requested)
            return {"figure_run": self._figure_run_view(run), "figure_run_id": requested,
                    "dataset": summary}
        runs = [
            {"figure_run_id": run.get("figure_run_id"), "chart": run.get("chart"),
             "chart_title": run.get("chart_title"), "data_digest": run.get("data_digest"),
             "at": run.get("at"), "figures": C.jsonable(run.get("figures") or [])}
            for run in self.figure_runs.values()
            if run.get("dataset_id") == dataset.get("id")
        ]
        return {"history": self._history_view(dataset), "count": len(dataset.get("history") or []),
                "figure_runs": runs, "figure_runs_count": len(runs),
                "need": dataset.get("need") or "", "dataset": summary}

    # 自检 / 建议
    def op_audit(self, args: Mapping[str, Any]) -> dict:
        checks: list[dict] = []

        def add(check_id: str, level: str, title: str, detail: str) -> None:
            checks.append({"id": check_id, "level": level, "title": title, "detail": detail})

        try:
            dataset = self.get(args.get("dataset"))
        except C.EngineError as error:
            return {"score": 0, "level": "bad",
                    "checks": [{"id": "dataset", "level": "fail", "title": "没有数据", "detail": error.message}],
                    "summary": {"bullets": ["还没有打开数据集：拖入 CSV/TSV/XLSX 或载入示例"]}}
        df = dataset["df"]
        add("dataset", "ok", "数据集就绪", f"{dataset.get('name')}：{df.shape[0]} 行 × {df.shape[1]} 列（{dataset.get('kind')}）")
        profile = pl_clean.profile(dataset)
        add("quality", "ok" if profile["score"] >= 80 else ("warn" if profile["score"] >= 60 else "fail"),
            "数据质量", f"{profile['score']}/100")
        if dataset.get("cleaning"):
            add("cleaning", "ok", "已做清洗", f"{len(dataset['cleaning'])} 步：" +
                "、".join(item.get("op", "") for item in dataset["cleaning"]))
        else:
            add("cleaning", "warn", "尚未清洗", "建议先跑 action=clean auto=true")
        history = dataset.get("history") or []
        if history:
            last = history[-1]
            figure = last.get("figure") or {}
            exists = os.path.exists(str(figure.get("file")))
            add("figure", "ok" if exists else "warn", "已出图",
                f"{last.get('chart_title')} → {figure.get('file')}" if exists else "历史图片文件已被清理，请重新出图")
            add("code", "ok" if self.last_plot and self.last_plot.get("code", {}).get("python") else "warn",
                "关键代码", "已生成可复现 Python 代码" if self.last_plot and self.last_plot.get("code") else "出图时会自动生成代码")
        else:
            add("figure", "warn", "还没有出图", "建议 action=plot 或 action=auto need=...")
        if dataset.get("need"):
            add("need", "ok", "已记录需求", dataset["need"])
        score = 100 - sum(18 if c["level"] == "fail" else (8 if c["level"] == "warn" else 0) for c in checks)
        score = int(max(0, min(100, score)))
        return {"score": score, "level": "good" if score >= 85 else ("warn" if score >= 60 else "bad"),
                "checks": checks, "summary": {"bullets": [c["title"] + "：" + c["detail"] for c in checks]}}

    def op_suggest(self, args: Mapping[str, Any]) -> dict:
        suggestions = []
        try:
            dataset = self.get(args.get("dataset"))
        except C.EngineError:
            return {"suggestions": [
                {"id": "open", "title": "导入数据", "detail": "拖入 CSV/TSV/XLSX/JSON，或 action=demo 载入示例"},
            ]}
        if not dataset.get("cleaning"):
            suggestions.append({"id": "clean", "title": "自动清洗", "detail": "profile 显示的问题可一键处理（缺失/异常/类型/去重）"})
        if not dataset.get("history"):
            rec = pl_clean.profile(dataset)["recommend"]["charts"]
            if rec:
                suggestions.append({"id": "plot", "title": f"推荐图表：{rec[0]['title']}", "detail": rec[0]["why"]})
        else:
            suggestions.append({"id": "stat", "title": "做统计分析", "detail": "组间比较/相关/回归/ROC/生存，结果自动进报告"})
            suggestions.append({"id": "export", "title": "导出图片与代码", "detail": "PNG 300dpi + 可复现 Python 代码 + report.md"})
        return {"suggestions": suggestions}

    def op_restart_noop(self, args: Mapping[str, Any]) -> dict:
        return {"ok": True}


def _has_stats() -> bool:
    try:
        import pl_stats  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _has_io() -> bool:
    try:
        import pl_io  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _python_exe() -> str:
    import sys
    return sys.executable


def _read_sample_table(path: str) -> pd.DataFrame:
    """读样本表（CSV/TSV/XLSX），列名规整：首列或 sample 列作为样本名。"""
    text = str(path)
    lower = text.lower()
    if lower.endswith((".xlsx", ".xls", ".xlsm")):
        frame = pd.read_excel(text)
    else:
        raw = None
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"):
            try:
                with open(text, "r", encoding=encoding) as handle:
                    head = handle.read(65536)
                try:
                    dialect = csv.Sniffer().sniff(head, delimiters=",\t;|")
                    sep = dialect.delimiter
                except csv.Error:
                    sep = "\t" if head.count("\t") > head.count(",") else ","
                frame = pd.read_csv(text, sep=sep, encoding=encoding)
                break
            except UnicodeDecodeError:
                raw = None
                continue
            except Exception:  # noqa: BLE001
                continue
        else:
            raise C.bad_request(f"无法读取样本表 {path}（编码/分隔符无法识别）")
    if frame.empty:
        raise C.bad_request("样本表为空")
    columns = [str(c) for c in frame.columns]
    sample_col = None
    for column in columns:
        if str(column).lower() in ("sample", "sample_id", "sampleid", "id", "name", "样本"):
            sample_col = column
            break
    if sample_col is None:
        sample_col = columns[0]
    frame = frame.rename(columns={sample_col: "sample"})
    frame["sample"] = frame["sample"].astype(str)
    return frame
