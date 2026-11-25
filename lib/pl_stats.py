"""PlotLab 统计分析模块。

对外契约（DESIGN.md §6）：

    STATS_CATALOG = [{"id", "title", "group", "desc", "needs", "roles", "params"}, ...]
    run(stat_id, dataset, data, params, style, outdir) -> {
        "title", "method", "summary": {"bullets", "key_numbers"},
        "tables": [pl_common.table(...)], "figures": [figure],
        "report": "markdown", "annotation": {...} | None, "warnings": [],
    }

约定：
- 数值可 JSON 序列化（经 pl_common.table / jsonable）；
- p 值用 pl_common.fmt_p（< 1e-4 显示 "< 1e-4" 字符串），数值用 pl_common.fmt_num；
- bootstrap 一律 np.random.default_rng(0)，保证同一输入结果完全一致；
- 数据不满足前提时抛 pl_common.prerequisite / bad_request，可选步骤失败降级为 warnings；
- 图内文字英文，表格/结论中文。

本模块只允许 import 标准库 + numpy/pandas/scipy/sklearn/statsmodels/lifelines/matplotlib +
pl_common + pl_prep。模块内禁止 import pl_state / pl_registry / pl_engine / pl_charts。
"""
from __future__ import annotations

import math
import os
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sstats

import pl_common as C
import pl_prep as P

# ---------------------------------------------------------------------------
# 统计目录
# ---------------------------------------------------------------------------

STATS_CATALOG: list[dict] = [
    {
        "id": "describe",
        "title": "描述统计",
        "group": "描述",
        "desc": "按分组或整体汇总数值列的 n、均值、标准差、中位数、四分位数、极值与缺失值。",
        "needs": "tidy",
        "roles": {"x": "optional", "y": "optional"},
        "params": [
            {"key": "columns", "label": "统计列（多选，留空=自动取 y 或全部数值列）",
             "type": "columns", "default": []},
            {"key": "group", "label": "分组列（可选）", "type": "text", "default": ""},
        ],
    },
    {
        "id": "compare",
        "title": "组间比较",
        "group": "比较",
        "desc": "两组自动 Welch t / Mann-Whitney，多组自动 ANOVA / Kruskal-Wallis；"
                "报告统计量、p、效应量、均值差 95%CI 与事后两两比较（多重校正），并返回图上标星用 annotation。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required"},
        "params": [
            {"key": "method", "label": "检验方法", "type": "select", "default": "auto",
             "options": ["auto", "welch", "student", "wilcox", "anova", "kruskal"]},
            {"key": "paired", "label": "配对设计（默认按配对 ID；无 ID 会报错）", "type": "boolean", "default": False},
            {"key": "paired_id", "label": "配对 ID 列（配对设计用）", "type": "text", "default": ""},
            {"key": "pairing", "label": "配对依据", "type": "select", "default": "auto",
             "options": ["auto", "id", "row_order"]},
            {"key": "independent_samples", "label": "矩阵数据的独立样本单位（默认拒绝检验）",
             "type": "text", "default": ""},
            {"key": "adjust", "label": "多重校正", "type": "select", "default": "bh",
             "options": ["bh", "bonferroni", "holm", "none"]},
            {"key": "bootstrap", "label": "bootstrap 次数", "type": "number", "default": 2000,
             "min": 200, "max": 20000},
        ],
    },
    {
        "id": "assumption",
        "title": "前提检查",
        "group": "比较",
        "desc": "逐组 Shapiro-Wilk 正态性、Levene 方差齐性、样本量、IQR 离群点比例，"
                "给出是否建议参数检验的结论与建议。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required"},
        "params": [
            {"key": "alpha", "label": "显著性水平", "type": "number", "default": 0.05,
             "min": 0.001, "max": 0.2},
            {"key": "outlier_k", "label": "离群点 IQR 倍数", "type": "number", "default": 1.5,
             "min": 0.5, "max": 5},
        ],
    },
    {
        "id": "corr",
        "title": "相关性分析",
        "group": "关系",
        "desc": "Pearson / Spearman / Kendall 相关矩阵，含 p 值、显著性星号、强相关长表；"
                "可选相关系数热图。矩阵数据按特征（基因/代谢物）计算。",
        "needs": "wide",
        "roles": {},
        "params": [
            {"key": "method", "label": "相关系数", "type": "select", "default": "pearson",
             "options": ["pearson", "spearman", "kendall"]},
            {"key": "columns", "label": "变量列（多选，留空=自动）", "type": "columns", "default": []},
            {"key": "plot", "label": "输出热图", "type": "boolean", "default": False},
            {"key": "max_cols", "label": "自动选取变量上限", "type": "number", "default": 12,
             "min": 2, "max": 40},
        ],
    },
    {
        "id": "regression",
        "title": "线性回归 / Logistic 回归",
        "group": "关系",
        "desc": "OLS 线性回归（系数/SE/t/p/95%CI/标准化 β/R²/adj R²/F/VIF）；"
                "因变量为二分类时自动改为 logistic 回归并报告 OR 与 95%CI；可选诊断图。",
        "needs": "tidy",
        "roles": {"y": "optional"},
        "params": [
            {"key": "y", "label": "因变量 y", "type": "text", "default": ""},
            {"key": "x", "label": "自变量 x（多选）", "type": "columns", "default": []},
            {"key": "covariates", "label": "协变量（多选）", "type": "columns", "default": []},
            {"key": "positive", "label": "logistic 阳性水平（可选）", "type": "text", "default": ""},
            {"key": "plot", "label": "输出诊断图（OLS）", "type": "boolean", "default": False},
        ],
    },
    {
        "id": "chi2",
        "title": "卡方 / Fisher 检验",
        "group": "分布",
        "desc": "两分类变量列联表；2×2 用 Fisher 精确检验，其余用 Pearson 卡方；"
                "输出期望频数、Cramér's V 与最小期望频数警告。",
        "needs": "tidy",
        "roles": {"x": "required"},
        "params": [
            {"key": "x", "label": "行变量", "type": "text", "default": ""},
            {"key": "y", "label": "列变量", "type": "text", "default": ""},
            {"key": "method", "label": "检验方法", "type": "select", "default": "auto",
             "options": ["auto", "chi2", "fisher"]},
        ],
    },
    {
        "id": "normality",
        "title": "正态性检验",
        "group": "分布",
        "desc": "逐列 Shapiro-Wilk 检验 + 偏度/峰度 + 样本量，给出是否符合正态的结论；"
                "可选直方图+正态曲线与 QQ 图。",
        "needs": "tidy",
        "roles": {"y": "optional"},
        "params": [
            {"key": "columns", "label": "变量列（多选，留空=自动）", "type": "columns", "default": []},
            {"key": "plot", "label": "输出直方图/QQ 图", "type": "boolean", "default": False},
            {"key": "max_plots", "label": "最多出几张图", "type": "number", "default": 4,
             "min": 1, "max": 12},
        ],
    },
    {
        "id": "roc",
        "title": "ROC 曲线",
        "group": "诊断",
        "desc": "二分类标签 + 分数列：AUC、bootstrap 95%CI、Youden 最佳阈值、"
                "敏感度/特异度/PPV/NPV，输出 ROC 曲线图。",
        "needs": "tidy",
        "roles": {"y": "optional", "group": "optional"},
        "params": [
            {"key": "label", "label": "标签列（0/1 或二分类）", "type": "text", "default": ""},
            {"key": "score", "label": "分数列（连续）", "type": "text", "default": ""},
            {"key": "positive", "label": "阳性水平（可选）", "type": "text", "default": ""},
            {"key": "bootstrap", "label": "AUC bootstrap 次数", "type": "number", "default": 1000,
             "min": 100, "max": 5000},
            {"key": "plot", "label": "输出 ROC 曲线", "type": "boolean", "default": True},
        ],
    },
    {
        "id": "survival",
        "title": "生存分析",
        "group": "生存",
        "desc": "Kaplan-Meier 曲线 + log-rank 检验，可按分组与置信带绘制并附风险人数；"
                "指定协变量时做 Cox 比例风险回归（HR/95%CI/p）。",
        "needs": "tidy",
        "roles": {"time": "optional", "event": "optional", "group": "optional"},
        "params": [
            {"key": "time", "label": "时间列", "type": "text", "default": ""},
            {"key": "event", "label": "事件列（0/1）", "type": "text", "default": ""},
            {"key": "group", "label": "分组列（可选）", "type": "text", "default": ""},
            {"key": "covariates", "label": "Cox 协变量（多选）", "type": "columns", "default": []},
            {"key": "risk_table", "label": "图上附风险人数", "type": "boolean", "default": True},
            {"key": "ci", "label": "KM 置信带", "type": "boolean", "default": True},
            {"key": "plot", "label": "输出 KM 曲线", "type": "boolean", "default": True},
        ],
    },
    {
        "id": "adjust",
        "title": "多重检验校正",
        "group": "校正",
        "desc": "对一串 p 值做 Benjamini-Hochberg FDR / Bonferroni / Holm 校正，"
                "输出原 p、校正 p、是否显著。",
        "needs": "tidy",
        "roles": {},
        "params": [
            {"key": "pvalues", "label": "p 值（逗号/空格分隔）", "type": "text", "default": ""},
            {"key": "column", "label": "或从数据列读取 p 值", "type": "text", "default": ""},
            {"key": "method", "label": "校正方法", "type": "select", "default": "bh",
             "options": ["bh", "bonferroni", "holm", "none"]},
            {"key": "alpha", "label": "显著性水平", "type": "number", "default": 0.05,
             "min": 0.001, "max": 0.5},
        ],
    },
    {
        "id": "pca",
        "title": "主成分分析",
        "group": "降维",
        "desc": "对矩阵/多数值列做 PCA（可选标准化）：解释方差、累计解释率、载荷 top 特征与得分图。",
        "needs": "wide",
        "roles": {},
        "params": [
            {"key": "scale", "label": "标准化（z-score）", "type": "boolean", "default": True},
            {"key": "n_components", "label": "保留主成分数", "type": "number", "default": 2,
             "min": 2, "max": 10},
            {"key": "max_features", "label": "最大特征数（0=全部）", "type": "number", "default": 0,
             "min": 0, "max": 5000},
            {"key": "sample_group_col", "label": "样本分组列（矩阵数据）", "type": "text", "default": ""},
            {"key": "plot", "label": "输出得分图", "type": "boolean", "default": False},
        ],
    },
]

STATS_INDEX: dict[str, dict] = {str(item["id"]): item for item in STATS_CATALOG}


def stats_catalog() -> list[dict]:
    """返回统计目录副本（供 registry/UI 使用）。"""
    return [dict(item) for item in STATS_CATALOG]


def stat_meta(stat_id: str) -> dict:
    """取单个统计方法的目录项；未知 id → not_found。"""
    stat = STATS_INDEX.get(str(stat_id or "").strip())
    if stat is None:
        raise C.not_found(f"未知统计方法：{stat_id}；可用：{'、'.join(STATS_INDEX)}")
    return dict(stat)


# ---------------------------------------------------------------------------
# 运行上下文
# ---------------------------------------------------------------------------


class _Ctx:
    """单次统计运行的上下文（dataset/data/params/style/outdir + warnings）。"""

    __slots__ = ("dataset", "data", "params", "style", "outdir", "warnings")

    def __init__(self, dataset: Mapping[str, Any], data: Mapping[str, Any],
                 params: Mapping[str, Any], style: Mapping[str, Any], outdir: str) -> None:
        self.dataset = dict(dataset or {})
        self.data = dict(data or {})
        self.params = dict(params or {})
        self.style = dict(style or {})
        self.outdir = str(outdir)
        self.warnings: list[str] = []

    @property
    def df(self) -> pd.DataFrame:
        frame = self.dataset.get("df")
        if frame is None or len(frame) == 0:
            raise C.prerequisite("没有数据，请先导入数据")
        return frame

    @property
    def kind(self) -> str:
        return str(self.dataset.get("kind") or "tidy")

    def roles(self) -> dict:
        return dict(self.data.get("roles") or {})

    def labels(self) -> dict:
        return dict(self.data.get("labels") or {})

    def levels(self, role: str = "group") -> list:
        return list((self.data.get("levels") or {}).get(role) or [])


# ---------------------------------------------------------------------------
# 参数读取
# ---------------------------------------------------------------------------


def _params(value: Any) -> dict:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _p_str(params: Mapping[str, Any], key: str, default: str = "") -> str:
    value = params.get(key)
    if value is None:
        return str(default)
    text = str(value).strip()
    return text if text else str(default)


def _p_float(params: Mapping[str, Any], key: str, default: float,
             lo: float | None = None, hi: float | None = None) -> float:
    try:
        value = float(params.get(key))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return value


def _p_int(params: Mapping[str, Any], key: str, default: int,
           lo: int | None = None, hi: int | None = None) -> int:
    return int(round(_p_float(params, key, float(default),
                              None if lo is None else float(lo),
                              None if hi is None else float(hi))))


def _p_bool(params: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = params.get(key)
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "y", "是", "真"):
        return True
    if text in ("0", "false", "no", "off", "n", "否", "假", ""):
        return False
    return bool(default)


def _p_select(params: Mapping[str, Any], key: str, allowed: Iterable[str], default: str) -> str:
    text = str(params.get(key) or "").strip().lower()
    return text if text in set(allowed) else str(default)


def _p_list(params: Mapping[str, Any], key: str) -> list[str]:
    value = params.get(key)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value]
    else:
        items = [part.strip() for part in re.split(r"[,;，、\s]+", str(value))]
    return [item for item in items if item]


def _p_columns(params: Mapping[str, Any], key: str) -> list[str]:
    return _p_list(params, key)


# ---------------------------------------------------------------------------
# 表格 / 报告
# ---------------------------------------------------------------------------

Spec = tuple[str, str, str]  # (key, label, type)


def _fmt_cell(value: Any, kind: str) -> Any:
    if value is None:
        return "—"
    if kind == "p":
        return C.fmt_p(value)
    if kind == "num":
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return str(int(value))
        return C.fmt_num(value, 4)
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _make_table(tid: str, title: str, specs: Sequence[Spec], rows: Iterable[Mapping[str, Any]],
                note: str | None = None, max_rows: int = 200) -> dict:
    """按 specs 统一格式化单元格后构造 pl_common.table。"""
    columns = [{"key": key, "label": label, "type": kind} for key, label, kind in specs]
    formatted: list[dict] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, _label, kind in specs:
            item[key] = _fmt_cell(row.get(key), kind)
        formatted.append(item)
    return C.table(tid, title, columns, formatted, note=note, max_rows=max_rows)


def _md_cell(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|")


def _md_table(table_obj: Mapping[str, Any], max_rows: int = 20) -> str:
    columns = list(table_obj.get("columns") or [])
    rows = list(table_obj.get("rows") or [])
    if not columns:
        return ""
    head = "| " + " | ".join(str(column.get("label") or column.get("key")) for column in columns) + " |"
    align = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_md_cell(row.get(column["key"])) for column in columns) + " |"
            for row in rows[:max_rows]]
    return "\n".join([head, align] + body)


def _report(title: str, method: str, bullets: Sequence[str],
            tables: Sequence[Mapping[str, Any]], extra_md: str | None = None) -> str:
    parts = [f"### {title}", ""]
    for bullet in bullets:
        parts.append(f"- {bullet}")
    if method:
        parts.extend(["", f"**方法**：{method}"])
    for table_obj in tables[:2]:
        markdown = _md_table(table_obj)
        if not markdown:
            continue
        parts.extend(["", f"**{table_obj.get('title') or '结果表'}**", "", markdown])
    if extra_md:
        parts.extend(["", str(extra_md).strip()])
    return "\n".join(parts).strip() + "\n"


def _pack(*, title: str, method: str, bullets: Sequence[str],
          key_numbers: Mapping[str, Any] | None = None,
          tables: Sequence[Mapping[str, Any] | None] | None = None,
          figures: Sequence[Mapping[str, Any] | None] | None = None,
          annotation: Mapping[str, Any] | None = None,
          warnings: Sequence[str] | None = None,
          extra_md: str | None = None) -> dict:
    clean_tables = [table for table in (tables or []) if table]
    clean_bullets = [str(bullet) for bullet in (bullets or []) if str(bullet).strip()]
    return {
        "title": str(title),
        "method": str(method),
        "summary": {"bullets": clean_bullets, "key_numbers": C.jsonable(dict(key_numbers or {}))},
        "tables": clean_tables,
        "figures": [figure for figure in (figures or []) if figure],
        "report": _report(title, method, clean_bullets, clean_tables, extra_md),
        "annotation": dict(annotation) if annotation else None,
        "warnings": [str(w) for w in (warnings or [])],
    }


def _try(ctx: _Ctx, label: str, func: Callable[[], Any], default: Any = None) -> Any:
    """可选步骤：失败降级为 warning，EngineError 继续上抛。"""
    try:
        return func()
    except C.EngineError:
        raise
    except Exception as exc:  # noqa: BLE001 - 可选步骤降级
        ctx.warnings.append(f"{label}未完成：{type(exc).__name__}: {exc}")
        return default


# ---------------------------------------------------------------------------
# 基础数值工具
# ---------------------------------------------------------------------------


def _label(value: Any) -> str:
    """分组/水平显示名：数值用 fmt_num，其余 str。"""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        return C.fmt_num(number)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return str(value)


def _arr(df: pd.DataFrame, column: str) -> np.ndarray:
    values = P.to_numeric(df[column]).to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _numeric_array(df: pd.DataFrame, column: str, label: str | None = None) -> np.ndarray:
    values = P.to_numeric(df[column]).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise C.prerequisite(
            f"列「{label or column}」没有可用数值（全部缺失或无法解析为数字）"
        )
    return finite


def _numeric_columns(ctx: _Ctx) -> list[str]:
    df = ctx.df
    columns = [str(c) for c in P.numeric_columns(ctx.dataset) if str(c) in df.columns]
    if not columns:
        columns = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return columns


def _categorical_columns(ctx: _Ctx) -> list[str]:
    df = ctx.df
    columns = [str(c) for c in P.categorical_columns(ctx.dataset) if str(c) in df.columns]
    if not columns:
        columns = []
        for column in df.columns:
            dtype = P.infer_dtype(df[column])
            if dtype in ("category", "text", "datetime") and 2 <= int(df[column].nunique(dropna=True)) <= 60:
                columns.append(str(column))
    return columns


def _param_col(ctx: _Ctx, *names: str) -> str | None:
    """严格读取单列参数：参数给了但列不存在 → bad_request。"""
    df = ctx.df
    for name in names:
        if name not in ctx.params:
            continue
        raw = ctx.params.get(name)
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else None
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if text not in df.columns:
            raise C.bad_request(f"参数 {name} 指定的列「{text}」不在数据里")
        return text
    return None


def _param_cols(ctx: _Ctx, name: str) -> list[str]:
    """严格读取多列参数：任一列不存在 → bad_request。"""
    items = _p_columns(ctx.params, name)
    if not items:
        return []
    missing = [item for item in items if item not in ctx.df.columns]
    if missing:
        raise C.bad_request(f"参数 {name} 中的列不存在：{'、'.join(missing)}")
    return items


def _summary_stats(values: Sequence[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None,
                "q3": None, "iqr": None, "min": None, "max": None}
    q1, median, q3 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if n > 1 else None,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _mean_sd_text(values: np.ndarray) -> str:
    stats_dict = _summary_stats(values)
    mean = stats_dict["mean"]
    sd = stats_dict["sd"]
    if mean is None:
        return "—"
    mean_text = C.fmt_num(mean, 3)
    if sd is None:
        return mean_text
    return f"{mean_text}±{C.fmt_num(sd, 3)}"


def _shapiro_p(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3 or arr.size > 5000:
        return None
    try:
        return float(sstats.shapiro(arr).pvalue)
    except Exception:  # noqa: BLE001 - 数值退化
        return None


def _shapiro_w(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3 or arr.size > 5000:
        return None
    try:
        return float(sstats.shapiro(arr).statistic)
    except Exception:  # noqa: BLE001
        return None


def _levene(groups: Sequence[np.ndarray]) -> dict | None:
    usable = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(usable) < 2:
        return None
    try:
        result = sstats.levene(*usable, center="median")
        return {"statistic": float(result.statistic), "p": float(result.pvalue)}
    except Exception:  # noqa: BLE001
        return None


def _skew_kurt(values: np.ndarray) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return None, None
    try:
        skew = float(sstats.skew(arr, bias=False))
        kurt = float(sstats.kurtosis(arr, fisher=True, bias=False))
        return skew, kurt
    except Exception:  # noqa: BLE001
        return None, None


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = int(len(a)), int(len(b))
    if na < 2 or nb < 2:
        return None
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if not math.isfinite(pooled) or pooled == 0:
        return None
    return float((np.mean(a) - np.mean(b)) / pooled)


def _hedges_g(d: float | None, a: np.ndarray, b: np.ndarray) -> float | None:
    if d is None:
        return None
    na, nb = int(len(a)), int(len(b))
    total = na + nb
    if total <= 3:
        return None
    return float(d * (1.0 - 3.0 / (4.0 * total - 9.0)))


def _effect_label_d(d: float | None) -> str:
    if d is None:
        return "无法估计"
    value = abs(float(d))
    if value < 0.2:
        return "可忽略"
    if value < 0.5:
        return "小"
    if value < 0.8:
        return "中"
    return "大"


def _effect_label_eta(value: float | None) -> str:
    if value is None:
        return "无法估计"
    v = abs(float(value))
    if v < 0.01:
        return "可忽略"
    if v < 0.06:
        return "小"
    if v < 0.14:
        return "中"
    return "大"


def _effect_label_r(value: float | None) -> str:
    if value is None:
        return "无法估计"
    v = abs(float(value))
    if v < 0.1:
        return "可忽略"
    if v < 0.3:
        return "小"
    if v < 0.5:
        return "中"
    return "大"


def _welch_test(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = int(a.size), int(b.size)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    diff = ma - mb
    se = math.sqrt(va / na + vb / nb) if na > 1 and nb > 1 else float("nan")
    if not math.isfinite(se) or se == 0:
        t = 0.0 if diff == 0 else math.copysign(float("inf"), diff)
        dof = float(na + nb - 2)
        p = 1.0 if diff == 0 else 0.0
    else:
        t = diff / se
        denom = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        dof = (va / na + vb / nb) ** 2 / denom if denom > 0 else float(na + nb - 2)
        p = float(2.0 * sstats.t.sf(abs(t), dof))
    try:
        tcrit = float(sstats.t.ppf(0.975, dof))
    except Exception:  # noqa: BLE001
        tcrit = 1.959963984540054
    ci = (float(diff - tcrit * se), float(diff + tcrit * se)) if math.isfinite(se) else (None, None)
    return {"t": float(t), "df": float(dof), "p": float(p), "mean_a": ma, "mean_b": mb,
            "diff": float(diff), "ci": ci, "d": _cohens_d(a, b)}


def _student_test(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = int(a.size), int(b.size)
    result = sstats.ttest_ind(a, b, equal_var=True)
    dof = float(na + nb - 2)
    diff = float(np.mean(a) - np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) if na + nb > 2 else float("nan")
    se = pooled * math.sqrt(1.0 / na + 1.0 / nb) if math.isfinite(pooled) else float("nan")
    tcrit = float(sstats.t.ppf(0.975, dof))
    ci = (float(diff - tcrit * se), float(diff + tcrit * se)) if math.isfinite(se) else (None, None)
    return {"t": float(result.statistic), "df": dof, "p": float(result.pvalue),
            "diff": diff, "ci": ci, "d": _cohens_d(a, b)}


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = int(a.size), int(b.size)
    result = sstats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    u = float(result.statistic)
    r_rb = 2.0 * u / (na * nb) - 1.0 if na and nb else None
    return {"u": u, "p": float(result.pvalue), "r": r_rb, "n_a": na, "n_b": nb}


def _paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = int(diff.size)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    se = sd_diff / math.sqrt(n) if n > 0 and math.isfinite(sd_diff) else float("nan")
    tcrit = float(sstats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    ci = (float(mean_diff - tcrit * se), float(mean_diff + tcrit * se)) if math.isfinite(se) else (None, None)
    result = sstats.ttest_rel(a, b)
    dz = mean_diff / sd_diff if (math.isfinite(sd_diff) and sd_diff > 0) else None
    return {"t": float(result.statistic), "df": float(n - 1), "p": float(result.pvalue),
            "mean_diff": mean_diff, "ci": ci, "dz": dz,
            "norm_p": _shapiro_p(diff)}


def _wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if diff.size < 6 or np.allclose(diff, 0):
        return None
    try:
        result = sstats.wilcoxon(a, b)
        return {"statistic": float(result.statistic), "p": float(result.pvalue)}
    except Exception:  # noqa: BLE001
        return None


def _anova(a_groups: Sequence[np.ndarray]) -> dict:
    arrays = [np.asarray(group, dtype=float) for group in a_groups]
    k = len(arrays)
    n = int(sum(arr.size for arr in arrays))
    grand = float(np.mean(np.concatenate(arrays)))
    result = sstats.f_oneway(*arrays)
    ss_between = float(sum(arr.size * (float(np.mean(arr)) - grand) ** 2 for arr in arrays))
    ss_within = float(sum(float(np.sum((arr - float(np.mean(arr))) ** 2)) for arr in arrays))
    ss_total = ss_between + ss_within
    df1 = k - 1
    df2 = n - k
    ms_between = ss_between / df1 if df1 > 0 else None
    ms_within = ss_within / df2 if df2 > 0 else None
    eta2 = ss_between / ss_total if ss_total > 0 else None
    omega2 = None
    if ss_between is not None and ms_within is not None and (ss_total + ms_within) > 0:
        omega2 = (ss_between - df1 * ms_within) / (ss_total + ms_within)
        omega2 = max(0.0, min(1.0, float(omega2)))
    return {"f": float(result.statistic), "df1": df1, "df2": df2, "p": float(result.pvalue),
            "eta2": eta2, "omega2": omega2, "ss_between": ss_between,
            "ss_within": ss_within, "ss_total": ss_total,
            "ms_between": ms_between, "ms_within": ms_within, "n": n, "k": k}


def _kruskal(groups: Sequence[np.ndarray]) -> dict:
    arrays = [np.asarray(group, dtype=float) for group in groups]
    k = len(arrays)
    n = int(sum(arr.size for arr in arrays))
    result = sstats.kruskal(*arrays)
    h = float(result.statistic)
    epsilon = (h - k + 1) / (n - k) if n > k else None
    if epsilon is not None:
        epsilon = max(0.0, min(1.0, float(epsilon)))
    return {"h": h, "df": k - 1, "p": float(result.pvalue), "epsilon2": epsilon, "n": n, "k": k}


def _bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, reps: int, seed: int = 0,
                       conf: float = 0.95) -> tuple[float, float] | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = int(a.size), int(b.size)
    if na < 2 or nb < 2 or reps < 10:
        return None
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(reps), dtype=float)
    for index in range(int(reps)):
        diffs[index] = float(np.mean(rng.choice(a, na, replace=True))
                             - np.mean(rng.choice(b, nb, replace=True)))
    low = float(np.percentile(diffs, (1.0 - conf) / 2.0 * 100.0))
    high = float(np.percentile(diffs, (1.0 + conf) / 2.0 * 100.0))
    return low, high


def _bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, reps: int,
                      seed: int = 0, conf: float = 0.95) -> tuple[float, float] | None:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if pos.size < 2 or neg.size < 2 or reps < 10:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(reps)):
        pi = rng.choice(pos, pos.size, replace=True)
        ni = rng.choice(neg, neg.size, replace=True)
        idx = np.concatenate([pi, ni])
        labels = np.concatenate([np.ones(pi.size), np.zeros(ni.size)])
        try:
            values.append(float(roc_auc_score(labels, score[idx])))
        except ValueError:
            continue
    if len(values) < max(20, reps // 10):
        return None
    arr = np.asarray(values)
    low = float(np.percentile(arr, (1.0 - conf) / 2.0 * 100.0))
    high = float(np.percentile(arr, (1.0 + conf) / 2.0 * 100.0))
    return low, high


def _corr_pair(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float | None, float | None, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    a = x[mask]
    b = y[mask]
    n = int(a.size)
    if n < 3:
        return None, None, n
    try:
        if method == "spearman":
            result = sstats.spearmanr(a, b)
        elif method == "kendall":
            result = sstats.kendalltau(a, b)
        else:
            result = sstats.pearsonr(a, b)
        return float(result.statistic), float(result.pvalue), n
    except Exception:  # noqa: BLE001 - 常量序列等
        return None, None, n


def _corr_stars(p: float | None) -> str:
    if p is None:
        return ""
    try:
        value = float(p)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def _outliers(values: np.ndarray, factor: float = 1.5) -> int:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return 0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return 0
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    return int(np.sum((arr < low) | (arr > high)))


def _annot_style(style: Mapping[str, Any]) -> str:
    value = str(style.get("annot_style") or "stars")
    return value if value in ("stars", "p", "both") else "stars"


def _p_phrase(p: Any) -> str:
    """p 值短语：p< 1e-4 / p=0.012（用于中文结论）。"""
    text = C.fmt_p(p)
    if text.startswith("<"):
        return "p" + text
    return "p=" + text


def _annot_label(p: float | None, style: Mapping[str, Any]) -> str:
    stars = C.p_stars(p)
    mode = _annot_style(style)
    if p is None or not math.isfinite(float(p)):
        return "NA"
    p_text = C.fmt_p(p)
    phrase = f"P {p_text}" if p_text.startswith("<") else f"P = {p_text}"
    if mode == "p":
        return phrase
    if mode == "both":
        return f"{stars}  {phrase}"
    return stars


def _save(fig: Any, ctx: _Ctx, name: str, title: str) -> dict:
    return C.save_fig(fig, ctx.outdir, name, title, dpi=int(ctx.style.get("dpi", 300) or 300))


def _finish(fig: Any, ctx: _Ctx, title: str, note: str | None = None,
            legend: bool = True) -> None:
    C.finalize_figure(fig, ctx.style, title=title, legend=legend, note=note)


# ---------------------------------------------------------------------------
# 共用：tidy 数据 / 分组
# ---------------------------------------------------------------------------


def _tidy_frame(ctx: _Ctx) -> pd.DataFrame:
    frame = ctx.data.get("tidy")
    if frame is None or len(frame) == 0:
        raise C.prerequisite("没有可用的观测数据（可能全部缺失）；请检查角色映射或清洗结果")
    return frame


def _grouped_values(ctx: _Ctx, min_n: int = 2,
                    purpose: str = "组间比较") -> tuple[list[tuple[str, np.ndarray]], dict]:
    """从 tidy 视图取 (分组标签, 数值数组) 列表与 labels。"""
    frame = _tidy_frame(ctx)
    labels = ctx.labels()
    for role in ("x", "y"):
        if f"__{role}" not in frame.columns:
            hint = "分组列（x）" if role == "x" else "数值列（y）"
            raise C.bad_request(f"{purpose}需要 {role} 角色：请在左侧选择{hint}")
    frame = frame.loc[frame["__y"].notna() & frame["__x"].notna()]
    if frame.empty:
        raise C.prerequisite("过滤缺失值后没有可用于分析的观测")
    x_series = frame["__x"]
    y_values = frame["__y"].to_numpy(dtype=float)
    if pd.api.types.is_numeric_dtype(x_series) and int(x_series.nunique(dropna=True)) > 12:
        raise C.prerequisite(
            f"x 角色「{labels.get('x') or 'x'}」是连续变量（{int(x_series.nunique())} 个不同取值），"
            f"{purpose}需要分类分组列（建议 2–12 组）"
        )
    groups: list[tuple[str, np.ndarray]] = []
    dropped: list[tuple[str, int]] = []
    for raw in pd.unique(x_series.dropna()):
        mask = (x_series == raw).to_numpy()
        arr = y_values[mask]
        label = _label(raw) or "NA"
        if arr.size < min_n:
            dropped.append((label, int(arr.size)))
            continue
        groups.append((label, arr))
    if dropped:
        text = "、".join(f"{label}(n={size})" for label, size in dropped)
        ctx.warnings.append(f"有效观测不足 {min_n} 的分组已剔除：{text}")
    if len(groups) < 2:
        raise C.prerequisite(f"至少需要 2 个有效分组（每组 n≥{min_n}），当前只有 {len(groups)} 个")
    return groups, labels


def _group_summary_row(label: str, values: np.ndarray) -> dict:
    info = _summary_stats(values)
    return {"group": label, **info}


def _group_summary_table(tid: str, y_label: str, groups: Sequence[tuple[str, np.ndarray]]) -> dict:
    specs: list[Spec] = [
        ("group", "分组", "str"),
        ("n", "n", "num"),
        ("mean", "均值", "num"),
        ("sd", "标准差", "num"),
        ("median", "中位数", "num"),
        ("q1", "Q1", "num"),
        ("q3", "Q3", "num"),
        ("iqr", "IQR", "num"),
        ("min", "最小值", "num"),
        ("max", "最大值", "num"),
    ]
    rows = [_group_summary_row(label, values) for label, values in groups]
    return _make_table(tid, f"分组概况（{y_label}）", specs, rows)


_TEST_LABELS = {
    "welch": "Welch t-test",
    "student": "Student t-test",
    "wilcox": "Mann-Whitney U",
    "paired_t": "Paired t-test",
    "anova": "One-way ANOVA",
    "kruskal": "Kruskal-Wallis",
}

_METHOD_TEXT = {
    "welch": "Welch 两样本 t 检验（scipy.stats.ttest_ind, equal_var=False）；主结果/CI/Cohen's d 均取自 Welch；"
             "均值差 95%CI 用 Welch–Satterthwaite 自由度 t 近似，并用 percentile bootstrap 复核",
    "student": "Student 两样本 t 检验（scipy.stats.ttest_ind, equal_var=True）；主结果/CI/Cohen's d 均取自等方差版本；"
               "均值差 95%CI 用 n₁+n₂−2 自由度 t 近似，并用 percentile bootstrap 复核",
    "wilcox": "Mann-Whitney U 秩和检验（scipy.stats.mannwhitneyu, two-sided）；"
              "效应量秩二列相关 r = 2U/(n₁·n₂) − 1",
    "paired_t": "配对 t 检验（scipy.stats.ttest_rel）；效应量 Cohen's dz；配对依据在结果与 annotation 中显式记录",
    "anova": "单因素方差分析（scipy.stats.f_oneway；SS 表由组均值/平方和计算）；效应量 η²/ω²；"
             "事后两两比较用 Welch t 检验 + 多重校正（近似 Games-Howell）",
    "kruskal": "Kruskal-Wallis 秩和检验（scipy.stats.kruskal）；效应量 ε² = (H − k + 1)/(n − k)；"
               "事后两两比较用 Mann-Whitney U + 多重校正",
}


# ---------------------------------------------------------------------------
# describe：描述统计
# ---------------------------------------------------------------------------


def _run_describe(ctx: _Ctx) -> dict:
    df = ctx.df
    roles = ctx.roles()
    params = ctx.params
    columns = _param_cols(ctx, "columns")
    group_col = _param_col(ctx, "group") or roles.get("group") or roles.get("x")
    if group_col and group_col not in df.columns:
        group_col = None

    specs: list[Spec] = [
        ("variable", "变量", "str"),
        ("group", "分组", "str"),
        ("n", "n", "num"),
        ("missing", "缺失", "num"),
        ("mean", "均值", "num"),
        ("sd", "标准差", "num"),
        ("median", "中位数", "num"),
        ("q1", "Q1", "num"),
        ("q3", "Q3", "num"),
        ("iqr", "IQR", "num"),
        ("min", "最小值", "num"),
        ("max", "最大值", "num"),
    ]
    rows: list[dict] = []
    bullets: list[str] = []
    key_numbers: dict[str, Any] = {}

    tidy = ctx.data.get("tidy")
    if (not columns and ctx.kind == "matrix" and tidy is not None
            and "__y" in tidy.columns and "__x" in tidy.columns):
        # 矩阵数据：融化成 样本 × 数值，按样本描述
        value_label = str(ctx.labels().get("y") or "value")
        tidy = tidy.loc[tidy["__y"].notna()]
        for raw in pd.unique(tidy["__x"].dropna()):
            values = tidy.loc[tidy["__x"] == raw, "__y"].to_numpy(dtype=float)
            info = _summary_stats(values)
            info["missing"] = 0
            rows.append({"variable": value_label, "group": _label(raw), **info})
        columns = [value_label]
        n_features = int(tidy["__id"].nunique()) if "__id" in tidy.columns else (
            int(rows[0]["n"]) if rows else 0)
        bullets.append(
            f"矩阵数据按样本汇总：{len(rows)} 个样本 × 每个样本的数值分布；"
            f"每行 n={n_features} 来自特征数（同一生物样本内的技术重复），不代表生物学重复数"
        )
        ctx.warnings.append(
            "矩阵融化视图：描述统计的 n 是特征数（技术重复），不能当作生物学重复；"
            "组间检验请先显式声明独立样本（compare 的 independent_samples）"
        )
    else:
        if not columns:
            y_col = roles.get("y")
            if y_col and y_col in df.columns:
                columns = [str(y_col)]
        numeric = _numeric_columns(ctx)
        if not columns:
            columns = numeric[:12]
            if len(numeric) > 12:
                ctx.warnings.append(f"数值列共 {len(numeric)} 个，仅统计前 12 个；可用参数 columns 指定")
        if not columns:
            raise C.prerequisite("数据里没有数值列，无法做描述统计")
        invalid = [c for c in columns if c not in df.columns or P.infer_dtype(df[c]) != "numeric"]
        if invalid:
            raise C.bad_request(f"列 {'、'.join(invalid)} 不是数值列，无法计算描述统计")

        level_pairs: list[tuple[str, Any]] = []
        if group_col:
            level_pairs = [(_label(v), v) for v in pd.unique(df[group_col].dropna())][:20]
        for column in columns:
            series = P.to_numeric(df[column])
            if level_pairs:
                for label, raw in level_pairs:
                    mask = (df[group_col] == raw).to_numpy()
                    sub = series[mask]
                    info = _summary_stats(sub.to_numpy(dtype=float))
                    info["missing"] = int(sub.isna().sum())
                    rows.append({"variable": column, "group": label, **info})
            else:
                info = _summary_stats(series.to_numpy(dtype=float))
                info["missing"] = int(series.isna().sum())
                rows.append({"variable": column, "group": "全部", **info})

        total_missing = int(sum(P.to_numeric(df[column]).isna().sum() for column in columns))
        denominator = max(1, len(df) * len(columns))
        if level_pairs:
            bullets.append(f"数据 {len(df)} 行 × {len(columns)} 个数值列，按「{group_col}」分 {len(level_pairs)} 组")
        else:
            bullets.append(f"数据 {len(df)} 行 × {len(columns)} 个数值列（整体汇总）")
        bullets.append(
            f"缺失值合计 {total_missing} 个（{total_missing / denominator * 100.0:.1f}%）"
        )
        for column in columns[:3]:
            overall = _summary_stats(P.to_numeric(df[column]).to_numpy(dtype=float))
            text = (f"{column}：n={overall['n']}，均值 {C.fmt_num(overall['mean'])}，"
                    f"中位数 {C.fmt_num(overall['median'])}，范围 {C.fmt_num(overall['min'])}–{C.fmt_num(overall['max'])}")
            if level_pairs and len(level_pairs) >= 2:
                means = []
                for label, raw in level_pairs:
                    values = P.to_numeric(df.loc[df[group_col] == raw, column]).to_numpy(dtype=float)
                    info = _summary_stats(values)
                    means.append((label, info["mean"]))
                means = [(label, mean) for label, mean in means if mean is not None]
                if len(means) >= 2:
                    high = max(means, key=lambda item: item[1])
                    low = min(means, key=lambda item: item[1])
                    text += f"；{high[0]} 组最高（{C.fmt_num(high[1])}），{low[0]} 组最低（{C.fmt_num(low[1])}）"
            bullets.append(text)
        key_numbers = {
            "n_rows": int(len(df)),
            "n_columns": int(len(columns)),
            "n_missing": total_missing,
            "n_groups": int(len(level_pairs)) if level_pairs else 1,
        }

    table = _make_table("describe", "描述统计", specs, rows, max_rows=400)
    extra = ""
    if group_col:
        extra = f"分组变量：`{group_col}`"
    return _pack(
        title="描述统计",
        method="描述统计（n/mean/sd/median/quartile/min/max/missing）；pandas",
        bullets=bullets,
        key_numbers=key_numbers,
        tables=[table],
        warnings=ctx.warnings,
        extra_md=extra,
    )


# ---------------------------------------------------------------------------
# compare：组间比较
# ---------------------------------------------------------------------------


def _two_group_payload(ctx: _Ctx, groups: list[tuple[str, np.ndarray]],
                       method_req: str, paired: bool, adjust: str, reps: int,
                       y_label: str, x_label: str = "x",
                       pairing_basis: str = "independent",
                       n_pairs: int | None = None) -> dict:
    (la, a), (lb, b) = groups[0], groups[1]
    na, nb = int(a.size), int(b.size)
    warnings = ctx.warnings
    bullets: list[str] = []
    test_rows: list[dict] = []
    estimate_rows: list[dict] = []
    factor = str(x_label or "x")
    contrast_text = f"{la} vs {lb}"
    contrast_id = f"{factor}:{la}|{lb}"
    pair_count = int(n_pairs) if (paired and n_pairs is not None) else (na if paired else None)

    stats_a = _summary_stats(a)
    stats_b = _summary_stats(b)
    bullets.append(
        f"{la} 组 n={na}，{y_label}={_mean_sd_text(a)}；{lb} 组 n={nb}，{y_label}={_mean_sd_text(b)}"
    )

    if paired:
        if na != nb:
            raise C.prerequisite(
                f"配对检验要求两组样本量一致（{la} n={na}，{lb} n={nb}）；请检查数据是否成对完整"
            )
        if na < 3:
            raise C.prerequisite("配对样本少于 3 对，无法做配对检验")
        if pairing_basis == "id":
            bullets.append(f"配对方式：按配对 ID 对齐（{pair_count} 对完整观测，ID 集合两侧一致）")
        else:
            warnings.append(
                "paired=true 且 pairing=row_order：按各组在数据中的出现顺序（行序）配对，"
                "请确认行序即配对顺序（如需按 ID 配对请传 paired_id 或 pairing=id）"
            )
        paired_res = _paired_test(a, b)
        wil = _wilcoxon_paired(a, b)
        diff_norm = paired_res["norm_p"]
        if method_req == "wilcox":
            primary = "wilcox"
        elif method_req in ("welch", "student"):
            primary = "paired_t"
        elif method_req in ("anova", "kruskal"):
            warnings.append(f"参数 method={method_req} 不适用于配对设计，已自动选择配对检验")
            primary = "paired_t" if (diff_norm is None or diff_norm >= 0.05) else "wilcox"
        else:
            primary = "paired_t" if (diff_norm is None or diff_norm >= 0.05) else "wilcox"

        test_rows.append({
            "test": "Paired t-test" + ("（主检验）" if primary == "paired_t" else "（参考）"),
            "statistic": paired_res["t"], "df": paired_res["df"], "p": paired_res["p"],
            "effect_name": "Cohen's dz", "effect_value": paired_res["dz"],
            "note": f"配对差均值 {C.fmt_num(paired_res['mean_diff'])}",
        })
        if wil is not None:
            test_rows.append({
                "test": "Wilcoxon signed-rank" + ("（主检验）" if primary == "wilcox" else "（兜底）"),
                "statistic": wil["statistic"], "df": None, "p": wil["p"],
                "effect_name": "—", "effect_value": None,
                "note": f"配对差正态性 Shapiro {_p_phrase(diff_norm)}",
            })
        else:
            warnings.append("配对差少于 6 对或全为 0，未做 Wilcoxon 符号秩检验")
        estimate_rows.extend([
            {"metric": "平均配对差（a − b）", "value": paired_res["mean_diff"], "note": f"{la} − {lb}"},
            {"metric": "95%CI 下限（t 近似）", "value": paired_res["ci"][0], "note": ""},
            {"metric": "95%CI 上限（t 近似）", "value": paired_res["ci"][1], "note": ""},
            {"metric": "Cohen's dz", "value": paired_res["dz"],
             "note": _effect_label_d(paired_res["dz"])},
        ])
        primary_p = paired_res["p"] if primary == "paired_t" else (wil["p"] if wil else paired_res["p"])
        primary_effect = paired_res["dz"]
        bullets.extend([
            f"平均配对差 {C.fmt_num(paired_res['mean_diff'])}（95%CI "
            f"{C.fmt_num(paired_res['ci'][0])}–{C.fmt_num(paired_res['ci'][1])}），"
            f"配对 t={C.fmt_num(paired_res['t'])}，df={C.fmt_num(paired_res['df'])}，"
            f"{_p_phrase(paired_res['p'])}，Cohen's dz={C.fmt_num(paired_res['dz'])}"
            f"（{_effect_label_d(paired_res['dz'])}）",
        ])
        if wil is not None:
            bullets.append(f"Wilcoxon 符号秩检验：W={C.fmt_num(wil['statistic'])}，{_p_phrase(wil['p'])}")
        if pairing_basis == "id":
            pairing_text = f"配对按登记 ID 对齐（{pair_count} 对完整观测，两侧 ID 集合一致）"
        else:
            pairing_text = f"配对按各组在数据中的出现顺序（row_order 逃生门，{pair_count} 对，可能错配）"
        method = (
            "配对 t 检验（scipy.stats.ttest_rel）；效应量 Cohen's dz；"
            f"n_pairs={pair_count}；{pairing_text}；配对差 95%CI 用 t 近似"
        )
        primary_name = _TEST_LABELS["paired_t"] if primary == "paired_t" else _TEST_LABELS["wilcox"]
    else:
        welch = _welch_test(a, b)
        student = _student_test(a, b)
        mw = _mannwhitney(a, b)
        normal_a = _shapiro_p(a)
        normal_b = _shapiro_p(b)
        normal_ok = all(p is None or p >= 0.05 for p in (normal_a, normal_b))
        if method_req == "auto":
            primary = "welch" if normal_ok else "wilcox"
        elif method_req in ("welch", "student", "wilcox"):
            primary = method_req
        else:
            warnings.append(f"参数 method={method_req} 需要 ≥3 组；本次只有 2 组，已按正态性自动选择")
            primary = "welch" if normal_ok else "wilcox"
        if not normal_ok:
            bullets.append(
                "正态性提示：Shapiro-Wilk p="
                f"{C.fmt_p(normal_a)}（{la}）/ {C.fmt_p(normal_b)}（{lb}），"
                + ("已优先采用非参数检验" if primary == "wilcox" else "仍报告参数检验结果，可参考非参数兜底")
            )

        rows_by_key = {
            "welch": {
                "statistic": welch["t"], "df": welch["df"], "p": welch["p"],
                "effect_name": "Cohen's d", "effect_value": welch["d"],
                "note": "Welch–Satterthwaite 自由度；Cohen's d 用合并标准差",
            },
            "student": {
                "statistic": student["t"], "df": student["df"], "p": student["p"],
                "effect_name": "Cohen's d", "effect_value": student["d"],
                "note": "等方差假设版本；df=n₁+n₂−2",
            },
            "wilcox": {
                "statistic": mw["u"], "df": None, "p": mw["p"],
                "effect_name": "rank-biserial r", "effect_value": mw["r"],
                "note": "秩和检验，不依赖正态性",
            },
        }
        for key, row in rows_by_key.items():
            if key == primary:
                suffix = "（主检验）"
            elif key == "wilcox":
                suffix = "（非参数兜底）"
            else:
                suffix = "（参考）"
            row["test"] = _TEST_LABELS[key] + suffix
        order = {
            "welch": ["welch", "wilcox", "student"],
            "student": ["student", "welch", "wilcox"],
            "wilcox": ["wilcox", "welch", "student"],
        }[primary]
        test_rows.extend([rows_by_key[key] for key in order])

        boot = _try(ctx, "均值差 bootstrap 95%CI",
                    lambda: _bootstrap_diff_ci(a, b, reps), None)
        primary_res = {"welch": welch, "student": student}.get(primary)
        diff = primary_res["diff"] if primary_res is not None else student["diff"]
        estimate_rows.append({"metric": "均值差（a − b）", "value": diff, "note": f"{la} − {lb}"})
        ci_low = ci_high = None
        if primary_res is not None:
            ci_low, ci_high = primary_res["ci"]
            if primary == "student":
                ci_kind = "Student t 近似"
                ci_note = f"df={C.fmt_num(student['df'])}（等方差）"
            else:
                ci_kind = "Welch–Satterthwaite t 近似"
                ci_note = f"df={C.fmt_num(welch['df'])}（Welch）"
            estimate_rows.extend([
                {"metric": f"95%CI 下限（{ci_kind}）", "value": ci_low, "note": ci_note},
                {"metric": f"95%CI 上限（{ci_kind}）", "value": ci_high, "note": ci_note},
            ])
        if boot is not None:
            estimate_rows.extend([
                {"metric": "bootstrap 95%CI 下限", "value": boot[0],
                 "note": f"{reps} 次重抽样，seed=0"},
                {"metric": "bootstrap 95%CI 上限", "value": boot[1],
                 "note": f"{reps} 次重抽样，seed=0"},
            ])
        if primary == "wilcox":
            primary_effect = mw["r"]
            estimate_rows.extend([
                {"metric": "rank-biserial r", "value": mw["r"], "note": _effect_label_r(mw["r"])},
                {"metric": "Cohen's d（参考）", "value": welch["d"], "note": "Welch 版本，仅作对比"},
            ])
        else:
            primary_effect = primary_res["d"]
            estimate_rows.extend([
                {"metric": "Cohen's d", "value": primary_res["d"],
                 "note": _effect_label_d(primary_res["d"])},
                {"metric": "Hedges' g", "value": _hedges_g(primary_res["d"], a, b),
                 "note": "小样本偏倚校正"},
                {"metric": "rank-biserial r", "value": mw["r"], "note": _effect_label_r(mw["r"])},
            ])
        stats_high, stats_low = ((la, stats_a), (lb, stats_b)) if stats_a["mean"] >= stats_b["mean"] else (
            (lb, stats_b), (la, stats_a))
        diff_abs = abs(stats_a["mean"] - stats_b["mean"])
        interval_text = "95%CI 无法估计"
        if ci_low is not None and ci_high is not None:
            if ci_low > ci_high:
                ci_low, ci_high = ci_high, ci_low
            if stats_high[0] == la:
                interval_low, interval_high = ci_low, ci_high
            else:
                interval_low, interval_high = -ci_high, -ci_low
            interval_text = f"95%CI {C.fmt_num(interval_low)}–{C.fmt_num(interval_high)}"
        elif boot is not None:
            interval_text = f"bootstrap 95%CI {C.fmt_num(boot[0])}–{C.fmt_num(boot[1])}"
        direction = "高" if diff_abs > 0 else "持平"
        bullets.append(
            f"均值差 {C.fmt_num(diff_abs)}（{stats_high[0]} 较 {stats_low[0]} {direction}；{interval_text}）"
        )
        if primary in ("welch", "student"):
            name = "Student t" if primary == "student" else "Welch t"
            bullets.append(
                f"{name}={C.fmt_num(primary_res['t'])}，df={C.fmt_num(primary_res['df'])}，"
                f"{_p_phrase(primary_res['p'])}，"
                f"Cohen's d={C.fmt_num(primary_res['d'])}（{_effect_label_d(primary_res['d'])}）"
            )
        bullets.append(
            f"Mann-Whitney U={C.fmt_num(mw['u'])}，{_p_phrase(mw['p'])}，"
            f"rank-biserial r={C.fmt_num(mw['r'])}（{_effect_label_r(mw['r'])}）"
        )
        if boot is not None:
            bullets.append(
                f"bootstrap {reps} 次：均值差 95%CI {C.fmt_num(boot[0])}–{C.fmt_num(boot[1])}"
            )
        primary_p = {"welch": welch["p"], "student": student["p"], "wilcox": mw["p"]}[primary]
        primary_name = _TEST_LABELS[primary]
        method = _METHOD_TEXT[primary]

    adjusted = float(C.adjust_pvalues([primary_p], adjust)[0])
    annotation = {
        "groups": [la, lb],
        "factor": factor,
        "x_label": x_label,
        "contrasts": [{
            "contrast_id": contrast_id,
            "contrast": contrast_text,
            "a": la, "b": lb,
            "p_adj": adjusted,
            "test": primary_name,
        }],
        "pairs": [{
            "a": la, "b": lb,
            "contrast_id": contrast_id,
            "contrast": contrast_text,
            "p_raw": float(primary_p),
            "p_adj": adjusted,
            "stars": C.p_stars(adjusted),
            "label": _annot_label(adjusted, ctx.style),
            "test": primary_name,
            "n_a": na,
            "n_b": nb,
        }],
        "test": primary_name,
        "n": {la: na, lb: nb},
        "n_pairs": pair_count,
        "pairing_basis": pairing_basis,
        "y_label": y_label,
    }
    tables = [
        _group_summary_table("compare_groups", y_label, groups),
        _make_table("compare_tests", "检验结果", [
            ("test", "检验", "str"),
            ("statistic", "统计量", "num"),
            ("df", "df", "num"),
            ("p", "p", "p"),
            ("effect_name", "效应量", "str"),
            ("effect_value", "效应值", "num"),
            ("note", "说明", "str"),
        ], test_rows),
        _make_table("compare_effects", "差异与效应量", [
            ("metric", "指标", "str"),
            ("value", "数值", "num"),
            ("note", "说明", "str"),
        ], estimate_rows),
    ]
    key_numbers = {
        "n": int(na + nb),
        "n_a": na,
        "n_b": nb,
        "p": float(primary_p),
        "p_adj": adjusted,
        "effect": primary_effect,
        "test": primary_name,
        "method": primary if not paired else ("paired_t" if primary == "paired_t" else "wilcox"),
        "pairing_basis": pairing_basis,
        "n_pairs": pair_count,
        "factor": factor,
        "contrast_id": contrast_id,
    }
    return {
        "bullets": bullets,
        "tables": tables,
        "annotation": annotation,
        "key_numbers": key_numbers,
        "method": method,
        "primary_p": float(primary_p),
        "n_sig": int(adjusted < 0.05),
    }


def _multi_group_payload(ctx: _Ctx, groups: list[tuple[str, np.ndarray]],
                         method_req: str, adjust: str, y_label: str,
                         x_label: str = "x") -> dict:
    warnings = ctx.warnings
    factor = str(x_label or "x")
    labels_list = [label for label, _ in groups]
    arrays = [values for _, values in groups]
    k = len(arrays)
    n = int(sum(arr.size for arr in arrays))
    shapiro_ps = [_shapiro_p(arr) for arr in arrays]
    normal_ok = all(p is None or p >= 0.05 for p in shapiro_ps)
    if method_req == "auto":
        primary = "anova" if normal_ok else "kruskal"
    elif method_req in ("anova", "kruskal"):
        primary = method_req
    else:
        warnings.append(f"参数 method={method_req} 只适用于 2 组；{k} 组时已按正态性自动选择")
        primary = "anova" if normal_ok else "kruskal"

    anova = _anova(arrays)
    kruskal = _kruskal(arrays)
    test_rows = [
        {
            "test": _TEST_LABELS["anova"] + ("（主检验）" if primary == "anova" else "（参考）"),
            "statistic": anova["f"], "df": anova["df1"], "p": anova["p"],
            "effect_name": "η²", "effect_value": anova["eta2"],
            "note": f"df2={anova['df2']}；ω²={C.fmt_num(anova['omega2'])}",
        },
        {
            "test": _TEST_LABELS["kruskal"] + ("（主检验）" if primary == "kruskal" else "（非参数兜底）"),
            "statistic": kruskal["h"], "df": kruskal["df"], "p": kruskal["p"],
            "effect_name": "ε²", "effect_value": kruskal["epsilon2"],
            "note": "秩和检验，不依赖正态性",
        },
    ]
    pair_rows: list[dict] = []
    pair_raw: list[dict] = []
    contrast_ids: list[str] = []
    for i in range(k):
        for j in range(i + 1, k):
            if primary == "anova":
                res = _welch_test(arrays[i], arrays[j])
                statistic = res["t"]
                p_value = res["p"]
                effect = res["d"]
                effect_name = "Cohen's d"
            else:
                res = _mannwhitney(arrays[i], arrays[j])
                statistic = res["u"]
                p_value = res["p"]
                effect = res["r"]
                effect_name = "rank-biserial r"
            contrast_id = f"{factor}:{labels_list[i]}|{labels_list[j]}"
            contrast_ids.append(contrast_id)
            pair_raw.append({
                "a": labels_list[i], "b": labels_list[j],
                "contrast_id": contrast_id,
                "contrast": f"{labels_list[i]} vs {labels_list[j]}",
                "statistic": statistic, "p": p_value, "effect": effect,
                "effect_name": effect_name,
            })
    adjusted = C.adjust_pvalues([item["p"] for item in pair_raw], adjust)
    for item, p_adj in zip(pair_raw, adjusted):
        pair_rows.append({
            "pair": f"{item['a']} vs {item['b']}",
            "contrast_id": item["contrast_id"],
            "statistic": item["statistic"],
            "p_raw": item["p"],
            "p_adj": float(p_adj),
            "stars": C.p_stars(p_adj),
            "effect_name": item["effect_name"],
            "effect": item["effect"],
        })
    n_sig = int(sum(1 for p_adj in adjusted if p_adj < 0.05))
    ss_rows = [
        {"source": "组间（Between）", "ss": anova["ss_between"], "df": anova["df1"],
         "ms": anova["ms_between"], "f": anova["f"], "p": anova["p"]},
        {"source": "组内（Within）", "ss": anova["ss_within"], "df": anova["df2"],
         "ms": anova["ms_within"], "f": None, "p": None},
        {"source": "合计（Total）", "ss": anova["ss_total"], "df": anova["df1"] + anova["df2"],
         "ms": None, "f": None, "p": None},
    ]
    means = [(_label(label), _summary_stats(arr)["mean"]) for label, arr in groups]
    means = [(label, mean) for label, mean in means if mean is not None]
    high = max(means, key=lambda item: item[1])
    low = min(means, key=lambda item: item[1])
    bullets = [
        f"{k} 组共 n={n}：" + "、".join(f"{label}(n={arr.size})" for label, arr in groups),
        f"{y_label} 均值范围 {C.fmt_num(low[1])}–{C.fmt_num(high[1])}；"
        f"{high[0]} 最高，{low[0]} 最低",
        f"单因素 ANOVA F({anova['df1']}, {anova['df2']})={C.fmt_num(anova['f'])}，"
        f"{_p_phrase(anova['p'])}，η²={C.fmt_num(anova['eta2'])}"
        f"（{_effect_label_eta(anova['eta2'])}）",
        f"Kruskal-Wallis H={C.fmt_num(kruskal['h'])}，df={kruskal['df']}，"
        f"{_p_phrase(kruskal['p'])}，ε²={C.fmt_num(kruskal['epsilon2'])}"
        f"（{_effect_label_eta(kruskal['epsilon2'])}）",
    ]
    if not normal_ok:
        bullets.append(
            "正态性提示：Shapiro-Wilk 至少一组 p<0.05，"
            + ("已优先采用 Kruskal-Wallis" if primary == "kruskal" else "仍报告 ANOVA 结果，可参考秩和检验")
        )
    top_pairs = sorted(pair_rows, key=lambda row: row["p_adj"])[:3]
    if top_pairs:
        pieces = [f"{row['pair']}（p_adj={C.fmt_p(row['p_adj'])}）" for row in top_pairs]
        bullets.append(
            f"事后两两比较 {len(pair_rows)} 对（{adjust} 校正），{n_sig} 对显著；最显著：" + "、".join(pieces)
        )
    primary_p = anova["p"] if primary == "anova" else kruskal["p"]
    primary_effect = anova["eta2"] if primary == "anova" else kruskal["epsilon2"]
    annotation = {
        "groups": labels_list,
        "factor": factor,
        "x_label": x_label,
        "contrasts": [
            {"contrast_id": item["contrast_id"], "contrast": item["contrast"],
             "a": item["a"], "b": item["b"], "p_adj": float(p_adj),
             "test": item["effect_name"]}
            for item, p_adj in zip(pair_raw, adjusted)
        ],
        "pairs": [
            {"a": item["a"], "b": item["b"],
             "contrast_id": item["contrast_id"], "contrast": item["contrast"],
             "p_raw": float(item["p"]), "p_adj": float(p_adj),
             "stars": C.p_stars(p_adj), "label": _annot_label(p_adj, ctx.style),
             "test": _TEST_LABELS[primary],
             "n_a": int(next(arr.size for label, arr in groups if label == item["a"])),
             "n_b": int(next(arr.size for label, arr in groups if label == item["b"]))}
            for item, p_adj in zip(pair_raw, adjusted)
        ],
        "test": _TEST_LABELS[primary],
        "n": {label: int(arr.size) for label, arr in groups},
        "y_label": y_label,
    }
    tables = [
        _group_summary_table("compare_groups", y_label, groups),
        _make_table("compare_tests", "整体检验结果", [
            ("test", "检验", "str"),
            ("statistic", "统计量", "num"),
            ("df", "df", "num"),
            ("p", "p", "p"),
            ("effect_name", "效应量", "str"),
            ("effect_value", "效应值", "num"),
            ("note", "说明", "str"),
        ], test_rows),
        _make_table("compare_anova", "ANOVA 方差分析表", [
            ("source", "变异来源", "str"),
            ("ss", "平方和 SS", "num"),
            ("df", "df", "num"),
            ("ms", "均方 MS", "num"),
            ("f", "F", "num"),
            ("p", "p", "p"),
        ], ss_rows),
        _make_table("compare_posthoc", "事后两两比较", [
            ("pair", "比较", "str"),
            ("statistic", "统计量", "num"),
            ("p_raw", "原始 p", "p"),
            ("p_adj", "校正 p", "p"),
            ("stars", "显著性", "str"),
            ("effect_name", "效应量", "str"),
            ("effect", "效应值", "num"),
        ], pair_rows, note=f"多重校正：{adjust}"),
    ]
    key_numbers = {
        "n": n,
        "k": k,
        "p": float(primary_p),
        "p_adj": float(min(adjusted)) if adjusted else None,
        "effect": primary_effect,
        "n_sig_pairs": n_sig,
        "test": _TEST_LABELS[primary],
        "factor": factor,
        "contrasts": len(pair_rows),
    }
    return {
        "bullets": bullets,
        "tables": tables,
        "annotation": annotation,
        "key_numbers": key_numbers,
        "method": _METHOD_TEXT[primary],
        "primary_p": float(primary_p),
        "n_sig": n_sig,
    }


def _matrix_view_guard(ctx: _Ctx) -> bool:
    """矩阵融化视图的生物学 n 保护：默认拒绝推断，除非显式声明独立样本。"""
    if not ctx.data.get("melted"):
        return False
    declared = _p_str(ctx.params, "independent_samples").lower()
    if declared not in ("sample", "independent"):
        raise C.prerequisite(
            "矩阵数据（特征×样本）已融化为长表：同一个样本里的多个特征值是技术重复，"
            "不是独立生物学重复，默认拒绝做组间检验。若确认每个样本本身就是独立观测单位，"
            "请显式传 independent_samples=\"sample\"（结果会记录 sample_unit，并保留技术重复提示）；"
            "否则请先在样本水平汇总，或改用热图/相关/PCA 等描述性图表。"
        )
    ctx.warnings.append(
        "已按 independent_samples=\"sample\" 显式声明样本为独立观测单位：本次比较的重复单位是样本，"
        "样本内多个特征值仍属技术重复；n 按样本数计，不按特征数计。"
    )
    return True


def _aggregate_melted_samples(ctx: _Ctx) -> None:
    """矩阵融化视图在显式声明独立样本后：先按样本聚合（样本内特征均值），再按样本分组比较。

    这样 n = 样本数，而不是把同一生物样本的 5 个特征值当成 5 个独立重复。
    """
    data = dict(ctx.data)
    tidy = data.get("tidy")
    if tidy is None or "__x" not in tidy.columns or "__y" not in tidy.columns:
        return
    if "__group" not in tidy.columns or int(tidy["__group"].nunique(dropna=True)) < 2:
        raise C.prerequisite(
            "矩阵数据没有可用的样本分组：请在样本表里指定 group 列（或先挂载样本信息），"
            "否则无法按样本单位做组间比较"
        )
    work = tidy.loc[tidy["__y"].notna() & tidy["__x"].notna()]
    if work.empty:
        raise C.prerequisite("矩阵数据融化后没有可用于聚合的数值")
    grouped = work.groupby("__x", observed=True, sort=False).agg(
        __group=("__group", "first"), __y=("__y", "mean"))
    frame = pd.DataFrame({
        "__x": grouped["__group"].astype(object).to_numpy(),
        "__y": grouped["__y"].to_numpy(dtype=float),
        "__sample": [str(index) for index in grouped.index],
    })
    labels = dict(data.get("labels") or {})
    group_name = labels.get("group") or labels.get("x") or "group"
    data["tidy"] = frame.reset_index(drop=True)
    data["labels"] = {**labels, "x": str(group_name), "y": labels.get("y") or "value"}
    data["levels"] = {"x": [str(value) for value in pd.unique(frame["__x"])]}
    data["notes"] = list(data.get("notes") or []) + [
        "矩阵数据已按样本聚合（每个样本取其全部特征的均值）后再按样本分组比较："
        "n=样本数，样本内特征值仍为技术重复"
    ]
    ctx.data = data
    ctx.warnings.append(
        "矩阵数据已按样本聚合（样本内特征均值）后比较：生物学重复单位是样本，"
        "每组 n=样本数，不把样本内特征值当独立重复"
    )


def _pairing_id_column(ctx: _Ctx, params: Mapping[str, Any]) -> str | None:
    explicit = _p_str(params, "paired_id") or _p_str(params, "id")
    if explicit:
        return explicit
    candidate = ctx.dataset.get("id_col")
    if candidate and candidate in ctx.df.columns:
        return str(candidate)
    return None


def _align_pair_groups(ctx: _Ctx, groups: list[tuple[str, np.ndarray]],
                       id_col: str) -> tuple[list[tuple[str, np.ndarray]], int]:
    """按配对 ID 把两组观测对齐；集合不一致、组内重复、ID 缺失都报 bad_request。"""
    escape = (
        "；如确认行顺序就是配对顺序，可显式设 pairing=\"row_order\" 走逃生门"
        "（按各组出现顺序配对，行序不同会静默错配，结果不可信）"
    )
    if id_col not in ctx.df.columns:
        raise C.bad_request(f"配对列「{id_col}」不在数据里：请检查 paired_id" + escape)
    roles = dict(ctx.roles())
    roles["id"] = id_col
    frame, _labels, _levels, _notes = P.tidy_frame(
        ctx.dataset, roles, ("x", "y"), ("id",), drop_na=True)
    if "__id" not in frame.columns:
        raise C.bad_request(f"无法从列「{id_col}」建立配对 ID" + escape)
    if frame["__id"].isna().any():
        raise C.bad_request(f"配对 ID 列「{id_col}」存在缺失值：配对要求每个观测都有 ID" + escape)
    work = frame.loc[frame["__y"].notna() & frame["__x"].notna()]
    maps: dict[str, dict[str, float]] = {}
    for raw in pd.unique(work["__x"].dropna()):
        label = _label(raw)
        sub = work.loc[work["__x"] == raw]
        ids = sub["__id"].astype(str)
        duplicated = sorted(set(ids[ids.duplicated()].tolist()))
        if duplicated:
            raise C.bad_request(
                f"配对 ID 在「{label}」组内重复：{'、'.join(duplicated[:5])}"
                f"（配对要求每个 ID 在每组各出现一次）" + escape
            )
        maps[label] = dict(zip(ids.tolist(), sub["__y"].to_numpy(dtype=float)))
    la, lb = groups[0][0], groups[1][0]
    if la not in maps or lb not in maps:
        raise C.bad_request(f"配对 ID 对齐失败：分组「{la}」「{lb}」与配对框架不一致" + escape)
    left, right = maps[la], maps[lb]
    missing_left = sorted(set(right) - set(left))
    missing_right = sorted(set(left) - set(right))
    if missing_left or missing_right:
        pieces = []
        if missing_left:
            pieces.append(f"仅 {lb} 有 {len(missing_left)} 个 ID（如 {'、'.join(missing_left[:5])}）")
        if missing_right:
            pieces.append(f"仅 {la} 有 {len(missing_right)} 个 ID（如 {'、'.join(missing_right[:5])}）")
        raise C.bad_request("配对 ID 集合两侧不一致：" + "；".join(pieces) + escape)
    common = sorted(set(left) & set(right))
    if len(common) < 3:
        raise C.prerequisite(f"完整配对数只有 {len(common)} 对（<3），无法做配对检验")
    arr_a = np.asarray([left[key] for key in common], dtype=float)
    arr_b = np.asarray([right[key] for key in common], dtype=float)
    ctx.warnings.append(
        f"配对按 ID 列「{id_col}」对齐：{len(common)} 对完整观测，两侧 ID 集合一致"
    )
    return [(la, arr_a), (lb, arr_b)], len(common)


def _run_compare(ctx: _Ctx) -> dict:
    params = ctx.params
    melted = _matrix_view_guard(ctx)
    if melted:
        _aggregate_melted_samples(ctx)
    groups, labels = _grouped_values(ctx, min_n=2, purpose="组间比较")
    method_req = _p_select(params, "method", ("auto", "welch", "student", "wilcox", "anova", "kruskal"), "auto")
    paired = _p_bool(params, "paired", False)
    adjust = _p_select(params, "adjust", ("bh", "bonferroni", "holm", "none"),
                       str(ctx.style.get("p_adjust") or "bh"))
    reps = _p_int(params, "bootstrap", 2000, 200, 20000)
    y_label = str(labels.get("y") or "数值")
    x_label = str(labels.get("x") or "分组")
    if paired and len(groups) > 2:
        raise C.bad_request("配对检验目前只支持 2 组；多组配对请改用重复测量模型")
    pairing_basis = "independent"
    n_pairs: int | None = None
    if paired:
        pairing = _p_select(params, "pairing", ("auto", "id", "row_order"), "auto")
        if pairing == "row_order":
            pairing_basis = "row_order"
            ctx.warnings.append(
                "pairing=\"row_order\" 逃生门：按各组在数据中的出现顺序（行序）配对；"
                "若行序不是配对顺序，会静默错配，务必人工确认"
            )
        else:
            id_col = _pairing_id_column(ctx, params)
            if not id_col:
                raise C.bad_request(
                    "配对检验需要配对 ID（参数 paired_id，或数据里的 ID 列）：当前没有可用 ID 列。"
                    "请指定 paired_id=列名 / pairing=\"id\"，或显式设 pairing=\"row_order\" "
                    "走逃生门——它按各组出现顺序配对，行序不同会静默错配，结果不可信。"
                )
            groups, n_pairs = _align_pair_groups(ctx, groups, id_col)
            pairing_basis = "id"
    payload = (_two_group_payload(ctx, groups, method_req, paired, adjust, reps, y_label,
                                  x_label=x_label, pairing_basis=pairing_basis,
                                  n_pairs=n_pairs)
               if len(groups) == 2 else _multi_group_payload(ctx, groups, method_req, adjust, y_label,
                                                             x_label=x_label))
    if melted:
        payload["key_numbers"]["sample_unit"] = "sample"
    extra = f"分组变量：`{x_label}`；因变量：`{y_label}`"
    result = _pack(
        title="组间比较",
        method=payload["method"],
        bullets=payload["bullets"],
        key_numbers=payload["key_numbers"],
        tables=payload["tables"],
        annotation=payload["annotation"],
        warnings=ctx.warnings,
        extra_md=extra,
    )
    return result


# ---------------------------------------------------------------------------
# assumption：前提检查
# ---------------------------------------------------------------------------


def _run_assumption(ctx: _Ctx) -> dict:
    params = ctx.params
    alpha = _p_float(params, "alpha", 0.05, 0.001, 0.2)
    factor = _p_float(params, "outlier_k", 1.5, 0.5, 5.0)
    groups, labels = _grouped_values(ctx, min_n=2, purpose="前提检查")
    y_label = str(labels.get("y") or "数值")
    alpha_text = C.fmt_num(alpha)

    rows: list[dict] = []
    for label, values in groups:
        info = _summary_stats(values)
        w_value = _shapiro_w(values)
        p_value = _shapiro_p(values)
        skew, kurt = _skew_kurt(values)
        outliers = _outliers(values, factor)
        abnormal = p_value is not None and p_value < alpha
        rows.append({
            "group": label,
            "n": info["n"],
            "mean": info["mean"],
            "sd": info["sd"],
            "median": info["median"],
            "shapiro_w": w_value,
            "shapiro_p": p_value,
            "skew": skew,
            "kurtosis": kurt,
            "outliers": outliers,
            "outlier_pct": outliers / info["n"] * 100.0 if info["n"] else None,
            "normal": "否（偏离正态）" if abnormal else ("是" if p_value is not None else "样本量不足"),
        })

    min_n = min(int(values.size) for _, values in groups)
    total_outliers = sum(int(row["outliers"] or 0) for row in rows)
    total_n = sum(int(row["n"] or 0) for row in rows)
    shapiro_ps = [p for p in (row["shapiro_p"] for row in rows) if p is not None]
    all_normal = all(p >= alpha for p in shapiro_ps) if shapiro_ps else False
    any_normal = any(p >= alpha for p in shapiro_ps) if shapiro_ps else False
    levene = _levene([values for _, values in groups])
    variance_ok = levene is not None and levene["p"] >= alpha
    two_groups = len(groups) == 2

    if all_normal and (variance_ok or two_groups):
        recommendation = ("满足参数检验前提：建议 Welch t 检验（2 组，不要求方差齐）"
                          if two_groups else "满足参数检验前提：建议单因素 ANOVA（方差不齐时用 Welch ANOVA）")
        parametric = "是"
    elif all_normal and not variance_ok and not two_groups:
        recommendation = "正态性满足但方差不齐：建议 Welch ANOVA 或先做变量变换，也可用 Kruskal-Wallis"
        parametric = "部分"
    elif min_n >= 30:
        recommendation = "样本量较大（n≥30），中心极限定理下参数检验较稳健；同时报告非参数检验作敏感性分析"
        parametric = "较稳健"
    else:
        recommendation = ("偏离正态且样本量偏小：建议非参数检验（Mann-Whitney U / Kruskal-Wallis），"
                          "或对变量做变换后复检")
        parametric = "否"

    test_rows = [
        {
            "item": "正态性（逐组 Shapiro-Wilk）",
            "statistic": None,
            "p": min(shapiro_ps) if shapiro_ps else None,
            "conclusion": ("各组均未拒绝正态（p≥" + alpha_text + "）" if all_normal
                           else ("部分组偏离正态" if any_normal else "各组均偏离正态")),
            "suggestion": recommendation if not all_normal else "正态性前提满足",
        },
        {
            "item": "方差齐性（Levene，中心=中位数）",
            "statistic": levene["statistic"] if levene else None,
            "p": levene["p"] if levene else None,
            "conclusion": ("方差齐" if variance_ok else "方差不齐") if levene else "无法检验",
            "suggestion": "2 组时 Welch t 不要求方差齐；多组方差不齐用 Welch ANOVA",
        },
        {
            "item": "样本量（最小分组 n）",
            "statistic": min_n,
            "p": None,
            "conclusion": f"最小 n={min_n}",
            "suggestion": "n≥30 时 t/F 检验对正态偏离较稳健；n<10 建议非参数检验",
        },
        {
            "item": f"离群点比例（IQR×{C.fmt_num(factor)}）",
            "statistic": total_outliers / total_n * 100.0 if total_n else None,
            "p": None,
            "conclusion": f"{total_outliers} / {total_n} 个观测",
            "suggestion": "离群点比例>5% 时建议检查数据或用稳健方法（Wilcoxon/Kruskal）",
        },
        {
            "item": "参数检验适用性",
            "statistic": None,
            "p": None,
            "conclusion": f"参数检验建议：{parametric}",
            "suggestion": recommendation,
        },
    ]

    bullets = [
        "各组 n：" + "、".join(f"{label} {int(values.size)}" for label, values in groups)
        + f"；最小分组 n={min_n}",
        "Shapiro-Wilk：" + "；".join(
            f"{row['group']} W={C.fmt_num(row['shapiro_w'])}，{_p_phrase(row['shapiro_p'])}"
            for row in rows
        ) + ("（p≥" + alpha_text + "，未拒绝正态）" if all_normal else "（存在偏离正态的分组）"),
        ("Levene 方差齐性：F=" + C.fmt_num(levene["statistic"]) + f"，{_p_phrase(levene['p'])}，"
         + ("方差齐" if variance_ok else "方差不齐")) if levene
        else f"Levene 方差齐性：样本不足以检验（{len(groups)} 组）",
        f"离群点（IQR×{C.fmt_num(factor)}）：{total_outliers} / {total_n} 个"
        f"（{total_outliers / total_n * 100.0 if total_n else 0:.1f}%）",
        f"结论：{recommendation}（{len(groups)} 组，n={total_n}）",
    ]

    tables = [
        _make_table("assumption_groups", f"逐组前提检查（{y_label}）", [
            ("group", "分组", "str"),
            ("n", "n", "num"),
            ("mean", "均值", "num"),
            ("sd", "标准差", "num"),
            ("median", "中位数", "num"),
            ("shapiro_w", "Shapiro W", "num"),
            ("shapiro_p", "Shapiro p", "p"),
            ("skew", "偏度", "num"),
            ("kurtosis", "峰度", "num"),
            ("outliers", "离群点 n", "num"),
            ("outlier_pct", "离群点 %", "num"),
            ("normal", "正态性", "str"),
        ], rows, note=f"显著性水平 α={alpha_text}；峰度为 Fisher 定义（正态=0）"),
        _make_table("assumption_tests", "前提检验与建议", [
            ("item", "检查项", "str"),
            ("statistic", "统计量", "num"),
            ("p", "p", "p"),
            ("conclusion", "结论", "str"),
            ("suggestion", "建议", "str"),
        ], test_rows),
    ]
    key_numbers = {
        "n": total_n,
        "n_groups": len(groups),
        "min_group_n": min_n,
        "shapiro_min_p": min(shapiro_ps) if shapiro_ps else None,
        "levene_p": levene["p"] if levene else None,
        "outlier_pct": total_outliers / total_n * 100.0 if total_n else None,
    }
    return _pack(
        title="前提检查",
        method="Shapiro-Wilk 正态性检验（scipy.stats.shapiro）+ Levene 方差齐性检验"
               "（scipy.stats.levene, center=median，即 Brown-Forsythe）+ IQR 离群点准则",
        bullets=bullets,
        key_numbers=key_numbers,
        tables=tables,
        warnings=ctx.warnings,
    )


# ---------------------------------------------------------------------------
# corr：相关性
# ---------------------------------------------------------------------------


def _diverging_cmap(style: Mapping[str, Any]):
    import matplotlib

    name = str(style.get("palette") or "")
    cmap_name = C.PALETTE_CMAPS.get(name)
    try:
        if cmap_name in ("RdBu_r", "coolwarm"):
            return matplotlib.colormaps[cmap_name]
        return matplotlib.colormaps["RdBu_r"]
    except Exception:  # noqa: BLE001
        return matplotlib.colormaps["RdBu_r"]


def _corr_heatmap(ctx: _Ctx, names: Sequence[str], r_matrix: np.ndarray,
                  p_matrix: np.ndarray, method_label: str, n_text: str) -> dict | None:
    with C.style_context(ctx.style):
        fig, ax = C.new_fig(ctx.style)
        data = np.ma.masked_invalid(np.asarray(r_matrix, dtype=float))
        image = ax.imshow(data, cmap=_diverging_cmap(ctx.style), vmin=-1.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        rotate = 45 if len(names) > 4 else 0
        ax.set_xticklabels(list(names), rotation=rotate, ha="right" if rotate else "center",
                           fontsize=max(6.0, float(ctx.style.get("tick_size", 10)) - 1))
        ax.set_yticklabels(list(names), fontsize=max(6.0, float(ctx.style.get("tick_size", 10)) - 1))
        if len(names) <= 14:
            size = max(5.5, float(ctx.style.get("text_size", 10)) - 3.5)
            for i in range(len(names)):
                for j in range(len(names)):
                    value = float(r_matrix[i, j])
                    if not math.isfinite(value):
                        continue
                    stars = "" if i == j else _corr_stars(p_matrix[i, j])
                    color = "white" if abs(value) > 0.62 else "#1A1A1A"
                    ax.text(j, i, f"{_corr_text(value)}{stars}", ha="center", va="center",
                            fontsize=size, color=color)
        fig.colorbar(image, ax=ax, shrink=0.82, label="Correlation coefficient")
        C.finalize_figure(fig, ctx.style,
                          title=f"Correlation matrix ({method_label})",
                          note=f"{n_text}")
        return _save(fig, ctx, f"stat_corr_{method_label}", f"Correlation matrix ({method_label})")


def _run_corr(ctx: _Ctx) -> dict:
    params = ctx.params
    method = _p_select(params, "method", ("pearson", "spearman", "kendall"), "pearson")
    method_label = {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"}[method]
    max_cols = _p_int(params, "max_cols", 12, 2, 40)
    columns = _param_cols(ctx, "columns")
    names: list[str] = []
    frame: pd.DataFrame | None = None
    n_rows = 0

    if columns:
        invalid = [c for c in columns if P.infer_dtype(ctx.df[c]) != "numeric"]
        if invalid:
            raise C.bad_request(f"列 {'、'.join(invalid)} 不是数值列，无法计算相关")
        names = columns
        frame = pd.DataFrame({c: P.to_numeric(ctx.df[c]).to_numpy(dtype=float) for c in names})
        n_rows = int(len(ctx.df))
    elif ctx.kind == "matrix" and ctx.data.get("wide"):
        wide = ctx.data["wide"]
        matrix = np.asarray(wide.get("matrix"), dtype=float)
        features = [str(f) for f in wide.get("features") or []]
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise C.prerequisite("矩阵数据不足（至少需要 2 个特征）才能计算相关")
        if len(features) > max_cols:
            variances = np.nanvar(matrix, axis=0)
            keep = np.sort(np.argsort(-np.nan_to_num(variances))[:max_cols])
            matrix = matrix[:, keep]
            features = [features[index] for index in keep]
            ctx.warnings.append(f"特征较多，仅保留方差最大的 {max_cols} 个特征")
        names = features
        frame = pd.DataFrame(matrix, columns=names)
        n_rows = int(matrix.shape[0])
    else:
        numeric = _numeric_columns(ctx)
        if len(numeric) < 2:
            raise C.prerequisite("至少需要 2 个数值列才能计算相关矩阵")
        if len(numeric) > max_cols:
            ctx.warnings.append(f"数值列较多，仅使用前 {max_cols} 个；可用参数 columns 指定")
        names = numeric[:max_cols]
        frame = pd.DataFrame({c: P.to_numeric(ctx.df[c]).to_numpy(dtype=float) for c in names})
        n_rows = int(len(ctx.df))

    if len(names) < 2:
        raise C.prerequisite("至少需要 2 个变量才能计算相关矩阵")
    k = len(names)
    r_matrix = np.full((k, k), np.nan, dtype=float)
    p_matrix = np.full((k, k), np.nan, dtype=float)
    pair_rows: list[dict] = []
    for i in range(k):
        r_matrix[i, i] = 1.0
        p_matrix[i, i] = 0.0
        for j in range(i + 1, k):
            r_value, p_value, n_pair = _corr_pair(
                frame[names[i]].to_numpy(dtype=float),
                frame[names[j]].to_numpy(dtype=float), method,
            )
            if r_value is not None and p_value is not None:
                r_matrix[i, j] = r_matrix[j, i] = r_value
                p_matrix[i, j] = p_matrix[j, i] = p_value
            pair_rows.append({
                "variable_a": names[i], "variable_b": names[j], "n": n_pair,
                "r": r_value, "p": p_value, "stars": _corr_stars(p_value),
            })
    if not pair_rows:
        raise C.prerequisite("没有可计算的有效变量对")
    valid_pairs = [row for row in pair_rows if row["r"] is not None]
    if not valid_pairs:
        raise C.prerequisite("所有变量对的有效样本都少于 3 个（可能缺失过多或为常量）")
    valid_pairs.sort(key=lambda row: abs(row["r"]), reverse=True)

    matrix_columns: list[dict] = [{"key": "variable", "label": "变量", "type": "str"}]
    matrix_rows: list[dict] = []
    for i in range(k):
        row: dict[str, Any] = {"variable": names[i]}
        for j in range(k):
            value = float(r_matrix[i, j])
            if not math.isfinite(value):
                row[names[j]] = "—"
            elif i == j:
                row[names[j]] = "1.00"
            else:
                row[names[j]] = f"{_corr_text(value)}{_corr_stars(p_matrix[i, j])}"
        matrix_rows.append(row)
    for name in names:
        matrix_columns.append({"key": name, "label": name, "type": "str"})
    matrix_table = C.table("corr_matrix", f"{method_label} 相关系数矩阵（*p<0.05 **p<0.01 ***p<0.001）",
                           matrix_columns, matrix_rows, note=f"行数 n={n_rows}；成对完整观测")

    pairs_table = _make_table("corr_pairs", "相关系数（按 |r| 排序）", [
        ("variable_a", "变量 A", "str"),
        ("variable_b", "变量 B", "str"),
        ("n", "n", "num"),
        ("r", "r", "num"),
        ("p", "p", "p"),
        ("stars", "显著性", "str"),
    ], valid_pairs, max_rows=120)

    strongest = valid_pairs[0]
    strong_count = int(sum(1 for row in valid_pairs if abs(row["r"]) >= 0.5))
    sig_count = int(sum(1 for row in valid_pairs if row["p"] is not None and row["p"] < 0.05))
    bullets = [
        f"{method_label} 相关：{k} 个变量、{len(valid_pairs)} 个变量对，样本 n={n_rows}",
        f"最强相关：{strongest['variable_a']} – {strongest['variable_b']}，"
        f"r={C.fmt_num(strongest['r'])}（{_p_phrase(strongest['p'])}，n={strongest['n']}）",
        f"|r|≥0.5 的强相关变量对 {strong_count} 个；p<0.05 显著变量对 {sig_count} 个（未校正）",
    ]
    if k > 2:
        second = valid_pairs[1] if len(valid_pairs) > 1 else None
        if second is not None:
            bullets.append(
                f"次强相关：{second['variable_a']} – {second['variable_b']}，"
                f"r={C.fmt_num(second['r'])}（{_p_phrase(second['p'])}）"
            )
    figures: list[dict] = []
    if _p_bool(params, "plot", False):
        figure = _try(ctx, "相关热图", lambda: _corr_heatmap(
            ctx, names, r_matrix, p_matrix, method_label, f"n={n_rows}; pairwise-complete"))
        if figure:
            figures.append(figure)

    key_numbers = {
        "n_variables": k,
        "n_pairs": len(valid_pairs),
        "n_rows": n_rows,
        "max_r": float(strongest["r"]),
        "max_r_p": float(strongest["p"]) if strongest["p"] is not None else None,
        "n_sig_pairs": sig_count,
    }
    return _pack(
        title="相关性分析",
        method=f"{method_label} 相关分析（scipy.stats."
               + {"pearson": "pearsonr", "spearman": "spearmanr", "kendall": "kendalltau"}[method]
               + "，成对完整观测）；显著性星号 *p<0.05 **p<0.01 ***p<0.001",
        bullets=bullets,
        key_numbers=key_numbers,
        tables=[matrix_table, pairs_table],
        figures=figures,
        warnings=ctx.warnings,
    )


# ---------------------------------------------------------------------------
# regression：线性回归 / logistic 回归
# ---------------------------------------------------------------------------


_POSITIVE_HINTS = ("yes", "positive", "case", "1", "true", "dead", "death", "event",
                   "responder", "response", "复发", "死亡", "阳性", "有")


def _guess_positive(levels: Sequence[str]) -> str:
    lowered = [str(v).strip().lower() for v in levels]
    for hint in _POSITIVE_HINTS:
        for value, low in zip(levels, lowered):
            if low == hint:
                return str(value)
    return str(levels[-1])


def _design_frame(df: pd.DataFrame, numeric_terms: Sequence[str],
                  factor_terms: Sequence[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """构建设计矩阵：数值列保持原值，分类列 one-hot（drop_first，记录参考水平）。"""
    parts: list[pd.Series] = []
    dropped: list[str] = []
    reference_notes: list[str] = []
    for column in numeric_terms:
        series = P.to_numeric(df[column]).astype(float)
        finite = series.dropna()
        if finite.size < 2:
            dropped.append(f"{column}(有效值不足)")
            continue
        sd = float(finite.std(ddof=1))
        if not math.isfinite(sd) or sd == 0:
            dropped.append(f"{column}(常数)")
            continue
        parts.append(series.rename(str(column)))
    for column in factor_terms:
        text = df[column].astype("string")
        levels = sorted({str(v) for v in text.dropna().tolist()})
        if len(levels) < 2:
            dropped.append(f"{column}(单一水平)")
            continue
        dummies = pd.get_dummies(text, prefix=str(column), drop_first=True, dtype=float)
        reference_notes.append(f"{column} 参考水平={levels[0]}")
        for name in dummies.columns:
            parts.append(dummies[name].astype(float).rename(str(name)))
    if parts:
        design = pd.concat(parts, axis=1)
    else:
        design = pd.DataFrame(index=df.index)
    return design, dropped, reference_notes


def _qq_plot(ax: Any, values: np.ndarray, color: str, point_size: float) -> None:
    """按 style 颜色绘制正态 QQ 图（不用 scipy 默认配色）。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return
    (osm, osr), (slope, intercept, _corr) = sstats.probplot(arr, dist="norm", fit=True)
    osm_array = np.asarray(osm, dtype=float)
    ax.scatter(osm_array, np.asarray(osr, dtype=float), s=point_size, color=color,
               alpha=0.85, edgecolor="white", linewidth=0.35, zorder=3)
    ax.plot(osm_array, slope * osm_array + intercept, color="#B24745",
            linewidth=1.0, zorder=2)


def _corr_text(value: float) -> str:
    """相关系数文本，避免出现 "-0.00"。"""
    if not math.isfinite(float(value)):
        return "—"
    if abs(float(value)) < 0.005:
        return "0.00"
    return f"{float(value):.2f}"


def _regression_diagnostics(ctx: _Ctx, fitted: np.ndarray, residuals: np.ndarray,
                            title: str) -> dict | None:
    with C.style_context(ctx.style):
        fig, axes = C.new_fig(ctx.style, 1, 2)
        ax1, ax2 = np.ravel(axes)[:2]
        color = C.color_list(ctx.style, 1)[0]
        ax1.scatter(fitted, residuals, s=float(ctx.style.get("point_size", 28)),
                    alpha=float(ctx.style.get("alpha", 0.85)), color=color, edgecolor="none")
        ax1.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
        ax1.set_xlabel("Fitted values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs fitted", fontsize=float(ctx.style.get("title_size", 14)) * 0.72)
        _qq_plot(ax2, np.asarray(residuals, dtype=float), color,
                 float(ctx.style.get("point_size", 28)))
        ax2.set_title("Normal Q-Q", fontsize=float(ctx.style.get("title_size", 14)) * 0.72)
        ax2.set_xlabel("Theoretical quantiles")
        ax2.set_ylabel("Sample quantiles")
        C.finalize_figure(fig, ctx.style, title=title, legend=False)
        return _save(fig, ctx, "stat_regression_diagnostics", title)


def _run_regression(ctx: _Ctx) -> dict:
    import statsmodels.api as sm

    df = ctx.df
    params = ctx.params
    roles = ctx.roles()
    warnings = ctx.warnings
    y_col = _param_col(ctx, "y") or roles.get("y")
    if not y_col:
        raise C.bad_request("回归需要因变量：请用参数 y 指定一列（数值列或二分类列）")
    if y_col not in df.columns:
        raise C.bad_request(f"因变量列「{y_col}」不在数据里")

    x_cols = [c for c in _param_cols(ctx, "x") if c != y_col]
    cov_cols = [c for c in _param_cols(ctx, "covariates") if c != y_col]
    if not x_cols:
        roles_x = roles.get("x")
        if roles_x and roles_x != y_col and roles_x in df.columns:
            x_cols = [str(roles_x)]
            warnings.append(f"未指定自变量，已使用角色 x：{roles_x}")
        else:
            fallback = [c for c in _numeric_columns(ctx) if c != y_col][:2]
            if fallback:
                x_cols = fallback
                warnings.append(f"未指定自变量，已自动使用：{'、'.join(fallback)}")
    if not x_cols and not cov_cols:
        raise C.bad_request("回归需要至少 1 个自变量：请用参数 x 指定（可多选）")

    raw_series = df[y_col]
    dtype = P.infer_dtype(raw_series)
    binary: dict | None = None
    y = pd.Series(np.nan, index=df.index, dtype=float)
    if dtype != "numeric":
        levels = [str(v) for v in pd.unique(raw_series.dropna().astype(str))]
        if len(levels) != 2:
            raise C.bad_request(
                f"因变量「{y_col}」不是数值列（{len(levels)} 个水平：{'、'.join(levels[:6])}）；"
                "回归需要数值列或二分类列"
            )
        positive = _p_str(params, "positive") or _guess_positive(levels)
        if positive not in levels:
            raise C.bad_request(f"positive 参数「{positive}」不在因变量水平中（{'、'.join(levels)}）")
        text = raw_series.astype(str).str.strip()
        y = text.map(lambda value: 1.0 if value == positive else 0.0).astype(float)
        y[raw_series.isna()] = np.nan
        binary = {"positive": positive, "levels": levels}
    else:
        y = P.to_numeric(raw_series).astype(float)
        finite = y.dropna()
        unique_values = sorted({float(v) for v in pd.unique(finite)}) if len(finite) else []
        if len(unique_values) == 2 and set(unique_values) <= {0.0, 1.0}:
            pos_text = _p_str(params, "positive")
            positive_value = 0.0 if pos_text in ("0", "0.0") else 1.0
            binary = {"positive": C.fmt_num(positive_value),
                      "levels": [C.fmt_num(v) for v in unique_values]}
    if binary is None and P.infer_dtype(raw_series) != "numeric":
        raise C.bad_request(f"因变量「{y_col}」无法作为数值或二分类变量使用")
    if y.notna().sum() == 0:
        raise C.prerequisite(f"因变量「{y_col}」没有有效取值")

    predictors = x_cols + cov_cols
    numeric_terms = [c for c in predictors if P.infer_dtype(df[c]) == "numeric"]
    factor_terms = [c for c in predictors if c not in numeric_terms]
    design, dropped, reference_notes = _design_frame(df, numeric_terms, factor_terms)
    if dropped:
        warnings.append(f"以下预测变量被剔除：{'、'.join(dropped)}")
    if design.shape[1] == 0:
        raise C.prerequisite("没有可用的预测变量（全部是常数或单一水平）")
    if design.shape[1] > 12:
        warnings.append(f"预测变量较多（{design.shape[1]} 列），注意过拟合与多重共线性")

    mask = y.notna() & design.notna().all(axis=1)
    n_used = int(mask.sum())
    if n_used < max(8, design.shape[1] + 3):
        raise C.prerequisite(
            f"有效样本过少（n={n_used}），不足以拟合 {design.shape[1]} 个预测变量"
        )
    y_used = y.loc[mask]
    design_used = design.loc[mask]
    design_const = sm.add_constant(design_used, has_constant="add")

    figures: list[dict] = []
    coefficient_rows: list[dict] = []
    fit_rows: list[dict] = []
    bullets: list[str] = []
    linear_terms = list(design_used.columns)
    formula_text = f"{y_col} ~ {' + '.join(predictors) if predictors else '1'}"
    reference_text = "；".join(reference_notes)

    if binary is None:
        try:
            model = sm.OLS(y_used, design_const).fit()
        except Exception as exc:  # noqa: BLE001
            raise C.prerequisite(f"线性回归拟合失败（可能存在完全共线）：{exc}") from exc
        beta = model.params
        se = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues
        conf = model.conf_int()
        y_sd = float(np.std(y_used.to_numpy(dtype=float), ddof=1)) if n_used > 1 else 0.0
        vif_values: dict[str, float] = {}
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            matrix = np.asarray(design_const, dtype=float)
            if design_used.shape[1] >= 2 and n_used > design_used.shape[1] + 2:
                for index, term in enumerate(design_const.columns):
                    if str(term) == "const":
                        continue
                    try:
                        vif_values[str(term)] = float(variance_inflation_factor(matrix, index))
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            pass
        for term in design_const.columns:
            name = str(term)
            display = "截距" if name == "const" else name
            std_beta = None
            vif = None
            if name != "const":
                x_values = design_used[name].to_numpy(dtype=float)
                x_sd = float(np.std(x_values, ddof=1))
                if y_sd > 0 and math.isfinite(x_sd):
                    std_beta = float(beta[name] * x_sd / y_sd)
                vif = vif_values.get(name)
            coefficient_rows.append({
                "term": display,
                "beta": float(beta[name]),
                "se": float(se[name]),
                "stat": float(tvalues[name]),
                "p": float(pvalues[name]),
                "ci_low": float(conf.loc[name, 0]),
                "ci_high": float(conf.loc[name, 1]),
                "std_beta": std_beta,
                "vif": vif,
            })
        r2 = float(model.rsquared)
        adj_r2 = float(model.rsquared_adj)
        f_stat = float(model.fvalue) if model.fvalue is not None else None
        f_p = float(model.f_pvalue) if model.f_pvalue is not None else None
        fit_rows.extend([
            {"metric": "模型", "value": None, "note": "普通最小二乘 OLS 线性回归"},
            {"metric": "因变量", "value": None, "note": y_col},
            {"metric": "样本量 n", "value": n_used, "note": ""},
            {"metric": "R²", "value": r2, "note": f"解释 {r2 * 100.0:.1f}% 方差"},
            {"metric": "调整 R²", "value": adj_r2, "note": ""},
            {"metric": "F 统计量", "value": f_stat, "note": f"df=({model.df_model}, {model.df_resid})"},
            {"metric": "F 检验 p", "value": f_p, "note": ""},
            {"metric": "残差标准误", "value": float(math.sqrt(float(model.mse_resid))), "note": "RMSE"},
            {"metric": "AIC", "value": float(model.aic), "note": ""},
            {"metric": "BIC", "value": float(model.bic), "note": ""},
        ])
        bullets.append(
            f"{formula_text}：n={n_used}，R²={C.fmt_num(r2)}（调整 R²={C.fmt_num(adj_r2)}），"
            f"F({int(model.df_model)}, {int(model.df_resid)})={C.fmt_num(f_stat)}，{_p_phrase(f_p)}"
        )
        significant = [row for row in coefficient_rows
                       if row["term"] != "截距" and row["p"] is not None and row["p"] < 0.05]
        if significant:
            pieces = [
                f"{row['term']}（β={C.fmt_num(row['beta'])}，95%CI {C.fmt_num(row['ci_low'])}–"
                f"{C.fmt_num(row['ci_high'])}，{_p_phrase(row['p'])}"
                + (f"，标准化 β={C.fmt_num(row['std_beta'])}" if row["std_beta"] is not None else "")
                + "）"
                for row in significant[:3]
            ]
            bullets.append(f"显著预测变量 {len(significant)} 个：" + "；".join(pieces))
        else:
            bullets.append("没有显著预测变量（所有 p≥0.05）")
        high_vif = [row for row in coefficient_rows
                    if row["vif"] is not None and row["vif"] > 10]
        if high_vif:
            bullets.append(
                "共线性警告：" + "、".join(f"{row['term']}(VIF={C.fmt_num(row['vif'])})" for row in high_vif)
                + "（VIF>10，系数可能不稳定）"
            )
        if f_p is not None and f_p < 0.05:
            bullets.append(f"模型整体显著（F 检验 {_p_phrase(f_p)}）")
        if _p_bool(params, "plot", False):
            figure = _try(ctx, "回归诊断图", lambda: _regression_diagnostics(
                ctx, np.asarray(model.fittedvalues, dtype=float),
                np.asarray(model.resid, dtype=float),
                f"Regression diagnostics: {y_col}"))
            if figure:
                figures.append(figure)
        coefficient_table = _make_table("regression_coef", "回归系数（OLS）", [
            ("term", "变量", "str"),
            ("beta", "β", "num"),
            ("se", "标准误", "num"),
            ("stat", "t", "num"),
            ("p", "p", "p"),
            ("ci_low", "95%CI 下限", "num"),
            ("ci_high", "95%CI 上限", "num"),
            ("std_beta", "标准化 β", "num"),
            ("vif", "VIF", "num"),
        ], coefficient_rows)
        method = ("普通最小二乘线性回归（statsmodels.OLS）；95%CI 为 t 分布区间；"
                  "标准化 β = β·sd(x)/sd(y)；VIF 用于共线性诊断（>10 提示严重共线）")
        key_numbers = {
            "n": n_used,
            "r2": r2,
            "adj_r2": adj_r2,
            "f": f_stat,
            "p": f_p,
            "n_terms": int(design_used.shape[1]),
        }
    else:
        try:
            model = sm.Logit(y_used, design_const).fit(disp=0, maxiter=300)
        except Exception as exc:  # noqa: BLE001
            raise C.prerequisite(f"logistic 回归拟合失败（可能存在完全分离或完全共线）：{exc}") from exc
        beta = model.params
        se = model.bse
        zvalues = model.tvalues
        pvalues = model.pvalues
        conf = model.conf_int()
        n_events = int(y_used.sum())
        for term in design_const.columns:
            name = str(term)
            display = "截距" if name == "const" else name
            coefficient_rows.append({
                "term": display,
                "beta": float(beta[name]),
                "se": float(se[name]),
                "stat": float(zvalues[name]),
                "p": float(pvalues[name]),
                "or": float(math.exp(beta[name])),
                "or_low": float(math.exp(conf.loc[name, 0])),
                "or_high": float(math.exp(conf.loc[name, 1])),
            })
        llf = float(model.llf)
        llnull = float(model.llnull) if model.llnull is not None else None
        pseudo_r2 = float(model.prsquared)
        lr_stat = 2.0 * (llf - llnull) if llnull is not None else None
        lr_p = None
        if lr_stat is not None:
            try:
                lr_p = float(sstats.chi2.sf(max(0.0, lr_stat), int(model.df_model)))
            except Exception:  # noqa: BLE001
                lr_p = None
        fit_rows.extend([
            {"metric": "模型", "value": None, "note": "二分类 logistic 回归（最大似然）"},
            {"metric": "因变量", "value": None, "note": f"{y_col}（阳性={binary['positive']}）"},
            {"metric": "样本量 n", "value": n_used, "note": ""},
            {"metric": "事件数（阳性）", "value": n_events,
             "note": f"{n_events / max(1, n_used) * 100.0:.1f}%"},
            {"metric": "McFadden 伪 R²", "value": pseudo_r2, "note": ""},
            {"metric": "对数似然 LLF", "value": llf, "note": ""},
            {"metric": "似然比检验 χ²", "value": lr_stat, "note": "vs 截距模型"},
            {"metric": "似然比检验 p", "value": lr_p, "note": ""},
            {"metric": "AIC", "value": float(model.aic), "note": ""},
            {"metric": "BIC", "value": float(model.bic), "note": ""},
        ])
        bullets.append(
            f"{formula_text}（logistic，阳性={binary['positive']}）：n={n_used}"
            f"（事件 {n_events}，{n_events / max(1, n_used) * 100.0:.1f}%），"
            f"McFadden R²={C.fmt_num(pseudo_r2)}，似然比检验 {_p_phrase(lr_p)}"
        )
        significant = [row for row in coefficient_rows
                       if row["term"] != "截距" and row["p"] is not None and row["p"] < 0.05]
        if significant:
            pieces = [
                f"{row['term']}（OR={C.fmt_num(row['or'])}，95%CI {C.fmt_num(row['or_low'])}–"
                f"{C.fmt_num(row['or_high'])}，{_p_phrase(row['p'])}）"
                for row in significant[:3]
            ]
            bullets.append(f"显著预测变量 {len(significant)} 个：" + "；".join(pieces))
        else:
            bullets.append("没有显著预测变量（所有 p≥0.05）")
        unstable = [row for row in coefficient_rows
                    if row["term"] != "截距" and (abs(row["beta"]) > 10 or row["se"] > 10)]
        if unstable:
            warnings.append("部分系数绝对值/标准误很大，可能存在完全分离或强共线，OR 不可靠")
        if _p_bool(params, "plot", False):
            warnings.append("logistic 回归暂不提供残差诊断图（可改用校准曲线/ROC 评估）")
        coefficient_table = _make_table("regression_coef", "回归系数（logistic，OR）", [
            ("term", "变量", "str"),
            ("beta", "β", "num"),
            ("se", "标准误", "num"),
            ("stat", "z", "num"),
            ("p", "p", "p"),
            ("or", "OR", "num"),
            ("or_low", "OR 95%CI 下限", "num"),
            ("or_high", "OR 95%CI 上限", "num"),
        ], coefficient_rows)
        method = ("二分类 logistic 回归（statsmodels.Logit，最大似然，Newton-Raphson）；"
                  "OR = exp(β)，95%CI 为 Wald 区间；伪 R² 为 McFadden 定义")
        key_numbers = {
            "n": n_used,
            "n_events": n_events,
            "pseudo_r2": pseudo_r2,
            "lr_p": lr_p,
            "n_terms": int(design_used.shape[1]),
        }

    fit_table = _make_table("regression_fit", "模型概况", [
        ("metric", "指标", "str"),
        ("value", "数值", "num"),
        ("note", "说明", "str"),
    ], fit_rows)
    extra_lines = []
    if reference_text:
        extra_lines.append(f"分类变量编码：{reference_text}（one-hot，drop_first）")
    if x_cols:
        extra_lines.append(f"自变量：{', '.join(x_cols)}")
    if cov_cols:
        extra_lines.append(f"协变量：{', '.join(cov_cols)}")
    return _pack(
        title="线性回归" if binary is None else "Logistic 回归",
        method=method,
        bullets=bullets,
        key_numbers=key_numbers,
        tables=[fit_table, coefficient_table],
        figures=figures,
        warnings=warnings,
        extra_md="\n".join(f"- {line}" for line in extra_lines),
    )


# ---------------------------------------------------------------------------
# chi2：列联表卡方 / Fisher
# ---------------------------------------------------------------------------


def _categorical_pair(ctx: _Ctx) -> tuple[str, str]:
    df = ctx.df
    roles = ctx.roles()
    x_col = _param_col(ctx, "x") or roles.get("x")
    if not x_col:
        candidates = _categorical_columns(ctx)
        if not candidates:
            raise C.prerequisite("卡方检验需要分类变量：数据里没有可用分类列")
        x_col = candidates[0]
        ctx.warnings.append(f"未指定行变量，已自动使用：{x_col}")
    if x_col not in df.columns:
        raise C.bad_request(f"行变量列「{x_col}」不在数据里")
    y_col = _param_col(ctx, "y") or roles.get("group")
    if not y_col or y_col == x_col or y_col not in df.columns:
        candidates = [c for c in _categorical_columns(ctx) if c != x_col]
        if not candidates:
            raise C.bad_request("卡方检验需要两个分类变量：请用参数 y 指定第二个分类列")
        y_col = candidates[0]
        ctx.warnings.append(f"未指定列变量，已自动使用：{y_col}")
    for column, label in ((x_col, "行变量"), (y_col, "列变量")):
        dtype = P.infer_dtype(df[column])
        unique = int(df[column].nunique(dropna=True))
        if dtype == "numeric" and unique > 10:
            raise C.bad_request(
                f"{label}「{column}」是连续变量（{unique} 个取值），卡方检验需要分类变量"
            )
    return str(x_col), str(y_col)


def _run_chi2(ctx: _Ctx) -> dict:
    params = ctx.params
    x_col, y_col = _categorical_pair(ctx)
    df = ctx.df
    work = df[[x_col, y_col]].dropna()
    if work.empty:
        raise C.prerequisite("过滤缺失后没有可用观测，无法构建列联表")
    work = work.copy()
    work[x_col] = work[x_col].astype(str).str.strip()
    work[y_col] = work[y_col].astype(str).str.strip()
    row_levels = [str(v) for v in pd.unique(work[x_col])]
    col_levels = [str(v) for v in pd.unique(work[y_col])]
    if len(row_levels) < 2 or len(col_levels) < 2:
        raise C.prerequisite(
            f"列联表需要至少 2×2（当前 {len(row_levels)}×{len(col_levels)}）；建议先合并类别"
        )
    if len(row_levels) > 12 or len(col_levels) > 12:
        raise C.prerequisite(
            f"类别过多（{len(row_levels)}×{len(col_levels)}），不适合卡方检验；建议先合并类别"
        )
    observed = np.zeros((len(row_levels), len(col_levels)), dtype=float)
    for i, row_level in enumerate(row_levels):
        row_mask = work[x_col] == row_level
        for j, col_level in enumerate(col_levels):
            observed[i, j] = float(np.sum(row_mask & (work[y_col] == col_level)))
    n = float(observed.sum())
    if n < 4:
        raise C.prerequisite(f"有效观测过少（n={int(n)}），无法做卡方/Fisher 检验")

    is_2x2 = observed.shape == (2, 2)
    method_req = _p_select(params, "method", ("auto", "chi2", "fisher"), "auto")
    if method_req == "fisher" and not is_2x2:
        raise C.bad_request(f"Fisher 精确检验仅支持 2×2 列联表（当前 {observed.shape[0]}×{observed.shape[1]}）")
    try:
        chi2, p_chi2, dof, expected = sstats.chi2_contingency(observed, correction=is_2x2)
    except ValueError as exc:
        raise C.prerequisite(f"卡方检验无法进行：{exc}") from exc
    fisher = None
    if is_2x2:
        try:
            odds_ratio, p_fisher = sstats.fisher_exact(observed)
            fisher = {"or": float(odds_ratio), "p": float(p_fisher)}
        except Exception:  # noqa: BLE001
            fisher = None
            ctx.warnings.append("Fisher 精确检验计算失败，仅报告卡方结果")
    expected_min = float(np.min(expected))
    low_cells = int(np.sum(expected < 5))
    low_pct = low_cells / expected.size * 100.0
    denominator = float(min(observed.shape[0] - 1, observed.shape[1] - 1))
    cramers_v = math.sqrt(float(chi2) / (n * denominator)) if n > 0 and denominator > 0 else None

    if low_pct > 20.0:
        ctx.warnings.append(
            f"{low_cells}/{expected.size} 个格子期望频数<5（{low_pct:.0f}%），卡方近似不可靠；"
            "建议用 Fisher 精确检验或合并类别"
        )
    if expected_min < 1.0:
        ctx.warnings.append(f"最小期望频数 {C.fmt_num(expected_min)}<1，卡方结果不可靠")
    if method_req == "fisher" and fisher is None:
        raise C.prerequisite("Fisher 精确检验失败，无法按指定方法给出结果")

    primary_p = float(fisher["p"]) if (is_2x2 and fisher is not None) else float(p_chi2)
    primary_name = "Fisher exact test" if (is_2x2 and fisher is not None) else "Pearson chi-square"

    # 列联表
    contingency_columns: list[dict] = [{"key": "row", "label": x_col, "type": "str"}]
    for index, col_level in enumerate(col_levels):
        contingency_columns.append({"key": f"c{index + 1}", "label": col_level, "type": "num"})
    contingency_columns.append({"key": "total", "label": "合计", "type": "num"})
    contingency_rows: list[dict] = []
    for i, row_level in enumerate(row_levels):
        row: dict[str, Any] = {"row": row_level}
        for j in range(len(col_levels)):
            row[f"c{j + 1}"] = int(round(observed[i, j]))
        row["total"] = int(round(float(observed[i].sum())))
        contingency_rows.append(row)
    total_row: dict[str, Any] = {"row": "合计"}
    for j in range(len(col_levels)):
        total_row[f"c{j + 1}"] = int(round(float(observed[:, j].sum())))
    total_row["total"] = int(round(n))
    contingency_rows.append(total_row)
    contingency_table = C.table("chi2_contingency", f"列联表（{x_col} × {y_col}）",
                               contingency_columns, contingency_rows)

    expected_rows = []
    for i, row_level in enumerate(row_levels):
        row = {"row": row_level}
        for j in range(len(col_levels)):
            row[f"c{j + 1}"] = round(float(expected[i, j]), 2)
        expected_rows.append(row)
    expected_rows.append({"row": "合计"})
    expected_table = _make_table("chi2_expected", "期望频数", [
        ("row", x_col, "str"),
        *[(f"c{index + 1}", col_level, "num") for index, col_level in enumerate(col_levels)],
    ], expected_rows, note="期望频数<5 的格子超过 20% 时卡方近似不可靠")

    test_rows = [
        {"metric": "Pearson 卡方 χ²", "value": float(chi2), "p_value": float(p_chi2),
         "note": f"df={int(dof)}" + ("，Yates 连续性校正" if is_2x2 else "")},
        {"metric": "Fisher 精确检验", "value": fisher["or"] if fisher else None,
         "p_value": fisher["p"] if fisher else None,
         "note": "OR（2×2；仅 2×2 可用）" if is_2x2 else "仅支持 2×2"},
        {"metric": "Cramér's V", "value": cramers_v, "p_value": None,
         "note": _effect_label_r(cramers_v) + "关联强度"},
        {"metric": "样本量 n", "value": n, "p_value": None, "note": ""},
        {"metric": "最小期望频数", "value": expected_min, "p_value": None,
         "note": "≥5 时卡方近似良好"},
        {"metric": "期望<5 的格子占比（%）", "value": low_pct, "p_value": None,
         "note": f"{low_cells}/{expected.size} 个格子"},
    ]
    test_table = _make_table("chi2_tests", "检验结果", [
        ("metric", "指标", "str"),
        ("value", "统计量", "num"),
        ("p_value", "p", "p"),
        ("note", "说明", "str"),
    ], test_rows, note=f"主检验：{primary_name}")

    bullets = [
        f"{x_col} × {y_col}：{len(row_levels)}×{len(col_levels)} 列联表，n={int(n)}",
        f"Pearson χ²={C.fmt_num(chi2)}，df={int(dof)}，{_p_phrase(p_chi2)}"
        + ("（Yates 连续性校正）" if is_2x2 else ""),
    ]
    if fisher is not None:
        bullets.append(
            f"Fisher 精确检验：{_p_phrase(fisher['p'])}，OR={C.fmt_num(fisher['or'])}"
        )
    bullets.append(f"Cramér's V={C.fmt_num(cramers_v)}（{_effect_label_r(cramers_v)}关联强度）")
    if len(row_levels) <= 6:
        target = col_levels[-1]
        pieces = []
        for i, row_level in enumerate(row_levels):
            row_total = float(observed[i].sum())
            if row_total > 0:
                pieces.append(f"{row_level} {observed[i, -1] / row_total * 100.0:.1f}%")
        bullets.append(f"「{target}」占比：" + "、".join(pieces))
    bullets.append(
        f"最小期望频数 {C.fmt_num(expected_min)}"
        + ("（满足卡方前提）" if expected_min >= 5 else "（部分格子偏小，结论谨慎）")
    )
    key_numbers = {
        "n": int(n),
        "chi2": float(chi2),
        "dof": int(dof),
        "p": primary_p,
        "p_chi2": float(p_chi2),
        "p_fisher": fisher["p"] if fisher else None,
        "cramers_v": cramers_v,
        "min_expected": expected_min,
        "primary_test": primary_name,
    }
    return _pack(
        title="卡方 / Fisher 检验",
        method="Pearson 卡方检验（scipy.stats.chi2_contingency，2×2 用 Yates 连续性校正）"
               + ("；2×2 时以 Fisher 精确检验（scipy.stats.fisher_exact）为主" if is_2x2 else "")
               + "；效应量 Cramér's V = √(χ²/(n·min(r−1,c−1)))",
        bullets=bullets,
        key_numbers=key_numbers,
        tables=[contingency_table, expected_table, test_table],
        annotation=None,
        warnings=ctx.warnings,
    )


# ---------------------------------------------------------------------------
# normality：单变量正态性
# ---------------------------------------------------------------------------


def _skew_kurt_tests(values: np.ndarray) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return None, None
    skew_p = kurt_p = None
    try:
        skew_p = float(sstats.skewtest(arr).pvalue)
    except Exception:  # noqa: BLE001
        skew_p = None
    try:
        kurt_p = float(sstats.kurtosistest(arr).pvalue)
    except Exception:  # noqa: BLE001
        kurt_p = None
    return skew_p, kurt_p


def _normality_figure(ctx: _Ctx, name: str, values: np.ndarray) -> dict | None:
    with C.style_context(ctx.style):
        fig, axes = C.new_fig(ctx.style, 1, 2)
        ax1, ax2 = np.ravel(axes)[:2]
        color = C.color_list(ctx.style, 1)[0]
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        edges = np.histogram_bin_edges(arr, bins="auto")
        ax1.hist(arr, bins=edges, color=color, alpha=float(ctx.style.get("alpha", 0.85)),
                 edgecolor="white", linewidth=0.6)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        if sd > 0 and len(edges) > 1:
            width = float(edges[1] - edges[0])
            xs = np.linspace(float(arr.min()), float(arr.max()), 200)
            ax1.plot(xs, sstats.norm.pdf(xs, mean, sd) * arr.size * width,
                     color="#B24745", linewidth=float(ctx.style.get("line_width", 1.4)))
        ax1.set_xlabel(name)
        ax1.set_ylabel("Count")
        ax1.set_title("Histogram + normal fit",
                      fontsize=float(ctx.style.get("title_size", 14)) * 0.72)
        _qq_plot(ax2, arr, color, float(ctx.style.get("point_size", 28)))
        ax2.set_title("Normal Q-Q", fontsize=float(ctx.style.get("title_size", 14)) * 0.72)
        ax2.set_xlabel("Theoretical quantiles")
        ax2.set_ylabel("Sample quantiles")
        C.finalize_figure(fig, ctx.style, title=f"Normality: {name}", legend=False)
        return _save(fig, ctx, f"stat_normality_{C.slug(name)}", f"Normality: {name}")


def _run_normality(ctx: _Ctx) -> dict:
    params = ctx.params
    df = ctx.df
    roles = ctx.roles()
    columns = _param_cols(ctx, "columns")
    if not columns:
        y_col = roles.get("y")
        if y_col and y_col in df.columns:
            columns = [str(y_col)]
    if not columns:
        numeric = _numeric_columns(ctx)
        columns = numeric[:8]
        if len(numeric) > 8:
            ctx.warnings.append(f"数值列共 {len(numeric)} 个，仅检验前 8 个；可用参数 columns 指定")
    if not columns:
        raise C.prerequisite("数据里没有数值列，无法做正态性检验")
    invalid = [c for c in columns if c not in df.columns or P.infer_dtype(df[c]) != "numeric"]
    if invalid:
        raise C.bad_request(f"列 {'、'.join(invalid)} 不是数值列，无法做正态性检验")

    rows: list[dict] = []
    bullets: list[str] = []
    normal_count = 0
    plotted = 0
    max_plots = _p_int(params, "max_plots", 4, 1, 12)
    figures: list[dict] = []
    for column in columns:
        series = P.to_numeric(df[column])
        values = series.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        info = _summary_stats(finite)
        w_value = _shapiro_w(finite)
        p_value = _shapiro_p(finite)
        skew, kurt = _skew_kurt(finite)
        skew_p, kurt_p = _skew_kurt_tests(finite)
        if p_value is None:
            conclusion = "样本量不足（3–5000）" if info["n"] < 3 else "n>5000，Shapiro 不适用"
        elif p_value >= 0.05:
            conclusion = "未拒绝正态"
            normal_count += 1
        else:
            conclusion = "偏离正态"
        rows.append({
            "variable": column,
            "n": info["n"],
            "missing": int(series.isna().sum()),
            "mean": info["mean"],
            "sd": info["sd"],
            "median": info["median"],
            "skew": skew,
            "kurtosis": kurt,
            "shapiro_w": w_value,
            "shapiro_p": p_value,
            "skew_p": skew_p,
            "kurt_p": kurt_p,
            "conclusion": conclusion,
        })
        if len(bullets) < 4:
            bullets.append(
                f"{column}：n={info['n']}，W={C.fmt_num(w_value)}，{_p_phrase(p_value)}，"
                f"偏度={C.fmt_num(skew)}，峰度={C.fmt_num(kurt)} → {conclusion}"
            )
        if _p_bool(params, "plot", False) and plotted < max_plots:
            figure = _try(ctx, f"{column} 正态性图", lambda col=column, val=finite: _normality_figure(
                ctx, col, val))
            if figure:
                figures.append(figure)
                plotted += 1

    if not bullets:
        bullets.append(f"共检验 {len(columns)} 个数值列")
    bullets.insert(0, f"正态性检验 {len(columns)} 列（Shapiro-Wilk，α=0.05）："
                      f"{normal_count} 列未拒绝正态，{len(columns) - normal_count} 列偏离正态")
    if normal_count == len(columns):
        bullets.append(f"结论：{len(columns)}/{len(columns)} 列未拒绝正态，可优先使用参数检验（t/F 检验）")
    elif normal_count == 0:
        bullets.append(f"结论：{len(columns)} 列均偏离正态，建议使用非参数检验（Mann-Whitney U / Kruskal-Wallis）")
    else:
        bullets.append(
            f"结论：{normal_count}/{len(columns)} 列未拒绝正态；偏离正态的变量建议做变换（log/Box-Cox）"
            "或改用非参数方法"
        )

    table = _make_table("normality", "正态性检验", [
        ("variable", "变量", "str"),
        ("n", "n", "num"),
        ("missing", "缺失", "num"),
        ("mean", "均值", "num"),
        ("sd", "标准差", "num"),
        ("median", "中位数", "num"),
        ("skew", "偏度", "num"),
        ("kurtosis", "峰度", "num"),
        ("shapiro_w", "Shapiro W", "num"),
        ("shapiro_p", "Shapiro p", "p"),
        ("skew_p", "偏度检验 p", "p"),
        ("kurt_p", "峰度检验 p", "p"),
        ("conclusion", "结论", "str"),
    ], rows, note="峰度为 Fisher 定义（正态=0）；偏度/峰度检验需 n≥8")
    key_numbers = {
        "n_columns": len(columns),
        "n_normal": normal_count,
        "min_shapiro_p": min((row["shapiro_p"] for row in rows if row["shapiro_p"] is not None),
                             default=None),
    }
    return _pack(
        title="正态性检验",
        method="Shapiro-Wilk 正态性检验（scipy.stats.shapiro）+ 偏度/峰度（scipy.stats.skew/kurtosis，"
               "无偏估计）+ D'Agostino 偏度/峰度检验（scipy.stats.skewtest/kurtosistest）",
        bullets=bullets,
        key_numbers=key_numbers,
        tables=[table],
        figures=figures,
        warnings=ctx.warnings,
    )


# ---------------------------------------------------------------------------
# roc：ROC 曲线
# ---------------------------------------------------------------------------


def _auc_label(auc_value: float) -> str:
    if auc_value >= 0.9:
        return "优秀"
    if auc_value >= 0.8:
        return "好"
    if auc_value >= 0.7:
        return "较好"
    if auc_value >= 0.6:
        return "一般"
    return "较差"


def _roc_figure(ctx: _Ctx, fpr: np.ndarray, tpr: np.ndarray, auc_value: float,
                ci: tuple[float, float] | None, threshold: float, sens: float,
                spec: float) -> dict | None:
    with C.style_context(ctx.style):
        fig, ax = C.new_fig(ctx.style)
        color = C.color_list(ctx.style, 1)[0]
        label = f"AUC = {auc_value:.3f}"
        if ci is not None:
            label += f" (95%CI {ci[0]:.3f}–{ci[1]:.3f})"
        ax.plot(fpr, tpr, color=color, linewidth=float(ctx.style.get("line_width", 1.4)) + 0.3,
                label=label)
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=0.9, label="Chance")
        ax.scatter([1.0 - spec], [sens], s=float(ctx.style.get("point_size", 28)) * 1.6,
                   color="#B24745", zorder=5, edgecolor="white",
                   label=f"Youden cutoff = {threshold:.4g}")
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        C.finalize_figure(fig, ctx.style, title="ROC curve")
        return _save(fig, ctx, "stat_roc_curve", "ROC curve")


def _run_roc(ctx: _Ctx) -> dict:
    params = ctx.params
    df = ctx.df
    roles = ctx.roles()
    labels = ctx.labels()
    label_col = _param_col(ctx, "label")
    score_col = _param_col(ctx, "score")
    frame = ctx.data.get("tidy")

    if label_col and score_col:
        work = pd.DataFrame({
            "label": df[label_col],
            "score": P.to_numeric(df[score_col]),
        })
        label_name, score_name = label_col, score_col
    elif label_col or score_col:
        missing_name = "score" if label_col else "label"
        raise C.bad_request(f"ROC 需要同时指定标签列与分数列：缺少参数 {missing_name}")
    else:
        if frame is None or "__y" not in frame.columns or "__group" not in frame.columns:
            raise C.bad_request(
                "ROC 需要标签列与分数列：请用参数 label/score 指定，或指定 group（标签）与 y（分数）角色"
            )
        work = pd.DataFrame({
            "label": frame["__group"].astype(object),
            "score": frame["__y"].astype(float),
        })
        label_name = str(labels.get("group") or "group")
        score_name = str(labels.get("y") or "score")

    work = work.dropna(subset=["label", "score"])
    work = work.loc[np.isfinite(work["score"].to_numpy(dtype=float))]
    if work.empty:
        raise C.prerequisite("过滤缺失后没有可用于 ROC 的观测")
    score = work["score"].to_numpy(dtype=float)
    label_series = work["label"]

    positive_name: str
    if P.infer_dtype(label_series) == "numeric":
        unique = sorted({float(v) for v in pd.unique(label_series)})
        if len(unique) < 2:
            raise C.prerequisite(f"标签列「{label_name}」只有 1 个取值，无法计算 ROC")
        if len(unique) == 2:
            positive_text = _p_str(params, "positive")
            if set(unique) <= {0.0, 1.0}:
                positive_value = 0.0 if positive_text in ("0", "0.0") else 1.0
            else:
                try:
                    positive_value = float(positive_text) if positive_text else unique[1]
                except ValueError as exc:
                    raise C.bad_request(f"positive 参数「{positive_text}」无法解析为数值") from exc
            if positive_value not in unique:
                raise C.bad_request(f"positive={positive_value} 不在标签取值中（{unique}）")
        else:
            raise C.prerequisite(
                f"标签列「{label_name}」是连续/多分类变量（{len(unique)} 个取值），"
                "ROC 需要二分类标签（或先用阈值二值化）"
            )
        y = (label_series.to_numpy(dtype=float) == positive_value).astype(int)
        positive_name = C.fmt_num(positive_value)
    else:
        levels = [str(v) for v in pd.unique(label_series.astype(str))]
        if len(levels) < 2:
            raise C.prerequisite(f"标签列「{label_name}」只有 1 个类别，无法计算 ROC")
        positive = _p_str(params, "positive")
        if not positive:
            if len(levels) > 2:
                raise C.prerequisite(
                    f"标签列「{label_name}」是多分类（{len(levels)} 类），请用参数 positive 指定阳性类别"
                )
            positive = _guess_positive(levels)
        if positive not in levels:
            raise C.bad_request(f"positive 参数「{positive}」不在标签类别中（{'、'.join(levels)}）")
        y = (label_series.astype(str) == positive).astype(int).to_numpy()
        positive_name = positive

    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos < 2 or n_neg < 2:
        raise C.prerequisite(
            f"阳性/阴性样本不足（阳性 {n_pos}，阴性 {n_neg}），至少各需 2 例才能计算 ROC"
        )
    if n_pos < 5 or n_neg < 5:
        ctx.warnings.append(f"阳性/阴性样本偏少（{n_pos}/{n_neg}），AUC 置信区间会很宽")

    from sklearn.metrics import roc_auc_score, roc_curve

    try:
        fpr, tpr, thresholds = roc_curve(y, score)
        auc_value = float(roc_auc_score(y, score))
    except ValueError as exc:
        raise C.prerequisite(f"ROC 计算失败：{exc}") from exc
    reps = _p_int(params, "bootstrap", 1000, 100, 5000)
    ci = _try(ctx, "AUC bootstrap 95%CI", lambda: _bootstrap_auc_ci(y, score, reps), None)

    youden = tpr - fpr
    index = int(np.argmax(youden))
    if len(thresholds) > 1 and not math.isfinite(float(thresholds[index])):
        index = int(np.argmax(youden[1:])) + 1
    threshold = float(thresholds[index])
    sensitivity = float(tpr[index])
    specificity = float(1.0 - fpr[index])
    predicted = score >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None
    accuracy = (tp + tn) / max(1, int(y.size))

    figures: list[dict] = []
    if _p_bool(params, "plot", True):
        figure = _try(ctx, "ROC 曲线图", lambda: _roc_figure(
            ctx, np.asarray(fpr), np.asarray(tpr), auc_value, ci,
            threshold, sensitivity, specificity))
        if figure:
            figures.append(figure)

    metric_rows = [
        {"metric": "AUC", "value": auc_value, "note": _auc_label(auc_value)},
        {"metric": "AUC 95%CI 下限", "value": ci[0] if ci else None,
         "note": f"bootstrap {reps} 次，seed=0"},
        {"metric": "AUC 95%CI 上限", "value": ci[1] if ci else None,
         "note": f"bootstrap {reps} 次，seed=0"},
        {"metric": "样本量 n", "value": int(y.size), "note": f"阳性 {n_pos} / 阴性 {n_neg}"},
        {"metric": "最佳阈值（Youden）", "value": threshold, "note": "max(sensitivity − (1−specificity))"},
        {"metric": "敏感度 Sensitivity", "value": sensitivity, "note": f"{tp}/({tp}+{fn})"},
        {"metric": "特异度 Specificity", "value": specificity, "note": f"{tn}/({tn}+{fp})"},
        {"metric": "阳性预测值 PPV", "value": ppv, "note": f"{tp}/({tp}+{fp})"},
        {"metric": "阴性预测值 NPV", "value": npv, "note": f"{tn}/({tn}+{fn})"},
        {"metric": "准确率 Accuracy", "value": accuracy, "note": f"({tp}+{tn})/{int(y.size)}"},
    ]
    metric_table = _make_table("roc_metrics", "ROC 与阈值表现", [
        ("metric", "指标", "str"),
        ("value", "数值", "num"),
        ("note", "说明", "str"),
    ], metric_rows)
    confusion_rows = [
        {"actual": f"阳性（{positive_name}）", "pred_pos": tp, "pred_neg": fn},
        {"actual": "阴性", "pred_pos": fp, "pred_neg": tn},
    ]
    confusion_table = _make_table("roc_confusion", f"最佳阈值 {C.fmt_num(threshold)} 处的混淆矩阵", [
        ("actual", "实际类别", "str"),
        ("pred_pos", "预测阳性", "num"),
        ("pred_neg", "预测阴性", "num"),
    ], confusion_rows)

    bullets = [
        f"分数「{score_name}」预测标签「{label_name}」（阳性={positive_name}）："
        f"n={int(y.size)}（阳性 {n_pos}，阴性 {n_neg}）",
        f"AUC={C.fmt_num(auc_value)}（{_auc_label(auc_value)}判别能力）"
        + (f"，bootstrap {reps} 次 95%CI {C.fmt_num(ci[0])}–{C.fmt_num(ci[1])}" if ci else "，95%CI 计算失败"),
        f"Youden 最佳阈值 {C.fmt_num(threshold)}：敏感度 {sensitivity:.3f}，特异度 {specificity:.3f}，"
        f"PPV {C.fmt_num(ppv)}，NPV {C.fmt_num(npv)}，准确率 {accuracy:.3f}",
    ]
    key_numbers = {
        "n": int(y.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc": auc_value,
        "auc_low": ci[0] if ci else None,
        "auc_high": ci[1] if ci else None,
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
    }
    return _pack(
        title="ROC 曲线",
        method="ROC 分析（scikit-learn roc_curve / roc_auc_score）；AUC 95%CI 为分层 percentile bootstrap"
               f"（{reps} 次，seed=0）；最佳阈值按 Youden 指数 J = sensitivity + specificity − 1",
        bullets=bullets,
        key_numbers=key_numbers,
        tables=[metric_table, confusion_table],
        figures=figures,
        warnings=ctx.warnings,
    )


# ---------------------------------------------------------------------------
# survival：KM + log-rank + Cox
# ---------------------------------------------------------------------------


def _matches_hint(key: Any, hints: Sequence[str]) -> bool:
    lowered = str(key).lower()
    return any(str(hint).lower() in lowered for hint in hints)


def _event_array(series: pd.Series) -> np.ndarray:
    numeric = P.to_numeric(series)
    valid = numeric.dropna()
    if len(valid) and float(valid.isin([0, 1]).mean()) >= 0.8:
        return numeric.to_numpy(dtype=float)
    text = series.astype(str).str.strip().str.lower()
    positive = {"dead", "death", "event", "yes", "true", "1", "复发", "死亡", "进展",
                "progression", "progressed", "recurrence", "1.0"}
    values = text.isin(positive).astype(float)
    values[series.isna()] = np.nan
    return values.to_numpy(dtype=float)


def _km_figure(ctx: _Ctx, groups: list[tuple[str, np.ndarray, np.ndarray]],
               time: np.ndarray, event: np.ndarray, p_value: float | None,
               time_label: str, risk_table: bool, ci: bool) -> dict | None:
    from lifelines import KaplanMeierFitter

    with C.style_context(ctx.style):
        fig, ax = C.new_fig(ctx.style)
        line_width = float(ctx.style.get("line_width", 1.4))
        if groups:
            colors = C.palette_colors(ctx.style, [label for label, _, _ in groups])
            for label, t_values, e_values in groups:
                fitter = KaplanMeierFitter()
                fitter.fit(t_values, e_values, label=f"{label} (n={t_values.size})")
                try:
                    fitter.plot_survival_function(
                        ax=ax, ci_show=ci, color=colors.get(label),
                        linewidth=line_width, ci_alpha=0.15,
                    )
                except TypeError:
                    fitter.plot_survival_function(ax=ax, ci_show=False,
                                                  color=colors.get(label), linewidth=line_width)
        else:
            fitter = KaplanMeierFitter()
            fitter.fit(time, event, label=f"All (n={time.size})")
            fitter.plot_survival_function(ax=ax, ci_show=ci,
                                          color=C.color_list(ctx.style, 1)[0],
                                          linewidth=line_width)
        ax.set_xlabel(time_label)
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0.0, 1.02)
        title = "Kaplan-Meier curve"
        if p_value is not None:
            p_text = C.fmt_p(p_value)
            p_phrase = f"P {p_text}" if p_text.startswith("<") else f"P = {p_text}"
            title += f"   log-rank {p_phrase}"
        if risk_table:
            max_time = float(np.max(time)) if time.size else 0.0
            points = [max_time * factor for factor in (0.0, 0.25, 0.5, 0.75, 1.0)]
            text_size = max(6.0, float(ctx.style.get("tick_size", 10)) - 3.0)
            header = "At risk        " + "   ".join(f"{value:.0f}" for value in points)
            ax.text(-0.02, -0.11, header, transform=ax.transAxes, ha="left", va="top",
                    fontsize=text_size, color="#333333")
            rows = groups if groups else [("All", time, event)]
            for offset, (label, t_values, _e_values) in enumerate(rows[:8]):
                counts = [int(np.sum(t_values >= point)) for point in points]
                ax.text(-0.02, -0.17 - offset * 0.055,
                        f"{label}: " + "   ".join(str(value) for value in counts),
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=text_size, color="#333333")
        C.finalize_figure(fig, ctx.style, title=title)
        return _save(fig, ctx, "stat_survival_km", "Kaplan-Meier curve")


def _run_survival(ctx: _Ctx) -> dict:
    params = ctx.params
    if ctx.kind == "matrix":
        raise C.prerequisite("矩阵数据不做生存分析；请使用 tidy 数据（每行一个样本，含时间与事件列）")
    df = ctx.df
    roles = ctx.roles()
    warnings = ctx.warnings
    time_col = _param_col(ctx, "time") or roles.get("time")
    event_col = _param_col(ctx, "event") or roles.get("event")
    group_col = _param_col(ctx, "group") or roles.get("group")

    if not time_col:
        for column in df.columns:
            if _matches_hint(column, P.TIME_HINTS):
                valid = P.to_numeric(df[column])
                if int(valid.notna().sum()) >= 3:
                    time_col = str(column)
                    break
    if not event_col:
        for column in df.columns:
            if column == time_col:
                continue
            candidate = P.to_numeric(df[column]).dropna()
            if _matches_hint(column, P.EVENT_HINTS) and len(candidate) and \
                    float(candidate.isin([0, 1]).mean()) >= 0.8:
                event_col = str(column)
                break
    if not time_col:
        raise C.prerequisite("生存分析需要时间列：请用参数 time 指定（或列名含 time/survival/months 等）")
    if not event_col:
        raise C.prerequisite("生存分析需要事件列（0/1）：请用参数 event 指定（或列名含 event/status/death 等）")
    if group_col and group_col in (time_col, event_col):
        group_col = None

    time_values = P.to_numeric(df[time_col]).to_numpy(dtype=float)
    event_values = _event_array(df[event_col])
    mask = (np.isfinite(time_values) & np.isfinite(event_values)
            & (time_values > 0) & np.isin(event_values, [0.0, 1.0]))
    if group_col and group_col in df.columns:
        mask = mask & df[group_col].notna().to_numpy()
    n_total = int(len(df))
    n_valid = int(np.sum(mask))
    dropped = n_total - n_valid
    if dropped:
        warnings.append(f"丢弃 {dropped} 行（时间为空/≤0 或事件不是 0/1）")
    if n_valid < 5:
        raise C.prerequisite(f"有效样本过少（n={n_valid}），无法做生存分析")
    time = time_values[mask]
    event = event_values[mask].astype(int)
    n_events = int(event.sum())
    if n_events < 2:
        raise C.prerequisite(f"事件数只有 {n_events} 个（<2），无法估计生存曲线或做 log-rank 检验")

    groups: list[tuple[str, np.ndarray, np.ndarray]] = []
    if group_col and group_col in df.columns:
        group_values = df[group_col].astype(str).to_numpy()[mask]
        levels = [str(v) for v in pd.unique(group_values)]
        if len(levels) > 8:
            raise C.prerequisite(f"分组列「{group_col}」有 {len(levels)} 个水平（>8），KM 曲线不可读；请先合并分组")
        candidate_groups = []
        for level in levels:
            level_mask = group_values == level
            if int(level_mask.sum()) < 2:
                warnings.append(f"分组「{level}」有效样本 <2，已剔除")
                continue
            candidate_groups.append((level, time[level_mask], event[level_mask]))
        if len(candidate_groups) >= 2:
            groups = candidate_groups
        else:
            warnings.append("有效分组不足 2 个，改为整体生存曲线")

    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    p_value: float | None = None
    logrank_stat: float | None = None
    logrank_df: int | None = None
    group_rows: list[dict] = []
    if groups:
        try:
            if len(groups) == 2:
                result = logrank_test(groups[0][1], groups[1][1],
                                      event_observed_A=groups[0][2],
                                      event_observed_B=groups[1][2])
                logrank_df = 1
            else:
                label_index = {label: index for index, (label, _, _) in enumerate(groups)}
                group_labels = df[group_col].astype(str).to_numpy()[mask]
                keep_rows = np.array([label in label_index for label in group_labels])
                codes = np.array([label_index[label] for label in group_labels[keep_rows]])
                result = multivariate_logrank_test(time[keep_rows], codes, event[keep_rows])
                logrank_df = len(groups) - 1
            p_value = float(result.p_value)
            logrank_stat = float(result.test_statistic)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"log-rank 检验失败：{type(exc).__name__}: {exc}")
            p_value = None
        for label, t_values, e_values in groups:
            fitter = KaplanMeierFitter()
            try:
                fitter.fit(t_values, e_values)
                median = float(fitter.median_survival_time_)
                if not math.isfinite(median):
                    median = None
            except Exception:  # noqa: BLE001
                median = None
            group_rows.append({
                "group": label,
                "n": int(t_values.size),
                "events": int(e_values.sum()),
                "censored": int(t_values.size - e_values.sum()),
                "median_survival": median,
            })
    else:
        fitter = KaplanMeierFitter()
        fitter.fit(time, event)
        median = float(fitter.median_survival_time_)
        group_rows.append({
            "group": "全部",
            "n": int(time.size),
            "events": n_events,
            "censored": int(time.size - n_events),
            "median_survival": None if not math.isfinite(median) else median,
        })

    # Cox 回归
    cox_rows: list[dict] = []
    cox_note = ""
    covariate_cols = [c for c in _param_cols(ctx, "covariates") if c not in (time_col, event_col)]
    cox_concordance: float | None = None
    if covariate_cols:
        numeric_terms = [c for c in covariate_cols if P.infer_dtype(df[c]) == "numeric"]
        factor_terms = [c for c in covariate_cols if c not in numeric_terms]
        design, dropped, reference_notes = _design_frame(df, numeric_terms, factor_terms)
        if dropped:
            warnings.append(f"Cox 协变量被剔除：{'、'.join(dropped)}")
        if design.shape[1] == 0:
            raise C.prerequisite("Cox 回归没有可用协变量（全部常数或单一水平）")
        cox_data = pd.DataFrame({
            "__time": time_values,
            "__event": event_values,
        })
        cox_data = pd.concat([cox_data.reset_index(drop=True),
                              design.reset_index(drop=True)], axis=1)
        cox_mask = (cox_data["__time"].notna() & cox_data["__event"].notna()
                    & cox_data["__time"].gt(0) & cox_data["__event"].isin([0, 1])
                    & cox_data.drop(columns=["__time", "__event"]).notna().all(axis=1))
        cox_data = cox_data.loc[cox_mask]
        n_cox = int(len(cox_data))
        events_cox = int(cox_data["__event"].sum()) if n_cox else 0
        if n_cox < 20 or events_cox < 5:
            warnings.append(f"Cox 回归样本不足（n={n_cox}，事件={events_cox}），已跳过")
        else:
            try:
                from lifelines import CoxPHFitter

                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(cox_data, duration_col="__time", event_col="__event")
                summary = cph.summary
                for term in design.columns:
                    if term not in summary.index:
                        continue
                    row = summary.loc[term]
                    cox_rows.append({
                        "term": str(term),
                        "hr": float(np.exp(row["coef"])),
                        "ci_low": float(np.exp(row["coef lower 95%"])),
                        "ci_high": float(np.exp(row["coef upper 95%"])),
                        "z": float(row["z"]),
                        "p": float(row["p"]),
                    })
                cox_concordance = float(getattr(cph, "concordance_index_", float("nan")))
                cox_note = f"n={n_cox}，事件={events_cox}，penalizer=0.01"
                if reference_notes:
                    cox_note += "；" + "；".join(reference_notes)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Cox 回归拟合失败：{type(exc).__name__}: {exc}")

    figures: list[dict] = []
    if _p_bool(params, "plot", True):
        figure = _try(ctx, "KM 生存曲线", lambda: _km_figure(
            ctx, groups, time, event, p_value, str(time_col),
            _p_bool(params, "risk_table", True), _p_bool(params, "ci", True)))
        if figure:
            figures.append(figure)

    median_time = float(np.median(time))
    bullets = [
        f"共 {n_valid} 例（事件 {n_events}，删失 {n_valid - n_events}）；"
        f"随访时间中位数 {C.fmt_num(median_time)}（{time_col}）",
    ]
    if groups:
        pieces = []
        for row in group_rows:
            median_text = C.fmt_num(row["median_survival"]) if row["median_survival"] is not None else "未达到"
            pieces.append(f"{row['group']}：n={row['n']}，事件 {row['events']}，中位生存 {median_text}")
        bullets.append("；".join(pieces))
        if p_value is not None:
            bullets.append(
                f"log-rank 检验：χ²={C.fmt_num(logrank_stat)}，df={logrank_df}，{_p_phrase(p_value)}，"
                + ("组间生存差异显著" if p_value < 0.05 else "组间生存差异不显著（p≥0.05）")
            )
        else:
            bullets.append(f"log-rank 检验失败，仅报告描述性生存曲线（n={n_valid}，{len(groups)} 组）")
    else:
        row = group_rows[0]
        median_text = C.fmt_num(row["median_survival"]) if row["median_survival"] is not None else "未达到"
        bullets.append(f"整体中位生存时间 {median_text}；未指定分组，未做 log-rank 检验")
    if cox_rows:
        top = sorted(cox_rows, key=lambda item: item["p"])[:3]
        pieces = [
            f"{row['term']} HR={C.fmt_num(row['hr'])}（95%CI {C.fmt_num(row['ci_low'])}–"
            f"{C.fmt_num(row['ci_high'])}，{_p_phrase(row['p'])}）"
            for row in top
        ]
        bullets.append(
            "Cox 比例风险回归：" + "；".join(pieces)
            + (f"；一致性指数 C={C.fmt_num(cox_concordance)}" if cox_concordance is not None
               and math.isfinite(cox_concordance) else "")
        )

    tables: list[dict] = [
        _make_table("survival_groups", "分组生存概况", [
            ("group", "分组", "str"),
            ("n", "n", "num"),
            ("events", "事件数", "num"),
            ("censored", "删失数", "num"),
            ("median_survival", "中位生存时间", "num"),
        ], group_rows),
    ]
    if groups:
        tables.append(_make_table("survival_logrank", "log-rank 检验", [
            ("test", "检验", "str"),
            ("statistic", "统计量", "num"),
            ("df", "df", "num"),
            ("p", "p", "p"),
        ], [{
            "test": "Log-rank (Mantel-Cox)",
            "statistic": logrank_stat,
            "df": logrank_df,
            "p": p_value,
        }]))
    else:
        tables.append(_make_table("survival_logrank", "log-rank 检验", [
            ("test", "检验", "str"),
            ("statistic", "统计量", "num"),
            ("df", "df", "num"),
            ("p", "p", "p"),
        ], [{"test": "Log-rank (Mantel-Cox)", "statistic": None, "df": None, "p": None}],
            note="未指定分组，未做 log-rank 检验"))
    if cox_rows:
        tables.append(_make_table("survival_cox", "Cox 比例风险回归", [
            ("term", "变量", "str"),
            ("hr", "HR", "num"),
            ("ci_low", "95%CI 下限", "num"),
            ("ci_high", "95%CI 上限", "num"),
            ("z", "z", "num"),
            ("p", "p", "p"),
        ], cox_rows, note=cox_note))
    key_numbers = {
        "n": n_valid,
        "n_events": n_events,
        "n_groups": len(groups) if groups else 1,
        "p": p_value,
        "logrank_stat": logrank_stat,
        "concordance": cox_concordance,
    }
    return _pack(
        title="生存分析",
        method="Kaplan-Meier 生存估计与 log-rank 检验（lifelines.KaplanMeierFitter / "
               "lifelines.statistics.logrank_test / multivariate_logrank_test）；"
               "Cox 比例风险回归（lifelines.CoxPHFitter，penalizer=0.01 提升数值稳定性）",
        bullets=bullets,
        key_numbers=key_numbers,
        tables=tables,
        figures=figures,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# pca：主成分分析
# ---------------------------------------------------------------------------


def _wide_groups(ctx: _Ctx, wide: Mapping[str, Any],
                 samples: Sequence[str]) -> tuple[np.ndarray | None, str | None]:
    groups = wide.get("sample_groups")
    group_name = wide.get("group_name")
    if groups is not None and len(groups) == len(samples):
        return np.asarray(groups, dtype=object), group_name
    info = ctx.dataset.get("sample_info")
    if isinstance(info, pd.DataFrame) and "sample" in info.columns and len(info):
        candidate = (_p_str(ctx.params, "sample_group_col")
                     or ctx.roles().get("group")
                     or ("group" if "group" in info.columns else None))
        if candidate and candidate in info.columns:
            mapping = {str(sample): str(value) for sample, value in
                       zip(info["sample"].astype(str), info[candidate].astype(str))}
            values = [mapping.get(str(sample)) for sample in samples]
            if all(value is not None for value in values):
                return np.asarray(values, dtype=object), str(candidate)
    return None, None


def _pca_figure(ctx: _Ctx, scores: np.ndarray, ratios: np.ndarray,
                groups: np.ndarray | None, group_name: str | None) -> dict | None:
    with C.style_context(ctx.style):
        fig, ax = C.new_fig(ctx.style)
        point_size = float(ctx.style.get("point_size", 28))
        alpha = float(ctx.style.get("alpha", 0.85))
        if groups is not None:
            labels = [str(v) for v in pd.unique(pd.Series(groups))]
            colors = C.palette_colors(ctx.style, labels)
            for label in labels:
                mask = np.asarray([str(value) == label for value in groups])
                ax.scatter(scores[mask, 0], scores[mask, 1], s=point_size,
                           color=colors.get(label), alpha=alpha, edgecolor="white",
                           linewidth=0.4, label=f"{label} (n={int(mask.sum())})")
        else:
            ax.scatter(scores[:, 0], scores[:, 1], s=point_size,
                       color=C.color_list(ctx.style, 1)[0], alpha=alpha,
                       edgecolor="white", linewidth=0.4,
                       label=f"n={scores.shape[0]}")
        ax.axhline(0.0, color="#CCCCCC", linewidth=0.6, zorder=0)
        ax.axvline(0.0, color="#CCCCCC", linewidth=0.6, zorder=0)
        ax.set_xlabel(f"PC1 ({ratios[0] * 100.0:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({ratios[1] * 100.0:.1f}% variance)")
        title = "PCA score plot" + (f" ({group_name})" if group_name else "")
        C.finalize_figure(fig, ctx.style, title=title)
        return _save(fig, ctx, "stat_pca_scores", title)


def _run_pca(ctx: _Ctx) -> dict:
    params = ctx.params
    wide = ctx.data.get("wide")
    if not wide:
        raise C.prerequisite(
            "PCA 需要矩阵数据或 ≥2 个数值列；当前数据集无法构建宽表视图"
        )
    matrix = np.asarray(wide.get("matrix"), dtype=float)
    features = [str(f) for f in wide.get("features") or []]
    samples = [str(s) for s in wide.get("samples") or []]
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise C.prerequisite(
            f"PCA 至少需要 3 个样本 × 2 个特征（当前 {matrix.shape[0]}×{matrix.shape[1]}）"
        )
    max_features = _p_int(params, "max_features", 0, 0, 5000)
    if max_features and matrix.shape[1] > max_features:
        variances = np.nanvar(matrix, axis=0)
        keep = np.sort(np.argsort(-np.nan_to_num(variances))[:max_features])
        matrix = matrix[:, keep]
        features = [features[index] for index in keep]
        ctx.warnings.append(f"特征过多，仅保留方差最大的 {max_features} 个特征")

    finite_counts = np.sum(np.isfinite(matrix), axis=0)
    valid = finite_counts >= 2
    if int(valid.sum()) < 2:
        raise C.prerequisite("有效（非缺失）特征不足 2 个，无法做 PCA")
    if not bool(valid.all()):
        dropped = [features[index] for index in np.where(~valid)[0]]
        ctx.warnings.append(f"剔除有效值不足的特征：{'、'.join(dropped[:8])}")
        matrix = matrix[:, valid]
        features = [features[index] for index in np.where(valid)[0]]

    column_means = np.nanmean(matrix, axis=0)
    missing_index = np.where(~np.isfinite(matrix))
    matrix[missing_index] = np.take(column_means, missing_index[1])
    std = np.nanstd(matrix, axis=0, ddof=1)
    keep_variance = std > 0
    if int(keep_variance.sum()) < 2:
        raise C.prerequisite("所有特征的方差都为 0，无法做 PCA")
    if not bool(keep_variance.all()):
        dropped = [features[index] for index in np.where(~keep_variance)[0]]
        ctx.warnings.append(f"剔除零方差特征：{'、'.join(dropped[:8])}")
        matrix = matrix[:, keep_variance]
        std = std[keep_variance]
        features = [features[index] for index in np.where(keep_variance)[0]]

    n_samples, n_features = matrix.shape
    scale = _p_bool(params, "scale", True)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if scale:
        scaled = centered / std.reshape(1, -1)
    else:
        scaled = centered
    u_matrix, s_values, vt_matrix = np.linalg.svd(scaled, full_matrices=False)
    eigenvalues = (s_values ** 2) / max(1, n_samples - 1)
    total = float(eigenvalues.sum())
    if total <= 0:
        raise C.prerequisite("数据方差为 0，无法做 PCA")
    ratios = eigenvalues / total
    scores = u_matrix * s_values
    loadings = vt_matrix.T
    n_components = int(min(max(2, _p_int(params, "n_components", 2, 2, 10)), eigenvalues.size))

    groups, group_name = _wide_groups(ctx, wide, samples)

    variance_rows = []
    for index in range(min(10, eigenvalues.size)):
        variance_rows.append({
            "pc": f"PC{index + 1}",
            "eigenvalue": float(eigenvalues[index]),
            "var_pct": float(ratios[index] * 100.0),
            "cum_pct": float(np.sum(ratios[: index + 1]) * 100.0),
        })
    variance_table = _make_table("pca_variance", "解释方差", [
        ("pc", "主成分", "str"),
        ("eigenvalue", "特征值", "num"),
        ("var_pct", "解释方差 %", "num"),
        ("cum_pct", "累计 %", "num"),
    ], variance_rows, note=f"共 {n_features} 个特征、{n_samples} 个样本；"
                          f"{'z-score 标准化' if scale else '仅中心化'}")

    importance = np.sqrt(np.sum(loadings[:, :2] ** 2, axis=1))
    top_index = np.argsort(-importance)[:10]
    loading_rows = []
    for index in top_index:
        row = {"feature": features[index]}
        for component in range(min(3, loadings.shape[1])):
            row[f"pc{component + 1}"] = float(loadings[index, component])
        loading_rows.append(row)
    loading_specs: list[Spec] = [("feature", "特征", "str")]
    for component in range(min(3, loadings.shape[1])):
        loading_specs.append((f"pc{component + 1}", f"PC{component + 1}", "num"))
    loading_table = _make_table("pca_loadings", "载荷 top 特征（按 PC1–PC2 综合载荷排序）",
                                loading_specs, loading_rows)

    pc1_top = sorted(range(n_features), key=lambda index: -abs(loadings[index, 0]))[:5]
    bullets = [
        f"PCA（{n_samples} 样本 × {n_features} 特征，{'z-score 标准化' if scale else '仅中心化'}）："
        f"PC1 解释 {ratios[0] * 100.0:.1f}% 方差，PC2 解释 {ratios[1] * 100.0:.1f}%，"
        f"前 2 个主成分累计 {float(np.sum(ratios[:2])) * 100.0:.1f}%"
    ]
    pieces = [f"{features[index]}（{C.fmt_num(loadings[index, 0])}）" for index in pc1_top]
    bullets.append("PC1 载荷绝对值最大的特征：" + "、".join(pieces))
    if len(ratios) >= 3:
        bullets.append(
            f"PC3 解释 {ratios[2] * 100.0:.1f}%，前 3 个主成分累计 "
            f"{float(np.sum(ratios[:3])) * 100.0:.1f}%"
        )
    if groups is not None:
        labels_unique = [str(v) for v in pd.unique(pd.Series(groups))]
        if len(labels_unique) == 2:
            left = np.asarray([str(value) == labels_unique[0] for value in groups])
            right = np.asarray([str(value) == labels_unique[1] for value in groups])
            if left.sum() and right.sum():
                difference = float(scores[right, 0].mean() - scores[left, 0].mean())
                bullets.append(
                    f"分组「{group_name}」：{labels_unique[1]} 的 PC1 得分比 {labels_unique[0]} "
                    f"{'高' if difference >= 0 else '低'} {C.fmt_num(abs(difference))}"
                )
        else:
            bullets.append(f"分组「{group_name}」共 {len(labels_unique)} 个水平，见得分图着色")

    figures: list[dict] = []
    if _p_bool(params, "plot", False):
        figure = _try(ctx, "PCA 得分图", lambda: _pca_figure(
            ctx, scores, ratios, groups, group_name))
        if figure:
            figures.append(figure)

    key_numbers = {
        "n_samples": n_samples,
        "n_features": n_features,
        "pc1_var_pct": float(ratios[0] * 100.0),
        "pc2_var_pct": float(ratios[1] * 100.0),
        "cum_var_pct": float(np.sum(ratios[:2]) * 100.0),
        "n_components": n_components,
    }
    return _pack(
        title="主成分分析",
        method="主成分分析（numpy SVD；中心化"
               + ("，z-score 标准化" if scale else "")
               + "）；特征值 = S²/(n−1)；载荷 = 右奇异向量",
        bullets=bullets,
        key_numbers=key_numbers,
        tables=[variance_table, loading_table],
        figures=figures,
        warnings=ctx.warnings,
    )


# ---------------------------------------------------------------------------
# adjust：多重检验校正
# ---------------------------------------------------------------------------


_METHOD_LABELS = {
    "bh": "Benjamini-Hochberg FDR",
    "bonferroni": "Bonferroni",
    "holm": "Holm",
    "none": "不校正",
}


def _run_adjust(ctx: _Ctx) -> dict:
    params = ctx.params
    alpha = _p_float(params, "alpha", 0.05, 0.001, 0.5)
    method = _p_select(params, "method", ("bh", "bonferroni", "holm", "none"), "bh")
    raw_value = params.get("pvalues")
    name_hints = _p_list(params, "labels")
    values: list[float] = []
    names: list[str] = []

    def parse_p(token: Any, index: int) -> float:
        raw_token = str(token).strip().strip("*").lstrip("<≤>≥").strip()
        try:
            return float(raw_token)
        except ValueError as exc:
            raise C.bad_request(f"第 {index + 1} 个 p 值无法解析：{token}") from exc

    if isinstance(raw_value, (list, tuple)):
        parts = [part for part in raw_value if str(part).strip()]
        for index, part in enumerate(parts):
            values.append(parse_p(part, index))
            names.append(name_hints[index] if index < len(name_hints) else f"检验 {index + 1}")
    elif _p_str(params, "pvalues"):
        raw_text = _p_str(params, "pvalues")
        parts = [part for part in re.split(r"[,;，、\s]+", raw_text) if part]
        for index, part in enumerate(parts):
            values.append(parse_p(part, index))
            names.append(name_hints[index] if index < len(name_hints) else f"检验 {index + 1}")
    else:
        column = _param_col(ctx, "column")
        if not column:
            raise C.bad_request(
                "多重检验校正需要 p 值：请用参数 pvalues（逗号分隔）或 column 指定数据中的 p 值列"
            )
        series = P.to_numeric(ctx.df[column])
        for index, value in enumerate(series.tolist()):
            if value is None or not math.isfinite(float(value)):
                continue
            values.append(float(value))
            names.append(name_hints[index] if index < len(name_hints) else f"第 {index + 1} 行")
        if not values:
            raise C.prerequisite(f"列「{column}」里没有有效 p 值（全部缺失或无法解析）")

    if len(values) > 1000:
        ctx.warnings.append(
            f"p 值家族较大（{len(values)} > 1000）：已按整个家族一次性校正（FDR 的家族定义必须覆盖全部受检假设）；"
            "如需分批校正，请自行拆分，但分批会改变校正家族、使 q 值不可直接比较"
        )
    invalid = [(name, value) for name, value in zip(names, values) if not (0.0 <= value <= 1.0)]
    if invalid:
        text = "、".join(f"{name}={C.fmt_num(value)}" for name, value in invalid[:5])
        raise C.bad_request(f"p 值必须在 [0, 1] 区间：{text}")
    if not values:
        raise C.prerequisite("没有可校正的 p 值")

    adjusted = [float(p) for p in C.adjust_pvalues(values, method)]
    order = np.argsort(values, kind="stable")
    rank = {int(index): position + 1 for position, index in enumerate(order)}
    rows = []
    for index, (name, raw_value, adj_value) in enumerate(zip(names, values, adjusted)):
        rows.append({
            "label": name,
            "p_raw": raw_value,
            "p_adj": adj_value,
            "rank": rank[index],
            "significant": "是" if adj_value < alpha else "否",
            "note": "显著" if adj_value < alpha else "",
        })
    n_sig_raw = int(sum(1 for value in values if value < alpha))
    n_sig_adj = int(sum(1 for value in adjusted if value < alpha))
    smallest_raw = min(values)
    smallest_adj = min(adjusted)
    bullets = [
        f"共 {len(values)} 个 p 值，方法 {_METHOD_LABELS[method]}，显著性水平 α={C.fmt_num(alpha)}",
        f"校正前显著 {n_sig_raw} 个，校正后显著 {n_sig_adj} 个"
        + (f"（减少 {n_sig_raw - n_sig_adj} 个）" if n_sig_adj < n_sig_raw else ""),
        f"最小 p 值 {C.fmt_p(smallest_raw)} → 校正后 {C.fmt_p(smallest_adj)}",
    ]
    if method == "none":
        bullets.append(f"未做校正：{len(values)} 个 p 值原样输出，多假设检验时结论需谨慎")
    table = _make_table("adjust", f"多重检验校正（{_METHOD_LABELS[method]}）", [
        ("label", "检验项", "str"),
        ("p_raw", "原始 p", "p"),
        ("p_adj", "校正 p", "p"),
        ("rank", "秩", "num"),
        ("significant", f"是否显著（α={C.fmt_num(alpha)}）", "str"),
    ], rows, max_rows=500)
    key_numbers = {
        "n": len(values),
        "n_sig_raw": n_sig_raw,
        "n_sig_adjusted": n_sig_adj,
        "min_p": smallest_raw,
        "min_p_adjusted": smallest_adj,
        "alpha": alpha,
        "method": method,
    }
    method_text = {
        "bh": "Benjamini-Hochberg FDR 校正（pl_common.bh_fdr；Benjamini & Hochberg 1995）",
        "bonferroni": "Bonferroni 校正（p × 检验数，上限 1）",
        "holm": "Holm 逐步拒绝校正（pl_common.holm_adjust；Holm 1979）",
        "none": "不校正（原始 p 值）",
    }[method]
    return _pack(
        title="多重检验校正",
        method=method_text,
        bullets=bullets,
        key_numbers=key_numbers,
        tables=[table],
        warnings=ctx.warnings,
    )


# ---------------------------------------------------------------------------
# 分发
# ---------------------------------------------------------------------------

_RUNNERS: dict[str, Callable[[_Ctx], dict]] = {
    "describe": _run_describe,
    "compare": _run_compare,
    "assumption": _run_assumption,
    "corr": _run_corr,
    "regression": _run_regression,
    "chi2": _run_chi2,
    "normality": _run_normality,
    "roc": _run_roc,
    "survival": _run_survival,
    "pca": _run_pca,
    "adjust": _run_adjust,
}


def _ensure_outdir(outdir: str | None) -> str:
    text = str(outdir or "").strip()
    if not text:
        text = os.path.join(tempfile.gettempdir(), "plotlab_stats")
    return C.ensure_dir(text)


def _fallback_bullet(ctx: _Ctx, stat: Mapping[str, Any]) -> str:
    try:
        n_rows = int(len(ctx.df))
    except C.EngineError:
        n_rows = 0
    return f"{stat.get('title') or '统计'}已完成：n={n_rows}（未产生显著结果）"


def run(stat_id: str, dataset: Mapping[str, Any] | None, data: Mapping[str, Any] | None,
        params: Mapping[str, Any] | None, style: Mapping[str, Any] | None,
        outdir: str = "") -> dict:
    """执行一个统计方法，返回契约 §6 规定的结构。"""
    key = str(stat_id or "").strip()
    stat = STATS_INDEX.get(key)
    if stat is None:
        raise C.not_found(f"未知统计方法：{key or '（空）'}；可用：{'、'.join(STATS_INDEX)}")
    runner = _RUNNERS.get(key)
    if runner is None:
        raise C.unsupported(f"统计方法 {key} 暂未实现")
    ctx = _Ctx(dataset or {}, data or {}, _params(params),
               C.resolve_style(style), _ensure_outdir(outdir))
    payload = runner(ctx)

    warnings: list[str] = []
    for message in list(ctx.warnings) + list(payload.get("warnings") or []):
        text = str(message)
        if text and text not in warnings:
            warnings.append(text)
    summary = payload.get("summary") or {}
    bullets = [str(bullet) for bullet in (summary.get("bullets") or []) if str(bullet).strip()]
    if not bullets:
        bullets = [_fallback_bullet(ctx, stat)]
    tables = [table for table in (payload.get("tables") or []) if table]
    figures = [figure for figure in (payload.get("figures") or []) if figure]
    report = str(payload.get("report") or "").strip()
    if not report:
        report = _report(str(payload.get("title") or stat["title"]),
                         str(payload.get("method") or ""), bullets, tables)
    annotation = payload.get("annotation")
    return {
        "title": str(payload.get("title") or stat["title"]),
        "method": str(payload.get("method") or ""),
        "summary": {
            "bullets": bullets,
            "key_numbers": C.jsonable(summary.get("key_numbers") or {}),
        },
        "tables": tables,
        "figures": figures,
        "report": report,
        "annotation": C.jsonable(annotation) if annotation else None,
        "warnings": C.jsonable(warnings),
    }
