"""PlotLab 数据剖析与清洗。

- profile(dataset)：质量分 0-100、逐项检查、列统计、分组汇总、相关、推荐图表/统计
- auto_plan(dataset)：根据数据问题生成清洗计划（带中文理由）
- apply_clean(dataset, plan/auto)：执行清洗并写回 dataset（df + cleaning 记录 + columns）
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import pl_common as C
import pl_prep as P
import pl_state as S

MAX_CORR_COLUMNS = 12

# 端点 / 结局 / 生存类列名：自动清洗不得删除、转数值、填补或变换（否则会洗白结局）
_ENDPOINT_RE = re.compile(
    r"(response|outcome|event|status|death|dead|vital|censor|survival|follow|recur|relapse"
    r"|\b(?:os|dfs|rfs|pfs)\b|结局|事件|随访|生存|复发)",
    re.IGNORECASE,
)


def protected_columns(dataset: Mapping[str, Any]) -> set[str]:
    """自动清洗需要保护的列：ID、分组、端点/结局、生存时间与事件。

    这些列可以参与展示，但自动清洗不能删除、改类型、填补或做离群值处理，
    否则会破坏数据身份或伪造结局。
    """
    df = dataset.get("df")
    if df is None:
        return set()
    columns = {str(c) for c in df.columns}
    protected: set[str] = set()

    def keep(value: Any) -> None:
        if value is not None and str(value) in columns:
            protected.add(str(value))

    keep(dataset.get("id_col"))
    keep(dataset.get("group_col"))
    roles = dataset.get("roles") or {}
    for role in ("id", "group", "color", "facet", "time", "event"):
        keep(roles.get(role))
    for column in columns:
        if _ENDPOINT_RE.search(column):
            protected.add(column)
        if P._matches(column, P.GROUP_NAME_HINTS):
            protected.add(column)
    return protected


def endpoint_columns(dataset: Mapping[str, Any]) -> set[str]:
    """端点/结局/生存类列：可以删行（缺失无法分析），但绝不填补/改类型/缩尾。"""
    df = dataset.get("df")
    if df is None:
        return set()
    columns = {str(c) for c in df.columns}
    endpoints: set[str] = set()

    def keep(value: Any) -> None:
        if value is not None and str(value) in columns:
            endpoints.add(str(value))

    roles = dataset.get("roles") or {}
    for role in ("time", "event"):
        keep(roles.get(role))
    for column in columns:
        if _ENDPOINT_RE.search(column):
            endpoints.add(column)
    return endpoints




# ---------------------------------------------------------------------------
# 剖析
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _numeric_frame(dataset: Mapping[str, Any]) -> pd.DataFrame:
    df = dataset.get("df")
    columns = P.numeric_columns(dataset)
    if df is None or not columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    for column in columns[:40]:
        out[column] = P.to_numeric(df[column]).replace([np.inf, -np.inf], np.nan)
    return out


def _outlier_flags(series: pd.Series, method: str = "iqr", k: float = 1.5) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    valid = values[~np.isnan(values)]
    if len(valid) < 4:
        return np.zeros(len(values), dtype=bool)
    if method == "zscore":
        mean = np.nanmean(values)
        sd = np.nanstd(values, ddof=1) or 1e-12
        return np.abs((values - mean) / sd) > k
    if method == "mad":
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median)) * 1.4826
        mad = mad or 1e-12
        return np.abs((values - median) / mad) > k
    q1, q3 = np.nanpercentile(valid, [25, 75])
    iqr = q3 - q1 or 1e-12
    return (values < q1 - k * iqr) | (values > q3 + k * iqr)


def profile(dataset: Mapping[str, Any]) -> dict:
    df = dataset.get("df")
    if df is None or df.empty:
        raise C.prerequisite("没有数据，请先导入数据")
    n_rows, n_cols = df.shape
    columns = list(dataset.get("columns") or P.column_summaries(df))
    numeric_cols = [c["key"] for c in columns if c.get("dtype") == "numeric"]
    category_cols = [c["key"] for c in columns if c.get("dtype") in ("category", "text")]
    total_cells = max(1, n_rows * n_cols)
    missing_cells = int(sum(int(c.get("missing") or 0) for c in columns))
    missing_rate = missing_cells / total_cells

    checks: list[dict] = []

    def add(check_id: str, level: str, title: str, detail: str, fix: str | None = None) -> None:
        item = {"id": check_id, "level": level, "title": title, "detail": detail}
        if fix:
            item["fix"] = fix
        checks.append(item)

    # 1. 空列 / 全缺失
    empty_cols = [c["key"] for c in columns if int(c.get("missing") or 0) >= n_rows]
    if empty_cols:
        add("empty_columns", "fail", "存在全空列", f"{len(empty_cols)} 列全部为空：{'、'.join(empty_cols[:6])}",
            "用「清洗 → 删除空列」移除")
    else:
        add("empty_columns", "ok", "无全空列", "所有列都有有效值")

    # 2. 缺失率
    worst = sorted(columns, key=lambda c: -(c.get("missing_pct") or 0))[:3]
    if missing_rate > 0.2:
        add("missing", "fail", "缺失率偏高", f"整体缺失 {missing_rate * 100:.1f}%（{missing_cells}/{total_cells} 个单元格）",
            "用「清洗 → 缺失填补」按列处理")
    elif missing_rate > 0.02:
        add("missing", "warn", "存在缺失值", f"整体缺失 {missing_rate * 100:.1f}%，最高："
            + "、".join(f"{c['key']} {c.get('missing_pct')}%" for c in worst if c.get("missing_pct")))
    else:
        add("missing", "ok", "缺失很少", f"整体缺失 {missing_rate * 100:.2f}%")

    # 3. 重复行
    try:
        dup = int(df.duplicated().sum())
    except TypeError:
        dup = 0
    if dup:
        add("duplicate_rows", "warn", "存在完全重复行", f"{dup} 行与其它行完全重复（{dup / n_rows * 100:.1f}%）",
            "用「清洗 → 删除重复行」")
    else:
        add("duplicate_rows", "ok", "无重复行", "所有行互不相同")

    # 4. 常量列
    constant_cols = [c["key"] for c in columns if int(c.get("n_unique") or 0) <= 1]
    if constant_cols:
        add("constant_columns", "warn", "存在常量列", f"{len(constant_cols)} 列只有一个取值：{'、'.join(constant_cols[:6])}",
            "常量列无法用于分组或绘图，建议删除")
    else:
        add("constant_columns", "ok", "无常量列", "每列至少两个取值")

    # 5. ID 列
    id_like = [c["key"] for c in columns if int(c.get("n_unique") or 0) == n_rows and c.get("dtype") != "numeric"]
    if id_like:
        add("id_columns", "ok", "识别到 ID 列", f"{'、'.join(id_like[:5])} 每行唯一，通常只作标签不作数值")

    # 6. 数字被存成文本
    numeric_as_text = []
    for column in category_cols[:20]:
        series = df[column]
        sample = series.dropna().astype(str).head(300)
        if len(sample) == 0:
            continue
        hits = sum(1 for value in sample if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip().replace(",", "")))
        if hits / len(sample) >= 0.85:
            numeric_as_text.append(column)
    if numeric_as_text:
        add("numeric_as_text", "warn", "数字被识别成文本", f"{'、'.join(numeric_as_text[:6])} 看起来是数值但含非数字字符",
            "用「清洗 → 类型转换」转成数值（会自动去掉千分位/百分号）")
    else:
        add("numeric_as_text", "ok", "类型基本正确", "没有明显的数值文本混杂")

    # 7. 异常值
    numeric_frame = _numeric_frame(dataset)
    outlier_rates: list[tuple[str, float]] = []
    for column in numeric_frame.columns:
        flags = _outlier_flags(numeric_frame[column])
        rate = float(np.mean(flags)) if len(flags) else 0.0
        if rate > 0:
            outlier_rates.append((column, rate))
    outlier_rates.sort(key=lambda item: -item[1])
    heavy = [item for item in outlier_rates if item[1] > 0.05]
    if heavy:
        add("outliers", "warn", "异常值比例偏高",
            "、".join(f"{name} {rate * 100:.1f}%" for name, rate in heavy[:5]),
            "用「清洗 → 异常值处理」（标记/剔除/缩尾），或确认是否需要 log 变换")
    elif outlier_rates:
        add("outliers", "ok", "异常值比例正常",
            "、".join(f"{name} {rate * 100:.1f}%" for name, rate in outlier_rates[:5]))
    else:
        add("outliers", "ok", "未发现异常值", "IQR 法未检出离群点")

    # 8. 分组列与均衡
    group_col = dataset.get("roles", {}).get("group") or dataset.get("group_col")
    if group_col and group_col in df.columns:
        counts = df[group_col].astype(str).value_counts()
        smallest = int(counts.min())
        largest = int(counts.max())
        imbalance = largest / max(1, smallest)
        if smallest < 3:
            add("group_balance", "fail", "分组样本太少", f"最小组 {counts.idxmin()} 只有 {smallest} 个样本，无法做组间统计",
                "合并小分组或补充数据")
        elif smallest < 8 or imbalance > 4:
            title = "分组样本量偏少" if smallest < 8 and imbalance <= 4 else "分组不均衡"
            add("group_balance", "warn", title,
                f"{len(counts)} 组，最小 {smallest} / 最大 {largest}（比例 {imbalance:.1f}）",
                "解释结果时注意样本量差异；统计优先用 Welch/非参数方法")
        else:
            add("group_balance", "ok", "分组均衡", f"{len(counts)} 组，每组 {smallest}–{largest} 个样本")
    else:
        add("group_column", "warn", "未识别到分组列", "多数比较类图表需要一列分组（如 group/condition/sex）",
            "在角色映射里手动指定「分组」列")

    # 9. 样本量
    if n_rows < 6:
        add("sample_size", "fail", "样本量过少", f"只有 {n_rows} 行，统计检验几乎无法给出可靠结论")
    elif n_rows < 12:
        add("sample_size", "warn", "样本量偏少", f"{n_rows} 行，注意统计功效")
    else:
        add("sample_size", "ok", "样本量充足", f"{n_rows} 行")

    # 10. 高相关列
    corr_pairs: list[dict] = []
    if numeric_frame.shape[1] >= 2:
        corr = numeric_frame.corr(method="pearson")
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1:]:
                value = _num(corr.loc[left, right])
                if value is not None and abs(value) >= 0.95:
                    corr_pairs.append({"left": left, "right": right, "r": round(value, 4)})
    if corr_pairs:
        add("collinearity", "warn", "存在高度相关列",
            "、".join(f"{p['left']}≈{p['right']}（r={p['r']}）" for p in corr_pairs[:4]),
            "相关列同时入模会导致共线性；绘图时择一即可")
    else:
        add("collinearity", "ok", "无高度共线列", "|r| < 0.95")

    # 11. 负值（影响 log）
    negatives = [c["key"] for c in columns if c.get("dtype") == "numeric" and (c.get("negatives") or 0) > 0]
    if negatives:
        add("negative_values", "ok", "存在负值", f"{'、'.join(negatives[:5])} 含负数；如需 log 变换会自动平移")

    # 12. 内存 / 规模
    memory_mb = float(df.memory_usage(deep=True).sum()) / 1024 / 1024
    if n_rows > 200000 or n_cols > 500:
        add("size", "warn", "数据规模较大", f"{n_rows}×{n_cols}，绘图时会自动抽样/取 top 特征")
    else:
        add("size", "ok", "数据规模合适", f"{n_rows} 行 × {n_cols} 列，约 {memory_mb:.1f} MB")

    # 评分
    weight = {"fail": 18, "warn": 6, "ok": 0}
    penalty = sum(weight.get(item["level"], 0) for item in checks)
    score = int(max(0, min(100, round(100 - penalty))))
    level = "good" if score >= 85 else ("warn" if score >= 60 else "bad")

    # 数值列统计表
    numeric_rows = []
    for column in numeric_cols[:40]:
        values = numeric_frame[column] if column in numeric_frame.columns else P.to_numeric(df[column])
        valid = values.dropna()
        if not len(valid):
            continue
        numeric_rows.append(
            {
                "column": column,
                "n": int(len(valid)),
                "missing": int(values.isna().sum()),
                "mean": float(valid.mean()),
                "sd": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                "median": float(valid.median()),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "skew": float(valid.skew()) if len(valid) > 2 else 0.0,
                "outlier_pct": round(float(np.mean(_outlier_flags(values)) * 100), 2),
            }
        )
    numeric_table = C.table(
        "profile_numeric", "数值列统计",
        [
            {"key": "column", "label": "列"}, {"key": "n", "label": "n", "type": "num"},
            {"key": "missing", "label": "缺失", "type": "num"}, {"key": "mean", "label": "均值", "type": "num"},
            {"key": "sd", "label": "标准差", "type": "num"}, {"key": "median", "label": "中位数", "type": "num"},
            {"key": "min", "label": "最小", "type": "num"}, {"key": "max", "label": "最大", "type": "num"},
            {"key": "outlier_pct", "label": "异常值%", "type": "num"},
        ],
        numeric_rows,
        note="IQR 法（1.5×IQR）标记异常值",
    )

    # 分组汇总
    group_table = None
    if group_col and group_col in df.columns and numeric_cols:
        value_col = dataset.get("roles", {}).get("y") or numeric_cols[0]
        if value_col in df.columns:
            tmp = df[[group_col, value_col]].copy()
            tmp["__value"] = P.to_numeric(tmp[value_col])
            tmp = tmp.dropna(subset=["__value"])
            rows = []
            for group, sub in tmp.groupby(group_col, observed=True):
                values = sub["__value"]
                rows.append(
                    {
                        "group": str(group),
                        "n": int(len(values)),
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "median": float(values.median()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )
            group_table = C.table(
                "profile_group", f"分组汇总（{group_col} × {value_col}）",
                [
                    {"key": "group", "label": "组"}, {"key": "n", "label": "n", "type": "num"},
                    {"key": "mean", "label": "均值", "type": "num"}, {"key": "sd", "label": "标准差", "type": "num"},
                    {"key": "median", "label": "中位数", "type": "num"},
                    {"key": "min", "label": "最小", "type": "num"}, {"key": "max", "label": "最大", "type": "num"},
                ],
                rows,
            )

    # 相关矩阵
    corr_payload = None
    if numeric_frame.shape[1] >= 2:
        subset = numeric_frame.iloc[:, :MAX_CORR_COLUMNS]
        corr = subset.corr(method="pearson")
        pairs = []
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1:]:
                value = _num(corr.loc[left, right])
                if value is not None:
                    pairs.append({"left": left, "right": right, "r": round(value, 4)})
        pairs.sort(key=lambda item: -abs(item["r"]))
        corr_payload = {
            "columns": [str(c) for c in corr.columns],
            "matrix": [[_num(v) for v in row] for row in corr.to_numpy()],
            "pairs": pairs[:12],
        }

    recommend = _recommend(dataset, numeric_cols, category_cols, group_col)
    bullets = [
        f"{n_rows} 行 × {n_cols} 列（{dataset.get('kind', 'tidy')} 数据），数值列 {len(numeric_cols)} 个、分类列 {len(category_cols)} 个",
        f"数据质量 {score}/100（{level}）：" + "；".join(
            item["title"] for item in checks if item["level"] != "ok"
        ) if any(item["level"] != "ok" for item in checks) else f"数据质量 {score}/100，未发现明显问题",
        f"整体缺失率 {missing_rate * 100:.2f}%，重复行 {dup}",
    ]
    if group_col:
        bullets.append(f"分组列：{group_col}" + (f"（{dataset.get('groups') and len(dataset.get('groups')) or df[group_col].nunique()} 组）"))

    return {
        "score": score,
        "level": level,
        "checks": checks,
        "summary": {"bullets": bullets, "key_numbers": {
            "n_rows": n_rows, "n_cols": n_cols, "numeric_cols": len(numeric_cols),
            "missing_rate": round(missing_rate, 4), "duplicate_rows": dup, "score": score,
        }},
        "columns": C.jsonable(columns),
        "numeric_table": numeric_table,
        "group_table": group_table,
        "corr": corr_payload,
        "recommend": recommend,
        "kind": dataset.get("kind"),
        "group_col": group_col,
    }


def _recommend(dataset: Mapping[str, Any], numeric_cols: Sequence[str],
               category_cols: Sequence[str], group_col: str | None) -> dict:
    kind = dataset.get("kind")
    charts: list[dict] = []
    stats: list[dict] = []

    def chart(cid: str, title: str, why: str) -> None:
        charts.append({"id": cid, "title": title, "why": why})

    def stat(sid: str, title: str, why: str) -> None:
        stats.append({"id": sid, "title": title, "why": why})

    if kind == "matrix":
        chart("heatmap", "热图", "矩阵数据最适合展示特征×样本的整体模式")
        chart("pca", "PCA 散点", "看样本间整体差异与分组是否分离")
        chart("corr", "相关矩阵", "查看特征/样本之间的相关性")
        chart("box", "箱线图", "按样本分组比较特征分布")
        if group_col:
            stat("pca", "PCA", "矩阵数据降维看分组结构")
            stat("compare", "组间比较", "按分组比较特征均值差异")
        return {"charts": charts, "stats": stats}

    if group_col:
        chart("box", "箱线图", "分组比较首选，展示中位数与离散度")
        chart("bar", "柱状图（带误差线）", "分组均值±标准误，适合汇报")
        chart("violin", "小提琴图", "展示分组分布形状与样本密度")
        if len(numeric_cols) >= 2 and len(category_cols) >= 2:
            chart("dotplot", "点图", "多指标×多分组一屏比较")
        stat("describe", "描述统计", "先看各组 n/均值/标准差")
        stat("compare", "组间比较", "自动选择 t 检验/ANOVA/非参数检验并给效应量")
        stat("assumption", "前提检查", "正态性与方差齐性决定用参数还是非参数检验")
    if len(numeric_cols) >= 2:
        chart("scatter", "散点图", "两个连续变量的关系，可加回归线与相关系数")
        stat("corr", "相关性分析", f"数值列 {len(numeric_cols)} 个，可算 Pearson/Spearman")
        stat("regression", "线性回归", "量化一个变量对另一个变量的影响（含协变量）")
    if any(re.search(r"time|survival|os|dfs|follow", c, re.I) for c in numeric_cols) and \
            any(re.search(r"event|status|death", c, re.I) for c in numeric_cols + list(category_cols)):
        chart("km", "生存曲线", "检测到时间+事件列，可做 Kaplan-Meier")
        stat("survival", "生存分析", "log-rank 检验 + Cox HR")
    if len(numeric_cols) >= 1 and len(category_cols) >= 1:
        chart("roc", "ROC 曲线", "二分类结局与连续指标，可评估判别能力")
        stat("roc", "ROC/AUC", "AUC + 最佳阈值")
    if len(category_cols) >= 1:
        chart("hist", "直方图", "看连续变量的分布形状")
        chart("density", "密度图", "分组密度对比更直观")
    chart("line", "折线图", "有序变量（剂量/时间）的组间趋势")
    stat("chi2", "卡方检验", "两个分类变量的关联（如分组×缓解）")
    if len(numeric_cols) >= 2:
        stat("normality", "正态性检验", "绘图/建模前的分布检查")
    return {"charts": charts[:10], "stats": stats[:8]}


def _recommend_columns(dataset: Mapping[str, Any], limit: int = 12) -> list[dict]:
    columns = []
    for item in dataset.get("columns") or []:
        columns.append({"key": item.get("key"), "dtype": item.get("dtype"),
                        "n_unique": item.get("n_unique"), "missing_pct": item.get("missing_pct")})
    return columns[:limit]


# ---------------------------------------------------------------------------
# 清洗计划
# ---------------------------------------------------------------------------


def auto_plan(dataset: Mapping[str, Any]) -> list[dict]:
    """根据当前数据自动生成清洗计划（每步带 reason）。"""
    df = dataset.get("df")
    if df is None or df.empty:
        raise C.prerequisite("没有数据，无法生成清洗计划")
    columns = dataset.get("columns") or P.column_summaries(df)
    plan: list[dict] = []
    protected = protected_columns(dataset)

    empty = [c["key"] for c in columns
             if int(c.get("missing") or 0) >= len(df) and str(c["key"]) not in protected]
    if empty:
        plan.append({"op": "drop_empty", "params": {"columns": empty}, "reason": f"删除 {len(empty)} 个全空列"})
    kept_empty = [str(c["key"]) for c in columns
                  if int(c.get("missing") or 0) >= len(df) and str(c["key"]) in protected]
    if kept_empty:
        plan.append({"op": "none", "params": {},
                     "reason": f"全空列 {'、'.join(kept_empty[:5])} 属于 ID/分组/端点，受保护不删除；"
                               "请人工确认该列是否可移除"})

    dup = int(df.duplicated().sum()) if len(df) else 0
    if dup:
        plan.append({"op": "drop_duplicates", "params": {}, "reason": f"删除 {dup} 行完全重复记录"})

    const = [c["key"] for c in columns
             if int(c.get("n_unique") or 0) <= 1 and int(c.get("missing") or 0) < len(df)
             and str(c["key"]) not in protected]
    if const:
        plan.append({"op": "drop_columns", "params": {"columns": const},
                     "reason": f"{len(const)} 个常量列对分析无贡献，删除"})

    numeric_text = []
    for column in [c["key"] for c in columns
                   if c.get("dtype") in ("text", "category") and str(c["key"]) not in protected][:20]:
        series = df[column].dropna().astype(str).head(300)
        if not len(series):
            continue
        hits = sum(1 for value in series if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip().replace(",", "")))
        if hits / len(series) >= 0.85:
            numeric_text.append(column)
    if numeric_text:
        plan.append({"op": "coerce", "params": {"columns": {c: "numeric" for c in numeric_text}},
                     "reason": f"{len(numeric_text)} 列是数字文本，转成数值以便绘图"})

    # 缺失按列独立处理：数值列少缺失→中位数填补；分类列缺失→删除相应行；
    # 缺失过多→删除相应行。端点/ID/分组列受保护，既不填补也不删除列。
    numeric_missing = [
        c["key"] for c in columns
        if c.get("dtype") == "numeric" and str(c["key"]) not in protected
        and 0 < int(c.get("missing") or 0) < len(df) * 0.2
    ]
    if numeric_missing:
        plan.append({"op": "impute", "params": {"columns": numeric_missing, "method": "median"},
                     "reason": f"{len(numeric_missing)} 个数值列缺失 <20%，按列独立用中位数填补（保留样本量）"})
    categorical_missing = [
        c["key"] for c in columns
        if c.get("dtype") in ("category", "text") and str(c["key"]) not in protected
        and 0 < int(c.get("missing") or 0) < len(df)
    ]
    if categorical_missing:
        plan.append({"op": "dropna", "params": {"columns": categorical_missing, "how": "any"},
                     "reason": f"{len(categorical_missing)} 个分类列有缺失，按列独立删除这些列缺失的行"})
    drop_rows_cols = [
        c["key"] for c in columns
        if str(c["key"]) not in protected and 0 < int(c.get("missing") or 0) >= len(df) * 0.2
    ]
    if drop_rows_cols:
        plan.append({"op": "dropna", "params": {"columns": drop_rows_cols, "how": "any"},
                     "reason": f"{len(drop_rows_cols)} 个列缺失 ≥20%，删除这些列缺失的行"})
    endpoints = endpoint_columns(dataset)
    protected_missing = [
        c["key"] for c in columns
        if str(c["key"]) in protected and 0 < int(c.get("missing") or 0) < len(df)
        and str(c["key"]) not in endpoints
    ]
    if protected_missing:
        plan.append({"op": "dropna", "params": {"columns": protected_missing, "how": "any"},
                     "reason": f"受保护列 {'、'.join(str(c) for c in protected_missing[:5])} 缺失："
                               "不填补也不改类型，按列独立删除缺失行"})
    endpoint_missing = [
        c["key"] for c in columns
        if str(c["key"]) in endpoints and 0 < int(c.get("missing") or 0) < len(df)
    ]
    if endpoint_missing:
        plan.append({"op": "dropna", "params": {"columns": endpoint_missing, "how": "any"},
                     "reason": f"端点列 {'、'.join(str(c) for c in endpoint_missing[:5])} 有缺失："
                               "不填补（避免伪造结局），按列独立删除缺失行"})
    filled_protected = [
        c["key"] for c in columns
        if str(c["key"]) in protected and int(c.get("missing") or 0) >= len(df)
    ]
    if filled_protected:
        plan.append({"op": "none", "params": {},
                     "reason": f"受保护列 {'、'.join(str(c) for c in filled_protected[:5])} 全空，"
                               "自动清洗既不填补也不删除，请人工确认"})

    numeric = [c for c in columns if c.get("dtype") == "numeric" and str(c["key"]) not in protected]
    heavy_outlier = []
    for item in numeric[:20]:
        values = P.to_numeric(df[item["key"]]).replace([np.inf, -np.inf], np.nan)
        flags = _outlier_flags(values)
        if np.mean(flags) > 0.05:
            heavy_outlier.append(item["key"])
    if heavy_outlier:
        plan.append({"op": "outlier", "params": {"columns": heavy_outlier, "method": "iqr", "action": "winsorize"},
                     "reason": f"{len(heavy_outlier)} 列异常值 >5%，缩尾（winsorize）到 1.5×IQR 边界"})

    for step in plan:
        step["id"] = f"auto_{step['op']}"
    return plan


# ---------------------------------------------------------------------------
# 清洗执行
# ---------------------------------------------------------------------------


def _note_conv() -> None:
    pass


def _apply_transform(series: pd.Series, method: str, offset: float | None) -> tuple[pd.Series, str | None]:
    values = P.to_numeric(series).astype(float)
    method = str(method).lower()
    note = None
    if method in ("log2", "log10", "ln", "log"):
        minimum = values.min(skipna=True)
        if pd.isna(minimum):
            return values, "全列为空"
        if minimum <= 0:
            shift = float(offset) if offset is not None else (abs(minimum) + 1.0)
            values = values + shift
            note = f"含非正值，整体平移 +{C.fmt_num(shift)} 后再取对数"
        if method == "log2":
            return np.log2(values), note
        if method == "log10":
            return np.log10(values), note
        return np.log(values), note
    if method == "log1p":
        if values.min(skipna=True) < -1:
            return values, "存在 < -1 的值，log1p 无定义，已跳过"
        return np.log1p(values), note
    if method == "sqrt":
        if values.min(skipna=True) < 0:
            return values, "存在负值，sqrt 无定义，已跳过"
        return np.sqrt(values), note
    if method == "zscore":
        sd = values.std(ddof=1)
        if not sd or pd.isna(sd):
            return values, "标准差为 0，无法标准化"
        return (values - values.mean()) / sd, note
    if method == "minmax":
        lo, hi = values.min(skipna=True), values.max(skipna=True)
        if hi == lo:
            return values, "最大=最小，无法归一化"
        return (values - lo) / (hi - lo), note
    if method == "percent":
        total = values.sum(skipna=True)
        if not total:
            return values, "总和为 0，无法计算百分比"
        return values / total * 100.0, note
    if method == "rank":
        return values.rank(pct=True) * 100.0, note
    raise C.bad_request(f"未知变换 {method}，可选：log2/log10/ln/log1p/sqrt/zscore/minmax/percent/rank")


def _summarize_counts(df: pd.DataFrame, rows_before: int) -> dict:
    return {"rows_before": int(rows_before), "rows_after": int(len(df))}


def apply_clean(dataset: dict, plan: Sequence[Mapping[str, Any]] | None = None,
                auto: bool = False, dry_run: bool = False) -> dict:
    """执行清洗计划并写回 dataset。返回 {applied, skipped, warnings, preview}。

    dry_run=True（页面/接口的 preview 模式）：在副本上试算，返回同样的计划与实际行数变化，
    但不写回 dataset["df"]、不追加 cleaning 记录，用于先预览再决定是否落盘。
    """
    if dataset.get("df") is None:
        raise C.prerequisite("没有数据，无法清洗")
    steps = list(plan or [])
    if auto or not steps:
        steps = auto_plan(dataset)
    df = dataset["df"].copy()
    applied: list[dict] = []
    skipped: list[dict] = []
    warnings: list[str] = []

    for index, step in enumerate(steps):
        op = str(step.get("op") or "").strip()
        params = dict(step.get("params") or {})
        reason = str(step.get("reason") or "")
        before = len(df)
        note: list[str] = []
        try:
            if op == "reset":
                df = dataset["raw_df"].copy()
                note.append("已恢复到导入时的数据")
            elif op == "drop_empty":
                cols = params.get("columns") or [c for c in df.columns if df[c].isna().all()]
                cols = [c for c in cols if c in df.columns]
                df = df.drop(columns=cols)
                note.append(f"删除 {len(cols)} 个全空列")
            elif op == "drop_columns":
                cols = [c for c in (params.get("columns") or []) if c in df.columns]
                if not cols:
                    raise C.bad_request("drop_columns 没有匹配到任何列")
                df = df.drop(columns=cols)
                note.append(f"删除列：{'、'.join(cols[:8])}")
            elif op == "drop_duplicates":
                subset = [c for c in (params.get("subset") or []) if c in df.columns] or None
                df = df.drop_duplicates(subset=subset)
                note.append(f"删除 {before - len(df)} 行重复记录")
            elif op == "rename":
                mapping = {k: v for k, v in (params.get("mapping") or {}).items() if k in df.columns}
                if not mapping:
                    raise C.bad_request("rename 需要 params.mapping = {旧列名: 新列名}")
                df = df.rename(columns=mapping)
                note.append(f"重命名 {len(mapping)} 列")
            elif op == "coerce":
                targets = params.get("columns") or {}
                if isinstance(targets, str):
                    targets = {targets: params.get("type") or "numeric"}
                changed = 0
                for column, target in targets.items():
                    if column not in df.columns:
                        continue
                    if target == "numeric":
                        df[column] = P.to_numeric(df[column])
                    elif target == "category":
                        df[column] = df[column].astype(str).where(df[column].notna())
                    elif target == "datetime":
                        df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
                    elif target == "text":
                        df[column] = df[column].astype(str).where(df[column].notna())
                    changed += 1
                if not changed:
                    raise C.bad_request("coerce 没有匹配到任何列")
                note.append(f"转换 {changed} 列类型")
            elif op == "dropna":
                columns = [c for c in (params.get("columns") or []) if c in df.columns] or list(df.columns)
                how = str(params.get("how") or "any")
                if how not in ("any", "all"):
                    how = "any"
                df = df.dropna(subset=columns, how=how)
                note.append(f"按 {len(columns)} 列删除缺失行 {before - len(df)} 行")
            elif op == "impute":
                columns = [c for c in (params.get("columns") or []) if c in df.columns]
                method = str(params.get("method") or "median").lower()
                group_by = params.get("group_by")
                if not columns:
                    raise C.bad_request("impute 需要 params.columns（要填补的列）")
                filled = 0
                for column in columns:
                    if method in ("mean", "median", "mode", "zero"):
                        if group_by and group_by in df.columns and method in ("mean", "median"):
                            if method == "mean":
                                values = df.groupby(group_by, observed=True)[column].transform("mean")
                            else:
                                values = df.groupby(group_by, observed=True)[column].transform("median")
                            series = P.to_numeric(df[column])
                            filled += int(series.isna().sum())
                            df[column] = series.fillna(values)
                        else:
                            if method == "mode":
                                modes = df[column].mode(dropna=True)
                                if not len(modes):
                                    continue
                                fill_value = modes.iloc[0]
                            else:
                                numeric = P.to_numeric(df[column])
                                fill_value = {"mean": numeric.mean(), "median": numeric.median(), "zero": 0.0}[method]
                                df[column] = numeric
                            filled += int(df[column].isna().sum())
                            df[column] = df[column].fillna(fill_value)
                    else:
                        raise C.bad_request(f"未知填补方法 {method}，可选 mean/median/mode/zero")
                note.append(f"填补 {filled} 个缺失值（{method}）")
            elif op == "outlier":
                columns = [c for c in (params.get("columns") or []) if c in df.columns]
                method = str(params.get("method") or "iqr")
                action = str(params.get("action") or "flag")
                k = float(params.get("k") or (1.5 if method == "iqr" else 3.0))
                if not columns:
                    raise C.bad_request("outlier 需要 params.columns")
                flags_any = np.zeros(len(df), dtype=bool)
                for column in columns:
                    values = P.to_numeric(df[column])
                    flags = _outlier_flags(values, method=method, k=k)
                    flags_any |= flags
                    if action == "winsorize":
                        valid = values[~pd.isna(values)]
                        if len(valid) >= 4:
                            q1, q3 = np.nanpercentile(valid, [25, 75])
                            iqr = q3 - q1
                            df[column] = values.clip(q1 - k * iqr, q3 + k * iqr)
                    elif action == "flag":
                        df[f"{column}_outlier"] = flags
                if action == "remove":
                    df = df.loc[~flags_any]
                    note.append(f"删除 {int(flags_any.sum())} 行异常值（{method}, k={k}）")
                elif action == "winsorize":
                    note.append(f"{len(columns)} 列异常值缩尾（{method}, k={k}）")
                else:
                    note.append(f"标记 {int(flags_any.sum())} 行异常值（新增 *_outlier 列）")
            elif op == "filter":
                column = params.get("column")
                if column not in df.columns:
                    raise C.bad_request(f"filter 列 {column} 不存在")
                operator = str(params.get("op") or "==")
                value = params.get("value")
                series = df[column]
                if operator in ("==", "="):
                    mask = series.astype(str) == str(value)
                elif operator == "!=":
                    mask = series.astype(str) != str(value)
                else:
                    numeric = P.to_numeric(series)
                    threshold = float(value)  # type: ignore[arg-type]
                    if operator == ">":
                        mask = numeric > threshold
                    elif operator == ">=":
                        mask = numeric >= threshold
                    elif operator == "<":
                        mask = numeric < threshold
                    elif operator == "<=":
                        mask = numeric <= threshold
                    elif operator == "in":
                        items = value if isinstance(value, (list, tuple)) else str(value).split(",")
                        mask = series.astype(str).isin([str(v) for v in items])
                    elif operator == "contains":
                        mask = series.astype(str).str.contains(str(value), na=False)
                    elif operator == "notnull":
                        mask = series.notna()
                    else:
                        raise C.bad_request(f"未知过滤操作 {operator}")
                df = df.loc[mask]
                note.append(f"按 {column} {operator} {value} 保留 {len(df)} 行")
            elif op == "transform":
                columns = [c for c in (params.get("columns") or []) if c in df.columns]
                method = str(params.get("method") or "log2")
                offset = params.get("offset")
                if not columns:
                    raise C.bad_request("transform 需要 params.columns")
                used_offsets: dict[str, float] = {}
                for column in columns:
                    series, extra = _apply_transform(df[column], method, offset)
                    df[column] = series
                    if extra:
                        note.append(f"{column}: {extra}")
                        match = re.search(r"平移 \+([\d.eE+-]+)", str(extra))
                        if match:
                            used_offsets[column] = float(match.group(1))
                note.append(f"{len(columns)} 列 {method} 变换")
                if used_offsets:
                    params["_offsets"] = used_offsets
            elif op == "aggregate":
                group_by = params.get("group_by")
                value_cols = [c for c in (params.get("value_cols") or []) if c in df.columns]
                how = str(params.get("how") or "mean")
                if not group_by or group_by not in df.columns or not value_cols:
                    raise C.bad_request("aggregate 需要 params.group_by 与 params.value_cols")
                if how not in ("mean", "median", "sum", "min", "max"):
                    raise C.bad_request(f"未知聚合方式 {how}")
                others = [c for c in df.columns if c not in value_cols and c != group_by]
                agg_map = {column: how for column in value_cols}
                for column in others:
                    agg_map[column] = "first"
                df = df.groupby(group_by, observed=True).agg(agg_map).reset_index()
                note.append(f"按 {group_by} 聚合 {len(value_cols)} 列（{how}）")
            elif op in ("", "none"):
                continue
            else:
                raise C.bad_request(f"未知清洗操作 {op}")
        except C.EngineError as error:
            if error.code == "bad_request" and step.get("optional", True):
                skipped.append({"op": op, "params": params, "reason": error.message})
                if reason:
                    warnings.append(f"{reason}：{error.message}")
                continue
            raise
        except Exception as error:  # noqa: BLE001 - 单步失败不应该炸整个清洗
            skipped.append({"op": op, "params": params, "reason": str(error)})
            warnings.append(f"{op} 执行失败：{error}")
            continue

        record = {"op": op, "params": C.jsonable(params), "reason": reason or None,
                  **_summarize_counts(df, before), "notes": note}
        applied.append(record)

    if dry_run:
        preview_dataset = dict(dataset)
        preview_dataset["df"] = df.reset_index(drop=True)
        S.refresh_columns(preview_dataset)
        return {
            "dry_run": True,
            "applied": applied,
            "skipped": skipped,
            "warnings": warnings,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "n_rows_before": int(len(dataset.get("df"))),
            "preview": P.sample_rows(preview_dataset, 20),
            "columns": C.jsonable(preview_dataset.get("columns") or []),
            "plan": C.jsonable(steps),
        }
    dataset["df"] = df.reset_index(drop=True)
    dataset.setdefault("cleaning", []).extend(applied)
    dataset.setdefault("steps", []).append("clean")
    S.refresh_columns(dataset)
    preview = P.sample_rows(dataset, 20)
    return {
        "dry_run": False,
        "applied": applied,
        "skipped": skipped,
        "warnings": warnings,
        "n_rows": int(len(dataset["df"])),
        "n_cols": int(dataset["df"].shape[1]),
        "preview": preview,
        "columns": C.jsonable(dataset.get("columns") or []),
    }


def reset_data(dataset: dict) -> dict:
    """恢复到导入状态（保留列/角色推断）。"""
    raw = dataset.get("raw_df")
    if raw is None:
        raise C.prerequisite("没有原始数据可恢复")
    dataset["df"] = raw.copy()
    dataset["cleaning"] = []
    S.refresh_columns(dataset)
    return {"n_rows": int(len(dataset["df"])), "n_cols": int(dataset["df"].shape[1]),
            "preview": P.sample_rows(dataset, 20), "columns": C.jsonable(dataset.get("columns") or [])}
