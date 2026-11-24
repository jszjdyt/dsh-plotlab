"""PlotLab 数据装配：列类型、角色推荐、tidy/wide 两种视图。

图表与统计模块一律通过 build_data()/tidy_frame() 拿标准化的数据，不要自己猜列。

标准列（canonical）：__x __y __group __color __size __facet __time __event __id
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import pl_common as C

ROLE_KEYS = ("x", "y", "group", "color", "size", "facet", "time", "event", "id")
CANONICAL = {role: f"__{role}" for role in ROLE_KEYS}

GROUP_NAME_HINTS = ("group", "class", "type", "condition", "treatment", "group_name",
                    "status", "sex", "gender", "stage", "grade", "cohort", "arm", "category",
                    "分组", "组别", "性别")
TIME_HINTS = ("time", "survival", "os", "dfs", "rfs", "follow", "days", "months", "month",
              "surv_time", "fu_time", "生存时间", "随访")
EVENT_HINTS = ("event", "status", "death", "dead", "vital", "censor", "surv_event",
               "结局", "事件")
ID_HINTS = ("id", "sample", "sample_id", "subject", "patient", "case", "编号")
FEATURE_HINTS = ("gene", "gene_id", "symbol", "probe", "probe_id", "feature", "features",
                 "metabolite", "protein", "peptide", "id", "name", "mrna", "transcript")


# ---------------------------------------------------------------------------
# 列类型
# ---------------------------------------------------------------------------


def infer_dtype(series: pd.Series) -> str:
    """numeric/category/datetime/text。object 列多数可转数字→numeric，低基数→category。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "category"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "text"
    sample = values.head(500)
    numeric_hits = 0
    for text in sample:
        cleaned = re.sub(r"[,\s%￥$]", "", str(text))
        if cleaned.startswith("<") or cleaned.startswith(">"):
            cleaned = cleaned[1:]
        try:
            float(cleaned)
            numeric_hits += 1
        except ValueError:
            pass
    if numeric_hits / max(1, len(sample)) >= 0.85:
        return "numeric"
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed.notna().mean() >= 0.85 and re.search(r"[-/:]", " ".join(map(str, sample.head(5)))):
        return "datetime"
    unique = values.nunique()
    if unique <= min(30, max(2, int(len(values) * 0.5))):
        return "category"
    return "text"


def to_numeric(series: pd.Series) -> pd.Series:
    """宽松数值化：去千分位/百分号/单位空格；"<0.001" 取 0.0005；不可解析→NaN。"""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    def parse(value: Any) -> float:
        if value is None:
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        text = str(value).strip()
        if not text or text.lower() in ("na", "nan", "none", "null", "-", "—"):
            return np.nan
        halve = text.startswith("<") or text.startswith("≤")
        text = text.lstrip("<≥>≤").strip()
        text = re.sub(r"[,\s]", "", text)
        text = text.rstrip("%")
        match = re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not match:
            return np.nan
        try:
            number = float(match.group(0))
        except ValueError:
            return np.nan
        if halve:
            number = abs(number) / 2.0
        return number

    return series.map(parse).astype(float)


def column_summary(df: pd.DataFrame, key: str) -> dict:
    series = df[key]
    dtype = infer_dtype(series)
    total = int(len(series))
    missing = int(series.isna().sum())
    info: dict[str, Any] = {
        "key": str(key),
        "dtype": dtype,
        "missing": missing,
        "missing_pct": round(missing / total * 100.0, 2) if total else 0.0,
        "n_unique": int(series.nunique(dropna=True)),
    }
    if dtype == "numeric":
        numeric = to_numeric(series).replace([np.inf, -np.inf], np.nan)
        valid = numeric.dropna()
        if len(valid):
            info.update(
                {
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "mean": float(valid.mean()),
                    "sd": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                    "median": float(valid.median()),
                    "zeros": int((valid == 0).sum()),
                    "negatives": int((valid < 0).sum()),
                    "skew": float(valid.skew()) if len(valid) > 2 else 0.0,
                }
            )
        examples = [C.fmt_num(v) for v in valid.head(3).tolist()]
    elif dtype == "datetime":
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        valid = parsed.dropna()
        info.update({"min": str(valid.min())[:19], "max": str(valid.max())[:19]})
        examples = [str(v)[:19] for v in valid.head(3).tolist()]
    else:
        counts = series.astype(str).value_counts().head(8)
        info["levels"] = [str(v) for v in counts.index.tolist()]
        examples = info["levels"][:3]
    info["examples"] = C.jsonable(examples)
    return info


def column_summaries(df: pd.DataFrame, columns: Sequence[str] | None = None) -> list[dict]:
    keys = list(columns) if columns is not None else [str(c) for c in df.columns]
    out = []
    for key in keys:
        if key in df.columns:
            item = column_summary(df, key)
            item["key"] = str(key)
            out.append(item)
    return out


def column_index(dataset: Mapping[str, Any]) -> dict:
    return {str(item["key"]): item for item in dataset.get("columns") or []}


def numeric_columns(dataset: Mapping[str, Any]) -> list[str]:
    return [str(c["key"]) for c in dataset.get("columns") or [] if c.get("dtype") == "numeric"]


def categorical_columns(dataset: Mapping[str, Any]) -> list[str]:
    return [
        str(c["key"])
        for c in dataset.get("columns") or []
        if c.get("dtype") in ("category", "text", "datetime") and 2 <= int(c.get("n_unique") or 0) <= 60
    ]


def _matches(key: str, hints: Sequence[str]) -> bool:
    """列名匹配。短关键词（os/dfs/rfs/id 等）用词边界，避免 dose 命中 os。"""
    lowered = str(key).lower()
    for hint in hints:
        hint = str(hint).lower()
        if len(hint) <= 3:
            if re.search(r"(?<![a-z0-9])" + re.escape(hint) + r"(?![a-z0-9])", lowered):
                return True
        elif hint in lowered:
            return True
    return False


def P_kind_of(dataset: Mapping[str, Any], key: str) -> str:
    info = column_index(dataset).get(str(key)) or {}
    return str(info.get("dtype") or "")


def _is_id_column(dataset: Mapping[str, Any], key: str) -> bool:
    """每行唯一且偏文本的列视为 ID，不用于坐标/分组。"""
    columns = column_index(dataset)
    info = columns.get(str(key))
    if not info:
        return False
    n_rows = int((dataset.get("df").shape[0]) if dataset.get("df") is not None else 0)
    if n_rows < 4:
        return False
    if int(info.get("n_unique") or 0) < n_rows:
        return False
    if info.get("dtype") == "numeric":
        return False
    return True


# ---------------------------------------------------------------------------
# 角色推断
# ---------------------------------------------------------------------------


def detect_group_column(dataset: Mapping[str, Any], exclude: Sequence[str] = ()) -> str | None:
    columns = column_index(dataset)
    cats = categorical_columns(dataset)
    pool = [c for c in cats if c not in exclude and c in columns and not _is_id_column(dataset, c)]
    for key in pool:
        if _matches(key, GROUP_NAME_HINTS) and int(columns[key].get("n_unique") or 0) <= 20:
            return key
    for key in pool:
        if int(columns[key].get("n_unique") or 0) <= 8:
            return key
    return None


def _first_numeric(dataset: Mapping[str, Any], exclude: Sequence[str] = (), prefer_hints=(),
                   prefer_continuous: bool = False) -> str | None:
    nums = [c for c in numeric_columns(dataset) if c not in exclude]
    if not nums:
        return None
    index = column_index(dataset)
    if prefer_continuous:
        nums = sorted(nums, key=lambda key: -int(index.get(key, {}).get("n_unique") or 0))
    for hint in prefer_hints:
        for key in nums:
            if hint in key.lower():
                return key
    return nums[0]


def _first_categorical(dataset: Mapping[str, Any], exclude: Sequence[str] = ()) -> str | None:
    cats = [c for c in categorical_columns(dataset) if c not in exclude and not _is_id_column(dataset, c)]
    return cats[0] if cats else None


def infer_roles(dataset: Mapping[str, Any], chart_meta: Mapping[str, Any] | None = None) -> dict:
    """优先用 dataset 已保存的 roles，其次按图表语义推断。"""
    saved = dataset.get("roles") or {}
    roles = {key: saved.get(key) for key in ROLE_KEYS}
    chart_id = str((chart_meta or {}).get("id") or "")
    if chart_id:
        rec = _recommend_for_chart(dataset, chart_meta, base=roles)
        roles.update({k: v for k, v in rec.items() if not roles.get(k)})
    else:
        group = detect_group_column(dataset)
        num = _first_numeric(dataset)
        cat = _first_categorical(dataset, exclude=[group] if group else [])
        if not roles.get("group"):
            roles["group"] = group
        if not roles.get("x"):
            roles["x"] = cat or num or _first_categorical(dataset)
        if not roles.get("y"):
            roles["y"] = _first_numeric(dataset, exclude=[roles.get("x")] if roles.get("x") else [])
    # 只保留数据里真实存在的列
    existing = {str(c["key"]) for c in dataset.get("columns") or []}
    return {k: (v if v in existing else None) for k, v in roles.items()}


def recommend_roles(dataset: Mapping[str, Any], chart_meta: Mapping[str, Any]) -> dict:
    """图表语义优先给出角色；dataset 里保存的角色只补空缺（避免上一张图的角色带偏）。"""
    roles = {key: None for key in ROLE_KEYS}
    rec = _recommend_for_chart(dataset, chart_meta, base={})
    for key, value in rec.items():
        if value:
            roles[key] = value
    saved = dataset.get("roles") or {}
    for key in ROLE_KEYS:
        if not roles.get(key) and saved.get(key):
            roles[key] = saved[key]
    existing = {str(c["key"]) for c in dataset.get("columns") or []}
    return {k: (v if v in existing else None) for k, v in roles.items()}


COMPARISON_CHARTS = {
    "bar", "bar_dot", "box", "violin", "raincloud", "dotplot", "stack", "pie",
    "hist", "density", "qq",
}
RELATION_CHARTS = {"scatter", "bubble", "line", "area", "pairplot"}
WIDE_CHARTS = {"heatmap", "corr", "pca"}


def _recommend_for_chart(dataset: Mapping[str, Any], chart_meta: Mapping[str, Any],
                         base: Mapping[str, Any] | None = None) -> dict:
    chart_id = str(chart_meta.get("id") or "")
    roles: dict[str, Any] = {}
    group = detect_group_column(dataset)
    nums = numeric_columns(dataset)
    cats = categorical_columns(dataset)
    exclusions = [v for v in (base or {}).values() if v]

    if chart_id in WIDE_CHARTS:
        return roles

    if chart_id in ("km",):
        time_col = None
        event_col = None
        for key in [str(c["key"]) for c in dataset.get("columns") or []]:
            if _matches(key, TIME_HINTS) and time_col is None:
                time_col = key
            if _matches(key, EVENT_HINTS) and event_col is None:
                event_col = key
        roles["time"] = time_col
        roles["event"] = event_col
        roles["group"] = group
        return roles

    if chart_id == "roc":
        roles["y"] = _first_numeric(dataset, prefer_hints=("score", "prob", "risk", "value", "expr"))
        roles["group"] = group or _first_categorical(dataset)
        return roles

    if chart_id == "forest":
        time_col = None
        event_col = None
        for key in [str(c["key"]) for c in dataset.get("columns") or []]:
            if _matches(key, TIME_HINTS) and time_col is None:
                time_col = key
            if _matches(key, EVENT_HINTS) and event_col is None:
                event_col = key
        roles["time"] = time_col
        roles["event"] = event_col
        roles["group"] = group
        return roles

    if chart_id in ("scatter", "bubble", "line", "area", "pairplot"):
        x = None
        _keys = [str(c["key"]) for c in dataset.get("columns") or []]
        time_like = [c for c in _keys if _matches(c, TIME_HINTS) and P_kind_of(dataset, c) in ("numeric", "datetime")]
        if chart_id in ("line", "area") and time_like:
            x = time_like[0]
        elif chart_id in ("line", "area"):
            index = column_index(dataset)
            ordered = [c for c in numeric_columns(dataset) if c not in exclusions
                       and 2 <= int(index.get(c, {}).get("n_unique") or 0) <= 20]
            x = ordered[0] if ordered else _first_numeric(dataset, exclude=exclusions, prefer_continuous=True)
        else:
            x = _first_numeric(dataset, exclude=exclusions, prefer_continuous=True)
        y = _first_numeric(dataset, exclude=exclusions + ([x] if x else []), prefer_continuous=True)
        roles["x"] = x
        roles["y"] = y
        if chart_id in ("line", "area"):
            roles["group"] = group if group and group != x else (cats[0] if cats else None)
        else:
            roles["color"] = group
        if chart_id in ("bubble", "scatter"):
            size = _first_numeric(dataset, exclude=exclusions + [x, y])
            if chart_id == "bubble":
                roles["size"] = size
            roles["color"] = group
        return roles

    if chart_id in ("heatmap",):
        return roles

    if chart_id in ("hist", "density", "qq"):
        roles["x"] = _first_numeric(dataset, exclude=exclusions, prefer_continuous=True)
        if group and group != roles.get("x"):
            roles["color"] = group
            roles["group"] = group
        return roles

    # 比较 / 分布类：x=分组列（优先），y=数值列
    x = group or _first_categorical(dataset, exclude=exclusions)
    if chart_id in ("pie",):
        x = group or _first_categorical(dataset, exclude=exclusions)
        roles["x"] = x
        roles["y"] = _first_numeric(dataset, exclude=exclusions)
        return roles
    if chart_id in ("dotplot",):
        roles["x"] = group or _first_categorical(dataset, exclude=exclusions)
        roles["y"] = _first_categorical(dataset, exclude=exclusions + ([roles["x"]] if roles.get("x") else []))
        roles["color"] = _first_numeric(dataset, exclude=exclusions)
        roles["size"] = _first_numeric(dataset, exclude=exclusions + ([roles.get("color")] if roles.get("color") else []))
        return roles
    roles["x"] = x
    roles["y"] = _first_numeric(dataset, exclude=exclusions + ([x] if x else []), prefer_continuous=True)
    if chart_id in ("stack",):
        roles["group"] = _first_categorical(dataset, exclude=exclusions + ([x] if x else []))
    elif chart_id in ("hist",):
        roles["color"] = group if group and group != x else None
    else:
        roles["group"] = x if x in cats else group
    return roles


# ---------------------------------------------------------------------------
# tidy / wide 视图
# ---------------------------------------------------------------------------


def _bin_numeric_group(series: pd.Series, n_bins: int = 3) -> tuple[pd.Series, list]:
    valid = pd.to_numeric(series, errors="coerce")
    try:
        binned, edges = pd.qcut(valid, q=n_bins, retbins=True, duplicates="drop")
        labels = [f"{C.fmt_num(edges[i])}–{C.fmt_num(edges[i + 1])}" for i in range(len(edges) - 1)]
        categories = []
        for interval, label in zip(binned.cat.categories, labels):
            categories.append(label)
        mapped = binned.map(dict(zip(binned.cat.categories, labels)))
        return mapped, categories
    except (ValueError, TypeError):
        return series.astype(str), []


def group_series(series: pd.Series, max_levels: int = 12) -> tuple[pd.Series, list, str | None]:
    """把分组列规范化成 Categorical 并返回 (series, levels, note)。"""
    note = None
    dtype = infer_dtype(series)
    if dtype == "numeric":
        valid = to_numeric(series)
        unique = valid.dropna().unique()
        if len(unique) <= max_levels and len(unique) <= 8:
            levels = sorted(float(v) for v in unique)
            categorical = pd.Categorical(valid, categories=levels)
            return pd.Series(categorical, index=series.index), levels, note
        mapped, levels = _bin_numeric_group(valid)
        if not levels:
            return series.astype(str), sorted(series.dropna().astype(str).unique().tolist()), "数值分组列为空"
        note = f"数值列 {series.name} 已按分位数切成 {len(levels)} 组"
        return mapped, levels, note
    strings = series.astype("string")
    levels = [str(v) for v in pd.unique(strings.dropna())]
    levels = levels[:60]
    categorical = pd.Categorical(strings.astype(object), categories=levels)
    return pd.Series(categorical, index=series.index), levels, note


def tidy_frame(dataset: Mapping[str, Any], roles: Mapping[str, Any],
               required: Sequence[str] = (), optional: Sequence[str] = (),
               drop_na: bool = True) -> tuple[pd.DataFrame, dict, dict, list]:
    """按角色构造标准列 DataFrame。返回 (frame, labels, levels, notes)。"""
    df = dataset.get("df")
    if df is None:
        raise C.prerequisite("没有数据，请先导入数据")
    frame = pd.DataFrame(index=df.index)
    labels: dict[str, str] = {}
    levels: dict[str, list] = {}
    notes: list[str] = []
    for role in list(required) + list(optional):
        column = roles.get(role)
        if not column:
            if role in required:
                raise C.bad_request(f"缺少角色 {role} 的列：请在左侧选择一列作为 {role}")
            continue
        if column not in df.columns:
            raise C.bad_request(f"列 {column} 不在数据里（可能已被删除），请重新指定角色")
        series = df[column]
        target = CANONICAL[role]
        if role in ("group", "color", "facet"):
            grouped, group_levels, note = group_series(series.rename(column))
            frame[target] = grouped
            levels[role] = group_levels
            if note:
                notes.append(note)
        elif role in ("y", "size"):
            frame[target] = to_numeric(series)
        elif role == "time":
            frame[target] = to_numeric(series)
        elif role == "event":
            numeric = to_numeric(series)
            if numeric.notna().sum() == 0:
                text = series.astype(str).str.lower()
                numeric = text.isin(["dead", "death", "event", "yes", "true", "1", "复发", "死亡"]).astype(float)
                numeric[series.isna()] = np.nan
            frame[target] = numeric
        elif role == "x":
            dtype = infer_dtype(series)
            if dtype == "numeric":
                frame[target] = to_numeric(series)
            elif dtype == "datetime":
                frame[target] = pd.to_datetime(series, errors="coerce", format="mixed")
            else:
                grouped, group_levels, note = group_series(series.rename(column))
                frame[target] = grouped
                levels["x"] = group_levels
                if note:
                    notes.append(note)
        else:
            frame[target] = series.astype(object)
        labels[role] = column

    required_targets = [CANONICAL[role] for role in required]
    n_before = len(frame)
    if drop_na and required_targets:
        mask = frame[required_targets].notna().all(axis=1)
        frame = frame.loc[mask]
    n_dropped = int(n_before - len(frame))
    return frame.reset_index(drop=True), labels, levels, notes + ([f"丢弃 {n_dropped} 行缺失值"] if n_dropped else [])


def _matrix_from_wide_dataset(dataset: Mapping[str, Any], params: Mapping[str, Any]) -> dict | None:
    """matrix 数据集：首列=特征 id，其余数值列=样本。"""
    df = dataset.get("df")
    if df is None or df.empty:
        return None
    feature_col = dataset.get("feature_col")
    if not feature_col or feature_col not in df.columns:
        feature_col = str(df.columns[0])
    numeric_cols = [str(c) for c in df.columns if c != feature_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return None
    limit = int(params.get("max_features") or 0)
    features = df[feature_col].astype(str).tolist()
    values = df[numeric_cols].to_numpy(dtype=float).T  # (n_samples, n_features)
    samples = numeric_cols
    sample_groups = None
    group_name = None
    info = dataset.get("sample_info")
    group_col = str(params.get("sample_group_col") or dataset.get("group_col") or "")
    if isinstance(info, pd.DataFrame) and not info.empty and "sample" in info.columns:
        info = info.set_index("sample")
        if (not group_col or group_col not in info.columns) and "group" in info.columns:
            group_col = "group"
        if group_col and group_col in info.columns:
            ordered = [str(s) for s in samples]
            if all(s in info.index for s in ordered):
                sample_groups = info.loc[ordered, group_col].astype(str).to_numpy()
                group_name = group_col
            else:
                lookup = info[group_col].astype(str)
                sample_groups = np.array([lookup.get(str(s), "NA") for s in samples], dtype=object)
                group_name = group_col
    if limit and len(features) > limit:
        variances = np.nanvar(values, axis=0)
        top = np.argsort(-np.nan_to_num(variances))[:limit]
        top.sort()
        values = values[:, top]
        features = [features[i] for i in top]
    return {
        "matrix": values,
        "features": features,
        "samples": samples,
        "sample_groups": sample_groups,
        "group_name": group_name,
        "note": f"矩阵数据：{len(samples)} 样本 × {len(features)} 特征",
    }


def _matrix_from_tidy_dataset(dataset: Mapping[str, Any], roles: Mapping[str, Any],
                              params: Mapping[str, Any]) -> dict | None:
    """tidy 数据集（样本×特征 宽表）：数值列当特征，行当样本。"""
    df = dataset.get("df")
    if df is None or df.empty:
        return None
    nums = numeric_columns(dataset)
    if len(nums) < 2:
        return None
    limit = int(params.get("max_features") or 0)
    features = nums
    values = df[nums].apply(to_numeric).to_numpy(dtype=float)
    id_col = roles.get("id") or dataset.get("id_col")
    if id_col and id_col in df.columns:
        samples = df[id_col].astype(str).tolist()
    else:
        samples = [f"row{index + 1}" for index in range(len(df))]
    sample_groups = None
    group_name = None
    group_col = roles.get("group")
    if group_col and group_col in df.columns:
        sample_groups = df[group_col].astype(str).to_numpy()
        group_name = group_col
    if limit and len(features) > limit:
        variances = np.nanvar(values, axis=0)
        top = np.argsort(-np.nan_to_num(variances))[:limit]
        top.sort()
        values = values[:, top]
        features = [features[i] for i in top]
    return {
        "matrix": values,
        "features": features,
        "samples": samples,
        "sample_groups": sample_groups,
        "group_name": group_name,
        "note": f"宽表数据：{len(samples)} 行 × {len(features)} 个数值列",
    }


def build_wide(dataset: Mapping[str, Any], roles: Mapping[str, Any],
               params: Mapping[str, Any] | None = None) -> dict | None:
    params = dict(params or {})
    if dataset.get("kind") == "matrix":
        return _matrix_from_wide_dataset(dataset, params)
    return _matrix_from_tidy_dataset(dataset, roles, params)


def melt_matrix(dataset: Mapping[str, Any], wide: Mapping[str, Any], max_cells: int = 200000) -> dict:
    """矩阵融化为 tidy：x=样本，y=数值，group=样本分组。"""
    matrix = np.asarray(wide["matrix"], dtype=float)
    samples = list(wide["samples"])
    features = list(wide["features"])
    n_samples, n_features = matrix.shape
    if n_samples * n_features > max_cells:
        raise C.bad_request(
            f"矩阵太大（{n_samples}×{n_features}），融化成长表超过 {max_cells} 个值；"
            "请用热图/相关矩阵/PCA，或先过滤特征（params.max_features）"
        )
    frame = pd.DataFrame(
        {
            "__x": np.repeat(samples, n_features),
            "__y": matrix.reshape(-1),
            "__id": np.tile(features, n_samples),
        }
    )
    groups = wide.get("sample_groups")
    if groups is not None and len(groups) == n_samples:
        frame["__group"] = np.repeat(np.asarray(groups, dtype=object), n_features)
    else:
        frame["__group"] = frame["__x"]
    levels = {"group": [str(v) for v in pd.unique(frame["__group"])]}
    notes = ["矩阵数据已融化为长表：x=样本，y=数值"]
    return {
        "frame": frame,
        "labels": {"x": wide.get("group_name") or "sample", "y": "value", "group": wide.get("group_name") or "sample"},
        "levels": levels,
        "notes": notes,
    }


def build_data(dataset: Mapping[str, Any], chart_meta: Mapping[str, Any],
               params: Mapping[str, Any] | None = None,
               roles: Mapping[str, Any] | None = None) -> dict:
    """图表/统计统一数据装配。"""
    params = dict(params or {})
    chart_meta = dict(chart_meta or {})
    roles = dict(roles or recommend_roles(dataset, chart_meta))
    needs = str(chart_meta.get("needs") or "tidy")
    role_spec = chart_meta.get("roles") or {}
    required = [key for key, kind in role_spec.items() if kind == "required"]
    optional = [key for key, kind in role_spec.items() if kind == "optional"]

    dataset_meta = {
        "id": dataset.get("id"),
        "name": dataset.get("name"),
        "kind": dataset.get("kind"),
        "n_rows": int(dataset.get("df").shape[0]) if dataset.get("df") is not None else 0,
        "n_cols": int(dataset.get("df").shape[1]) if dataset.get("df") is not None else 0,
    }

    wide = build_wide(dataset, roles, params)
    tidy = None
    labels: dict[str, str] = {}
    levels: dict[str, list] = {}
    notes: list[str] = []
    n_dropped = 0

    if dataset.get("kind") == "matrix":
        if needs == "wide":
            if wide is None:
                raise C.prerequisite("矩阵视图构建失败：请确认首列是特征 id、其余列是数值样本")
            if wide.get("note"):
                notes.append(wide["note"])
        else:
            if wide is None:
                raise C.prerequisite("矩阵视图构建失败：请确认首列是特征 id、其余列是数值样本")
            melted = melt_matrix(dataset, wide, max_cells=int(params.get("max_cells") or 200000))
            tidy = melted["frame"]
            labels = melted["labels"]
            levels = melted["levels"]
            notes.extend(melted["notes"])
    else:
        if needs == "wide":
            if wide is None:
                raise C.prerequisite(
                    "这个数据集没有足够多的数值列来构建矩阵视图（至少 2 列数值）；"
                    "热图/相关/PCA 需要宽表或矩阵数据"
                )
            if wide.get("note"):
                notes.append(wide["note"])
        else:
            tidy, labels, levels, tidy_notes = tidy_frame(dataset, roles, required, optional)
            notes.extend(tidy_notes)
            if role_spec.get("group") and roles.get("group"):
                labels.setdefault("group", str(roles["group"]))

    if tidy is not None:
        if "x" in tidy.columns:
            values = tidy["x"].dropna()
            if len(values) and len(pd.unique(values)) <= 60 and not pd.api.types.is_numeric_dtype(values):
                levels.setdefault("x", [str(v) for v in pd.unique(values)])
        n_dropped = int(dataset.get("df").shape[0]) - int(len(tidy)) if needs != "wide" else 0

    return {
        "kind": str(needs),
        "dataset": dataset_meta,
        "tidy": tidy,
        "labels": labels,
        "levels": levels,
        "n_dropped": n_dropped,
        "wide": wide if (needs == "wide" or wide is not None) else None,
        "numeric_cols": numeric_columns(dataset),
        "categorical_cols": categorical_columns(dataset),
        "params": params,
        "roles": {k: v for k, v in roles.items() if v},
        "notes": notes,
    }


def require_tidy(data: Mapping[str, Any], columns: Sequence[str] = ()) -> pd.DataFrame:
    frame = data.get("tidy")
    if frame is None or frame.empty:
        raise C.prerequisite("没有可用的观测数据（可能全部缺失），请检查角色映射")
    for column in columns:
        if column not in frame.columns:
            raise C.bad_request(f"图表需要 {column} 角色，请在左侧指定一列")
    return frame


def numeric_column_values(dataset: Mapping[str, Any], column: str) -> np.ndarray:
    df = dataset.get("df")
    if df is None or column not in df.columns:
        raise C.bad_request(f"列 {column} 不存在")
    return to_numeric(df[column]).to_numpy(dtype=float)


def sample_rows(dataset: Mapping[str, Any], limit: int = 30) -> list[dict]:
    df = dataset.get("df")
    if df is None:
        return []
    head = df.head(max(1, min(int(limit or 30), 200)))
    rows = []
    for _, row in head.iterrows():
        item = {}
        for key in head.columns:
            value = row[key]
            if isinstance(value, (np.generic,)):
                value = value.item()
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                value = None
            if isinstance(value, (pd.Timestamp,)):
                value = str(value)[:19]
            item[str(key)] = value if isinstance(value, (int, float, str, type(None), bool)) else str(value)
        rows.append(item)
    return C.jsonable(rows)


def infer_kind_and_feature_col(df: pd.DataFrame, explicit: str | None = None) -> tuple[str, str | None]:
    """判断 tidy/matrix 与特征列。"""
    if explicit in ("tidy", "matrix"):
        feature = str(df.columns[0]) if explicit == "matrix" else None
        return explicit, feature
    if df.empty:
        return "tidy", None
    first = str(df.columns[0])
    n_rows, n_cols = df.shape
    numeric_others = [c for c in df.columns[1:] if infer_dtype(df[c]) == "numeric"]
    first_unique = df[first].nunique(dropna=True)
    ratio = first_unique / max(1, n_rows)
    looks_feature = _matches(first, FEATURE_HINTS) or first.lower().endswith(("id", "name"))
    if n_cols >= 3 and len(numeric_others) >= max(2, n_cols - 2) and (looks_feature or ratio >= 0.6) and n_rows >= 3:
        if n_rows > n_cols or looks_feature:
            return "matrix", first
    return "tidy", None
