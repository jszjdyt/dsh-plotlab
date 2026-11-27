"""PlotLab 绘图工作台 —— 22 个图表渲染 + Python/R 代码生成。

契约见 DESIGN.md §5。本模块顶只 import：标准库 + numpy/pandas/scipy/sklearn/matplotlib/
seaborn/lifelines（后两者按需惰性 import）+ pl_common + pl_prep。

约定：
- 图内文字（标题 / 轴标签 / 图例 / 图上注释）一律英文；返回 UI 的 summary.bullets 用中文。
- 所有 params 都做兜底：非法值回默认，不抛未处理异常（业务性错误用 pl_common 的 EngineError）。
- 显著性检验：2 组 Welch t / Wilcoxon，>2 组 one-way ANOVA / Kruskal + 两两检验，
  p 值按 style["p_adjust"] 校正；结果同时进返回的 stats / tables / 图上括号。
- 渲染入口 render() 负责：resolve style → style_context → 画图 → finalize_figure → save_fig。
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle

import pl_common as C
import pl_prep as P

_POSITIVE_HINTS = {
    "1", "yes", "true", "positive", "pos", "case", "event", "dead", "death", "tumor",
    "tumour", "high", "abnormal", "disease", "responder", "response", "阳", "阳性",
    "有病", "复发", "死亡", "事件",
}
_AGG_LABEL = {"mean": "均值", "median": "中位数", "sum": "总和"}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _params_dict(params: Any) -> dict:
    return dict(params) if isinstance(params, Mapping) else {}


def _resolve_style(style: Any) -> dict:
    try:
        return C.resolve_style(style if isinstance(style, Mapping) else None)
    except Exception:  # noqa: BLE001 - 风格兜底
        return C.resolve_style(None)


def _pbool(params: Mapping[str, Any], key: str, default: bool) -> bool:
    value = params.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y", "t", "是")
    if value is None:
        return bool(default)
    try:
        return bool(value)
    except Exception:  # noqa: BLE001
        return bool(default)


def _pfloat(params: Mapping[str, Any], key: str, default: float,
            lo: float | None = None, hi: float | None = None) -> float:
    value = params.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    if lo is not None:
        number = max(lo, number)
    if hi is not None:
        number = min(hi, number)
    return float(number)


def _pint(params: Mapping[str, Any], key: str, default: int,
          lo: int | None = None, hi: int | None = None) -> int:
    return int(round(_pfloat(params, key, default, lo, hi)))


def _pchoice(params: Mapping[str, Any], key: str, default: str,
             options: Sequence[str]) -> str:
    raw = params.get(key, default)
    text = str(raw if raw is not None else default).strip().lower()
    for option in options:
        if str(option).strip().lower() == text:
            return str(option)
    return str(default)


def _ptext(params: Mapping[str, Any], key: str, default: str = "") -> str:
    raw = params.get(key, default)
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        return ",".join(str(item) for item in raw)
    return str(raw)


def _split_terms(text: Any) -> list[str]:
    raw = _ptext({"v": text}, "v", "")
    for char in (";", "、", " ", "\n", "\t"):
        raw = raw.replace(char, ",")
    return [item.strip().strip("\"'") for item in raw.split(",") if item.strip()]


def _numeric(series: Any) -> np.ndarray:
    try:
        values = pd.to_numeric(series, errors="coerce")
        return values.to_numpy(dtype=float)
    except Exception:  # noqa: BLE001
        return np.asarray([], dtype=float)


def _finite(values: Any) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float).ravel()
    except Exception:  # noqa: BLE001
        arr = np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _fmt_level(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)[:10]
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "NA"
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return C.fmt_num(number, 4)
    text = str(value)
    return text if len(text) <= 48 else text[:45] + "..."


def _levels_of(frame: Any, column: str | None, max_levels: int = 30,
               warnings: list[str] | None = None) -> list:
    if not column:
        return []
    try:
        series = frame[column]
    except Exception:  # noqa: BLE001
        return []
    valid = series.dropna()
    if valid.empty:
        return []
    try:
        if (pd.api.types.is_numeric_dtype(valid)
                and not pd.api.types.is_bool_dtype(valid)
                and not pd.api.types.is_datetime64_any_dtype(valid)):
            levels = sorted(pd.unique(valid).tolist())
        else:
            levels = pd.unique(valid).tolist()
    except Exception:  # noqa: BLE001
        return []
    if len(levels) > max_levels:
        if warnings is not None:
            warnings.append(f"类别数 {len(levels)} 过多，仅展示前 {max_levels} 个")
        levels = levels[:max_levels]
    return levels


def _positions(levels: Sequence[Any]) -> dict[str, float]:
    return {_fmt_level(level): float(index + 1) for index, level in enumerate(levels)}


def _colors(style: Mapping[str, Any], levels: Sequence[Any]) -> dict[str, str]:
    """按 levels 生成 {格式化的类别名: #HEX}，支持 style["colors"] 覆盖。"""
    levels = list(levels)
    if not levels:
        return {}
    base = C.color_list(style, max(1, len(levels)))
    overrides = style.get("colors") or {}
    out: dict[str, str] = {}
    for index, level in enumerate(levels):
        name = _fmt_level(level)
        color = overrides.get(str(level)) or overrides.get(name) or base[index % len(base)]
        out[name] = str(color).upper()
    return out


def _maybe_n(label: Any, count: Any, style: Mapping[str, Any]) -> str:
    text = _fmt_level(label) if not isinstance(label, str) else label
    if style.get("show_n") and count:
        return f"{text}\n(n={int(count)})"
    return text


def _sample_frame(frame: pd.DataFrame, limit: int = 3000,
                  warnings: list[str] | None = None) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    rng = np.random.default_rng(20261124)
    index = rng.choice(len(frame), size=int(limit), replace=False)
    index.sort()
    if warnings is not None:
        warnings.append(
            f"数据点 {len(frame)} 个，散点已随机抽样 {limit} 个以保证渲染速度（统计量仍按全量计算）"
        )
    return frame.iloc[index]


def _top_variance(matrix: Any, names: Sequence[str], limit: int,
                  warnings: list[str] | None = None, label: str = "特征") -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise C.prerequisite("矩阵数据维度异常，无法绘制")
    if limit <= 0 or arr.shape[1] <= limit:
        return arr, [str(name) for name in names]
    variance = np.nanvar(arr, axis=0)
    order = np.argsort(-np.nan_to_num(variance, nan=-1.0))[:limit]
    order.sort()
    if warnings is not None:
        warnings.append(f"{label}数 {arr.shape[1]} 超过上限 {limit}，已按方差保留前 {limit} 个")
    return arr[:, order], [str(names[i]) for i in order]


def _is_numeric_series(series: Any) -> bool:
    try:
        valid = series.dropna()
        if valid.empty:
            return False
        if pd.api.types.is_numeric_dtype(valid) and not pd.api.types.is_bool_dtype(valid):
            return True
        return _finite(pd.to_numeric(valid, errors="coerce")).size >= max(3, int(len(valid) * 0.8))
    except Exception:  # noqa: BLE001
        return False


def _jsonable(value: Any) -> Any:
    return C.jsonable(value)


# ---------------------------------------------------------------------------
# 描述统计 / 误差线
# ---------------------------------------------------------------------------


def _summarize(values: Any, agg: str = "mean", error: str = "se") -> tuple[float, float, int]:
    arr = _finite(values)
    n = int(arr.size)
    if n == 0:
        return float("nan"), 0.0, 0
    if agg == "median":
        center = float(np.median(arr))
    elif agg == "sum":
        center = float(np.sum(arr))
    else:
        center = float(np.mean(arr))
    err = 0.0
    if error in ("sd", "se", "ci95") and n > 1:
        sd = float(np.std(arr, ddof=1))
        if error == "sd":
            err = sd
        elif error == "se":
            err = sd / math.sqrt(n)
        else:
            try:
                err = float(sps.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
            except Exception:  # noqa: BLE001
                err = 1.96 * sd / math.sqrt(n)
    return center, float(err), n


def _box_stats(values: Any) -> dict | None:
    arr = np.sort(_finite(values))
    if arr.size == 0:
        return None
    q1, med, q3 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
    iqr = q3 - q1
    low = arr[arr >= q1 - 1.5 * iqr]
    high = arr[arr <= q3 + 1.5 * iqr]
    whislo = float(low.min()) if low.size else float(arr.min())
    whishi = float(high.max()) if high.size else float(arr.max())
    fliers = [float(v) for v in arr[(arr < whislo) | (arr > whishi)]]
    return {"med": med, "q1": q1, "q3": q3, "whislo": whislo, "whishi": whishi,
            "fliers": fliers, "n": int(arr.size)}


# ---------------------------------------------------------------------------
# 组间统计（bar / bar_dot / box / violin / raincloud 共用）
# ---------------------------------------------------------------------------


def _p_text(p: float) -> str:
    text = C.fmt_p(p)
    if text.startswith("<") or text.startswith(">"):
        return "p " + text
    return "p = " + text


def _sig_label(p: Any, style: Mapping[str, Any] | None) -> str:
    try:
        value = float(p)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    mode = str((style or {}).get("annot_style") or "stars")
    stars = C.p_stars(value)
    if mode == "p":
        return _p_text(value)
    if mode == "both":
        return f"{stars} {_p_text(value)}".strip()
    return stars


def _normal_ok(values: Any) -> bool:
    arr = _finite(values)
    if arr.size < 3:
        return False
    if arr.size > 5000:
        arr = np.random.default_rng(0).choice(arr, 5000, replace=False)
    if float(np.std(arr)) <= 1e-12:
        return False
    try:
        return float(sps.shapiro(arr).pvalue) > 0.05
    except Exception:  # noqa: BLE001
        return False


def _t_p(a: np.ndarray, b: np.ndarray, unequal: bool = True) -> float:
    try:
        return float(sps.ttest_ind(a, b, equal_var=not unequal, nan_policy="omit").pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def _mw_p(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = int(a.size), int(b.size)
    if na < 2 or nb < 2:
        return float("nan")
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(1, na + nb - 2))
    if pooled <= 1e-12:
        return 0.0
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled)


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    try:
        u = float(sps.mannwhitneyu(a, b, alternative="two-sided").statistic)
        return float(2.0 * u / max(1.0, float(a.size * b.size)) - 1.0)
    except Exception:  # noqa: BLE001
        return float("nan")


def _safe_p(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _compare_groups(frame: pd.DataFrame, factor_col: str, value_col: str,
                    style: Mapping[str, Any],
                    levels: Sequence[Any] | None = None,
                    warnings: list[str] | None = None,
                    factor: str | None = None) -> dict | None:
    """组间比较：2 组 Welch t / Wilcoxon；>2 组 ANOVA / Kruskal + 两两检验。失败→None + warning。"""
    if warnings is None:
        warnings = []
    if factor_col not in frame.columns or value_col not in frame.columns:
        return None
    work = frame[[factor_col, value_col]].dropna().copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work[np.isfinite(work[value_col].to_numpy(dtype=float))]
    if work.empty:
        warnings.append("组间比较跳过：没有有效观测")
        return None
    if levels is None:
        levels = _levels_of(work, factor_col, max_levels=8)
    names: list[str] = []
    arrays: list[np.ndarray] = []
    for level in levels:
        values = work.loc[work[factor_col] == level, value_col].to_numpy(dtype=float)
        values = _finite(values)
        if values.size == 0:
            continue
        names.append(_fmt_level(level))
        arrays.append(values)
    if len(arrays) < 2:
        warnings.append("组间比较跳过：有效分组少于 2 组")
        return None
    if min(int(a.size) for a in arrays) < 3:
        warnings.append("组间比较跳过：至少 1 组有效样本少于 3 个")
        return None
    if len(arrays) > 8:
        warnings.append(f"分组数 {len(arrays)} 过多，已跳过组间显著性检验")
        return None

    method = str(style.get("stat_method") or "auto").lower()
    p_adjust = str(style.get("p_adjust") or "bh").lower()
    pairs: list[dict] = []
    if len(arrays) == 2:
        a, b = arrays
        if method == "wilcox":
            test, p_value = "Wilcoxon rank-sum", _mw_p(a, b)
            effect_name, effect = "rank-biserial r", _rank_biserial(a, b)
        elif method in ("welch", "student"):
            test = "Student t-test" if method == "student" else "Welch t-test"
            p_value, effect_name, effect = _t_p(a, b, method != "student"), "Cohen's d", _cohen_d(a, b)
        elif _normal_ok(a) and _normal_ok(b):
            test, p_value = "Welch t-test", _t_p(a, b, True)
            effect_name, effect = "Cohen's d", _cohen_d(a, b)
        else:
            test, p_value = "Wilcoxon rank-sum", _mw_p(a, b)
            effect_name, effect = "rank-biserial r", _rank_biserial(a, b)
        pairs.append({"a": names[0], "b": names[1], "test": test, "p": _safe_p(p_value),
                      "n_a": int(a.size), "n_b": int(b.size),
                      "effect_name": effect_name, "effect": float(effect)})
        global_info = {"test": test, "p": _safe_p(p_value), "statistic": float("nan"),
                       "n_groups": 2}
    else:
        parametric = method in ("welch", "student", "anova") or (
            method == "auto" and all(_normal_ok(a) for a in arrays)
        )
        if parametric:
            try:
                statistic, p_value = sps.f_oneway(*arrays)
                test = "One-way ANOVA"
            except Exception:  # noqa: BLE001
                statistic, p_value = float("nan"), float("nan")
                test = "One-way ANOVA"
            pairwise_test, effect_name = "Welch t-test", "Cohen's d"
        else:
            try:
                statistic, p_value = sps.kruskal(*arrays)
                test = "Kruskal-Wallis"
            except Exception:  # noqa: BLE001
                statistic, p_value = float("nan"), float("nan")
                test = "Kruskal-Wallis"
            pairwise_test, effect_name = "Wilcoxon rank-sum", "rank-biserial r"
        all_values = np.concatenate(arrays)
        grand = float(np.mean(all_values))
        ss_between = sum(a.size * (float(np.mean(a)) - grand) ** 2 for a in arrays)
        ss_total = float(np.sum((all_values - grand) ** 2))
        eta = float(ss_between / ss_total) if ss_total > 1e-12 else 0.0
        global_info = {"test": test, "p": _safe_p(p_value), "statistic": float(statistic),
                       "n_groups": len(arrays), "effect_name": "eta squared", "effect": eta}
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                a, b = arrays[i], arrays[j]
                if parametric:
                    p_raw, effect = _t_p(a, b, True), _cohen_d(a, b)
                else:
                    p_raw, effect = _mw_p(a, b), _rank_biserial(a, b)
                pairs.append({"a": names[i], "b": names[j], "test": pairwise_test,
                              "p": _safe_p(p_raw), "n_a": int(a.size), "n_b": int(b.size),
                              "effect_name": effect_name, "effect": float(effect)})

    factor_name = str(factor or "x")
    for pair in pairs:
        pair.setdefault("contrast_id", f"{factor_name}:{pair['a']}|{pair['b']}")
        pair.setdefault("contrast", f"{pair['a']} vs {pair['b']}")
    finite_pairs = [pair for pair in pairs if math.isfinite(pair["p"])]
    adjusted = iter(C.adjust_pvalues([pair["p"] for pair in finite_pairs], p_adjust)) if finite_pairs else iter([])
    adjusted = list(adjusted)
    cursor = 0
    for pair in pairs:
        if math.isfinite(pair["p"]):
            pair["p_adj"] = float(adjusted[cursor]) if cursor < len(adjusted) else pair["p"]
            cursor += 1
        else:
            pair["p_adj"] = float("nan")
        pair["label"] = _sig_label(pair["p_adj"], style)
    return {
        "method": str(global_info["test"]),
        "factor": factor_name,
        "global": global_info,
        "levels": names,
        "n": {name: int(values.size) for name, values in zip(names, arrays)},
        "pairs": pairs,
        "p_adjust": p_adjust,
        "stat_method": str(style.get("stat_method") or "auto"),
    }


def _compare_table(result: Mapping[str, Any] | None, tid: str = "compare") -> dict | None:
    if not result:
        return None
    rows = []
    for pair in result.get("pairs") or []:
        rows.append({
            "pair": f"{pair['a']} vs {pair['b']}",
            "test": pair["test"],
            "n1": pair["n_a"],
            "n2": pair["n_b"],
            "p": pair["p"],
            "p_adj": pair.get("p_adj"),
            "effect": f"{pair.get('effect_name', 'effect')} = {C.fmt_num(pair.get('effect'))}",
            "sig": pair.get("label") or "ns",
        })
    return C.table(
        tid,
        f"组间比较（{result.get('method')}，{result.get('p_adjust')} 校正）",
        [
            {"key": "pair", "label": "比较", "type": "str"},
            {"key": "test", "label": "检验方法", "type": "str"},
            {"key": "n1", "label": "n1", "type": "num"},
            {"key": "n2", "label": "n2", "type": "num"},
            {"key": "p", "label": "p", "type": "p"},
            {"key": "p_adj", "label": "校正 p", "type": "p"},
            {"key": "effect", "label": "效应量", "type": "str"},
            {"key": "sig", "label": "显著性", "type": "str"},
        ],
        rows,
    )


def _compare_bullets(result: Mapping[str, Any] | None) -> list[str]:
    if not result:
        return []
    bullets: list[str] = []
    levels = result.get("levels") or []
    global_info = result.get("global") or {}
    if len(levels) == 2 and result.get("pairs"):
        pair = result["pairs"][0]
        significant = math.isfinite(pair.get("p_adj", float("nan"))) and pair["p_adj"] < 0.05
        bullets.append(
            f"两组比较（{pair['test']}）：{pair['a']} vs {pair['b']}，p = {C.fmt_p(pair['p'])}"
            f"（校正后 {C.fmt_p(pair.get('p_adj'))}），"
            f"{pair.get('effect_name', 'effect')} = {C.fmt_num(pair.get('effect'))}，"
            f"{'差异有统计学意义' if significant else '差异无统计学意义'}"
        )
    elif len(levels) > 2:
        g_p = global_info.get("p", float("nan"))
        significant = math.isfinite(g_p) and g_p < 0.05
        bullets.append(
            f"{len(levels)} 组整体（{result.get('method')}）：p = {C.fmt_p(g_p)}，"
            f"{'组间差异有统计学意义' if significant else '组间差异无统计学意义'}"
            + (f"，{global_info.get('effect_name')} = {C.fmt_num(global_info.get('effect'))}"
               if global_info.get("effect_name") else "")
        )
        pairs = result.get("pairs") or []
        hits = [pair for pair in pairs if math.isfinite(pair.get("p_adj", float("nan"))) and pair["p_adj"] < 0.05]
        if pairs:
            bullets.append(
                f"两两比较中 {len(hits)}/{len(pairs)} 对达到显著（{result.get('p_adjust')} 校正）"
            )
    return bullets


def _annotate_stats(ax, result: Mapping[str, Any] | None, positions: Mapping[str, float],
                    style: Mapping[str, Any], warnings: list[str] | None = None) -> None:
    """在 ax 顶部画显著性括号；数据不足以定位/对数轴时跳过。"""
    if not result:
        return
    if style.get("y_log"):
        if warnings is not None:
            warnings.append("y 轴为对数刻度，已跳过显著性括号标注")
        return
    pairs = [pair for pair in (result.get("pairs") or [])
             if pair.get("label") and pair["a"] in positions and pair["b"] in positions]
    if not pairs:
        return
    if len(pairs) > 6:
        pairs = pairs[:6]
        if warnings is not None:
            warnings.append("两两比较超过 6 对，图上仅标注前 6 对（全部结果见统计表）")
    y0, y1 = ax.get_ylim()
    span = float(y1 - y0)
    if not math.isfinite(span) or span <= 0:
        span = max(abs(float(y1)), 1.0)
    step = span * 0.085
    top = float(y1)
    text_size = max(6.0, float(style.get("text_size", 10)) - 2)
    line_width = max(0.6, float(style.get("line_width", 1.4)) * 0.7)
    for index, pair in enumerate(pairs):
        xa = float(positions[pair["a"]])
        xb = float(positions[pair["b"]])
        y = top + span * 0.05 + index * step
        tick = span * 0.02
        ax.plot([xa, xa, xb, xb], [y - tick, y, y, y - tick], color="#3A3A3A",
                linewidth=line_width, clip_on=False, zorder=6)
        ax.text((xa + xb) / 2.0, y + tick * 0.35, str(pair["label"]), ha="center",
                va="bottom", fontsize=text_size, color="#1A1A1A", clip_on=False, zorder=7)
    ax.set_ylim(y0, top + span * 0.05 + len(pairs) * step + span * 0.05)


# ---------------------------------------------------------------------------
# 图例代理（finalize_figure 统一收集各 axes 的 handles）
# ---------------------------------------------------------------------------


def _legend_marker(ax, label: Any, color: str, marker: str = "o", size: float = 8.0,
                   alpha: float = 1.0) -> None:
    ax.plot([], [], linestyle="none", marker=marker, markersize=size,
            markerfacecolor=color, markeredgecolor="none", alpha=alpha,
            label=str(label))


def _legend_line(ax, label: Any, color: str, linestyle: str = "-",
                 linewidth: float = 1.4) -> None:
    ax.plot([], [], color=color, linestyle=linestyle, linewidth=linewidth,
            label=str(label))


def _legend_patch(ax, label: Any, color: str, alpha: float = 1.0) -> None:
    """图例色块代理：画在 axes 外（被裁剪掉），但 finalize 收进 fig 级图例。"""
    patch = Rectangle((2.5, 2.5), 1.0, 1.0, facecolor=color, edgecolor="none",
                      alpha=alpha, label=str(label), transform=ax.transAxes,
                      clip_on=True)
    ax.add_artist(patch)


# ---------------------------------------------------------------------------
# 轴标签 / 辅助轴
# ---------------------------------------------------------------------------


def _apply_labels(ax, style: Mapping[str, Any], x: Any = None, y: Any = None) -> None:
    size = float(style.get("label_size", 12))
    if x is not None:
        ax.set_xlabel(str(style.get("x_label") or x), fontsize=size)
    if y is not None:
        ax.set_ylabel(str(style.get("y_label") or y), fontsize=size)


def _restore_aux(ax, style: Mapping[str, Any], xlim, ylim, xscale, yscale) -> None:
    """finalize_figure 会把全局 style 套到 colorbar / 风险表等辅助轴上，这里恢复原状。"""
    try:
        # set_*scale 会重置 tick locator/formatter，只有确实被 finalize 改掉时才恢复
        if ax.get_xscale() != xscale:
            ax.set_xscale(xscale)
        if ax.get_yscale() != yscale:
            ax.set_yscale(yscale)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis="both", which="major", labelsize=float(style.get("tick_size", 10)))
    except Exception:  # noqa: BLE001
        pass


def _fallback_numeric(data: Mapping[str, Any],
                      warnings: list[str] | None = None) -> tuple[str, np.ndarray] | None:
    """x 不是数值型时，退而用数据集里的第一个数值列（宽表视图提供值）。"""
    wide = data.get("wide") or {}
    features = [str(name) for name in (wide.get("features") or [])]
    matrix = wide.get("matrix")
    if not features or matrix is None:
        return None
    numeric_cols = [str(name) for name in (data.get("numeric_cols") or [])]
    pool = [name for name in numeric_cols if name in features]
    pick = pool[0] if pool else features[0]
    index = features.index(pick)
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or index >= arr.shape[1]:
        return None
    if warnings is not None:
        warnings.append(
            f"x 列不是数值型，已自动改用数值列 {pick} 绘制；可在左侧把 x 角色换成需要的数值列"
        )
    return pick, arr[:, index]


# ---------------------------------------------------------------------------
# 排序 / 方差筛选
# ---------------------------------------------------------------------------


def _reorder(values: Any, names: Sequence[str], cluster_rows: bool = False,
             cluster_cols: bool = False) -> tuple[Any, list[str], list[int]]:
    """（热图用）按层次聚类排序矩阵行/列；失败则保持原顺序。"""
    arr = np.asarray(values, dtype=float)
    row_order = list(range(arr.shape[0]))
    col_order = list(range(arr.shape[1]))
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
    except Exception:  # noqa: BLE001
        return arr, list(names), col_order
    if cluster_rows and arr.shape[0] >= 3:
        try:
            row_order = [int(v) for v in leaves_list(linkage(arr, method="average"))]
        except Exception:  # noqa: BLE001
            row_order = list(range(arr.shape[0]))
    if cluster_cols and arr.shape[1] >= 3:
        try:
            col_order = [int(v) for v in leaves_list(linkage(arr.T, method="average"))]
        except Exception:  # noqa: BLE001
            col_order = list(range(arr.shape[1]))
    return arr[row_order][:, col_order], [str(names[i]) for i in row_order], col_order


# ---------------------------------------------------------------------------
# 图表目录（顺序即 UI 顺序，共 22 个）
# ---------------------------------------------------------------------------


def _param(key: str, label: str, kind: str, default: Any, **extra: Any) -> dict:
    item = {"key": key, "label": label, "type": kind, "default": default}
    item.update(extra)
    return item


def _select(key: str, label: str, default: str, options: Sequence[tuple[str, str]]) -> dict:
    return {
        "key": key,
        "label": label,
        "type": "select",
        "default": default,
        "options": [{"value": value, "label": text} for value, text in options],
    }


_ERROR_OPTIONS = [
    ("auto", "跟随风格设置"),
    ("sd", "标准差 SD"),
    ("se", "标准误 SE"),
    ("ci95", "95% 置信区间"),
    ("none", "不画误差线"),
]
_AGG_OPTIONS = [("mean", "均值"), ("median", "中位数"), ("sum", "总和")]
_STAT_METHOD_OPTIONS = [
    ("pearson", "Pearson 线性相关"),
    ("spearman", "Spearman 秩相关"),
    ("kendall", "Kendall 秩相关"),
]

CHARTS: list[dict] = [
    # ---------------- 比较 ----------------
    {
        "id": "bar",
        "title": "柱状图",
        "group": "比较",
        "desc": "按分类聚合数值画柱，支持均值/中位数/求和的误差线与散点叠加，可自动标注组间显著性。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "group": "optional"},
        "params": [
            _select("agg", "聚合方式", "mean", _AGG_OPTIONS),
            _select("error", "误差线", "auto", _ERROR_OPTIONS),
            _param("points", "叠加原始散点", "boolean", True),
        ],
        "stat": True,
    },
    {
        "id": "bar_dot",
        "title": "柱散点图",
        "group": "比较",
        "desc": "柱表示聚合值，叠加每个观测的散点，适合样本量不大、想同时看分布与均值的比较。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "group": "optional"},
        "params": [
            _select("agg", "聚合方式", "mean", _AGG_OPTIONS),
            _select("error", "误差线", "auto", _ERROR_OPTIONS),
        ],
        "stat": True,
    },
    {
        "id": "box",
        "title": "箱线图",
        "group": "比较",
        "desc": "箱线图展示中位数、四分位距与离群值，可叠加抖动散点，并自动标注组间显著性。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "group": "optional"},
        "params": [
            _param("outliers", "显示离群点", "boolean", True),
            _param("points", "叠加抖动散点", "boolean", True),
        ],
        "stat": True,
    },
    {
        "id": "violin",
        "title": "小提琴图",
        "group": "比较",
        "desc": "小提琴图用核密度展示分布形状，内嵌箱线/中位数，可分组并标注显著性。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "group": "optional"},
        "params": [
            _select("inner", "内部图形", "box", [("box", "迷你箱线"), ("quartile", "分位线"), ("none", "不画")]),
            _param("bw", "带宽系数（1=自动）", "number", 1.0, min=0.2, max=4.0, step=0.1),
            _param("trim", "截断到数据范围", "boolean", True),
        ],
        "stat": True,
    },
    {
        "id": "raincloud",
        "title": "雨云图",
        "group": "比较",
        "desc": "半小提琴（云）+ 箱线 + 抖动散点（雨），同时呈现分布形状、分位数与个体观测。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "group": "optional"},
        "params": [
            _param("box", "显示箱线", "boolean", True),
            _param("points", "显示散点", "boolean", True),
        ],
        "stat": True,
    },
    {
        "id": "dotplot",
        "title": "点图",
        "group": "比较",
        "desc": "分类矩阵点图：横轴为列分类、纵轴为分组，点的大小和颜色表示聚合数值（如均值）。",
        "needs": "tidy",
        "roles": {"x": "required", "group": "optional", "color": "optional", "size": "optional"},
        "params": [
            _select("agg", "聚合方式", "mean", _AGG_OPTIONS),
            _param("cmap", "连续色系（空=自动）", "text", ""),
            _param("show_values", "点上标数值", "boolean", False),
        ],
        "stat": False,
    },
    {
        "id": "stack",
        "title": "堆叠柱状图",
        "group": "比较",
        "desc": "按 x 分类堆积各组占比或绝对值，适合看构成比例随分组的变化。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "group": "optional"},
        "params": [
            _param("normalize", "归一化为百分比", "boolean", False),
            _select("agg", "聚合方式", "sum", _AGG_OPTIONS),
            _param("show_total", "柱顶标总数", "boolean", False),
        ],
        "stat": False,
    },
    {
        "id": "pie",
        "title": "饼图 / 环图",
        "group": "比较",
        "desc": "各类别的占比饼图，可切换为环图；标签带数量与百分比。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "optional"},
        "params": [
            _param("donut", "环图（中空）", "boolean", True),
            _param("show_pct", "显示百分比", "boolean", True),
            _param("start_angle", "起始角度", "number", 90.0, min=0.0, max=360.0, step=15.0),
        ],
        "stat": False,
    },
    # ---------------- 关系 ----------------
    {
        "id": "scatter",
        "title": "散点图",
        "group": "关系",
        "desc": "两连续变量的散点图，可选回归线 + 95% 置信带 + 相关系数标注，支持分组着色与气泡大小。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "color": "optional", "size": "optional"},
        "params": [
            _param("add_fit", "回归线", "boolean", True),
            _param("add_ci", "回归 95% 置信带", "boolean", True),
            _param("add_corr", "相关系数标注", "boolean", True),
            _select("method", "相关方法", "pearson", _STAT_METHOD_OPTIONS),
            _param("x_bins", "横向分箱均值点数（0=关闭）", "number", 0, min=0, max=30, step=1),
        ],
        "stat": False,
    },
    {
        "id": "bubble",
        "title": "气泡图",
        "group": "关系",
        "desc": "在散点基础上用气泡大小编码第三个连续变量，可同时按分组着色。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "color": "optional", "size": "optional"},
        "params": [
            _param("add_fit", "回归线", "boolean", False),
            _param("size_min", "最小气泡面积", "number", 20.0, min=4.0, max=1000.0, step=10.0),
            _param("size_max", "最大气泡面积", "number", 420.0, min=20.0, max=5000.0, step=20.0),
        ],
        "stat": False,
    },
    {
        "id": "line",
        "title": "折线图",
        "group": "关系",
        "desc": "按分组聚合后连线，可加误差带/误差线和数据点，适合时间或剂量趋势。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "group": "optional"},
        "params": [
            _select("agg", "聚合方式", "mean", _AGG_OPTIONS),
            _select("error", "误差带", "auto", _ERROR_OPTIONS),
            _param("marker", "显示数据点", "boolean", True),
        ],
        "stat": False,
    },
    {
        "id": "area",
        "title": "面积图",
        "group": "关系",
        "desc": "按 x 聚合后填色面积，可堆叠各组或分别绘制，适合看累计量与构成变化。",
        "needs": "tidy",
        "roles": {"x": "required", "y": "required", "group": "optional"},
        "params": [
            _select("agg", "聚合方式", "mean", _AGG_OPTIONS),
            _param("stack", "堆叠各组", "boolean", True),
            _param("normalize", "归一化为百分比", "boolean", False),
        ],
        "stat": False,
    },
    {
        "id": "heatmap",
        "title": "热图",
        "group": "关系",
        "desc": "矩阵热图：行=特征、列=样本，支持按行/列 z-score、层次聚类排序、数值标注与样本分组着色。",
        "needs": "wide",
        "roles": {},
        "params": [
            _param("top_n", "特征数上限（按方差）", "number", 60, min=2, max=500, step=10),
            _select("zscore", "标准化", "row", [("none", "不标准化"), ("row", "按行 z-score"), ("col", "按列 z-score")]),
            _param("cluster_rows", "行聚类排序", "boolean", True),
            _param("cluster_cols", "列聚类排序", "boolean", True),
            _param("annotate", "格子里标数值", "boolean", False),
            _param("cmap", "连续色系（空=自动）", "text", ""),
            _param("sample_group_col", "样本分组列（矩阵数据）", "text", "group"),
        ],
        "stat": False,
    },
    {
        "id": "corr",
        "title": "相关矩阵",
        "group": "关系",
        "desc": "多变量相关矩阵（Pearson/Spearman/Kendall），格子标注相关系数与显著性星号。",
        "needs": "wide",
        "roles": {},
        "params": [
            _select("method", "相关方法", "pearson", _STAT_METHOD_OPTIONS),
            _param("star", "显示显著性星号", "boolean", True),
            _param("annotate", "标注相关系数", "boolean", True),
            _param("top_n", "变量数上限（按方差）", "number", 40, min=2, max=200, step=5),
        ],
        "stat": False,
    },
    {
        "id": "pairplot",
        "title": "散点矩阵",
        "group": "关系",
        "desc": "多个数值变量的两两散点与对角分布，按分组着色；变量数上限 6 个，可在参数里选列。",
        "needs": "tidy",
        "roles": {"x": "optional", "y": "optional", "color": "optional"},
        "params": [
            _param("vars", "选列（逗号分隔，空=前 6 个数值列）", "text", ""),
            _select("diag", "对角图形", "hist", [("hist", "直方图"), ("kde", "密度曲线")]),
            _param("show_r", "上三角标相关系数", "boolean", False),
        ],
        "stat": False,
    },
    # ---------------- 分布 ----------------
    {
        "id": "hist",
        "title": "直方图",
        "group": "分布",
        "desc": "连续变量分布直方图，可分组叠加或堆叠，可切换为密度，并标注均值/中位数。",
        "needs": "tidy",
        "roles": {"x": "required", "color": "optional"},
        "params": [
            _param("bins", "分箱数（0=自动）", "number", 20, min=0, max=200, step=1),
            _param("density", "按密度归一化", "boolean", False),
            _param("stack", "堆叠分组", "boolean", False),
            _param("show_stats", "标注均值/中位数", "boolean", True),
        ],
        "stat": False,
    },
    {
        "id": "density",
        "title": "密度图",
        "group": "分布",
        "desc": "核密度曲线展示连续变量分布，支持分组填色、带宽调节与地毯线。",
        "needs": "tidy",
        "roles": {"x": "required", "color": "optional"},
        "params": [
            _param("fill", "曲线下填色", "boolean", True),
            _param("bw", "带宽系数（1=自动）", "number", 1.0, min=0.2, max=4.0, step=0.1),
            _param("rug", "底部地毯线", "boolean", False),
        ],
        "stat": False,
    },
    {
        "id": "qq",
        "title": "QQ 图",
        "group": "分布",
        "desc": "分位数-分位数图检验分布是否接近指定理论分布（默认正态），并给出 Shapiro-Wilk 检验。",
        "needs": "tidy",
        "roles": {"x": "required", "color": "optional"},
        "params": [
            _select("dist", "理论分布", "normal", [("normal", "正态"), ("uniform", "均匀"), ("exponential", "指数")]),
        ],
        "stat": False,
    },
    # ---------------- 统计 ----------------
    {
        "id": "roc",
        "title": "ROC 曲线",
        "group": "统计",
        "desc": "二分类诊断曲线：y=分数/风险值、group=0/1 标签，给出 AUC、bootstrap 置信区间与最佳阈值。",
        "needs": "tidy",
        "roles": {"y": "required", "group": "required"},
        "params": [
            _param("positive", "阳性水平（空=自动）", "text", ""),
            _param("add_ci", "bootstrap AUC 置信区间", "boolean", True),
        ],
        "stat": False,
    },
    {
        "id": "km",
        "title": "生存曲线",
        "group": "统计",
        "desc": "Kaplan-Meier 生存曲线，带置信带与删失标记，自动做 log-rank 检验，可加风险表。",
        "needs": "tidy",
        "roles": {"time": "required", "event": "required", "group": "optional"},
        "params": [
            _param("ci", "95% 置信带", "boolean", True),
            _param("censor", "标记删失", "boolean", True),
            _param("risk_table", "风险表", "boolean", False),
        ],
        "stat": False,
    },
    {
        "id": "forest",
        "title": "森林图",
        "group": "统计",
        "desc": "效应量（HR/OR/β）与置信区间的森林图；可用 params.estimates 直接传表，或用 time/event 角色做单因素 Cox。",
        "needs": "tidy",
        "roles": {"time": "optional", "event": "optional"},
        "params": [
            _select("model", "模型", "auto", [("auto", "自动（有生存列用 Cox）"), ("cox", "Cox 单因素 HR")]),
            _param("terms", "纳入变量（逗号分隔，空=全部）", "text", ""),
            _param("estimates", "直接传入估计值表（JSON）", "text", ""),
            _param("ref_line", "参考线", "number", 1.0, min=-10.0, max=10.0, step=0.1),
            _param("log_scale", "对数横轴", "boolean", True),
            _param("show_text", "显示数值标签", "boolean", True),
        ],
        "stat": False,
    },
    {
        "id": "pca",
        "title": "PCA 主成分分析",
        "group": "统计",
        "desc": "样本 PCA 散点（按样本分组着色、95% 置信椭圆、可选载荷箭头）配碎石图。",
        "needs": "wide",
        "roles": {},
        "params": [
            _param("scale", "变量标准化", "boolean", True),
            _param("ellipse", "95% 置信椭圆", "boolean", True),
            _param("loadings", "载荷箭头", "boolean", False),
            _param("sample_group_col", "样本分组列（矩阵数据）", "text", "group"),
        ],
        "stat": False,
    },
]

CHART_INDEX = {str(chart["id"]): chart for chart in CHARTS}


def chart_meta(chart_id: str) -> dict | None:
    return CHART_INDEX.get(str(chart_id or ""))


# ---------------------------------------------------------------------------
# 比较类图表的公共上下文
# ---------------------------------------------------------------------------


def _comparison_context(data: Mapping[str, Any], style: Mapping[str, Any],
                        warnings: list[str], keys: Sequence[str] = ("__group",)) -> dict:
    frame = P.require_tidy(data, ["__x", "__y"]).copy()
    frame["__y"] = pd.to_numeric(frame["__y"], errors="coerce")
    frame = frame[np.isfinite(frame["__y"].to_numpy(dtype=float))]
    if frame.empty:
        raise C.prerequisite("没有有效的 y 数值：请检查 y 角色对应的列是否为数值列")
    levels = _levels_of(frame, "__x", max_levels=24, warnings=warnings)
    if not levels:
        raise C.bad_request("x 角色没有可用的分类/数值取值，请重新指定 x 列")
    frame = frame[frame["__x"].isin(levels)]
    group_col = _pick_group_col(data, frame, keys=keys)
    group_levels = _levels_of(frame, group_col, max_levels=10, warnings=warnings) if group_col else []
    if group_col and len(group_levels) < 2:
        group_col, group_levels = None, []
    colors = _colors(style, group_levels if group_col else levels)
    labels = data.get("labels") or {}
    counts = {_fmt_level(level): int((frame["__x"] == level).sum()) for level in levels}
    gcounts = (
        {_fmt_level(level): int((frame[group_col] == level).sum()) for level in group_levels}
        if group_col else {}
    )
    return {
        "frame": frame,
        "levels": levels,
        "positions": _positions(levels),
        "group_col": group_col,
        "group_levels": group_levels,
        "group_positions": _positions(group_levels) if group_levels else {},
        "colors": colors,
        "counts": counts,
        "gcounts": gcounts,
        "labels": labels,
        "x_label": str(labels.get("x") or "x"),
        "y_label": str(labels.get("y") or "y"),
    }


def _pick_group_col(data: Mapping[str, Any], frame: pd.DataFrame,
                    keys: Sequence[str] = ("__group",)) -> str | None:
    labels = data.get("labels") or {}
    x_label = str(labels.get("x") or "")
    x_values = None
    if "__x" in frame.columns:
        try:
            x_values = frame["__x"].astype("string").fillna("").to_numpy()
        except Exception:  # noqa: BLE001
            x_values = None
    for key in keys:
        if key not in frame.columns:
            continue
        values = frame[key]
        if values.isna().all():
            continue
        try:
            unique = int(values.nunique(dropna=True))
        except Exception:  # noqa: BLE001
            continue
        if unique < 2:
            continue
        role = key[2:]
        label = str(labels.get(role) or "")
        if label and x_label and label == x_label:
            continue
        if x_values is not None and len(x_values) == len(values):
            try:
                same = (values.astype("string").fillna("").to_numpy() == x_values).all()
                if same:
                    continue
            except Exception:  # noqa: BLE001
                pass
        return key
    return None


def _stat_for_factor(ctx: Mapping[str, Any], style: Mapping[str, Any],
                     warnings: list[str]) -> dict | None:
    """组间比较按 x 主因子做（分组只影响画法，不改变比较对象）。"""
    if len(ctx.get("levels") or []) < 2:
        warnings.append("x 只有 1 个分类，已跳过组间显著性检验")
        return None
    return _compare_groups(ctx["frame"], "__x", "__y", style,
                           levels=ctx["levels"], warnings=warnings,
                           factor=str(ctx.get("x_label") or "x"))


def _comparison_tables(result: Mapping[str, Any] | None,
                       aggregate_rows: Sequence[Mapping[str, Any]],
                       aggregate_columns: Sequence[Mapping[str, Any]],
                       title: str = "分组聚合值") -> list[dict]:
    tables: list[dict] = []
    if aggregate_rows:
        tables.append(C.table("aggregate", title, aggregate_columns, aggregate_rows))
    compare = _compare_table(result)
    if compare:
        tables.append(compare)
    return tables


# ---------------------------------------------------------------------------
# stats / annotations 载荷（返回给 registry 的 stats 字段）
# ---------------------------------------------------------------------------


def _stats_payload(result: Mapping[str, Any] | None) -> dict | None:
    if not result:
        return None
    factor = result.get("factor")
    pairs = list(result.get("pairs") or [])
    return {
        "method": result.get("method"),
        "factor": factor,
        "p_adjust": result.get("p_adjust"),
        "levels": list(result.get("levels") or []),
        "n": dict(result.get("n") or {}),
        "global": dict(result.get("global") or {}),
        "contrasts": [
            {"contrast_id": pair.get("contrast_id"),
             "a": pair.get("a"), "b": pair.get("b"),
             "test": pair.get("test"),
             "p": pair.get("p"), "p_adj": pair.get("p_adj"),
             "effect_name": pair.get("effect_name"), "effect": pair.get("effect")}
            for pair in pairs
        ],
        "pairs": [
            {key: pair.get(key) for key in
             ("a", "b", "contrast_id", "test", "p", "p_adj", "effect_name", "effect", "n_a", "n_b")}
            for pair in pairs
        ],
    }


def _annotations_payload(result: Mapping[str, Any] | None) -> dict | None:
    if not result:
        return None
    factor = result.get("factor")
    pairs = list(result.get("pairs") or [])
    return {
        "method": result.get("method"),
        "factor": factor,
        "contrasts": [
            {"contrast_id": pair.get("contrast_id"), "a": pair.get("a"), "b": pair.get("b"),
             "label": pair.get("label"), "p_adj": pair.get("p_adj")}
            for pair in pairs
        ],
        "pairs": [
            {"a": pair.get("a"), "b": pair.get("b"), "contrast_id": pair.get("contrast_id"),
             "label": pair.get("label"), "p_adj": pair.get("p_adj")}
            for pair in pairs
        ],
    }


def _aggregate_columns(ctx: Mapping[str, Any], value_label: str) -> list[dict]:
    columns = [{"key": "x", "label": str(ctx.get("x_label") or "x"), "type": "str"}]
    if ctx.get("group_col"):
        group_name = str((ctx.get("labels") or {}).get("group") or "group")
        columns.append({"key": "group", "label": group_name, "type": "str"})
    columns.extend([
        {"key": "n", "label": "n", "type": "num"},
        {"key": "value", "label": value_label, "type": "num"},
        {"key": "error", "label": "误差", "type": "num"},
    ])
    return columns


def _pick_cmap(style: Mapping[str, Any], requested: Any = "", diverging: bool = False):
    """选择连续色系：params.cmap > style.palette（若为连续色系）> viridis/RdBu_r。"""
    text = str(requested or "").strip()
    if text:
        if text in C.PALETTE_CMAPS:
            text = C.PALETTE_CMAPS[str(text)]
        try:
            return matplotlib.colormaps[text]
        except Exception:  # noqa: BLE001
            pass
    palette = str(style.get("palette") or "")
    if palette in C.PALETTE_CMAPS:
        return matplotlib.colormaps[C.PALETTE_CMAPS[palette]]
    return matplotlib.colormaps["RdBu_r" if diverging else "viridis"]


def _jitter_sizes(style: Mapping[str, Any]) -> float:
    size = float(style.get("jitter_size", 2.2) or 2.2)
    return max(6.0, (size * 1.7) ** 2)


# ---------------------------------------------------------------------------
# 1) bar —— 柱状图
# ---------------------------------------------------------------------------


def _render_bar(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    agg = _pchoice(params, "agg", "mean", ["mean", "median", "sum"])
    error = _pchoice(params, "error", "auto", ["auto", "sd", "se", "ci95", "none"])
    if error == "auto":
        error = str(style.get("error") or "se")
    show_points = _pbool(params, "points", True)
    ctx = _comparison_context(data, style, warnings)
    frame, levels, positions = ctx["frame"], ctx["levels"], ctx["positions"]
    colors = ctx["colors"]
    group_col, group_levels = ctx["group_col"], ctx["group_levels"]
    bar_width = float(style.get("bar_width") or 0.72)
    alpha = float(style.get("alpha") or 0.85)
    edge = float(style.get("edge_width", 0.8) or 0.8)
    jitter_alpha = float(style.get("jitter_alpha", 0.55) or 0.0)
    point_area = _jitter_sizes(style)
    aggregate_rows: list[dict] = []

    fig, ax = C.new_fig(style)
    if group_col:
        n_groups = max(1, len(group_levels))
        width = bar_width / n_groups
        offsets = [(index - (n_groups - 1) / 2.0) * width for index in range(n_groups)]
        for gi, glevel in enumerate(group_levels):
            gname = _fmt_level(glevel)
            color = colors.get(gname, C.color_list(style, 1)[0])
            xs, heights, errs = [], [], []
            for level in levels:
                mask = (frame["__x"] == level) & (frame[group_col] == glevel)
                center, err, n = _summarize(frame.loc[mask, "__y"].to_numpy(dtype=float), agg, error)
                pos = positions[_fmt_level(level)] + offsets[gi]
                xs.append(pos)
                heights.append(center)
                errs.append(err)
                aggregate_rows.append({"x": _fmt_level(level), "group": gname, "n": n,
                                       "value": center, "error": err})
            ax.bar(xs, heights, width=width, color=color, edgecolor="white",
                   linewidth=edge, alpha=alpha, label=gname, zorder=2)
            if error != "none":
                ax.errorbar(xs, heights, yerr=errs, fmt="none", ecolor="#2B2B2B",
                            elinewidth=max(0.6, edge * 0.9), capsize=3, capthick=0.8, zorder=3)
            if show_points:
                rng = np.random.default_rng(11 + gi)
                for level, pos in zip(levels, xs):
                    values = _finite(frame.loc[(frame["__x"] == level) & (frame[group_col] == glevel), "__y"])
                    if values.size:
                        jitter = rng.normal(0.0, width * 0.13, size=values.size)
                        ax.scatter(pos + jitter, values, s=point_area, color="#3A3A3A",
                                   alpha=jitter_alpha, linewidths=0, zorder=4)
        ax.set_xticks([positions[_fmt_level(level)] for level in levels])
        ax.set_xticklabels([_maybe_n(_fmt_level(level), ctx["counts"].get(_fmt_level(level)), style)
                            for level in levels])
        for gname, color in colors.items():
            _legend_patch(ax, gname, color, alpha=alpha)
    else:
        xs, heights, errs = [], [], []
        for level in levels:
            name = _fmt_level(level)
            center, err, n = _summarize(frame.loc[frame["__x"] == level, "__y"].to_numpy(dtype=float),
                                        agg, error)
            pos = positions[name]
            xs.append(pos)
            heights.append(center)
            errs.append(err)
            aggregate_rows.append({"x": name, "group": "", "n": n, "value": center, "error": err})
            ax.bar(pos, center, width=bar_width, color=colors.get(name, C.color_list(style, 1)[0]),
                   edgecolor="white", linewidth=edge, alpha=alpha, zorder=2)
        if error != "none":
            ax.errorbar(xs, heights, yerr=errs, fmt="none", ecolor="#2B2B2B",
                        elinewidth=max(0.6, edge * 0.9), capsize=3, capthick=0.8, zorder=3)
        if show_points:
            rng = np.random.default_rng(11)
            for level, pos in zip(levels, xs):
                values = _finite(frame.loc[frame["__x"] == level, "__y"])
                if values.size:
                    jitter = rng.normal(0.0, bar_width * 0.12, size=values.size)
                    ax.scatter(pos + jitter, values, s=point_area, color="#3A3A3A",
                               alpha=jitter_alpha, linewidths=0, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([_maybe_n(_fmt_level(level), ctx["counts"].get(_fmt_level(level)), style)
                            for level in levels])

    ax.set_xlim(0.3, len(levels) + 0.7)
    _apply_labels(ax, style, x=ctx["x_label"], y=ctx["y_label"])
    title = f"Bar chart: {agg} {ctx['y_label']} by {ctx['x_label']}"

    stats = None
    if style.get("stat_annot"):
        stats = _stat_for_factor(ctx, style, warnings)
        if stats:
            _annotate_stats(ax, stats, positions, style, warnings)

    bullets = [f"共 {len(frame)} 个观测、x 轴 {len(levels)} 个类别"
               + (f"，按 {len(group_levels)} 个分组拆分" if group_col else "")]
    valid_rows = [row for row in aggregate_rows if math.isfinite(row["value"])]
    if valid_rows:
        best = max(valid_rows, key=lambda row: row["value"])
        worst = min(valid_rows, key=lambda row: row["value"])
        best_label = best["x"] + (f" / {best['group']}" if best["group"] else "")
        worst_label = worst["x"] + (f" / {worst['group']}" if worst["group"] else "")
        bullets.append(
            f"{_AGG_LABEL.get(agg, agg)}最高的是 {best_label}（{C.fmt_num(best['value'])}），"
            f"最低的是 {worst_label}（{C.fmt_num(worst['value'])}）"
        )
    bullets.extend(_compare_bullets(stats))
    key_numbers = {
        "n": int(len(frame)),
        "n_categories": len(levels),
        "max_value": (max(valid_rows, key=lambda row: row["value"])["value"] if valid_rows else None),
        "min_value": (min(valid_rows, key=lambda row: row["value"])["value"] if valid_rows else None),
    }
    if stats:
        key_numbers["p"] = (stats["global"] or {}).get("p")
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": key_numbers,
        "tables": _comparison_tables(stats, aggregate_rows,
                                     _aggregate_columns(ctx, _AGG_LABEL.get(agg, agg))),
        "warnings": warnings,
        "stats": _stats_payload(stats),
        "annotations": _annotations_payload(stats),
    }


# ---------------------------------------------------------------------------
# 2) bar_dot —— 柱散点图
# ---------------------------------------------------------------------------


def _render_bar_dot(data: Mapping[str, Any], params: Mapping[str, Any],
                    style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    agg = _pchoice(params, "agg", "mean", ["mean", "median", "sum"])
    error = _pchoice(params, "error", "auto", ["auto", "sd", "se", "ci95", "none"])
    if error == "auto":
        error = str(style.get("error") or "se")
    ctx = _comparison_context(data, style, warnings)
    frame, levels, positions = ctx["frame"], ctx["levels"], ctx["positions"]
    colors = ctx["colors"]
    group_col, group_levels = ctx["group_col"], ctx["group_levels"]
    bar_width = float(style.get("bar_width") or 0.72)
    alpha = float(style.get("alpha") or 0.85)
    edge = float(style.get("edge_width", 0.8) or 0.8)
    jitter_alpha = max(0.35, float(style.get("jitter_alpha", 0.55) or 0.55))
    point_area = _jitter_sizes(style)
    aggregate_rows: list[dict] = []

    fig, ax = C.new_fig(style)
    if group_col:
        n_groups = max(1, len(group_levels))
        width = bar_width / n_groups
        offsets = [(index - (n_groups - 1) / 2.0) * width for index in range(n_groups)]
        for gi, glevel in enumerate(group_levels):
            gname = _fmt_level(glevel)
            color = colors.get(gname, C.color_list(style, 1)[0])
            xs, heights, errs = [], [], []
            for level in levels:
                mask = (frame["__x"] == level) & (frame[group_col] == glevel)
                center, err, n = _summarize(frame.loc[mask, "__y"].to_numpy(dtype=float), agg, error)
                pos = positions[_fmt_level(level)] + offsets[gi]
                xs.append(pos)
                heights.append(center)
                errs.append(err)
                aggregate_rows.append({"x": _fmt_level(level), "group": gname, "n": n,
                                       "value": center, "error": err})
            ax.bar(xs, heights, width=width, color=color, edgecolor="white", linewidth=edge,
                   alpha=min(1.0, alpha * 0.45), label=gname, zorder=2)
            rng = np.random.default_rng(5 + gi)
            for level, pos in zip(levels, xs):
                values = _finite(frame.loc[(frame["__x"] == level) & (frame[group_col] == glevel), "__y"])
                if values.size:
                    jitter = rng.normal(0.0, width * 0.22, size=values.size)
                    ax.scatter(pos + jitter, values, s=point_area, color=color, alpha=jitter_alpha,
                               edgecolors="none", zorder=4)
            if error != "none":
                ax.errorbar(xs, heights, yerr=errs, fmt="none", ecolor="#2B2B2B",
                            elinewidth=max(0.6, edge * 0.9), capsize=3, zorder=5)
        ax.set_xticks([positions[_fmt_level(level)] for level in levels])
        ax.set_xticklabels([_maybe_n(_fmt_level(level), ctx["counts"].get(_fmt_level(level)), style)
                            for level in levels])
        for gname, color in colors.items():
            _legend_patch(ax, gname, color, alpha=alpha)
    else:
        xs, heights, errs = [], [], []
        for level in levels:
            name = _fmt_level(level)
            values = _finite(frame.loc[frame["__x"] == level, "__y"])
            center, err, n = _summarize(values, agg, error)
            pos = positions[name]
            xs.append(pos)
            heights.append(center)
            errs.append(err)
            aggregate_rows.append({"x": name, "group": "", "n": n, "value": center, "error": err})
            ax.bar(pos, center, width=bar_width, color=colors.get(name, C.color_list(style, 1)[0]),
                   edgecolor="white", linewidth=edge, alpha=min(1.0, alpha * 0.45), zorder=2)
            if values.size:
                jitter = np.random.default_rng(5).normal(0.0, bar_width * 0.2, size=values.size)
                ax.scatter(pos + jitter, values, s=point_area,
                           color=colors.get(name, C.color_list(style, 1)[0]),
                           alpha=jitter_alpha, edgecolors="none", zorder=4)
        if error != "none":
            ax.errorbar(xs, heights, yerr=errs, fmt="none", ecolor="#2B2B2B",
                        elinewidth=max(0.6, edge * 0.9), capsize=3, zorder=5)
        ax.set_xticks(xs)
        ax.set_xticklabels([_maybe_n(_fmt_level(level), ctx["counts"].get(_fmt_level(level)), style)
                            for level in levels])

    ax.set_xlim(0.3, len(levels) + 0.7)
    _apply_labels(ax, style, x=ctx["x_label"], y=ctx["y_label"])
    title = f"Bar + dot chart: {agg} {ctx['y_label']} by {ctx['x_label']}"

    stats = None
    if style.get("stat_annot"):
        stats = _stat_for_factor(ctx, style, warnings)
        if stats:
            _annotate_stats(ax, stats, positions, style, warnings)

    bullets = [f"共 {len(frame)} 个观测、{len(levels)} 个 x 类别"
               + (f"，{len(group_levels)} 个分组" if group_col else "")]
    valid_rows = [row for row in aggregate_rows if math.isfinite(row["value"])]
    if valid_rows:
        best = max(valid_rows, key=lambda row: row["value"])
        best_label = best["x"] + (f" / {best['group']}" if best["group"] else "")
        bullets.append(f"柱高最高的是 {best_label}（{C.fmt_num(best['value'])}）")
    bullets.extend(_compare_bullets(stats))
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "n_categories": len(levels),
                        "max_value": (max(valid_rows, key=lambda row: row["value"])["value"]
                                      if valid_rows else None)},
        "tables": _comparison_tables(stats, aggregate_rows,
                                     _aggregate_columns(ctx, _AGG_LABEL.get(agg, agg))),
        "warnings": warnings,
        "stats": _stats_payload(stats),
        "annotations": _annotations_payload(stats),
    }


# ---------------------------------------------------------------------------
# 3) box —— 箱线图
# ---------------------------------------------------------------------------


def _render_box(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    show_outliers = _pbool(params, "outliers", True)
    show_points = _pbool(params, "points", True)
    ctx = _comparison_context(data, style, warnings)
    frame, levels, positions = ctx["frame"], ctx["levels"], ctx["positions"]
    colors = ctx["colors"]
    group_col, group_levels = ctx["group_col"], ctx["group_levels"]
    alpha = float(style.get("alpha") or 0.85)
    edge = max(0.4, float(style.get("edge_width", 0.8) or 0.8))
    jitter_alpha = float(style.get("jitter_alpha", 0.55) or 0.55)
    point_area = _jitter_sizes(style)

    fig, ax = C.new_fig(style)
    slot = 0.82
    aggregate_rows: list[dict] = []
    bxp: list[dict] = []
    bxp_positions: list[float] = []
    bxp_colors: list[str] = []
    cells: list[tuple[float, float, Any, Any, str]] = []
    width = 0.6
    if group_col:
        n_groups = max(1, len(group_levels))
        width = slot / n_groups * 0.82
        offsets = [(index - (n_groups - 1) / 2.0) * (slot / n_groups) for index in range(n_groups)]
        for gi, glevel in enumerate(group_levels):
            gname = _fmt_level(glevel)
            color = colors.get(gname, C.color_list(style, 1)[0])
            for level in levels:
                values = _finite(frame.loc[(frame["__x"] == level) & (frame[group_col] == glevel), "__y"])
                stats = _box_stats(values)
                if stats is None:
                    continue
                pos = positions[_fmt_level(level)] + offsets[gi]
                stats["label"] = ""
                bxp.append(stats)
                bxp_positions.append(pos)
                bxp_colors.append(color)
                cells.append((pos, width, level, glevel, color))
                aggregate_rows.append({"x": _fmt_level(level), "group": gname, "n": stats["n"],
                                       "value": stats["med"],
                                       "error": round((stats["q3"] - stats["q1"]) / 2.0, 6)})
            _legend_patch(ax, gname, color, alpha=alpha)
    else:
        width = min(0.66, slot * 0.9)
        for level in levels:
            values = _finite(frame.loc[frame["__x"] == level, "__y"])
            stats = _box_stats(values)
            if stats is None:
                continue
            pos = positions[_fmt_level(level)]
            stats["label"] = ""
            bxp.append(stats)
            bxp_positions.append(pos)
            bxp_colors.append(colors.get(_fmt_level(level), C.color_list(style, 1)[0]))
            cells.append((pos, width, level, None, colors.get(_fmt_level(level), "#4E79A7")))
            aggregate_rows.append({"x": _fmt_level(level), "group": "", "n": stats["n"],
                                   "value": stats["med"],
                                   "error": round((stats["q3"] - stats["q1"]) / 2.0, 6)})
    if not bxp:
        raise C.prerequisite("没有可用于箱线图的有效数值：请检查 y 角色对应的列")

    artists = ax.bxp(bxp, positions=bxp_positions, widths=width, patch_artist=True,
                     showfliers=bool(show_outliers), manage_ticks=False,
                     boxprops={"linewidth": edge}, whiskerprops={"linewidth": edge},
                     capprops={"linewidth": edge}, medianprops={"linewidth": max(1.0, edge)},
                     flierprops={"markersize": 3.5, "marker": "o", "alpha": 0.7})
    for patch, color in zip(artists["boxes"], bxp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
        patch.set_edgecolor("#2B2B2B")
    for line in artists["medians"]:
        line.set_color("#1A1A1A")
    for line in artists["whiskers"] + artists["caps"]:
        line.set_color("#2B2B2B")
    for flier in artists["fliers"]:
        flier.set_markerfacecolor("#7F7F7F")
        flier.set_markeredgecolor("#7F7F7F")

    if show_points:
        rng = np.random.default_rng(19)
        for pos, cell_width, level, glevel, _color in cells:
            mask = frame["__x"] == level
            if glevel is not None and group_col:
                mask = mask & (frame[group_col] == glevel)
            values = _finite(frame.loc[mask, "__y"])
            if values.size:
                jitter = rng.normal(0.0, cell_width * 0.18, size=values.size)
                ax.scatter(pos + jitter, values, s=point_area, color="#2B2B2B",
                           alpha=jitter_alpha, linewidths=0, zorder=3)

    ax.set_xticks([positions[_fmt_level(level)] for level in levels])
    ax.set_xticklabels([_maybe_n(_fmt_level(level), ctx["counts"].get(_fmt_level(level)), style)
                        for level in levels])
    ax.set_xlim(0.35, len(levels) + 0.65)
    _apply_labels(ax, style, x=ctx["x_label"], y=ctx["y_label"])
    title = f"Box plot: {ctx['y_label']} by {ctx['x_label']}"

    stats = None
    if style.get("stat_annot"):
        stats = _stat_for_factor(ctx, style, warnings)
        if stats:
            _annotate_stats(ax, stats, positions, style, warnings)

    valid_rows = [row for row in aggregate_rows if math.isfinite(row["value"])]
    bullets = [f"共 {len(frame)} 个观测、{len(levels)} 个类别" +
               (f"，{len(group_levels)} 个分组" if group_col else "")]
    if valid_rows:
        best = max(valid_rows, key=lambda row: row["value"])
        worst = min(valid_rows, key=lambda row: row["value"])
        best_label = best["x"] + (f" / {best['group']}" if best["group"] else "")
        worst_label = worst["x"] + (f" / {worst['group']}" if worst["group"] else "")
        bullets.append(
            f"中位数最高的是 {best_label}（{C.fmt_num(best['value'])}），"
            f"最低的是 {worst_label}（{C.fmt_num(worst['value'])}）"
        )
    bullets.extend(_compare_bullets(stats))
    columns = _aggregate_columns(ctx, "中位数")
    if columns and columns[0]["key"] == "x":
        columns[-1] = {"key": "error", "label": "半 IQR", "type": "num"}
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "n_categories": len(levels),
                        "max_median": (max(valid_rows, key=lambda row: row["value"])["value"]
                                       if valid_rows else None)},
        "tables": _comparison_tables(stats, aggregate_rows, columns),
        "warnings": warnings,
        "stats": _stats_payload(stats),
        "annotations": _annotations_payload(stats),
    }


# ---------------------------------------------------------------------------
# 4) violin —— 小提琴图
# ---------------------------------------------------------------------------


def _kde_curve(values: Any, grid: np.ndarray, bw: float = 1.0,
               trim: bool = True) -> np.ndarray | None:
    arr = _finite(values)
    if arr.size < 3:
        return None
    if float(np.std(arr)) <= 1e-12:
        return None
    try:
        kde = sps.gaussian_kde(arr, bw_method="scott")
        if abs(float(bw) - 1.0) > 1e-9:
            try:
                kde.set_bandwidth(float(kde.factor) * max(0.1, float(bw)))
            except Exception:  # noqa: BLE001
                pass
        density = np.asarray(kde(grid), dtype=float)
    except Exception:  # noqa: BLE001
        return None
    if density.size == 0 or not np.all(np.isfinite(density)) or float(np.max(density)) <= 0:
        return None
    if trim:
        low, high = float(arr.min()), float(arr.max())
        density = np.where((grid >= low) & (grid <= high), density, 0.0)
    return density


def _render_violin(data: Mapping[str, Any], params: Mapping[str, Any],
                   style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    inner = _pchoice(params, "inner", "box", ["box", "quartile", "none"])
    bw = _pfloat(params, "bw", 1.0, 0.2, 4.0)
    trim = _pbool(params, "trim", True)
    ctx = _comparison_context(data, style, warnings)
    frame, levels, positions = ctx["frame"], ctx["levels"], ctx["positions"]
    colors = ctx["colors"]
    group_col, group_levels = ctx["group_col"], ctx["group_levels"]
    alpha = min(1.0, float(style.get("alpha") or 0.85) * 0.9)
    edge = max(0.4, float(style.get("edge_width", 0.8) or 0.8))
    all_values = _finite(frame["__y"])
    if all_values.size < 3:
        raise C.prerequisite("有效观测太少，无法绘制小提琴图")
    low, high = float(np.min(all_values)), float(np.max(all_values))
    pad = max((high - low) * 0.12, 1e-9)
    grid = np.linspace(low - pad, high + pad, 160)
    aggregate_rows: list[dict] = []
    cells: list[tuple[float, float, Any, Any, str]] = []
    fig, ax = C.new_fig(style)
    slot = 0.84
    if group_col:
        n_groups = max(1, len(group_levels))
        width = slot / n_groups * 0.9
        offsets = [(index - (n_groups - 1) / 2.0) * (slot / n_groups) for index in range(n_groups)]
        for gi, glevel in enumerate(group_levels):
            gname = _fmt_level(glevel)
            color = colors.get(gname, C.color_list(style, 1)[0])
            for level in levels:
                values = _finite(frame.loc[(frame["__x"] == level) & (frame[group_col] == glevel), "__y"])
                pos = positions[_fmt_level(level)] + offsets[gi]
                cells.append((pos, width, level, glevel, color))
                aggregate_rows.append({"x": _fmt_level(level), "group": gname, "n": int(values.size),
                                       "value": float(np.median(values)) if values.size else None,
                                       "error": None})
            _legend_patch(ax, gname, color, alpha=alpha)
    else:
        width = min(0.7, slot * 0.92)
        for level in levels:
            values = _finite(frame.loc[frame["__x"] == level, "__y"])
            pos = positions[_fmt_level(level)]
            cells.append((pos, width, level, None, colors.get(_fmt_level(level), C.color_list(style, 1)[0])))
            aggregate_rows.append({"x": _fmt_level(level), "group": "", "n": int(values.size),
                                   "value": float(np.median(values)) if values.size else None,
                                   "error": None})

    for pos, width, level, glevel, color in cells:
        mask = frame["__x"] == level
        if glevel is not None and group_col:
            mask = mask & (frame[group_col] == glevel)
        values = frame.loc[mask, "__y"].to_numpy(dtype=float)
        density = _kde_curve(values, grid, bw=bw, trim=trim)
        if density is None:
            warnings.append(f"{_fmt_level(level)} 组样本量或方差不足以估计密度，已跳过该小提琴")
            continue
        half = (density / float(np.max(density))) * (width / 2.0)
        ax.fill_betweenx(grid, pos - half, pos + half, facecolor=color, alpha=alpha,
                         edgecolor="#2B2B2B", linewidth=edge * 0.8, zorder=2)
        box = _box_stats(values)
        if inner == "box" and box is not None:
            box = dict(box)
            box["label"] = ""
            artists = ax.bxp([box], positions=[pos], widths=width * 0.16, patch_artist=True,
                             showfliers=False, manage_ticks=False,
                             boxprops={"facecolor": "white", "alpha": 0.9, "linewidth": edge * 0.8},
                             whiskerprops={"linewidth": edge * 0.8},
                             capprops={"linewidth": edge * 0.8},
                             medianprops={"linewidth": max(1.0, edge), "color": "#1A1A1A"})
            for patch in artists["boxes"]:
                patch.set_edgecolor("#2B2B2B")
            for line in artists["whiskers"] + artists["caps"]:
                line.set_color("#2B2B2B")
        elif inner == "quartile" and box is not None:
            ax.plot([pos, pos], [box["q1"], box["q3"]], color="#2B2B2B",
                    linewidth=max(1.0, edge), zorder=4, solid_capstyle="butt")
            ax.plot([pos], [box["med"]], marker="o", markersize=4.0, markerfacecolor="white",
                    markeredgecolor="#2B2B2B", markeredgewidth=0.8, zorder=5)

    ax.set_xticks([positions[_fmt_level(level)] for level in levels])
    ax.set_xticklabels([_maybe_n(_fmt_level(level), ctx["counts"].get(_fmt_level(level)), style)
                        for level in levels])
    ax.set_xlim(0.35, len(levels) + 0.65)
    ax.set_ylim(low - pad, high + pad)
    _apply_labels(ax, style, x=ctx["x_label"], y=ctx["y_label"])
    title = f"Violin plot: {ctx['y_label']} by {ctx['x_label']}"

    stats = None
    if style.get("stat_annot"):
        stats = _stat_for_factor(ctx, style, warnings)
        if stats:
            _annotate_stats(ax, stats, positions, style, warnings)

    valid_rows = [row for row in aggregate_rows if row["value"] is not None and math.isfinite(row["value"])]
    bullets = [f"共 {len(frame)} 个观测、{len(levels)} 个类别" +
               (f"，{len(group_levels)} 个分组" if group_col else "")]
    if valid_rows:
        best = max(valid_rows, key=lambda row: row["value"])
        best_label = best["x"] + (f" / {best['group']}" if best["group"] else "")
        bullets.append(f"中位数最高的是 {best_label}（{C.fmt_num(best['value'])}）")
    bullets.extend(_compare_bullets(stats))
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "n_categories": len(levels),
                        "max_median": (max(valid_rows, key=lambda row: row["value"])["value"]
                                       if valid_rows else None)},
        "tables": _comparison_tables(stats, aggregate_rows, _aggregate_columns(ctx, "中位数")),
        "warnings": warnings,
        "stats": _stats_payload(stats),
        "annotations": _annotations_payload(stats),
    }


# ---------------------------------------------------------------------------
# 5) raincloud —— 雨云图
# ---------------------------------------------------------------------------


def _render_raincloud(data: Mapping[str, Any], params: Mapping[str, Any],
                      style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    show_box = _pbool(params, "box", True)
    show_points = _pbool(params, "points", True)
    ctx = _comparison_context(data, style, warnings)
    frame, levels, positions = ctx["frame"], ctx["levels"], ctx["positions"]
    colors = ctx["colors"]
    group_col, group_levels = ctx["group_col"], ctx["group_levels"]
    alpha = min(1.0, float(style.get("alpha") or 0.85) * 0.9)
    edge = max(0.4, float(style.get("edge_width", 0.8) or 0.8))
    jitter_alpha = float(style.get("jitter_alpha", 0.55) or 0.55)
    point_area = _jitter_sizes(style) * 0.8
    all_values = _finite(frame["__y"])
    if all_values.size < 3:
        raise C.prerequisite("有效观测太少，无法绘制雨云图")
    low, high = float(np.min(all_values)), float(np.max(all_values))
    pad = max((high - low) * 0.12, 1e-9)
    grid = np.linspace(low - pad, high + pad, 160)
    aggregate_rows: list[dict] = []
    cells: list[tuple[float, float, Any, Any, str]] = []
    fig, ax = C.new_fig(style)
    slot = 0.86
    if group_col:
        n_groups = max(1, len(group_levels))
        cell_slot = slot / n_groups
        offsets = [(index - (n_groups - 1) / 2.0) * cell_slot for index in range(n_groups)]
        for gi, glevel in enumerate(group_levels):
            gname = _fmt_level(glevel)
            color = colors.get(gname, C.color_list(style, 1)[0])
            for level in levels:
                values = _finite(frame.loc[(frame["__x"] == level) & (frame[group_col] == glevel), "__y"])
                pos = positions[_fmt_level(level)] + offsets[gi]
                cells.append((pos, cell_slot * 0.9, level, glevel, color))
                aggregate_rows.append({"x": _fmt_level(level), "group": gname, "n": int(values.size),
                                       "value": float(np.median(values)) if values.size else None,
                                       "error": None})
            _legend_patch(ax, gname, color, alpha=alpha)
    else:
        cell_slot = slot * 0.92
        for level in levels:
            values = _finite(frame.loc[frame["__x"] == level, "__y"])
            pos = positions[_fmt_level(level)]
            cells.append((pos, cell_slot, level, None,
                          colors.get(_fmt_level(level), C.color_list(style, 1)[0])))
            aggregate_rows.append({"x": _fmt_level(level), "group": "", "n": int(values.size),
                                   "value": float(np.median(values)) if values.size else None,
                                   "error": None})

    rng = np.random.default_rng(23)
    for pos, width, level, glevel, color in cells:
        mask = frame["__x"] == level
        if glevel is not None and group_col:
            mask = mask & (frame[group_col] == glevel)
        values = frame.loc[mask, "__y"].to_numpy(dtype=float)
        density = _kde_curve(values, grid, bw=1.0, trim=True)
        if density is not None:
            half = (density / float(np.max(density))) * (width * 0.42)
            ax.fill_betweenx(grid, pos - half, pos, facecolor=color, alpha=alpha,
                             edgecolor="#2B2B2B", linewidth=edge * 0.8, zorder=2)
        box = _box_stats(values)
        if show_box and box is not None:
            box_center = pos + width * 0.16
            box_width = width * 0.16
            ax.add_patch(Rectangle((box_center - box_width / 2.0, box["q1"]), box_width,
                                   max(box["q3"] - box["q1"], 1e-9), facecolor=color,
                                   edgecolor="#2B2B2B", linewidth=edge * 0.9, alpha=0.95,
                                   zorder=3))
            ax.plot([box_center, box_center], [box["whislo"], box["q1"]], color="#2B2B2B",
                    linewidth=edge * 0.9, zorder=3)
            ax.plot([box_center, box_center], [box["q3"], box["whishi"]], color="#2B2B2B",
                    linewidth=edge * 0.9, zorder=3)
            cap = box_width * 0.35
            ax.plot([box_center - cap, box_center + cap], [box["whislo"]] * 2, color="#2B2B2B",
                    linewidth=edge * 0.9, zorder=3)
            ax.plot([box_center - cap, box_center + cap], [box["whishi"]] * 2, color="#2B2B2B",
                    linewidth=edge * 0.9, zorder=3)
            ax.plot([box_center - box_width / 2.0, box_center + box_width / 2.0],
                    [box["med"]] * 2, color="white", linewidth=max(1.2, edge),
                    zorder=4, solid_capstyle="butt")
        if show_points and values.size:
            point_center = pos + width * 0.34
            jitter = rng.normal(0.0, width * 0.055, size=values.size)
            ax.scatter(point_center + jitter, values, s=point_area, color=color,
                       alpha=jitter_alpha, edgecolors="none", zorder=5)

    ax.set_xticks([positions[_fmt_level(level)] for level in levels])
    ax.set_xticklabels([_maybe_n(_fmt_level(level), ctx["counts"].get(_fmt_level(level)), style)
                        for level in levels])
    ax.set_xlim(0.3, len(levels) + 0.7)
    ax.set_ylim(low - pad, high + pad)
    _apply_labels(ax, style, x=ctx["x_label"], y=ctx["y_label"])
    title = f"Raincloud plot: {ctx['y_label']} by {ctx['x_label']}"

    stats = None
    if style.get("stat_annot"):
        stats = _stat_for_factor(ctx, style, warnings)
        if stats:
            _annotate_stats(ax, stats, positions, style, warnings)

    valid_rows = [row for row in aggregate_rows if row["value"] is not None and math.isfinite(row["value"])]
    bullets = [f"共 {len(frame)} 个观测、{len(levels)} 个类别" +
               (f"，{len(group_levels)} 个分组" if group_col else "")]
    if valid_rows:
        best = max(valid_rows, key=lambda row: row["value"])
        best_label = best["x"] + (f" / {best['group']}" if best["group"] else "")
        bullets.append(f"中位数最高的是 {best_label}（{C.fmt_num(best['value'])}）")
    bullets.extend(_compare_bullets(stats))
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "n_categories": len(levels)},
        "tables": _comparison_tables(stats, aggregate_rows, _aggregate_columns(ctx, "中位数")),
        "warnings": warnings,
        "stats": _stats_payload(stats),
        "annotations": _annotations_payload(stats),
    }


# ---------------------------------------------------------------------------
# 6) dotplot —— 分类矩阵点图
# ---------------------------------------------------------------------------


def _render_dotplot(data: Mapping[str, Any], params: Mapping[str, Any],
                    style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    agg = _pchoice(params, "agg", "mean", ["mean", "median"])
    show_values = _pbool(params, "show_values", False)
    frame = P.require_tidy(data, ["__x"]).copy()
    levels = _levels_of(frame, "__x", max_levels=24, warnings=warnings)
    if len(levels) < 2:
        raise C.bad_request("点图至少需要 2 个 x 分类，请检查 x 角色对应列的取值")
    row_col = _pick_group_col(data, frame, keys=("__group", "__color"))
    row_levels = _levels_of(frame, row_col, max_levels=16, warnings=warnings) if row_col else []
    if row_col and len(row_levels) < 2:
        row_col, row_levels = None, []
    if not row_levels:
        row_levels = ["All"]
    labels = data.get("labels") or {}
    value_col = None
    value_label = ""
    for candidate in ("__y", "__size"):
        if candidate in frame.columns:
            values = pd.to_numeric(frame[candidate], errors="coerce")
            if _finite(values).size >= max(3, int(0.5 * len(frame))):
                value_col = candidate
                value_label = str(labels.get(candidate[2:]) or candidate[2:])
                break
    matrix = np.full((len(row_levels), len(levels)), np.nan, dtype=float)
    counts = np.zeros((len(row_levels), len(levels)), dtype=float)
    for ri, row in enumerate(row_levels):
        for ci, level in enumerate(levels):
            mask = frame["__x"] == level
            if row_col:
                mask = mask & (frame[row_col] == row)
            cell = frame.loc[mask]
            counts[ri, ci] = float(len(cell))
            if value_col:
                values = _finite(pd.to_numeric(cell[value_col], errors="coerce"))
                if values.size:
                    matrix[ri, ci] = float(np.median(values)) if agg == "median" else float(np.mean(values))
            else:
                matrix[ri, ci] = float(len(cell))
    if value_col is None:
        value_label = "count"
        warnings.append("没有可用数值列，点大小/颜色改为展示每个格子的观测数")

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise C.prerequisite("没有可绘制的数值：请给 color/size 角色指定一个数值列")
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    span = vmax - vmin
    norm_values = (matrix - vmin) / span if span > 1e-12 else np.full_like(matrix, 0.5)
    norm_values = np.where(np.isfinite(matrix), norm_values, np.nan)
    point_size = float(style.get("point_size", 28) or 28)
    size_min = max(6.0, point_size * 0.45)
    size_max = max(size_min + 12.0, point_size * 6.0)
    areas = size_min + np.nan_to_num(norm_values, nan=0.0) * (size_max - size_min)

    fig, ax = C.new_fig(style)
    cmap = _pick_cmap(style, params.get("cmap"))
    x_positions = np.arange(1, len(levels) + 1)
    y_positions = np.arange(len(row_levels), 0, -1)
    xx, yy = np.meshgrid(x_positions, y_positions)
    scatter = ax.scatter(xx.ravel(), yy.ravel(), s=areas.ravel(),
                         c=np.where(np.isfinite(matrix), matrix, np.nan).ravel(),
                         cmap=cmap, edgecolors="#2B2B2B",
                         linewidths=max(0.3, float(style.get("edge_width", 0.8) or 0.8) * 0.6),
                         alpha=float(style.get("alpha") or 0.85), zorder=3)
    aux_axes = []
    try:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.86, pad=0.02)
        cbar.set_label(value_label, fontsize=float(style.get("label_size", 12)))
        cbar.ax.tick_params(labelsize=float(style.get("tick_size", 10)))
        aux_axes.append(cbar.ax)
    except Exception:  # noqa: BLE001
        warnings.append("颜色条绘制失败（数据可能全为缺失）")
    for level_value in (vmin, (vmin + vmax) / 2.0, vmax):
        normalized = (level_value - vmin) / span if span > 1e-12 else 0.5
        area = size_min + normalized * (size_max - size_min)
        _legend_marker(ax, f"{C.fmt_num(level_value)}", "#7F7F7F", "o", math.sqrt(area) * 0.9, 0.85)
    if show_values:
        text_size = max(5.5, float(style.get("text_size", 10)) - 2.5)
        for ri in range(len(row_levels)):
            for ci in range(len(levels)):
                if math.isfinite(matrix[ri, ci]):
                    ax.text(x_positions[ci], y_positions[ri], C.fmt_num(matrix[ri, ci], 3),
                            ha="center", va="center", fontsize=text_size, color="#1A1A1A", zorder=5)
    if len(row_levels) > 10:
        for label in ax.get_yticklabels():
            label.set_fontsize(max(6.0, float(style.get("tick_size", 10)) - 2))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_fmt_level(level) for level in levels])
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_fmt_level(row) for row in row_levels])
    ax.set_xlim(0.4, len(levels) + 0.6)
    ax.set_ylim(0.4, len(row_levels) + 0.6)
    row_label = str(labels.get((row_col or "__group")[2:]) or "Group") if row_col else "Group"
    _apply_labels(ax, style, x=str(labels.get("x") or "x"), y=row_label)
    title = f"Dot plot: {value_label} by {labels.get('x') or 'x'}"

    rows = []
    for ri, row in enumerate(row_levels):
        for ci, level in enumerate(levels):
            if not math.isfinite(matrix[ri, ci]):
                continue
            rows.append({"x": _fmt_level(level), "row": _fmt_level(row),
                         "n": counts[ri, ci], "value": matrix[ri, ci]})
    table = C.table("dotplot_values", f"点图数值（{value_label}）",
                    [{"key": "x", "label": str(labels.get("x") or "x"), "type": "str"},
                     {"key": "row", "label": row_label, "type": "str"},
                     {"key": "n", "label": "n", "type": "num"},
                     {"key": "value", "label": value_label, "type": "num"}],
                    rows)
    best_index = np.unravel_index(int(np.nanargmax(matrix)), matrix.shape)
    bullets = [
        f"点图矩阵 {len(row_levels)} 行 × {len(levels)} 列，共 {int(np.isfinite(matrix).sum())} 个有效格子",
        f"{value_label} 取值范围 {C.fmt_num(vmin)} – {C.fmt_num(vmax)}，"
        f"最高值出现在 {_fmt_level(levels[best_index[1]])} / {_fmt_level(row_levels[best_index[0]])}",
        f"共 {len(frame)} 个观测参与聚合（{agg}）",
    ]
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "n_cells": int(np.isfinite(matrix).sum()),
                        "max_value": vmax, "min_value": vmin},
        "tables": [table],
        "warnings": warnings,
        "stats": None,
        "annotations": None,
        "aux": aux_axes,
    }


# ---------------------------------------------------------------------------
# 7) stack —— 堆叠柱状图
# ---------------------------------------------------------------------------


def _render_stack(data: Mapping[str, Any], params: Mapping[str, Any],
                  style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    normalize = _pbool(params, "normalize", False)
    agg = _pchoice(params, "agg", "sum", ["sum", "mean"])
    show_total = _pbool(params, "show_total", False)
    frame = P.require_tidy(data, ["__x", "__y"]).copy()
    frame["__y"] = pd.to_numeric(frame["__y"], errors="coerce")
    frame = frame[np.isfinite(frame["__y"].to_numpy(dtype=float))]
    levels = _levels_of(frame, "__x", max_levels=20, warnings=warnings)
    if not levels:
        raise C.bad_request("堆叠柱需要可用的 x 分类列")
    frame = frame[frame["__x"].isin(levels)]
    labels = data.get("labels") or {}
    group_col = _pick_group_col(data, frame, keys=("__group",))
    group_levels = _levels_of(frame, group_col, max_levels=12, warnings=warnings) if group_col else []
    if group_col and len(group_levels) < 2:
        group_col, group_levels = None, []
    if group_col:
        colors = _colors(style, group_levels)
        matrix = np.zeros((len(levels), len(group_levels)), dtype=float)
        counts = np.zeros_like(matrix)
        for ri, level in enumerate(levels):
            for ci, group in enumerate(group_levels):
                values = _finite(frame.loc[(frame["__x"] == level) & (frame[group_col] == group), "__y"])
                counts[ri, ci] = float(values.size)
                matrix[ri, ci] = _summarize(values, agg, "none")[0] if values.size else 0.0
        group_names = [_fmt_level(group) for group in group_levels]
    else:
        warnings.append("未映射 group 角色，堆叠柱退化为单色柱（每个 x 一个值）；请指定 group 以展示构成")
        colors = _colors(style, levels)
        matrix = np.array([[_summarize(_finite(frame.loc[frame["__x"] == level, "__y"]), agg, "none")[0]
                            if np.isfinite(_summarize(_finite(frame.loc[frame["__x"] == level, "__y"]), agg, "none")[0])
                            else 0.0] for level in levels], dtype=float)
        counts = matrix.copy()
        group_names = ["value"]

    if normalize:
        totals = np.nansum(matrix, axis=1)
        safe = np.where(totals > 1e-12, totals, 1.0)
        shown = matrix / safe[:, None] * 100.0
        value_label = "Percentage (%)"
    else:
        shown = matrix.copy()
        value_label = str(labels.get("y") or "y")

    alpha = float(style.get("alpha") or 0.85)
    edge = max(0.4, float(style.get("edge_width", 0.8) or 0.8))
    bar_width = float(style.get("bar_width") or 0.72)
    fig, ax = C.new_fig(style)
    x_positions = np.arange(1, len(levels) + 1)
    bottom = np.zeros(len(levels), dtype=float)
    for ci, group_name in enumerate(group_names):
        color = colors.get(group_name, C.color_list(style, len(group_names))[ci])
        ax.bar(x_positions, shown[:, ci], bottom=bottom, width=bar_width, color=color,
               edgecolor="white", linewidth=edge * 0.6, alpha=alpha,
               label=group_name if group_col else None, zorder=2)
        bottom = bottom + np.nan_to_num(shown[:, ci], nan=0.0)
        if group_col:
            _legend_patch(ax, group_name, color, alpha=alpha)
    if show_total:
        text_size = max(6.0, float(style.get("text_size", 10)) - 1)
        for position, total in zip(x_positions, bottom):
            ax.text(position, total, C.fmt_num(total, 3), ha="center", va="bottom",
                    fontsize=text_size, color="#1A1A1A")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_fmt_level(level) for level in levels])
    ax.set_xlim(0.35, len(levels) + 0.65)
    _apply_labels(ax, style, x=str(labels.get("x") or "x"), y=value_label)
    title = f"Stacked bar: {str(labels.get('y') or 'value')} by {str(labels.get('x') or 'x')}"
    if normalize:
        title += " (normalized)"

    rows = []
    totals = np.nansum(matrix, axis=1)
    for ri, level in enumerate(levels):
        for ci, group_name in enumerate(group_names):
            raw = matrix[ri, ci]
            pct = (raw / totals[ri] * 100.0) if totals[ri] > 1e-12 else float("nan")
            rows.append({"x": _fmt_level(level), "group": group_name, "n": counts[ri, ci],
                         "value": raw, "pct": pct})
    table = C.table(
        "stack_values", "堆叠柱数值",
        [{"key": "x", "label": str(labels.get("x") or "x"), "type": "str"},
         {"key": "group", "label": str(labels.get("group") or "group"), "type": "str"},
         {"key": "n", "label": "n", "type": "num"},
         {"key": "value", "label": value_label, "type": "num"},
         {"key": "pct", "label": "占比 %", "type": "num"}],
        rows)
    bullets = [f"共 {len(frame)} 个观测，{len(levels)} 个 x 类别"
               + (f" × {len(group_levels)} 个堆叠分组" if group_col else "")]
    if group_col and totals.size:
        overall = np.nansum(matrix, axis=0)
        grand = float(np.nansum(overall))
        if grand > 0:
            top_index = int(np.argmax(overall))
            bullets.append(
                f"总体上 {group_names[top_index]} 占比最高（{overall[top_index] / grand * 100:.1f}%）"
            )
        dominant = []
        for ri, level in enumerate(levels):
            if totals[ri] > 0:
                ci = int(np.nanargmax(matrix[ri]))
                dominant.append(f"{_fmt_level(level)}={group_names[ci]}")
        if dominant:
            bullets.append("各 x 类别的主导分组：" + "，".join(dominant[:6])
                           + (" 等" if len(dominant) > 6 else ""))
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "n_categories": len(levels),
                        "n_groups": len(group_names) if group_col else 1},
        "tables": [table],
        "warnings": warnings,
        "stats": None,
        "annotations": None,
    }


# ---------------------------------------------------------------------------
# 8) pie —— 饼图 / 环图
# ---------------------------------------------------------------------------


def _render_pie(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    donut = _pbool(params, "donut", True)
    show_pct = _pbool(params, "show_pct", True)
    start_angle = _pfloat(params, "start_angle", 90.0, 0.0, 360.0)
    frame = P.require_tidy(data, ["__x"]).copy()
    levels = _levels_of(frame, "__x", max_levels=16, warnings=warnings)
    if len(levels) < 2:
        raise C.bad_request("饼图至少需要 2 个类别，请检查 x 角色对应列的取值")
    frame = frame[frame["__x"].isin(levels)]
    labels = data.get("labels") or {}
    y_col = None
    if "__y" in frame.columns:
        values = pd.to_numeric(frame["__y"], errors="coerce")
        if _finite(values).size >= max(3, int(0.5 * len(frame))):
            y_col = "__y"
    counts = []
    values = []
    for level in levels:
        cell = frame.loc[frame["__x"] == level]
        counts.append(int(len(cell)))
        if y_col is not None:
            values.append(float(np.nansum(_finite(pd.to_numeric(cell[y_col], errors="coerce")))))
        else:
            values.append(float(len(cell)))
    if any(value < 0 for value in values):
        warnings.append("y 列含负值，饼图无法表示，已改用各类别观测数")
        y_col = None
        values = [float(count) for count in counts]
    total = float(np.sum(values))
    if total <= 0:
        raise C.prerequisite("饼图数值总和为 0，无法绘制")
    colors = _colors(style, levels)
    color_list = [colors.get(_fmt_level(level), C.color_list(style, 1)[0]) for level in levels]
    legend_pos = str(style.get("legend_pos") or "right")
    value_label = str(labels.get("y") or "") if y_col else "count"
    fig, ax = C.new_fig(style)
    autopct = (lambda pct: f"{pct:.1f}%") if show_pct else None
    pie_result = ax.pie(
        values,
        colors=color_list,
        startangle=start_angle,
        counterclock=False,
        labels=None,
        autopct=autopct,
        pctdistance=0.78 if donut else 0.62,
        wedgeprops={
            "width": 0.42 if donut else 1.0,
            "edgecolor": "white",
            "linewidth": max(0.5, float(style.get("edge_width", 0.8) or 0.8)),
        },
        textprops={"fontsize": max(6.0, float(style.get("text_size", 10)) - 1),
                   "color": "#1A1A1A"},
    )
    wedges = pie_result[0]
    ax.set_aspect("equal")
    if legend_pos == "none":
        for wedge, level, value in zip(wedges, levels, values):
            angle = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
            radius = 1.18
            ax.text(radius * math.cos(angle), radius * math.sin(angle),
                    f"{_fmt_level(level)} ({value / total * 100:.1f}%)",
                    ha="center", va="center",
                    fontsize=max(6.0, float(style.get("text_size", 10)) - 1))
    for level, value in zip(levels, values):
        _legend_patch(
            ax,
            f"{_fmt_level(level)} ({value / total * 100:.1f}%)",
            colors.get(_fmt_level(level), C.color_list(style, 1)[0]),
        )
    _apply_labels(ax, style)
    title = f"{'Donut' if donut else 'Pie'} chart: composition of {str(labels.get('x') or 'x')}"

    rows = []
    for level, value, count in zip(levels, values, counts):
        rows.append({"category": _fmt_level(level), "value": value,
                     "pct": value / total * 100.0, "n": count})
    table = C.table(
        "pie_values", "饼图数值",
        [{"key": "category", "label": str(labels.get("x") or "category"), "type": "str"},
         {"key": "value", "label": value_label or "value", "type": "num"},
         {"key": "pct", "label": "占比 %", "type": "num"},
         {"key": "n", "label": "n", "type": "num"}],
        rows)
    order = np.argsort(-np.array(values, dtype=float))
    top = [(levels[i], values[i]) for i in order[:3]]
    top_text = "，".join(f"{_fmt_level(level)} {value / total * 100:.1f}%"
                       for level, value in top)
    bullets = [
        f"共 {len(frame)} 个观测、{len(levels)} 个类别",
        f"占比最高的类别：{top_text}",
        f"数值口径：{value_label if y_col else '各类别观测数'}，合计 {C.fmt_num(total)}",
    ]
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "n_categories": len(levels),
                        "total": total,
                        "max_share_pct": float(values[int(np.argmax(values))] / total * 100.0)},
        "tables": [table],
        "warnings": warnings,
        "stats": None,
        "annotations": None,
    }


# ---------------------------------------------------------------------------
# 散点公共：数值化 x/y
# ---------------------------------------------------------------------------


def _xy_frame(data: Mapping[str, Any], warnings: list[str],
              min_points: int = 3) -> tuple[pd.DataFrame, bool]:
    frame = P.require_tidy(data, ["__x", "__y"]).copy()
    x_series = frame["__x"]
    x_is_time = bool(pd.api.types.is_datetime64_any_dtype(x_series))
    if x_is_time:
        x_num = np.asarray(mdates.date2num(pd.to_datetime(x_series, errors="coerce")), dtype=float)
    else:
        x_num = pd.to_numeric(x_series, errors="coerce").to_numpy(dtype=float)
    y_num = pd.to_numeric(frame["__y"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_num) & np.isfinite(y_num)
    if mask.sum() < min_points:
        raise C.bad_request("散点图需要数值型的 x 与 y（当前有效点不足 3 个），请在左侧换一个数值列")
    frame = frame.loc[mask].reset_index(drop=True)
    frame["_xn"] = np.asarray(x_num, dtype=float)[mask]
    frame["_yn"] = np.asarray(y_num, dtype=float)[mask]
    return frame, x_is_time


def _fit_line(ax, x: np.ndarray, y: np.ndarray, style: Mapping[str, Any],
              color: str, add_ci: bool = True, label: str = "Linear fit") -> dict | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return None
    try:
        slope, intercept = (float(value) for value in np.polyfit(x, y, 1))
    except Exception:  # noqa: BLE001
        return None
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 120)
    y_hat = slope * x_grid + intercept
    resid = y - (slope * x + intercept)
    dof = int(x.size - 2)
    if add_ci and dof > 0:
        s2 = float(np.sum(resid ** 2)) / dof
        xbar = float(np.mean(x))
        sxx = float(np.sum((x - xbar) ** 2))
        if s2 >= 0 and sxx > 1e-12:
            se = np.sqrt(s2 * (1.0 / x.size + (x_grid - xbar) ** 2 / sxx))
            try:
                tcrit = float(sps.t.ppf(0.975, dof))
            except Exception:  # noqa: BLE001
                tcrit = 1.96
            ax.fill_between(x_grid, y_hat - tcrit * se, y_hat + tcrit * se,
                            color=color, alpha=0.15, linewidth=0, zorder=3)
    ax.plot(x_grid, y_hat, color=color, linewidth=max(1.0, float(style.get("line_width", 1.4))),
            label=label, zorder=5)
    return {"slope": slope, "intercept": intercept}


def _corr_by_group(frame: pd.DataFrame, color_col: str | None, method: str,
                   warnings: list[str]) -> list[dict]:
    rows: list[dict] = []
    groups: list[tuple[str, pd.DataFrame]] = []
    if color_col and color_col in frame.columns:
        for level in _levels_of(frame, color_col, max_levels=12):
            cell = frame[frame[color_col] == level]
            if len(cell) >= 3:
                groups.append((_fmt_level(level), cell))
    if not groups:
        groups = [("Overall", frame)]
    for name, cell in groups:
        x = cell["_xn"].to_numpy(dtype=float)
        y = cell["_yn"].to_numpy(dtype=float)
        if x.size < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
            continue
        try:
            if method == "spearman":
                r, p = sps.spearmanr(x, y)
            elif method == "kendall":
                r, p = sps.kendalltau(x, y)
            else:
                r, p = sps.pearsonr(x, y)
            rows.append({"group": name, "n": int(x.size), "r": float(r), "p": float(p)})
        except Exception:  # noqa: BLE001
            warnings.append(f"{name} 组相关系数计算失败，已跳过")
    return rows


def _corr_annotation(rows: Sequence[Mapping[str, Any]], method: str) -> str:
    if not rows:
        return ""
    method_label = {"pearson": "Pearson r", "spearman": "Spearman rho", "kendall": "Kendall tau"}.get(method, method)
    overall = rows[0]
    text = f"{method_label} = {C.fmt_num(overall['r'])}, p = {C.fmt_p(overall['p'])}, n = {overall['n']}"
    return text


# ---------------------------------------------------------------------------
# 9) scatter —— 散点图
# ---------------------------------------------------------------------------


def _render_scatter(data: Mapping[str, Any], params: Mapping[str, Any],
                    style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    add_fit = _pbool(params, "add_fit", True)
    add_ci = _pbool(params, "add_ci", True)
    add_corr = _pbool(params, "add_corr", True)
    method = _pchoice(params, "method", "pearson", ["pearson", "spearman", "kendall"])
    x_bins = _pint(params, "x_bins", 0, 0, 30)
    frame, x_is_time = _xy_frame(data, warnings)
    labels = data.get("labels") or {}
    color_col = _pick_group_col(data, frame, keys=("__color", "__group"))
    color_levels = _levels_of(frame, color_col, max_levels=10, warnings=warnings) if color_col else []
    if color_col and len(color_levels) < 2:
        color_col, color_levels = None, []
    colors = _colors(style, color_levels) if color_col else {}
    alpha = float(style.get("alpha") or 0.85)
    edge = max(0.4, float(style.get("edge_width", 0.8) or 0.8))
    point_size = float(style.get("point_size", 28) or 28)
    plot = _sample_frame(frame, 3000, warnings)

    fig, ax = C.new_fig(style)
    if color_col:
        for level in color_levels:
            name = _fmt_level(level)
            color = colors.get(name, C.color_list(style, 1)[0])
            cell = plot[plot[color_col] == level]
            ax.scatter(cell["_xn"], cell["_yn"], s=point_size, color=color, alpha=alpha,
                       edgecolors="none", label=name, zorder=4)
            _legend_marker(ax, name, color, "o", math.sqrt(point_size), alpha)
    else:
        color = C.color_list(style, 1)[0]
        ax.scatter(plot["_xn"], plot["_yn"], s=point_size, color=color, alpha=alpha,
                   edgecolors="none", zorder=4)

    x_all = frame["_xn"].to_numpy(dtype=float)
    y_all = frame["_yn"].to_numpy(dtype=float)
    fits: list[dict] = []
    if add_fit and x_all.size >= 3:
        if color_col and len(color_levels) <= 4:
            for level in color_levels:
                name = _fmt_level(level)
                cell = frame[frame[color_col] == level]
                fit = _fit_line(ax, cell["_xn"].to_numpy(dtype=float),
                                cell["_yn"].to_numpy(dtype=float), style,
                                colors.get(name, C.color_list(style, 1)[0]), False,
                                label=f"{name} fit")
                if fit:
                    fit["group"] = name
                    fits.append(fit)
        else:
            fit = _fit_line(ax, x_all, y_all, style, "#333333", add_ci, label="Linear fit")
            if fit:
                fits.append(fit)
    if x_bins >= 2 and x_all.size >= x_bins:
        edges = np.linspace(float(np.min(x_all)), float(np.max(x_all)), x_bins + 1)
        centers, means, errors = [], [], []
        for index in range(x_bins):
            low, high = edges[index], edges[index + 1]
            mask = (x_all >= low) & (x_all <= high) if index == x_bins - 1 else (x_all >= low) & (x_all < high)
            values = y_all[mask]
            if values.size >= 2:
                centers.append((low + high) / 2.0)
                means.append(float(np.mean(values)))
                errors.append(float(np.std(values, ddof=1) / math.sqrt(values.size)))
        if centers:
            ax.errorbar(centers, means, yerr=errors, fmt="s", color="#B24745",
                        markersize=5, capsize=2.5, linewidth=max(0.8, edge),
                        label=f"Binned mean (n={x_bins})", zorder=6)

    corr_rows = _corr_by_group(frame, color_col, method, warnings)
    annotation_text = _corr_annotation(corr_rows, method) if add_corr else ""
    if annotation_text:
        text_size = max(6.5, float(style.get("text_size", 10)) - 0.5)
        ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes, ha="left", va="top",
                fontsize=text_size, color="#1A1A1A",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                      "edgecolor": "#CCCCCC", "alpha": 0.85, "linewidth": 0.6})
    _apply_labels(ax, style, x=str(labels.get("x") or "x"), y=str(labels.get("y") or "y"))
    title = f"Scatter plot: {str(labels.get('y') or 'y')} vs {str(labels.get('x') or 'x')}"

    overall = corr_rows[0] if corr_rows else None
    bullets = [f"共 {len(frame)} 个有效点"
               + (f"，按 {len(color_levels)} 个分组着色" if color_col else "")]
    if overall:
        significant = math.isfinite(overall["p"]) and overall["p"] < 0.05
        bullets.append(
            f"{'Pearson' if method == 'pearson' else method.capitalize()} 相关 r = "
            f"{C.fmt_num(overall['r'])}，p = {C.fmt_p(overall['p'])}，"
            f"{'相关有统计学意义' if significant else '相关无统计学意义'}"
        )
    if fits:
        fit = fits[0]
        bullets.append(f"线性拟合斜率 = {C.fmt_num(fit['slope'])}（每单位 x 的 y 变化）")
    rows = []
    for row in corr_rows:
        rows.append({"group": row["group"], "n": row["n"], "r": row["r"],
                     "p": row["p"], "r2": row["r"] ** 2})
    table = C.table(
        "correlation", f"相关系数与拟合（{method}）",
        [{"key": "group", "label": "分组", "type": "str"},
         {"key": "n", "label": "n", "type": "num"},
         {"key": "r", "label": "相关系数 r", "type": "num"},
         {"key": "r2", "label": "R²", "type": "num"},
         {"key": "p", "label": "p", "type": "p"}],
        rows)
    stats = {
        "method": method,
        "correlation": [{"group": row["group"], "r": row["r"], "p": row["p"], "n": row["n"]}
                        for row in corr_rows],
        "fit": fits[0] if fits else None,
        "n": int(len(frame)),
    }
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)),
                        "r": overall["r"] if overall else None,
                        "p": overall["p"] if overall else None},
        "tables": [table],
        "warnings": warnings,
        "stats": stats,
        "annotations": {"correlation": annotation_text} if annotation_text else None,
    }


# ---------------------------------------------------------------------------
# 10) bubble —— 气泡图
# ---------------------------------------------------------------------------


def _size_areas(values: Any, low: float, high: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, (low + high) / 2.0)
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    span = vmax - vmin
    if span <= 1e-12:
        norm = np.full(arr.shape, 0.5)
    else:
        norm = (arr - vmin) / span
    norm = np.clip(np.nan_to_num(norm, nan=0.5), 0.0, 1.0)
    return low + norm * (high - low)


def _render_bubble(data: Mapping[str, Any], params: Mapping[str, Any],
                   style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    add_fit = _pbool(params, "add_fit", False)
    size_min = _pfloat(params, "size_min", 20.0, 4.0, 1000.0)
    size_max = _pfloat(params, "size_max", 420.0, 20.0, 5000.0)
    if size_max <= size_min:
        size_max = size_min + 20.0
    frame, _x_is_time = _xy_frame(data, warnings)
    labels = data.get("labels") or {}
    if "__size" not in frame.columns or _finite(pd.to_numeric(frame["__size"], errors="coerce")).size < 3:
        raise C.bad_request("气泡图需要一个数值型的 size 角色（例如样本量/权重），请在左侧指定")
    frame["_size"] = pd.to_numeric(frame["__size"], errors="coerce")
    frame = frame[np.isfinite(frame["_size"].to_numpy(dtype=float))].reset_index(drop=True)
    if len(frame) < 3:
        raise C.prerequisite("有效点不足 3 个，无法绘制气泡图")
    color_col = _pick_group_col(data, frame, keys=("__color", "__group"))
    color_levels = _levels_of(frame, color_col, max_levels=10, warnings=warnings) if color_col else []
    if color_col and len(color_levels) < 2:
        color_col, color_levels = None, []
    colors = _colors(style, color_levels) if color_col else {}
    plot = _sample_frame(frame, 3000, warnings)
    areas = _size_areas(plot["_size"], size_min, size_max)
    alpha = float(style.get("alpha") or 0.85)
    edge = max(0.3, float(style.get("edge_width", 0.8) or 0.8) * 0.7)

    fig, ax = C.new_fig(style)
    if color_col:
        for level in color_levels:
            name = _fmt_level(level)
            color = colors.get(name, C.color_list(style, 1)[0])
            mask = plot[color_col] == level
            ax.scatter(plot.loc[mask, "_xn"], plot.loc[mask, "_yn"], s=areas[mask.to_numpy()],
                       color=color, alpha=alpha, edgecolors="#2B2B2B", linewidths=edge,
                       label=name, zorder=4)
            _legend_marker(ax, name, color, "o", 8.0, alpha)
    else:
        color = C.color_list(style, 1)[0]
        ax.scatter(plot["_xn"], plot["_yn"], s=areas, color=color, alpha=alpha,
                   edgecolors="#2B2B2B", linewidths=edge, zorder=4)
    if add_fit:
        fit = _fit_line(ax, frame["_xn"].to_numpy(dtype=float), frame["_yn"].to_numpy(dtype=float),
                        style, "#333333", True, label="Linear fit")
    else:
        fit = None
    size_values = _finite(frame["_size"])
    if size_values.size:
        for quantile, name in ((0.0, "min"), (0.5, "median"), (1.0, "max")):
            value = float(np.quantile(size_values, quantile))
            area = float(_size_areas(np.array([value]), size_min, size_max)[0])
            _legend_marker(ax, f"{name}: {C.fmt_num(value)}", "#7F7F7F", "o",
                           math.sqrt(area) * 0.9, 0.85)
    _apply_labels(ax, style, x=str(labels.get("x") or "x"), y=str(labels.get("y") or "y"))
    title = f"Bubble plot: {str(labels.get('y') or 'y')} vs {str(labels.get('x') or 'x')}"
    bullets = [
        f"共 {len(frame)} 个气泡，大小编码 {str(labels.get('size') or 'size')}"
        f"（{C.fmt_num(float(np.min(size_values)))} – {C.fmt_num(float(np.max(size_values)))}）",
        f"点面积范围 {C.fmt_num(size_min)} – {C.fmt_num(size_max)} 平方点",
    ]
    if color_col:
        bullets.append(f"按 {len(color_levels)} 个分组着色")
    if fit:
        bullets.append(f"线性拟合斜率 = {C.fmt_num(fit['slope'])}")
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "size_min": float(np.min(size_values)),
                        "size_max": float(np.max(size_values))},
        "tables": [],
        "warnings": warnings,
        "stats": {"fit": fit, "n": int(len(frame))} if fit else None,
        "annotations": None,
    }


# ---------------------------------------------------------------------------
# 11) line —— 折线图
# ---------------------------------------------------------------------------


def _aggregate_series(frame: pd.DataFrame, x_col: str, group_col: str | None,
                      agg: str, error: str) -> list[dict]:
    series: list[dict] = []
    groups: list[tuple[str | None, pd.DataFrame]] = []
    if group_col and group_col in frame.columns:
        for level in _levels_of(frame, group_col, max_levels=12):
            groups.append((_fmt_level(level), frame[frame[group_col] == level]))
    else:
        groups = [(None, frame)]
    for name, cell in groups:
        if cell.empty:
            continue
        unique_x = _levels_of(cell, x_col, max_levels=100000)
        xs, centers, errors, counts = [], [], [], []
        for x_value in unique_x:
            mask = cell[x_col] == x_value
            center, err, n = _summarize(cell.loc[mask, "__y"].to_numpy(dtype=float), agg, error)
            if not math.isfinite(center):
                continue
            xs.append(x_value)
            centers.append(center)
            errors.append(err)
            counts.append(n)
        if not xs:
            continue
        try:
            order = np.argsort(np.asarray(xs, dtype=float))
        except Exception:  # noqa: BLE001
            order = np.argsort(np.arange(len(xs)))
        series.append({
            "group": name,
            "x": [xs[i] for i in order],
            "center": [centers[i] for i in order],
            "error": [errors[i] for i in order],
            "n": [counts[i] for i in order],
        })
    return series


def _x_numbers(values: Sequence[Any], x_is_time: bool) -> np.ndarray | None:
    """把 x 转成数值坐标：数值列→float，时间列→matplotlib date number；失败返回 None。"""
    items = list(values)
    if not items:
        return np.asarray([], dtype=float)
    if x_is_time:
        try:
            parsed = pd.to_datetime(pd.Series(items), errors="coerce", format="mixed")
            numbers = np.asarray(mdates.date2num(parsed), dtype=float)
        except Exception:  # noqa: BLE001
            return None
        return numbers if np.isfinite(numbers).all() else None
    numeric = pd.to_numeric(pd.Series(items), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(numeric).all():
        return numeric
    try:
        parsed = pd.to_datetime(pd.Series(items), errors="coerce", format="mixed")
    except Exception:  # noqa: BLE001
        return None
    if parsed.notna().all():
        numbers = np.asarray(mdates.date2num(parsed), dtype=float)
        return numbers if np.isfinite(numbers).all() else None
    return None


def _format_time_axis(ax, style: Mapping[str, Any]) -> None:
    try:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
        ax.tick_params(axis="x", labelsize=float(style.get("tick_size", 10)))
    except Exception:  # noqa: BLE001
        pass


def _render_line(data: Mapping[str, Any], params: Mapping[str, Any],
                 style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    agg = _pchoice(params, "agg", "mean", ["mean", "median", "sum"])
    error = _pchoice(params, "error", "auto", ["auto", "sd", "se", "ci95", "none"])
    if error == "auto":
        error = str(style.get("error") or "se")
    marker = _pbool(params, "marker", True)
    frame = P.require_tidy(data, ["__x", "__y"]).copy()
    labels = data.get("labels") or {}
    x_is_time = bool(pd.api.types.is_datetime64_any_dtype(frame["__x"]))
    frame["__y"] = pd.to_numeric(frame["__y"], errors="coerce")
    frame = frame[np.isfinite(frame["__y"].to_numpy(dtype=float))]
    if frame.empty:
        raise C.prerequisite("没有有效的 y 数值可绘制折线图")
    group_col = _pick_group_col(data, frame, keys=("__group",))
    group_levels = _levels_of(frame, group_col, max_levels=12, warnings=warnings) if group_col else []
    if group_col and len(group_levels) < 2:
        group_col, group_levels = None, []
    series = _aggregate_series(frame, "__x", group_col, agg, error)
    if not series:
        raise C.prerequisite("按 x 聚合后没有有效数据，无法绘制折线图")
    colors = _colors(style, group_levels) if group_col else {}
    alpha = float(style.get("alpha") or 0.85)
    line_width = max(0.8, float(style.get("line_width", 1.4) or 1.4))
    point_size = float(style.get("point_size", 28) or 28)

    fig, ax = C.new_fig(style)
    for index, item in enumerate(series):
        name = item["group"]
        color = colors.get(name, C.color_list(style, len(series))[index]) if name else C.color_list(style, 1)[0]
        x_values = _x_numbers(item["x"], x_is_time)
        if x_values is None:
            raise C.bad_request("折线图需要数值型或时间型 x 轴；当前 x 取值无法转换为数值/日期，请改选 x 列")
        centers = np.asarray(item["center"], dtype=float)
        errors = np.asarray(item["error"], dtype=float)
        label = name if name else str(labels.get("y") or "value")
        ax.plot(x_values, centers, color=color, linewidth=line_width, alpha=alpha,
                label=label, marker="o" if marker else None,
                markersize=max(3.0, math.sqrt(point_size) * 0.7),
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)
        if error != "none" and np.any(errors > 0):
            ax.fill_between(x_values, centers - errors, centers + errors,
                            color=color, alpha=float(style.get("alpha", 0.85)) * 0.18,
                            linewidth=0, zorder=2)
        if name:
            _legend_line(ax, name, color, "-", line_width)
    if x_is_time:
        _format_time_axis(ax, style)
    _apply_labels(ax, style, x=str(labels.get("x") or "x"), y=str(labels.get("y") or "y"))
    title = f"Line chart: {agg} {str(labels.get('y') or 'y')} vs {str(labels.get('x') or 'x')}"

    rows = []
    for item in series:
        for x_value, center, err, n in zip(item["x"], item["center"], item["error"], item["n"]):
            rows.append({"x": _fmt_level(x_value), "group": item["group"] or "",
                         "n": n, "value": center, "error": err})
    table = C.table(
        "line_values", "折线聚合值",
        [{"key": "x", "label": str(labels.get("x") or "x"), "type": "str"},
         {"key": "group", "label": str(labels.get("group") or "group"), "type": "str"},
         {"key": "n", "label": "n", "type": "num"},
         {"key": "value", "label": _AGG_LABEL.get(agg, agg), "type": "num"},
         {"key": "error", "label": "误差", "type": "num"}],
        rows)
    bullets = [f"共 {len(frame)} 个观测、{len(series)} 条曲线，x 轴 {len({_fmt_level(v) for v in frame['__x']})} 个取值"]
    longest = max(series, key=lambda item: len(item["x"]))
    peak_index = int(np.argmax(longest["center"])) if longest["center"] else 0
    if longest["center"]:
        bullets.append(
            f"{'分组 ' + str(longest['group']) + ' 的' if longest['group'] else ''}峰值出现在 "
            f"x = {_fmt_level(longest['x'][peak_index])}（{C.fmt_num(longest['center'][peak_index])}）"
        )
    if len(series) > 1:
        bullets.append(f"共 {len(series)} 条分组曲线，颜色与图例对应分组")
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "n_series": len(series)},
        "tables": [table],
        "warnings": warnings,
        "stats": None,
        "annotations": None,
    }


# ---------------------------------------------------------------------------
# 12) area —— 面积图
# ---------------------------------------------------------------------------


def _render_area(data: Mapping[str, Any], params: Mapping[str, Any],
                 style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    agg = _pchoice(params, "agg", "mean", ["mean", "median", "sum"])
    stack = _pbool(params, "stack", True)
    normalize = _pbool(params, "normalize", False)
    frame = P.require_tidy(data, ["__x", "__y"]).copy()
    labels = data.get("labels") or {}
    x_is_time = bool(pd.api.types.is_datetime64_any_dtype(frame["__x"]))
    frame["__y"] = pd.to_numeric(frame["__y"], errors="coerce")
    frame = frame[np.isfinite(frame["__y"].to_numpy(dtype=float))]
    if frame.empty:
        raise C.prerequisite("没有有效的 y 数值可绘制面积图")
    group_col = _pick_group_col(data, frame, keys=("__group",))
    group_levels = _levels_of(frame, group_col, max_levels=10, warnings=warnings) if group_col else []
    if group_col and len(group_levels) < 2:
        group_col, group_levels = None, []
    series = _aggregate_series(frame, "__x", group_col, agg, "none")
    if not series:
        raise C.prerequisite("按 x 聚合后没有有效数据，无法绘制面积图")
    colors = _colors(style, group_levels) if group_col else {}
    alpha = float(style.get("alpha") or 0.85)

    # 以最长的 x 序列为基准，把各分组对齐到同一网格
    base_x = max(series, key=lambda item: len(item["x"]))["x"]
    x_values = _x_numbers(base_x, x_is_time)
    if x_values is None:
        raise C.bad_request("面积图需要数值型或时间型 x 轴；当前 x 取值无法转换为数值/日期，请改选 x 列")
    matrix = np.full((len(series), len(base_x)), np.nan, dtype=float)
    for si, item in enumerate(series):
        mapping = {_fmt_level(x): index for index, x in enumerate(item["x"])}
        for xi, x_value in enumerate(base_x):
            index = mapping.get(_fmt_level(x_value))
            if index is not None:
                matrix[si, xi] = item["center"][index]
    matrix = np.nan_to_num(matrix, nan=0.0)
    if normalize:
        totals = matrix.sum(axis=0)
        safe = np.where(totals > 1e-12, totals, 1.0)
        matrix = matrix / safe[None, :] * 100.0
    colors_list = [
        colors.get(item["group"], C.color_list(style, len(series))[index]) if item["group"]
        else C.color_list(style, 1)[0]
        for index, item in enumerate(series)
    ]
    fig, ax = C.new_fig(style)
    if stack:
        ax.stackplot(x_values, matrix, colors=colors_list, alpha=alpha,
                     labels=[item["group"] or str(labels.get("y") or "value") for item in series],
                     edgecolor="white", linewidth=0.4)
        for index, item in enumerate(series):
            _legend_patch(ax, item["group"] or str(labels.get("y") or "value"),
                          colors_list[index], alpha=alpha)
    else:
        for index, item in enumerate(series):
            color = colors_list[index]
            label = item["group"] or str(labels.get("y") or "value")
            ax.fill_between(x_values, 0, matrix[index], color=color, alpha=alpha * 0.7,
                            linewidth=0, label=label)
            ax.plot(x_values, matrix[index], color=color,
                    linewidth=max(0.8, float(style.get("line_width", 1.4) or 1.4)), zorder=4)
            _legend_patch(ax, label, color, alpha=alpha * 0.7)
    if x_is_time:
        _format_time_axis(ax, style)
    ax.set_ylim(bottom=0.0)
    _apply_labels(ax, style, x=str(labels.get("x") or "x"),
                  y="Percentage (%)" if normalize else str(labels.get("y") or "y"))
    title = f"Area chart: {str(labels.get('y') or 'y')} vs {str(labels.get('x') or 'x')}"
    if stack:
        title += " (stacked)"

    rows = []
    for si, item in enumerate(series):
        for xi, x_value in enumerate(base_x):
            rows.append({"x": _fmt_level(x_value), "group": item["group"] or "",
                         "value": matrix[si, xi]})
    table = C.table(
        "area_values", "面积图数值",
        [{"key": "x", "label": str(labels.get("x") or "x"), "type": "str"},
         {"key": "group", "label": str(labels.get("group") or "group"), "type": "str"},
         {"key": "value", "label": "Percentage (%)" if normalize else _AGG_LABEL.get(agg, agg), "type": "num"}],
        rows)
    bullets = [
        f"共 {len(frame)} 个观测、{len(series)} 条面积序列、{len(base_x)} 个 x 取值",
    ]
    if stack and len(series) > 1:
        overall = np.nansum(matrix, axis=1)
        total = float(np.sum(overall))
        if total > 0:
            top = int(np.argmax(overall))
            bullets.append(f"累计贡献最大的分组：{series[top]['group']}（{overall[top] / total * 100:.1f}%）")
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(frame)), "n_series": len(series), "n_x": len(base_x)},
        "tables": [table],
        "warnings": warnings,
        "stats": None,
        "annotations": None,
    }


# ---------------------------------------------------------------------------
# 矩阵类公共
# ---------------------------------------------------------------------------


def _unique_names(names: Sequence[Any]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = str(name)
        key = base
        index = 1
        while key in seen:
            index += 1
            key = f"{base}.{index}"
        seen[key] = 1
        out.append(key)
    return out


def _wide_matrix(data: Mapping[str, Any], chart: str) -> tuple[np.ndarray, list[str], list[str], dict]:
    wide = data.get("wide")
    if not isinstance(wide, Mapping) or wide.get("matrix") is None:
        raise C.prerequisite(f"{chart} 需要矩阵/宽表数据：请用矩阵数据（首列特征、其余为样本），或用至少有 2 个数值列的数据集")
    arr = np.asarray(wide.get("matrix"), dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise C.prerequisite(f"{chart} 需要至少 2 个样本 × 2 个特征的数据")
    features = _unique_names(wide.get("features") or [f"F{i + 1}" for i in range(arr.shape[1])])
    samples = [str(v) for v in (wide.get("samples") or [f"S{i + 1}" for i in range(arr.shape[0])])]
    if len(features) != arr.shape[1]:
        features = [f"F{i + 1}" for i in range(arr.shape[1])]
    if len(samples) != arr.shape[0]:
        samples = [f"S{i + 1}" for i in range(arr.shape[0])]
    return arr, features, samples, dict(wide)


def _text_color_for(cmap, norm, value: float) -> str:
    try:
        red, green, blue, _alpha = cmap(norm(value))
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        return "#FFFFFF" if luminance < 0.55 else "#1A1A1A"
    except Exception:  # noqa: BLE001
        return "#1A1A1A"


# ---------------------------------------------------------------------------
# 13) heatmap —— 热图
# ---------------------------------------------------------------------------


def _render_heatmap(data: Mapping[str, Any], params: Mapping[str, Any],
                    style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    arr, features, samples, wide = _wide_matrix(data, "热图")
    top_n = _pint(params, "top_n", 60, 2, 500)
    if arr.shape[1] > top_n:
        arr, features = _top_variance(arr, features, top_n, warnings, "特征")
    zscore = _pchoice(params, "zscore", "row", ["none", "row", "col"])
    cluster_rows = _pbool(params, "cluster_rows", True)
    cluster_cols = _pbool(params, "cluster_cols", True)
    annotate = _pbool(params, "annotate", False)

    values = arr.T.copy()  # 行=特征，列=样本
    if zscore in ("row", "col"):
        axis = 1 if zscore == "row" else 0
        mean = np.nanmean(values, axis=axis, keepdims=True)
        std = np.nanstd(values, axis=axis, ddof=0, keepdims=True)
        std = np.where(std > 1e-12, std, 1.0)
        values = (values - mean) / std
    values, row_names, col_order = _reorder(values, features, cluster_rows, cluster_cols)
    col_names = [samples[index] for index in col_order if index < len(samples)]

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise C.prerequisite("热图矩阵里没有有效数值")
    if zscore == "none":
        vmin = float(np.percentile(finite, 1))
        vmax = float(np.percentile(finite, 99))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
        color_label = "value"
    else:
        limit = float(np.nanpercentile(np.abs(values), 98))
        limit = max(1.5, min(3.0, limit)) if math.isfinite(limit) and limit > 0 else 2.0
        vmin, vmax = -limit, limit
        color_label = "z-score"
    cmap = _pick_cmap(style, params.get("cmap"), diverging=(zscore != "none"))
    tick_size = float(style.get("tick_size", 10))
    label_size = float(style.get("label_size", 12))

    fig, ax = C.new_fig(style)
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, fontsize=max(5.0, tick_size - 2))
    rotation = float(style.get("x_rotate") or (90.0 if len(col_names) > 8 else 0.0))
    if rotation:
        plt.setp(ax.get_xticklabels(), rotation=rotation,
                 ha="right" if rotation >= 30 else "center")
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=max(5.0, tick_size - 1))
    group_names: list[str] = []
    groups = wide.get("sample_groups")
    if groups is not None and len(groups) == len(samples):
        ordered = [_fmt_level(groups[index]) for index in col_order if index < len(groups)]
        group_names = list(dict.fromkeys(ordered))
        group_colors = _colors(style, group_names)
        for label, group in zip(ax.get_xticklabels(), ordered):
            label.set_color(group_colors.get(group, "#333333"))
        for group in group_names:
            _legend_patch(ax, group, group_colors.get(group, C.color_list(style, 1)[0]))
    if annotate:
        if values.size <= 400:
            font_size = max(4.5, min(8.0, 9.0 - 0.06 * max(values.shape)))
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            for ri in range(values.shape[0]):
                for ci in range(values.shape[1]):
                    value = float(values[ri, ci])
                    if not math.isfinite(value):
                        continue
                    ax.text(ci, ri, f"{value:.2f}", ha="center", va="center",
                            fontsize=font_size, color=_text_color_for(cmap, norm, value), zorder=4)
        else:
            warnings.append("格子数超过 400，已跳过数值标注")
    aux_axes = []
    try:
        cbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
        cbar.set_label(color_label, fontsize=label_size)
        cbar.ax.tick_params(labelsize=tick_size)
        aux_axes.append(cbar.ax)
    except Exception:  # noqa: BLE001
        warnings.append("颜色条绘制失败")
    _apply_labels(ax, style, x="Samples", y="Features")
    title = f"Heatmap: {len(row_names)} features × {len(col_names)} samples"

    variance = np.nanvar(arr, axis=0)
    order = np.argsort(-np.nan_to_num(variance, nan=-1.0))
    feature_rows = []
    for index in order[:20]:
        column = arr[:, int(index)]
        feature_rows.append({
            "feature": features[int(index)],
            "mean": float(np.nanmean(column)),
            "sd": float(np.nanstd(column, ddof=1)) if np.isfinite(column).sum() > 1 else None,
            "variance": float(variance[int(index)]),
        })
    table = C.table(
        "heatmap_features", "特征方差排序（前 20）",
        [{"key": "feature", "label": "特征", "type": "str"},
         {"key": "mean", "label": "均值", "type": "num"},
         {"key": "sd", "label": "标准差", "type": "num"},
         {"key": "variance", "label": "方差", "type": "num"}],
        feature_rows)
    bullets = [
        f"热图 {len(row_names)} 个特征 × {len(col_names)} 个样本",
        f"按{'行' if zscore == 'row' else '列'} z-score 标准化" if zscore != "none" else "未做标准化（原始值）",
    ]
    if cluster_rows or cluster_cols:
        parts = []
        if cluster_rows:
            parts.append("行")
        if cluster_cols:
            parts.append("列")
        bullets.append("已按层次聚类重排" + "/".join(parts) + "顺序")
    if feature_rows:
        bullets.append("方差最大的特征：" + "、".join(row["feature"] for row in feature_rows[:3]))
    if group_names:
        counts = {name: sum(1 for group in groups if _fmt_level(group) == name) for name in group_names}
        bullets.append("样本分组：" + "，".join(f"{name} n={counts[name]}" for name in group_names))
        _ = counts
    key_numbers = {"n_features": len(row_names), "n_samples": len(col_names),
                   "top_variance_feature": feature_rows[0]["feature"] if feature_rows else None}
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": key_numbers,
        "tables": [table],
        "warnings": warnings,
        "stats": None,
        "annotations": None,
        "aux": aux_axes,
    }


# ---------------------------------------------------------------------------
# 14) corr —— 相关矩阵
# ---------------------------------------------------------------------------


def _exact_rank_pvalue(a: np.ndarray, b: np.ndarray) -> float | None:
    """小样本 Spearman 精确置换 p（双侧 |rho|）；n>6 返回 None。

    scipy 的 spearmanr 用 t 渐近近似，rho=±1 且 n 很小时会给出 p=0（零分布被压扁）；
    斐查小样本（n<=6）用全部 n! 次秩置换枚举真实零分布。
    """
    n = int(a.size)
    if n < 3 or n > 6:
        return None
    from itertools import permutations

    try:
        from scipy.stats import rankdata

        ranks_a = rankdata(a)
        ranks_b = rankdata(b)

        def rho(left: np.ndarray, right: np.ndarray) -> float:
            x = left - left.mean()
            y = right - right.mean()
            denom = math.sqrt(float((x ** 2).sum()) * float((y ** 2).sum()))
            return float((x * y).sum() / denom) if denom > 1e-12 else 0.0

        observed = abs(rho(ranks_a, ranks_b))
        hits = 0
        total = 0
        for order in permutations(range(n)):
            total += 1
            if abs(rho(ranks_a, ranks_b[list(order)])) >= observed - 1e-12:
                hits += 1
        return hits / total if total else None
    except Exception:  # noqa: BLE001 - 退化数据退回渐近近似
        return None


# Kendall 精确零分布的置换枚举上限：n<=8 时 n! <= 40320，枚举全部排列得到的是
# 真实零分布（不是渐近近似），完全可接受；n>8 改走标准正态渐近式。
_KENDALL_EXACT_MAX_N = 8
# n -> 「逆序数 → 排列个数」分布（Mahonian 数）；每个 n 只枚举一次后缓存。
_KENDALL_EXACT_NULL_CACHE: dict[int, tuple[int, ...]] = {}


def _kendall_exact_null_counts(n: int) -> tuple[int, ...]:
    """枚举 n! 个排列，返回逆序数分布（第 i 项 = 逆序数为 i 的排列个数）。

    当 x 严格递增（无并列）时，Kendall tau 只由排列的逆序数 inv 决定：
        tau = 1 - 4*inv / (n*(n-1))
    于是枚举全部 n! 个排列就得到 tau 的精确零分布。单次 n=8 最多 40320 个排列，
    结果按 n 缓存，同一进程内不会重复枚举。
    """
    cached = _KENDALL_EXACT_NULL_CACHE.get(n)
    if cached is not None:
        return cached
    from itertools import permutations

    counts = [0] * (n * (n - 1) // 2 + 1)
    for order in permutations(range(n)):
        inversions = 0
        for i in range(n):
            head = order[i]
            for j in range(i + 1, n):
                if head > order[j]:
                    inversions += 1
        counts[inversions] += 1
    result = tuple(counts)
    _KENDALL_EXACT_NULL_CACHE[n] = result
    return result


def _kendall_exact_pvalue(tau: float, n: int) -> float | None:
    """小样本 Kendall 精确置换双侧 P；n 超出枚举上限时返回 None。

    精确 p = 「全部 n! 个排列中 |tau_perm| >= |tau_obs| 的比例」。
    以 tau=1、n=4 为例：4! = 24 个排列里只有完全同序（tau=+1）与完全反序
    （tau=-1）这 2 个满足 |tau_perm| >= 1，所以 p = 2/24 = 0.083333 —— 这正是
    scipy.stats.kendalltau 的精确口径，也是旧 Pearson t 近似在 tau=1 时发散成
    nan 的根因：t 近似假设 r<1，而秩完全同序时 r=1 恰好落在奇点。
    前提：无并列（tau-a 零分布）。若已知存在并列，请改传原始向量走 scipy 的
    并列校正口径（_corr_pvalue 的向量接口）。
    """
    if n < 3 or n > _KENDALL_EXACT_MAX_N:
        return None
    counts = _kendall_exact_null_counts(n)
    total = math.factorial(n)
    observed = abs(max(-1.0, min(1.0, float(tau))))  # 数值误差可能给出略超 ±1
    denominator = n * (n - 1)
    hits = 0
    for inversions, count in enumerate(counts):
        if not count:
            continue
        tau_perm = 1.0 - 4.0 * inversions / denominator
        if abs(tau_perm) >= observed - 1e-12:
            hits += count
    return hits / float(total)


def _kendall_asymptotic_pvalue(tau: float, n: int) -> float:
    """大样本 Kendall 双侧 P（标准正态渐近，无并列口径）。

        z = 3 * tau * sqrt(n*(n-1)) / sqrt(2*(2n+5))
        p = 2 * (1 - Phi(|z|)) = 2 * Phi(-|z|)（用 sf 计算，避免 1-Phi 的相消误差）

    小样本（n<=8）已由精确置换覆盖，所以这里不需要再做连续性/小样本修正：修正的
    必要性只存在于「小 n 却用渐近」的场景，而那种场景在本实现里不会发生。n>8 时
    scipy 在无并列且 n<=33 时默认仍走精确置换（尤其 |tau|→1 的尾部，精确 p 会比
    渐近 p 更小）；本函数面向「只拿得到 (tau, n)」的调用方，无法枚举/无法判断并列，
    因此明确采用标准渐近口径。需要更精确时请改用向量接口
    `_corr_pvalue(x, y, "kendall")`（内部走 scipy 的 exact/asymptotic 自动选择）。
    """
    try:
        z_value = 3.0 * float(tau) * math.sqrt(float(n) * (n - 1.0)) / math.sqrt(2.0 * (2.0 * n + 5.0))
        p_value = float(2.0 * sps.norm.sf(abs(z_value)))
    except Exception:  # noqa: BLE001 - 退化输入不给显著性
        return 1.0
    if not math.isfinite(p_value):
        return 1.0
    return max(0.0, min(1.0, p_value))


def _kendall_pvalue_from_statistic(tau: float, n: float) -> float:
    """只有 (tau, n)、没有原始观测时的 Kendall 双侧 P：小 n 精确、大 n 渐近。

    旧实现把所有相关方法都套 Pearson 的 t 近似：
        t = r*sqrt((n-2)/(1-r^2))
    tau=1 时分母为 0 → nan；tau 接近 1 时也被严重低估。Kendall 的零分布与
    Pearson 完全不同，必须按秩置换/渐近正态口径计算。本函数保证返回有限值
    （不返回 nan）：样本量不足以定义零分布时按最保守的 p=1 处理。
    """
    try:
        tau_value = float(tau)
        n_value = float(n)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(tau_value) or not math.isfinite(n_value) or n_value < 3.0:
        return 1.0  # n<3 时 tau 不携带任何可检验信息：保守地给 p=1，而不是 nan
    n_int = int(round(n_value))
    exact = _kendall_exact_pvalue(tau_value, n_int)
    if exact is not None:
        return max(0.0, min(1.0, float(exact)))
    return _kendall_asymptotic_pvalue(tau_value, n_int)


def _legacy_statistic_pvalue(statistic: float, n: float) -> float:
    """只有 (统计量, n) 时 Pearson/Spearman 沿用的 t 近似（保持既有行为不变）。"""
    if not math.isfinite(statistic) or not math.isfinite(n) or n < 3:
        return float("nan")
    if abs(statistic) >= 1.0:
        return 0.0
    try:
        t_value = abs(statistic) * math.sqrt((n - 2.0) / max(1e-300, 1.0 - statistic * statistic))
        return float(2.0 * sps.t.sf(t_value, n - 2.0))
    except Exception:  # noqa: BLE001
        return float("nan")


def _corr_pvalue(x: Any, y: Any = None, method: str = "pearson") -> float:
    """按方法分派真实零分布，支持两种调用形态（向后兼容）。

    1) 观测向量：_corr_pvalue(x_array, y_array, method)
       corr 热图走这条：Pearson=t 近似；Spearman=小样本精确置换/AS89；
       Kendall=scipy.stats.kendalltau（并列时自带并列校正）。
    2) 统计量：_corr_pvalue(statistic, n, method="kendall")
       只有统计量、没有原始数据时走这条（外部校验/报告/取证回查）。
       Kendall 用「小 n 精确置换、大 n 正态渐近」算 P（见上），tau=±1 也返回有限值。
       若两个入参都是标量（0 维或长度 1），按 (statistic, n) 解释。
    Pearson/Spearman 的既有行为不变。
    """
    left = np.asarray(x, dtype=float).ravel()
    right = np.asarray([] if y is None else y, dtype=float).ravel()
    name = str(method or "pearson").lower()
    if left.size == 1 and right.size == 1:
        statistic = float(left[0])
        size = float(right[0])
        if name == "kendall":
            return _kendall_pvalue_from_statistic(statistic, size)
        return _legacy_statistic_pvalue(statistic, size)
    mask = np.isfinite(left) & np.isfinite(right)
    a, b = left[mask], right[mask]
    if a.size < 3:
        return float("nan")
    try:
        if name == "kendall":
            return float(sps.kendalltau(a, b).pvalue)
        if name == "spearman":
            exact = _exact_rank_pvalue(a, b)
            if exact is not None:
                return float(exact)
            return float(sps.spearmanr(a, b).pvalue)
        return float(sps.pearsonr(a, b).pvalue)
    except Exception:  # noqa: BLE001 - 常数序列等退化情形
        return float("nan")


def _render_corr(data: Mapping[str, Any], params: Mapping[str, Any],
                 style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    arr, features, samples, _wide = _wide_matrix(data, "相关矩阵")
    top_n = _pint(params, "top_n", 40, 2, 200)
    if arr.shape[1] > top_n:
        arr, features = _top_variance(arr, features, top_n, warnings, "变量")
    method = _pchoice(params, "method", "pearson", ["pearson", "spearman", "kendall"])
    show_star = _pbool(params, "star", True)
    annotate = _pbool(params, "annotate", True)
    frame = pd.DataFrame(arr, columns=features)
    try:
        corr = frame.corr(method=method).to_numpy(dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise C.EngineError("internal", f"相关矩阵计算失败：{exc}") from exc
    k = len(features)
    pmatrix = np.ones((k, k), dtype=float)
    columns = [frame[features[index]].to_numpy(dtype=float) for index in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                pmatrix[i, j] = 0.0
            else:
                pmatrix[i, j] = _corr_pvalue(columns[i], columns[j], method)
    finite = corr[np.isfinite(corr)]
    if finite.size == 0:
        raise C.prerequisite("相关矩阵没有有效数值（变量可能全为缺失或常数）")
    cmap = _pick_cmap(style, "", diverging=True)
    tick_size = float(style.get("tick_size", 10))
    label_size = float(style.get("label_size", 12))
    fig, ax = C.new_fig(style)
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap=cmap, aspect="equal")
    rotation = float(style.get("x_rotate") or (45.0 if k > 6 else 0.0))
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(features, fontsize=max(5.0, tick_size - 2))
    ax.set_yticklabels(features, fontsize=max(5.0, tick_size - 1))
    if rotation:
        plt.setp(ax.get_xticklabels(), rotation=rotation,
                 ha="right" if rotation >= 30 else "center")
    if annotate:
        if k <= 25:
            font_size = max(4.5, min(8.5, 10.5 - 0.28 * k))
            norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
            for i in range(k):
                for j in range(k):
                    value = float(corr[i, j])
                    if not math.isfinite(value):
                        continue
                    text = f"{value:.2f}"
                    if show_star and i != j:
                        stars = C.p_stars(pmatrix[i, j])
                        if stars and stars != "ns":
                            text += stars
                    ax.text(j, i, text, ha="center", va="center", fontsize=font_size,
                            color=_text_color_for(cmap, norm, value), zorder=4)
        else:
            warnings.append("变量数超过 25，已跳过相关系数标注（可减少 top_n）")
    aux_axes = []
    try:
        cbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
        cbar.set_label(f"{method.capitalize()} correlation", fontsize=label_size)
        cbar.ax.tick_params(labelsize=tick_size)
        aux_axes.append(cbar.ax)
    except Exception:  # noqa: BLE001
        warnings.append("颜色条绘制失败")
    _apply_labels(ax, style, x="", y="")
    title = f"Correlation matrix: {k} variables ({method})"

    pair_rows = []
    for i in range(k):
        for j in range(i + 1, k):
            value = float(corr[i, j])
            if not math.isfinite(value):
                continue
            pair_rows.append({"var1": features[i], "var2": features[j], "r": value,
                              "p": float(pmatrix[i, j]),
                              "sig": (C.p_stars(pmatrix[i, j]) if show_star else "")})
    pair_rows.sort(key=lambda row: -abs(row["r"]))
    table = C.table(
        "corr_pairs", f"相关最强的前 20 对（{method}）",
        [{"key": "var1", "label": "变量 1", "type": "str"},
         {"key": "var2", "label": "变量 2", "type": "str"},
         {"key": "r", "label": "相关系数 r", "type": "num"},
         {"key": "p", "label": "p", "type": "p"},
         {"key": "sig", "label": "显著性", "type": "str"}],
        pair_rows[:20])
    significant = sum(1 for row in pair_rows
                      if math.isfinite(row["p"]) and row["p"] < 0.05)
    bullets = [
        f"{k} 个变量两两相关（{method}），基于 {len(samples)} 个样本",
        f"最强相关：{pair_rows[0]['var1']} vs {pair_rows[0]['var2']}，r = "
        f"{C.fmt_num(pair_rows[0]['r'])}（p = {C.fmt_p(pair_rows[0]['p'])}）" if pair_rows else "没有有效变量对",
        f"{len(pair_rows)} 对变量中 {significant} 对在 p<0.05 水平显著（未做多重校正）",
    ]
    stats = {
        "method": method,
        "n_variables": k,
        "n_samples": len(samples),
        "strongest": ({"var1": pair_rows[0]["var1"], "var2": pair_rows[0]["var2"],
                       "r": pair_rows[0]["r"], "p": pair_rows[0]["p"]} if pair_rows else None),
    }
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n_variables": k, "n_samples": len(samples),
                        "max_abs_r": (abs(pair_rows[0]["r"]) if pair_rows else None)},
        "tables": [table],
        "warnings": warnings,
        "stats": stats,
        "annotations": None,
        "aux": aux_axes,
    }


# ---------------------------------------------------------------------------
# 15) pairplot —— 散点矩阵
# ---------------------------------------------------------------------------


def _pair_colors(data: Mapping[str, Any], wide: Mapping[str, Any],
                 n_rows: int) -> tuple[list[str] | None, str | None]:
    groups = wide.get("sample_groups")
    if groups is not None and len(groups) == n_rows:
        labels = [_fmt_level(value) for value in list(groups)]
        if len(set(labels)) >= 2:
            return labels, str(wide.get("group_name") or "group")
    tidy = data.get("tidy")
    if isinstance(tidy, pd.DataFrame) and len(tidy) == n_rows:
        for key in ("__color", "__group"):
            if key in tidy.columns and tidy[key].notna().all():
                labels = [_fmt_level(value) for value in tidy[key].tolist()]
                if len(set(labels)) >= 2:
                    return labels, key[2:]
    return None, None


def _render_pairplot(data: Mapping[str, Any], params: Mapping[str, Any],
                     style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    arr, features, samples, wide = _wide_matrix(data, "散点矩阵")
    requested = _split_terms(params.get("vars") or "")
    if requested:
        keep = [index for index, name in enumerate(features) if name in requested]
        if len(keep) < 2:
            raise C.bad_request(
                f"params.vars 中至少需要 2 个存在的数值列；可选列：{', '.join(features[:12])}"
            )
        features = [features[index] for index in keep]
        arr = arr[:, keep]
    if len(features) > 6:
        raise C.bad_request(
            f"散点矩阵最多支持 6 个数值列（当前 {len(features)} 个）；"
            f"请在参数 vars 里指定要画的列，例如 \"{features[0]},{features[1]},{features[2]}\""
        )
    diag = _pchoice(params, "diag", "hist", ["hist", "kde"])
    show_r = _pbool(params, "show_r", False)
    group_labels, group_name = _pair_colors(data, wide, arr.shape[0])
    group_levels = list(dict.fromkeys(group_labels)) if group_labels else []
    group_colors = _colors(style, group_levels)
    n = len(features)
    frame = pd.DataFrame(arr, columns=features)
    if group_labels is not None:
        frame["__g"] = group_labels
    if len(frame) > 3000:
        rng = np.random.default_rng(20261124)
        index = rng.choice(len(frame), size=3000, replace=False)
        index.sort()
        frame = frame.iloc[index]
        warnings.append(f"样本数 {arr.shape[0]} 超过 3000，散点矩阵已抽样 3000 个点")
    alpha = float(style.get("alpha") or 0.85)
    point_size = float(style.get("point_size", 28) or 28) * 0.45
    tick_size = float(style.get("tick_size", 10))
    label_size = float(style.get("label_size", 12))
    side = max(float(style.get("fig_width") or 6.2), 1.5 * n + 1.8)
    fig, axes = plt.subplots(n, n, figsize=(side, side), layout="constrained")
    axes = np.atleast_2d(axes)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                if diag == "kde":
                    values = _finite(frame[features[i]])
                    if values.size >= 3 and group_labels is None:
                        low, high = float(np.min(values)), float(np.max(values))
                        pad = max((high - low) * 0.1, 1e-9)
                        grid = np.linspace(low - pad, high + pad, 160)
                        density = _kde_curve(values, grid, bw=1.0, trim=True)
                        if density is not None:
                            ax.plot(grid, density, color=C.color_list(style, 1)[0],
                                    linewidth=max(0.8, float(style.get("line_width", 1.4))))
                        else:
                            ax.hist(values, bins=14, color=C.color_list(style, 1)[0], alpha=alpha)
                    else:
                        for group in group_levels:
                            values = _finite(frame.loc[frame["__g"] == group, features[i]])
                            if values.size < 3:
                                continue
                            low, high = float(np.min(values)), float(np.max(values))
                            pad = max((high - low) * 0.1, 1e-9)
                            grid = np.linspace(low - pad, high + pad, 160)
                            density = _kde_curve(values, grid, bw=1.0, trim=True)
                            if density is not None:
                                ax.plot(grid, density, color=group_colors.get(group, "#4E79A7"),
                                        linewidth=max(0.8, float(style.get("line_width", 1.4))),
                                        alpha=alpha)
                else:
                    if group_labels is not None:
                        for group in group_levels:
                            values = _finite(frame.loc[frame["__g"] == group, features[i]])
                            if values.size:
                                ax.hist(values, bins=14, alpha=0.55, color=group_colors.get(group, "#4E79A7"))
                    else:
                        values = _finite(frame[features[i]])
                        if values.size:
                            ax.hist(values, bins=14, color=C.color_list(style, 1)[0], alpha=alpha)
            else:
                if group_labels is not None:
                    for group in group_levels:
                        cell = frame[frame["__g"] == group]
                        if cell.empty:
                            continue
                        ax.scatter(cell[features[j]], cell[features[i]], s=point_size,
                                   color=group_colors.get(group, "#4E79A7"), alpha=alpha,
                                   edgecolors="none")
                else:
                    ax.scatter(frame[features[j]], frame[features[i]], s=point_size,
                               color=C.color_list(style, 1)[0], alpha=alpha, edgecolors="none")
                if show_r:
                    values_x = _finite(frame[features[j]])
                    values_y = _finite(frame[features[i]])
                    if values_x.size >= 3 and values_y.size >= 3:
                        try:
                            r = float(sps.pearsonr(frame[features[j]], frame[features[i]])[0])
                            ax.text(0.96, 0.06, f"r = {C.fmt_num(r)}", transform=ax.transAxes,
                                    ha="right", va="bottom",
                                    fontsize=max(5.0, tick_size - 3.0),
                                    color="#1A1A1A")
                        except Exception:  # noqa: BLE001
                            pass
            ax.tick_params(labelsize=max(5.0, tick_size - 2), length=2.5)
            if j == 0 and i != 0:
                ax.set_ylabel(features[i], fontsize=max(6.0, label_size - 2))
            if i == n - 1:
                ax.set_xlabel(features[j], fontsize=max(6.0, label_size - 2))
    for group in group_levels:
        _legend_patch(axes[0, 0], group, group_colors.get(group, C.color_list(style, 1)[0]), alpha=alpha)
    title = f"Scatter matrix: {n} variables"
    corr_matrix = pd.DataFrame(arr, columns=features).corr().to_numpy(dtype=float)
    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            value = float(corr_matrix[i, j])
            if math.isfinite(value):
                pair_rows.append({"var1": features[i], "var2": features[j], "r": value})
    pair_rows.sort(key=lambda row: -abs(row["r"]))
    table = C.table(
        "pairplot_corr", "变量两两相关（Pearson）",
        [{"key": "var1", "label": "变量 1", "type": "str"},
         {"key": "var2", "label": "变量 2", "type": "str"},
         {"key": "r", "label": "相关系数 r", "type": "num"}],
        pair_rows)
    bullets = [
        f"{n} 个数值变量的散点矩阵，{arr.shape[0]} 个样本" +
        (f"，按 {group_name} 着色（{len(group_levels)} 组）" if group_labels else ""),
        "变量：" + "、".join(features),
    ]
    if pair_rows:
        bullets.append(f"最强相关：{pair_rows[0]['var1']} vs {pair_rows[0]['var2']}，"
                       f"r = {C.fmt_num(pair_rows[0]['r'])}")
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(arr.shape[0]), "n_variables": n,
                        "max_abs_r": (abs(pair_rows[0]["r"]) if pair_rows else None)},
        "tables": [table],
        "warnings": warnings,
        "stats": None,
        "annotations": None,
    }



# ---------------------------------------------------------------------------
# 分布类公共
# ---------------------------------------------------------------------------


def _describe(values: Any, label: str = "") -> dict | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    q1, q3 = (float(v) for v in np.percentile(arr, [25, 75]))
    try:
        skew = float(sps.skew(arr))
    except Exception:  # noqa: BLE001
        skew = None
    return {
        "group": label,
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
        "median": float(np.median(arr)),
        "q1": q1,
        "q3": q3,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "skew": skew,
    }


_DESCRIBE_COLUMNS = [
    {"key": "group", "label": "分组", "type": "str"},
    {"key": "n", "label": "n", "type": "num"},
    {"key": "mean", "label": "均值", "type": "num"},
    {"key": "sd", "label": "标准差", "type": "num"},
    {"key": "median", "label": "中位数", "type": "num"},
    {"key": "q1", "label": "Q1", "type": "num"},
    {"key": "q3", "label": "Q3", "type": "num"},
    {"key": "min", "label": "最小", "type": "num"},
    {"key": "max", "label": "最大", "type": "num"},
    {"key": "skew", "label": "偏度", "type": "num"},
]


def _distribution_inputs(data: Mapping[str, Any], warnings: list[str]) -> tuple[pd.DataFrame, str, str | None, list[str], dict]:
    """直方/密度/QQ 的输入：返回 (frame[x,g], x_label, group_col, group_levels, labels)。"""
    frame = P.require_tidy(data, ["__x"]).copy()
    labels = data.get("labels") or {}
    x_label = str(labels.get("x") or "x")
    group_col = _pick_group_col(data, frame, keys=("__color", "__group"))
    group_levels = _levels_of(frame, group_col, max_levels=10, warnings=warnings) if group_col else []
    if group_col and len(group_levels) < 2:
        group_col, group_levels = None, []
    if _is_numeric_series(frame["__x"]):
        plot = pd.DataFrame({"x": pd.to_numeric(frame["__x"], errors="coerce").to_numpy(dtype=float)})
        if group_col:
            plot["g"] = [_fmt_level(v) for v in frame[group_col].tolist()]
        return plot.dropna(subset=["x"]).reset_index(drop=True), x_label, group_col, group_levels, labels
    fallback = _fallback_numeric(data, warnings)
    if fallback is None:
        raise C.bad_request(
            f"x 角色（{x_label}）不是数值列，无法画分布图；请在左侧把 x 换成数值列"
        )
    name, values = fallback
    plot = pd.DataFrame({"x": np.asarray(values, dtype=float)})
    if group_col and len(frame) == len(plot):
        plot["g"] = [_fmt_level(v) for v in frame[group_col].tolist()]
    else:
        group_col, group_levels = None, []
    return plot.dropna(subset=["x"]).reset_index(drop=True), name, group_col, group_levels, labels


# ---------------------------------------------------------------------------
# 16) hist —— 直方图
# ---------------------------------------------------------------------------


def _render_hist(data: Mapping[str, Any], params: Mapping[str, Any],
                 style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    bins_param = _pint(params, "bins", 20, 0, 200)
    density = _pbool(params, "density", False)
    stacked = _pbool(params, "stack", False)
    show_stats = _pbool(params, "show_stats", True)
    plot, x_label, group_col, group_levels, _labels = _distribution_inputs(data, warnings)
    if plot.empty:
        raise C.prerequisite("没有有效数值可绘制直方图")
    values = plot["x"].to_numpy(dtype=float)
    colors = _colors(style, group_levels) if group_col else {}
    alpha = float(style.get("alpha") or 0.85)
    edge = max(0.4, float(style.get("edge_width", 0.8) or 0.8))
    line_width = max(0.8, float(style.get("line_width", 1.4) or 1.4))

    bins: Any = bins_param if bins_param > 0 else "auto"
    if style.get("x_log") and float(np.min(values)) > 0:
        low, high = float(np.min(values)), float(np.max(values))
        count = bins_param if bins_param > 0 else 20
        if high > low:
            bins = np.geomspace(low, high, count + 1)
    fig, ax = C.new_fig(style)
    if group_col:
        groups_arrays = [plot.loc[plot["g"] == name, "x"].to_numpy(dtype=float)
                         for name in group_levels]
        group_colors = [colors.get(name, C.color_list(style, len(group_levels))[index])
                        for index, name in enumerate(group_levels)]
        if stacked:
            ax.hist(groups_arrays, bins=bins, stacked=True, density=density,
                    color=group_colors, alpha=alpha, edgecolor="white",
                    linewidth=edge * 0.6, label=group_levels)
        else:
            for name, array, color in zip(group_levels, groups_arrays, group_colors):
                ax.hist(array, bins=bins, density=density, histtype="stepfilled",
                        color=color, alpha=alpha * 0.62, edgecolor="white",
                        linewidth=edge * 0.6, label=name)
        for name, color in zip(group_levels, group_colors):
            _legend_patch(ax, name, color, alpha=alpha * 0.75)
    else:
        ax.hist(values, bins=bins, density=density, color=C.color_list(style, 1)[0],
                alpha=alpha, edgecolor="white", linewidth=edge * 0.6)
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    if show_stats:
        ax.axvline(mean_value, color="#B24745", linestyle="--",
                   linewidth=max(0.8, line_width * 0.8), label="_mean")
        ax.axvline(median_value, color="#0072B2", linestyle=":",
                   linewidth=max(0.8, line_width * 0.9), label="_median")
        text = (f"mean = {C.fmt_num(mean_value)}\n"
                f"median = {C.fmt_num(median_value)}\n"
                f"SD = {C.fmt_num(np.std(values, ddof=1) if values.size > 1 else 0)}\n"
                f"n = {values.size}")
        ax.text(0.98, 0.96, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=max(6.0, float(style.get("text_size", 10)) - 1), color="#1A1A1A",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": "#CCCCCC", "alpha": 0.85, "linewidth": 0.6})
    _apply_labels(ax, style, x=x_label, y="Density" if density else "Count")
    title = f"Histogram: {x_label}"

    rows = []
    overall = _describe(values, "Overall")
    if overall:
        rows.append(overall)
    if group_col:
        for name in group_levels:
            described = _describe(plot.loc[plot["g"] == name, "x"], name)
            if described:
                rows.append(described)
    table = C.table("describe", "分布描述统计", _DESCRIBE_COLUMNS, rows)
    missing = int(plot["x"].isna().sum())
    bullets = [
        f"n = {values.size}，均值 {C.fmt_num(mean_value)} ± "
        f"{C.fmt_num(np.std(values, ddof=1) if values.size > 1 else 0)}，"
        f"中位数 {C.fmt_num(median_value)}（IQR {C.fmt_num(np.percentile(values, 25))}–"
        f"{C.fmt_num(np.percentile(values, 75))}）",
        f"范围 {C.fmt_num(float(np.min(values)))} – {C.fmt_num(float(np.max(values)))}"
        + (f"，偏度 {C.fmt_num(float(sps.skew(values)))}" if values.size > 2 else ""),
    ]
    if group_col:
        bullets.append(f"按 {len(group_levels)} 个分组叠加显示")
    if missing:
        bullets.append(f"有 {missing} 个缺失值未参与绘图")
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(values.size), "mean": mean_value, "median": median_value,
                        "sd": float(np.std(values, ddof=1)) if values.size > 1 else None},
        "tables": [table],
        "warnings": warnings,
        "stats": {"dist": "histogram", "describe": rows},
        "annotations": None,
    }


# ---------------------------------------------------------------------------
# 17) density —— 密度图
# ---------------------------------------------------------------------------


def _render_density(data: Mapping[str, Any], params: Mapping[str, Any],
                    style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    fill = _pbool(params, "fill", True)
    bw = _pfloat(params, "bw", 1.0, 0.2, 4.0)
    rug = _pbool(params, "rug", False)
    plot, x_label, group_col, group_levels, _labels = _distribution_inputs(data, warnings)
    if plot.empty:
        raise C.prerequisite("没有有效数值可绘制密度图")
    values_all = plot["x"].to_numpy(dtype=float)
    low, high = float(np.min(values_all)), float(np.max(values_all))
    pad = max((high - low) * 0.12, 1e-9)
    grid = np.linspace(low - pad, high + pad, 200)
    alpha = float(style.get("alpha") or 0.85)
    line_width = max(0.8, float(style.get("line_width", 1.4) or 1.4))
    colors = _colors(style, group_levels) if group_col else {}
    fig, ax = C.new_fig(style)
    drawn = 0
    describe_rows: list[dict] = []
    peak_info: list[dict] = []
    if group_col:
        for index, name in enumerate(group_levels):
            values = plot.loc[plot["g"] == name, "x"].to_numpy(dtype=float)
            color = colors.get(name, C.color_list(style, len(group_levels))[index])
            density = _kde_curve(values, grid, bw=bw, trim=True)
            if density is None:
                warnings.append(f"分组 {name} 样本量或方差不足以估计密度，已跳过该曲线")
                continue
            drawn += 1
            if fill:
                ax.fill_between(grid, 0.0, density, color=color, alpha=alpha * 0.35,
                                linewidth=0)
            ax.plot(grid, density, color=color, linewidth=line_width, label=name, zorder=4)
            _legend_line(ax, name, color, "-", line_width)
            peak_info.append({"group": name, "mode": float(grid[int(np.argmax(density))])})
            described = _describe(values, name)
            if described:
                describe_rows.append(described)
            if rug:
                ax.plot(values, np.full(values.size, 0.0), marker="|", linestyle="none",
                        markersize=5, color=color, alpha=0.5, zorder=3)
    else:
        color = C.color_list(style, 1)[0]
        density = _kde_curve(values_all, grid, bw=bw, trim=True)
        if density is None:
            raise C.prerequisite("数据方差过小或样本量不足，无法估计密度曲线")
        drawn = 1
        if fill:
            ax.fill_between(grid, 0.0, density, color=color, alpha=alpha * 0.35, linewidth=0)
        ax.plot(grid, density, color=color, linewidth=line_width, zorder=4)
        peak_info.append({"group": "", "mode": float(grid[int(np.argmax(density))])})
        described = _describe(values_all, "Overall")
        if described:
            describe_rows.append(described)
        if rug:
            ax.plot(values_all, np.full(values_all.size, 0.0), marker="|", linestyle="none",
                    markersize=5, color=color, alpha=0.5, zorder=3)
    if drawn == 0:
        raise C.prerequisite("所有分组都无法估计密度曲线（样本太少或方差为 0）")
    _apply_labels(ax, style, x=x_label, y="Density")
    title = f"Density plot: {x_label}"
    table = C.table("describe", "分布描述统计", _DESCRIBE_COLUMNS, describe_rows)
    bullets = []
    if len(peak_info) == 1:
        bullets.append(f"n = {values_all.size}，密度峰值约在 {C.fmt_num(peak_info[0]['mode'])}")
    else:
        bullets.append(f"{len(peak_info)} 条分组密度曲线，各自峰值：" +
                       "，".join(f"{item['group']}≈{C.fmt_num(item['mode'])}" for item in peak_info[:5]))
    overall = _describe(values_all, "Overall")
    if overall:
        bullets.append(f"整体中位数 {C.fmt_num(overall['median'])}，"
                       f"IQR {C.fmt_num(overall['q1'])}–{C.fmt_num(overall['q3'])}")
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(values_all.size),
                        "mode": peak_info[0]["mode"] if peak_info else None},
        "tables": [table],
        "warnings": warnings,
        "stats": {"dist": "density", "peaks": peak_info, "describe": describe_rows},
        "annotations": None,
    }


# ---------------------------------------------------------------------------
# 18) qq —— QQ 图
# ---------------------------------------------------------------------------


def _render_qq(data: Mapping[str, Any], params: Mapping[str, Any],
               style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    dist = _pchoice(params, "dist", "normal", ["normal", "uniform", "exponential"])
    scipy_dist = {"normal": "norm", "uniform": "uniform", "exponential": "expon"}.get(dist, "norm")
    plot, x_label, group_col, group_levels, _labels = _distribution_inputs(data, warnings)
    if plot.empty:
        raise C.prerequisite("没有有效数值可绘制 QQ 图")
    values_all = plot["x"].to_numpy(dtype=float)
    colors = _colors(style, group_levels) if group_col else {}
    line_width = max(0.8, float(style.get("line_width", 1.4) or 1.4))
    point_size = float(style.get("point_size", 28) or 28) * 0.8
    alpha = float(style.get("alpha") or 0.85)
    fig, ax = C.new_fig(style)
    stats_rows: list[dict] = []
    groups: list[tuple[str, np.ndarray, str]] = []
    if group_col:
        for index, name in enumerate(group_levels):
            values = plot.loc[plot["g"] == name, "x"].to_numpy(dtype=float)
            groups.append((name, values, colors.get(name, C.color_list(style, len(group_levels))[index])))
    else:
        groups = [("", values_all, C.color_list(style, 1)[0])]
    drawn = 0
    for name, values, color in groups:
        values = _finite(values)
        if values.size < 3:
            warnings.append(f"分组 {name or 'Overall'} 有效点数不足 3，已跳过")
            continue
        try:
            (osm, osr), (slope, intercept, r_value) = sps.probplot(
                values, dist=scipy_dist, fit=True)
        except Exception:  # noqa: BLE001
            warnings.append(f"分组 {name or 'Overall'} QQ 拟合失败，已跳过")
            continue
        drawn += 1
        label = name or f"QQ ({dist})"
        ax.scatter(osm, osr, s=point_size, color=color, alpha=alpha, edgecolors="none", label=label)
        ax.plot(osm, slope * np.asarray(osm) + intercept, color=color, linewidth=line_width,
                linestyle="-", label=f"_{name}_fit")
        if name:
            _legend_marker(ax, name, color, "o", math.sqrt(point_size), alpha)
        shapiro_p = float("nan")
        if dist == "normal" and values.size >= 3:
            try:
                shapiro_p = float(sps.shapiro(values).pvalue)
            except Exception:  # noqa: BLE001
                shapiro_p = float("nan")
        stats_rows.append({"group": name or "Overall", "n": int(values.size),
                           "r": float(r_value), "shapiro_p": shapiro_p,
                           "slope": float(slope)})
    if drawn == 0:
        raise C.prerequisite("所有分组都无法绘制 QQ 图（有效点不足）")
    _apply_labels(ax, style, x=f"Theoretical quantiles ({dist})", y="Sample quantiles")
    title = f"Q-Q plot: {x_label} vs {dist} distribution"
    if dist == "normal" and stats_rows:
        lines = []
        for row in stats_rows[:5]:
            p_text = C.fmt_p(row["shapiro_p"])
            verdict = ""
            if math.isfinite(row["shapiro_p"]):
                verdict = " (normal)" if row["shapiro_p"] > 0.05 else " (non-normal)"
            lines.append(f"{row['group']}: Shapiro-Wilk p = {p_text}{verdict}")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
                fontsize=max(6.0, float(style.get("text_size", 10)) - 1), color="#1A1A1A",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                      "edgecolor": "#CCCCCC", "alpha": 0.85, "linewidth": 0.6})
    table = C.table(
        "qq_stats", f"QQ 拟合与正态性检验（{dist}）",
        [{"key": "group", "label": "分组", "type": "str"},
         {"key": "n", "label": "n", "type": "num"},
         {"key": "r", "label": "QQ 拟合 r", "type": "num"},
         {"key": "slope", "label": "斜率", "type": "num"},
         {"key": "shapiro_p", "label": "Shapiro p", "type": "p"}],
        stats_rows)
    bullets = []
    for row in stats_rows[:3]:
        if dist == "normal" and math.isfinite(row["shapiro_p"]):
            verdict = "接近正态分布" if row["shapiro_p"] > 0.05 else "偏离正态分布"
            bullets.append(f"{row['group']}：n = {row['n']}，Shapiro-Wilk p = "
                           f"{C.fmt_p(row['shapiro_p'])}，{verdict}")
        else:
            bullets.append(f"{row['group']}：n = {row['n']}，QQ 拟合 r = {C.fmt_num(row['r'])}")
    stats = {"dist": dist, "groups": stats_rows}
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(values_all.size),
                        "shapiro_p": stats_rows[0]["shapiro_p"] if stats_rows else None},
        "tables": [table],
        "warnings": warnings,
        "stats": stats,
        "annotations": None,
    }


# ---------------------------------------------------------------------------
# 19) roc —— ROC 曲线
# ---------------------------------------------------------------------------


def _bootstrap_auc(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 200,
                   seed: int = 20261124) -> list[float] | None:
    from sklearn.metrics import auc as sk_auc, roc_curve as sk_roc_curve

    rng = np.random.default_rng(seed)
    total = len(y_true)
    values: list[float] = []
    for _ in range(int(n_boot)):
        index = rng.integers(0, total, size=total)
        y_boot = y_true[index]
        if np.min(y_boot) == np.max(y_boot):
            continue
        try:
            fpr, tpr, _thresholds = sk_roc_curve(y_boot, scores[index])
            values.append(float(sk_auc(fpr, tpr)))
        except Exception:  # noqa: BLE001
            continue
    if len(values) < 20:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _render_roc(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    from sklearn.metrics import auc as sk_auc, roc_curve as sk_roc_curve

    frame = P.require_tidy(data, ["__y"]).copy()
    score = pd.to_numeric(frame["__y"], errors="coerce").to_numpy(dtype=float)
    label_col = None
    for key in ("__group", "__color"):
        if key in frame.columns and frame[key].notna().any():
            label_col = key
            break
    if label_col is None:
        raise C.bad_request("ROC 需要一个二分类标签列；请把 group 角色映射到 0/1 或分组列")
    labels_raw = [_fmt_level(value) for value in frame[label_col].tolist()]
    work = pd.DataFrame({"score": score, "label": labels_raw})
    work = work[np.isfinite(work["score"])].reset_index(drop=True)
    if work.empty:
        raise C.prerequisite("ROC 没有有效分数（y 角色需要数值列）")
    levels = list(dict.fromkeys(work["label"].tolist()))
    if len(levels) != 2:
        raise C.bad_request(
            f"ROC 需要恰好 2 个标签水平（当前 {len(levels)} 个：{', '.join(levels[:6])}）；"
            "请把 group 角色映射到二分类列"
        )
    labels_cfg = data.get("labels") or {}
    positive = _ptext(params, "positive", "").strip()
    if positive and positive not in levels:
        matched = next((level for level in levels if level.lower() == positive.lower()), None)
        if matched is None:
            warnings.append(f"params.positive={positive} 不在标签水平 {levels} 中，已自动判断阳性水平")
            positive = ""
        else:
            positive = matched
    if not positive:
        for level in levels:
            if level.strip().lower() in _POSITIVE_HINTS:
                positive = level
                break
        if not positive:
            positive = levels[-1]
    y_true = (work["label"] == positive).astype(int).to_numpy()
    if int(y_true.sum()) == 0 or int((1 - y_true).sum()) == 0:
        raise C.prerequisite(f"阳性水平 {positive} 没有样本，无法计算 ROC")
    add_ci = _pbool(params, "add_ci", True)
    scores = work["score"].to_numpy(dtype=float)
    fpr, tpr, thresholds = sk_roc_curve(y_true, scores)
    auc_value = float(sk_auc(fpr, tpr))
    flipped = False
    if auc_value < 0.5:
        scores = -scores
        fpr, tpr, thresholds = sk_roc_curve(y_true, scores)
        auc_value = float(sk_auc(fpr, tpr))
        flipped = True
        warnings.append("AUC < 0.5，已自动翻转分数方向（阳性样本应有更高分数）")
    ci = _bootstrap_auc(y_true, scores, 200) if add_ci else None
    youden = tpr - fpr
    best_index = int(np.argmax(youden))
    best_threshold = float(thresholds[best_index])
    sensitivity = float(tpr[best_index])
    specificity = float(1.0 - fpr[best_index])
    color = C.color_list(style, 1)[0]
    line_width = max(1.0, float(style.get("line_width", 1.4) or 1.4))
    fig, ax = C.new_fig(style)
    auc_label = f"ROC (AUC = {auc_value:.3f})"
    if ci:
        auc_label += f", 95% CI {ci[0]:.3f}–{ci[1]:.3f}"
    ax.plot(fpr, tpr, color=color, linewidth=line_width + 0.4, label=auc_label, zorder=4)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9A9A9A", linewidth=1.0, label="_chance")
    ax.scatter([fpr[best_index]], [tpr[best_index]], s=float(style.get("point_size", 28)) * 1.4,
               color="#B24745", edgecolors="white", linewidths=0.8, zorder=6)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    text_lines = [f"AUC = {auc_value:.3f}"]
    if ci:
        text_lines.append(f"95% CI {ci[0]:.3f}–{ci[1]:.3f}")
    text_lines.append(f"n = {len(work)} (positive {int(y_true.sum())} / negative {int((1 - y_true).sum())})")
    text_lines.append(f"Youden threshold = {C.fmt_num(best_threshold)}")
    ax.text(0.98, 0.04, "\n".join(text_lines), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=max(6.0, float(style.get("text_size", 10)) - 1), color="#1A1A1A",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#CCCCCC", "alpha": 0.9, "linewidth": 0.6})
    _apply_labels(ax, style, x="False positive rate (1 - specificity)",
                  y="True positive rate (sensitivity)")
    title = f"ROC curve: {str(labels_cfg.get('y') or 'score')} (AUC = {auc_value:.3f})"

    selected = np.linspace(0, len(thresholds) - 1, min(8, len(thresholds))).astype(int)
    threshold_rows = []
    for index in selected:
        threshold = thresholds[index]
        threshold_rows.append({
            "threshold": (float(threshold) if np.isfinite(threshold) else None),
            "sensitivity": float(tpr[index]),
            "specificity": float(1.0 - fpr[index]),
            "youden": float(youden[index]),
        })
    table = C.table(
        "roc_thresholds", "ROC 阈值（按曲线采样）",
        [{"key": "threshold", "label": "阈值", "type": "num"},
         {"key": "sensitivity", "label": "敏感度", "type": "num"},
         {"key": "specificity", "label": "特异度", "type": "num"},
         {"key": "youden", "label": "Youden 指数", "type": "num"}],
        threshold_rows)
    bullets = [
        f"AUC = {auc_value:.3f}" + (f"（95% CI {ci[0]:.3f}–{ci[1]:.3f}）" if ci else "") +
        f"，样本 {len(work)} 例（阳性 {int(y_true.sum())} / 阴性 {int((1 - y_true).sum())}）",
        f"阳性水平取 {positive}；最佳阈值 {C.fmt_num(best_threshold)} 时敏感度 "
        f"{sensitivity:.1%}、特异度 {specificity:.1%}",
        f"判别能力" + ("优秀（AUC ≥ 0.9）" if auc_value >= 0.9 else
                     "良好（AUC ≥ 0.8）" if auc_value >= 0.8 else
                     "一般（AUC ≥ 0.7）" if auc_value >= 0.7 else "较弱（AUC < 0.7）"),
    ]
    if flipped:
        bullets.append("原始分数方向与阳性相反，已自动翻转后计算")
    stats = {
        "method": "ROC (DeLong-free bootstrap CI)",
        "auc": auc_value,
        "ci": ci,
        "n": int(len(work)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "positive_level": positive,
        "flipped": flipped,
        "best_threshold": best_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden": float(youden[best_index]),
    }
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(work)), "auc": auc_value,
                        "best_threshold": best_threshold,
                        "sensitivity": sensitivity, "specificity": specificity},
        "tables": [table],
        "warnings": warnings,
        "stats": stats,
        "annotations": {"auc": auc_value, "threshold": best_threshold},
    }


# ---------------------------------------------------------------------------
# 20) km —— Kaplan-Meier 生存曲线
# ---------------------------------------------------------------------------


def _render_km(data: Mapping[str, Any], params: Mapping[str, Any],
               style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    show_ci = _pbool(params, "ci", True)
    show_censors = _pbool(params, "censor", True)
    risk_table = _pbool(params, "risk_table", False)
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test, multivariate_logrank_test
    except Exception as exc:  # noqa: BLE001
        raise C.EngineError("prerequisite", f"缺少 lifelines 依赖，无法绘制生存曲线：{exc}") from exc

    frame = P.require_tidy(data, ["__time", "__event"]).copy()
    time_values = pd.to_numeric(frame["__time"], errors="coerce").to_numpy(dtype=float)
    event_values = pd.to_numeric(frame["__event"], errors="coerce").to_numpy(dtype=float)
    group_col = None
    for key in ("__group", "__color"):
        if key in frame.columns and frame[key].notna().any():
            group_col = key
            break
    work = pd.DataFrame({"time": time_values, "event": event_values})
    if group_col is not None:
        work["group"] = [_fmt_level(value) for value in frame[group_col].tolist()]
    work = work[np.isfinite(work["time"]) & np.isfinite(work["event"])]
    work = work[work["time"] >= 0]
    work["event"] = (work["event"] > 0).astype(int)
    if len(work) < 5:
        raise C.prerequisite("生存数据有效行数不足（< 5），无法绘制 KM 曲线")
    if int(work["event"].sum()) == 0:
        raise C.prerequisite("没有任何事件发生（event 全为 0），无法绘制 KM 曲线")
    labels = data.get("labels") or {}
    time_label = str(labels.get("time") or "Time")
    group_names = list(dict.fromkeys(work["group"].tolist())) if group_col is not None else []
    groups: list[tuple[str, pd.DataFrame]] = (
        [(name, work[work["group"] == name]) for name in group_names] if group_names else [("All", work)]
    )
    colors = _colors(style, group_names) if group_names else {}
    line_width = max(1.0, float(style.get("line_width", 1.4) or 1.4))
    tick_size = float(style.get("tick_size", 10))
    label_size = float(style.get("label_size", 12))

    aux_axes: list = []
    if risk_table:
        fig, (ax, ax_risk) = plt.subplots(
            2, 1, figsize=(float(style.get("fig_width", 6.2)), float(style.get("fig_height", 4.4)) * 1.32),
            sharex=True, layout="constrained", gridspec_kw={"height_ratios": [3.4, 1.0]},
        )
        aux_axes.append(ax_risk)
    else:
        fig, ax = C.new_fig(style)
        ax_risk = None

    curves: list[dict] = []
    for index, (name, cell) in enumerate(groups):
        color = (colors.get(name, C.color_list(style, len(groups))[index])
                 if group_names else C.color_list(style, 1)[0])
        label = f"{name} (n={len(cell)})"
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(cell["time"].to_numpy(dtype=float), cell["event"].to_numpy(dtype=int), label=label)
            kmf.plot_survival_function(
                ax=ax, ci_show=bool(show_ci), show_censors=bool(show_censors),
                censor_styles={"marker": "+", "ms": 4.5, "mew": 0.9},
                color=color, linewidth=line_width,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"分组 {name} 的 KM 拟合失败：{exc}")
            continue
        median = getattr(kmf, "median_survival_time_", None)
        try:
            median_value = float(median) if median is not None and np.isfinite(float(median)) else None
        except Exception:  # noqa: BLE001
            median_value = None
        curves.append({"group": name, "n": int(len(cell)),
                       "events": int(cell["event"].sum()), "median": median_value})
    if not curves:
        raise C.prerequisite("所有分组的 KM 拟合都失败了，无法绘制生存曲线")

    logrank_p = None
    if len(groups) >= 2:
        try:
            if len(groups) == 2:
                first, second = groups[0][1], groups[1][1]
                result = logrank_test(first["time"].to_numpy(dtype=float),
                                      second["time"].to_numpy(dtype=float),
                                      first["event"].to_numpy(dtype=int),
                                      second["event"].to_numpy(dtype=int))
            else:
                result = multivariate_logrank_test(
                    work["time"].to_numpy(dtype=float),
                    work["group"].to_numpy(),
                    work["event"].to_numpy(dtype=int),
                )
            logrank_p = float(getattr(result, "p_value", float("nan")))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"log-rank 检验失败：{exc}")
            logrank_p = None
        if logrank_p is not None and math.isfinite(logrank_p):
            p_text = C.fmt_p(logrank_p)
            ax.text(0.02, 0.04, f"Log-rank p = {p_text}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=max(6.0, float(style.get("text_size", 10)) - 1),
                    color="#1A1A1A",
                    bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                          "edgecolor": "#CCCCCC", "alpha": 0.85, "linewidth": 0.6})
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0.0)
    _apply_labels(ax, style, x=time_label, y="Survival probability")
    title = "Kaplan-Meier survival curves"

    if risk_table and ax_risk is not None:
        times = np.linspace(0.0, float(work["time"].max()), 5)
        y_positions = [len(groups) - index for index in range(len(groups))]
        for (name, cell), y_position in zip(groups, y_positions):
            for time_point in times:
                count = int((cell["time"] >= time_point).sum())
                ax_risk.text(time_point, y_position, str(count), ha="center", va="center",
                             fontsize=max(5.5, tick_size - 2))
        ax_risk.set_yticks(y_positions)
        ax_risk.set_yticklabels([name for name, _cell in groups],
                                fontsize=max(5.5, tick_size - 2))
        ax_risk.set_ylim(0.4, len(groups) + 0.6)
        ax_risk.set_xlabel(time_label, fontsize=label_size)
        ax_risk.set_ylabel("At risk", fontsize=max(6.0, label_size - 2))
        ax_risk.tick_params(axis="x", labelsize=max(5.5, tick_size - 2))
        ax_risk.tick_params(axis="y", length=0)

    table_rows = [{"group": item["group"], "n": item["n"], "events": item["events"],
                   "median": item["median"]} for item in curves]
    table = C.table(
        "km_summary", "分组生存汇总",
        [{"key": "group", "label": "分组", "type": "str"},
         {"key": "n", "label": "n", "type": "num"},
         {"key": "events", "label": "事件数", "type": "num"},
         {"key": "median", "label": "中位生存", "type": "num"}],
        table_rows)
    bullets = []
    for item in curves[:5]:
        median_text = (C.fmt_num(item["median"]) if item["median"] is not None else "未达到")
        bullets.append(f"{item['group']}：n = {item['n']}，事件 {item['events']} 例，"
                       f"中位生存 {median_text}")
    if logrank_p is not None and math.isfinite(logrank_p):
        significant = logrank_p < 0.05
        bullets.append(f"log-rank 检验 p = {C.fmt_p(logrank_p)}，"
                       f"{'各组生存曲线差异有统计学意义' if significant else '各组生存曲线差异无统计学意义'}")
    elif len(groups) == 1:
        bullets.append("只有 1 个分组，未做 log-rank 检验（可映射 group 角色以比较生存）")
    stats = {
        "method": "Kaplan-Meier",
        "logrank_p": (float(logrank_p) if logrank_p is not None and math.isfinite(logrank_p) else None),
        "groups": [dict(item) for item in curves],
    }
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n": int(len(work)), "events": int(work["event"].sum()),
                        "logrank_p": stats["logrank_p"]},
        "tables": [table],
        "warnings": warnings,
        "stats": stats,
        "annotations": ({"logrank_p": stats["logrank_p"]}
                        if stats["logrank_p"] is not None else None),
        "aux": aux_axes,
    }


# ---------------------------------------------------------------------------
# 21) forest —— 森林图
# ---------------------------------------------------------------------------


_FOREST_ESTIMATE_KEYS = ("estimate", "hr", "or", "beta", "est", "effect", "value", "exp_coef")
_FOREST_LOWER_KEYS = ("lower", "lo", "lcl", "ci_lower", "ci_lo", "lower_95", "ci95_lower")
_FOREST_UPPER_KEYS = ("upper", "hi", "ucl", "ci_upper", "ci_hi", "upper_95", "ci95_upper")
_FOREST_P_KEYS = ("p", "p_value", "pvalue")


def _first_number(item: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in item and item[key] is not None:
            try:
                number = float(item[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _parse_estimates(raw: Any) -> list[dict]:
    if raw is None:
        return []
    payload: Any = None
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], Mapping):
        payload = list(raw)
    else:
        text = _ptext({"v": raw}, "v", "").strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001
            payload = None
        if payload is None:
            rows: list[dict] = []
            for line in text.replace("；", ";").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.replace(";", ",").split(",")]
                if len(parts) < 3:
                    continue
                try:
                    numbers = [float(part) for part in parts[1:]]
                except ValueError:
                    continue
                label = parts[0]
                if len(numbers) >= 3:
                    rows.append({"label": label, "estimate": numbers[0],
                                 "lower": numbers[1], "upper": numbers[2]})
                elif len(numbers) == 2:
                    rows.append({"label": label, "estimate": numbers[0], "lower": numbers[0],
                                 "upper": numbers[1]})
            return rows
    if isinstance(payload, Mapping):
        payload = [payload]
    rows = []
    for item in payload or []:
        if not isinstance(item, Mapping):
            continue
        estimate = _first_number(item, _FOREST_ESTIMATE_KEYS)
        lower = _first_number(item, _FOREST_LOWER_KEYS)
        upper = _first_number(item, _FOREST_UPPER_KEYS)
        if estimate is None or lower is None or upper is None:
            continue
        label = (item.get("label") or item.get("term") or item.get("name")
                 or item.get("variable") or item.get("feature") or "term")
        rows.append({"label": str(label), "estimate": float(estimate),
                     "lower": float(lower), "upper": float(upper),
                     "p": _first_number(item, _FOREST_P_KEYS)})
    return rows


def _forest_rows_from_cox(data: Mapping[str, Any], params: Mapping[str, Any],
                          warnings: list[str]) -> list[dict]:
    frame = data.get("tidy")
    wide = data.get("wide")
    if not isinstance(frame, pd.DataFrame) or not isinstance(wide, Mapping):
        return []
    if "__time" not in frame.columns or "__event" not in frame.columns:
        return []
    matrix = wide.get("matrix")
    features = [str(name) for name in (wide.get("features") or [])]
    if matrix is None or not features:
        return []
    matrix = np.asarray(matrix, dtype=float)
    time_values = pd.to_numeric(frame["__time"], errors="coerce").to_numpy(dtype=float)
    event_values = pd.to_numeric(frame["__event"], errors="coerce").to_numpy(dtype=float)
    if matrix.shape[0] != len(time_values) or matrix.shape[1] != len(features):
        return []
    try:
        from lifelines import CoxPHFitter
    except Exception as exc:  # noqa: BLE001
        raise C.EngineError("prerequisite", f"缺少 lifelines 依赖，无法拟合 Cox：{exc}") from exc
    labels = data.get("labels") or {}
    exclude = {str(labels.get("time") or ""), str(labels.get("event") or "")}
    terms = _split_terms(params.get("terms") or "")
    rows: list[dict] = []
    for index, feature in enumerate(features):
        if feature in exclude:
            continue
        if terms and feature not in terms:
            continue
        values = matrix[:, index]
        mask = np.isfinite(values) & np.isfinite(time_values) & np.isfinite(event_values)
        if int(mask.sum()) < 12:
            continue
        selected = values[mask]
        sd = float(np.std(selected, ddof=1))
        if sd <= 1e-12:
            continue
        standardized = (selected - float(np.mean(selected))) / sd
        work = pd.DataFrame({"T": time_values[mask], "E": (event_values[mask] > 0).astype(int),
                             "z": standardized})
        if work["E"].nunique() < 2:
            continue
        try:
            fitter = CoxPHFitter()
            fitter.fit(work, duration_col="T", event_col="E")
            summary = fitter.summary.loc["z"]
            rows.append({
                "label": feature,
                "estimate": float(summary["exp(coef)"]),
                "lower": float(summary["exp(coef) lower 95%"]),
                "upper": float(summary["exp(coef) upper 95%"]),
                "p": float(summary["p"]),
                "n": int(mask.sum()),
            })
        except Exception:  # noqa: BLE001
            warnings.append(f"变量 {feature} 的 Cox 拟合失败，已跳过")
    if len(rows) > 10:
        rows.sort(key=lambda row: -abs(math.log(max(row["estimate"], 1e-12))))
        rows = rows[:10]
        warnings.append("单因素 Cox 变量超过 10 个，仅展示效应最强的前 10 个（可在 params.terms 指定变量）")
    return rows


def _render_forest(data: Mapping[str, Any], params: Mapping[str, Any],
                   style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    ref_line = _pfloat(params, "ref_line", 1.0, -1000.0, 1000.0)
    log_scale = _pbool(params, "log_scale", True)
    show_text = _pbool(params, "show_text", True)
    model = _pchoice(params, "model", "auto", ["auto", "cox"])
    rows = _parse_estimates(params.get("estimates"))
    source = "user"
    if rows:
        warnings.append("使用 params.estimates 传入的效应量（不重新拟合模型）")
    else:
        rows = _forest_rows_from_cox(data, params, warnings)
        source = "cox"
    if not rows:
        raise C.prerequisite(
            "森林图没有可用的效应量：请在参数 estimates 里传入 JSON 列表"
            "（如 [{\"label\":\"age\",\"estimate\":1.42,\"lower\":1.10,\"upper\":1.83,\"p\":0.01}]），"
            "或映射 time/event 角色后用数值列做单因素 Cox"
        )
    for row in rows:
        if row["lower"] > row["upper"]:
            row["lower"], row["upper"] = row["upper"], row["lower"]
    finite = [row for row in rows if row["lower"] > 0 and row["estimate"] > 0 and row["upper"] > 0]
    use_log = bool(log_scale and len(finite) == len(rows) and ref_line > 0)
    color_ns = "#4E79A7"
    color_sig = "#B24745"
    marker_size = max(5.0, math.sqrt(float(style.get("point_size", 28) or 28)) * 1.1)
    line_width = max(1.0, float(style.get("line_width", 1.4) or 1.4))
    tick_size = float(style.get("tick_size", 10))
    label_size = float(style.get("label_size", 12))

    fig, ax = C.new_fig(style)
    y_positions = np.arange(len(rows), 0, -1)
    for row, y_position in zip(rows, y_positions):
        significant = row.get("p") is not None and math.isfinite(row["p"]) and row["p"] < 0.05
        color = color_sig if significant else color_ns
        ax.plot([row["lower"], row["upper"]], [y_position, y_position], color="#4D4D4D",
                linewidth=line_width, solid_capstyle="butt", zorder=3)
        ax.plot([row["estimate"]], [y_position], marker="s", markersize=marker_size,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        if show_text:
            label = (f"{row['estimate']:.2f} ({row['lower']:.2f}–{row['upper']:.2f})")
            if row.get("p") is not None and math.isfinite(row["p"]):
                stars = C.p_stars(row["p"])
                if stars and stars != "ns":
                    label += f" {stars}"
            if use_log:
                text_x = row["upper"] * 1.12
            else:
                text_x = row["upper"] + max(abs(row["upper"]) * 0.05, 0.05)
            ax.text(text_x, y_position, label, ha="left", va="center",
                    fontsize=max(5.5, tick_size - 2), color="#1A1A1A", clip_on=False)
    ax.axvline(ref_line, color="#6A6A6A", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=max(5.5, tick_size - 1))
    ax.set_ylim(0.4, len(rows) + 0.6)
    if use_log:
        ax.set_xscale("log")
    _legend_marker(ax, "p < 0.05", color_sig, "s", marker_size)
    _legend_marker(ax, "p ≥ 0.05", color_ns, "s", marker_size)
    x_label = "Hazard ratio (per SD)" if source == "cox" else "Effect estimate (HR / OR / β)"
    _apply_labels(ax, style, x=x_label, y="")
    title = ("Forest plot: hazard ratios (univariate Cox)" if source == "cox"
             else "Forest plot: effect estimates")
    if show_text:
        try:
            low = min(row["lower"] for row in rows)
            high = max(row["upper"] for row in rows)
            if use_log:
                ax.set_xlim(max(low * 0.7, 1e-6), high * 1.9)
            else:
                span = high - low if high > low else 1.0
                ax.set_xlim(low - span * 0.05, high + span * 0.75)
        except Exception:  # noqa: BLE001
            pass

    significant_rows = [row for row in rows
                        if row.get("p") is not None and math.isfinite(row["p"]) and row["p"] < 0.05]
    table = C.table(
        "forest_estimates", "效应量与置信区间",
        [{"key": "label", "label": "变量", "type": "str"},
         {"key": "estimate", "label": "效应量", "type": "num"},
         {"key": "lower", "label": "下限", "type": "num"},
         {"key": "upper", "label": "上限", "type": "num"},
         {"key": "p", "label": "p", "type": "p"},
         {"key": "sig", "label": "显著性", "type": "str"}],
        [{**row, "sig": (C.p_stars(row["p"]) if row.get("p") is not None else "")}
         for row in rows])
    bullets = [f"共 {len(rows)} 个变量；{len(significant_rows)} 个达到 p<0.05"]
    if rows:
        strongest = max(rows, key=lambda row: abs(math.log(max(row["estimate"], 1e-12))))
        bullets.append(
            f"效应最强的是 {strongest['label']}：{strongest['estimate']:.2f} "
            f"（95% CI {strongest['lower']:.2f}–{strongest['upper']:.2f}）"
        )
    if source == "cox":
        bullets.append("模型：单因素 Cox，效应量为标准化后每 1 SD 的 HR")
    stats = {
        "method": "Cox regression (univariate, per SD)" if source == "cox" else "User-provided estimates",
        "reference": ref_line,
        "log_scale": use_log,
        "estimates": [{"term": row["label"], "estimate": row["estimate"],
                       "lower": row["lower"], "upper": row["upper"], "p": row.get("p")}
                      for row in rows],
    }
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n_terms": len(rows), "n_significant": len(significant_rows),
                        "reference": ref_line},
        "tables": [table],
        "warnings": warnings,
        "stats": stats,
        "annotations": {"reference": ref_line},
    }


# ---------------------------------------------------------------------------
# 22) pca —— 主成分分析
# ---------------------------------------------------------------------------


def _render_pca(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    do_scale = _pbool(params, "scale", True)
    show_ellipse = _pbool(params, "ellipse", True)
    show_loadings = _pbool(params, "loadings", False)
    arr, features, samples, wide = _wide_matrix(data, "PCA")
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # noqa: BLE001
        raise C.EngineError("prerequisite", f"缺少 scikit-learn 依赖，无法做 PCA：{exc}") from exc
    matrix = np.asarray(arr, dtype=float).copy()
    column_mean = np.nanmean(matrix, axis=0)
    column_mean = np.where(np.isfinite(column_mean), column_mean, 0.0)
    invalid = ~np.isfinite(matrix)
    if invalid.any():
        matrix[invalid] = np.take(column_mean, np.where(invalid)[1])
        warnings.append(f"有 {int(invalid.sum())} 个缺失值，已用该特征均值填补")
    if do_scale:
        matrix = StandardScaler().fit_transform(matrix)
    n_components = int(min(matrix.shape[0], matrix.shape[1], 10))
    if n_components < 2:
        raise C.prerequisite("样本或特征太少，无法做主成分分析（至少需要 2×2）")
    try:
        model = PCA(n_components=n_components, random_state=0)
        scores = model.fit_transform(matrix)
    except Exception as exc:  # noqa: BLE001
        raise C.EngineError("internal", f"PCA 拟合失败：{exc}") from exc
    ratio = np.asarray(model.explained_variance_ratio_, dtype=float) * 100.0
    cumulative = np.cumsum(ratio)
    group_labels, group_name = _pair_colors(data, wide, arr.shape[0])
    group_levels = list(dict.fromkeys(group_labels)) if group_labels else []
    group_colors = _colors(style, group_levels)
    alpha = float(style.get("alpha") or 0.85)
    point_size = float(style.get("point_size", 28) or 28)
    line_width = max(0.8, float(style.get("line_width", 1.4) or 1.4))
    tick_size = float(style.get("tick_size", 10))
    label_size = float(style.get("label_size", 12))

    fig, (ax, ax_scree) = plt.subplots(
        1, 2, figsize=(float(style.get("fig_width", 6.2)) * 1.62,
                       float(style.get("fig_height", 4.4))),
        layout="constrained", gridspec_kw={"width_ratios": [2.45, 1.0]},
    )
    if group_labels is not None:
        for level in group_levels:
            indexes = [index for index, label in enumerate(group_labels) if label == level]
            if not indexes:
                continue
            color = group_colors.get(level, C.color_list(style, len(group_levels))[0])
            ax.scatter(scores[indexes, 0], scores[indexes, 1], s=point_size, color=color,
                       alpha=alpha, edgecolors="white", linewidths=0.5, label=level, zorder=4)
            _legend_marker(ax, level, color, "o", math.sqrt(point_size), alpha)
            if show_ellipse and len(indexes) >= 3:
                points = scores[indexes, :2]
                try:
                    covariance = np.cov(points.T)
                    if covariance.shape == (2, 2) and np.all(np.isfinite(covariance)):
                        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                        order = np.argsort(eigenvalues)[::-1]
                        eigenvalues = eigenvalues[order]
                        eigenvectors = eigenvectors[:, order]
                        chi2 = float(sps.chi2.ppf(0.95, df=2))
                        width = 2.0 * math.sqrt(max(eigenvalues[0], 0.0) * chi2)
                        height = 2.0 * math.sqrt(max(eigenvalues[1], 0.0) * chi2)
                        angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                        ax.add_patch(Ellipse(xy=(float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))),
                                             width=width, height=height, angle=angle,
                                             facecolor=color, edgecolor=color,
                                             alpha=0.12, linewidth=1.0, zorder=2))
                except Exception:  # noqa: BLE001
                    warnings.append(f"分组 {level} 的置信椭圆计算失败，已跳过该椭圆")
    else:
        color = C.color_list(style, 1)[0]
        ax.scatter(scores[:, 0], scores[:, 1], s=point_size, color=color, alpha=alpha,
                   edgecolors="white", linewidths=0.5, zorder=4)
    if show_loadings:
        loadings = model.components_[:, :2]
        norms = np.linalg.norm(loadings, axis=1)
        top = np.argsort(-norms)[: min(8, len(features))]
        max_score = float(np.max(np.abs(scores[:, :2])))
        max_loading = float(np.max(norms[top])) if len(top) else 0.0
        scale_factor = (0.85 * max_score / max_loading) if max_loading > 1e-12 else 1.0
        for index in top:
            dx = float(loadings[index, 0]) * scale_factor
            dy = float(loadings[index, 1]) * scale_factor
            ax.annotate("", xy=(dx, dy), xytext=(0.0, 0.0),
                        arrowprops={"arrowstyle": "->", "color": "#555555",
                                    "linewidth": max(0.6, line_width * 0.6)},
                        zorder=5)
            ax.text(dx * 1.06, dy * 1.06, features[int(index)],
                    fontsize=max(5.0, tick_size - 3), color="#333333", zorder=6)
        warnings.append("载荷箭头已按得分范围缩放，方向表示变量在主成分上的贡献")
    ax.set_xlabel(f"PC1 ({ratio[0]:.1f}% variance)", fontsize=label_size)
    ax.set_ylabel(f"PC2 ({ratio[1]:.1f}% variance)", fontsize=label_size)
    ax.axhline(0.0, color="#D0D0D0", linewidth=0.7, zorder=1)
    ax.axvline(0.0, color="#D0D0D0", linewidth=0.7, zorder=1)
    ax.tick_params(labelsize=tick_size)

    positions = np.arange(1, len(ratio) + 1)
    bar_colors = C.color_list(style, len(ratio))
    ax_scree.bar(positions, ratio, color=bar_colors, alpha=alpha, edgecolor="white",
                 linewidth=0.5)
    ax_scree.plot(positions, cumulative, color="#B24745", marker="o", markersize=3.2,
                  linewidth=line_width, zorder=4)
    ax_scree.set_xticks(positions)
    ax_scree.set_xticklabels([str(int(p)) for p in positions], fontsize=max(5.5, tick_size - 2))
    ax_scree.set_xlabel("Principal component", fontsize=max(7.0, label_size - 2))
    ax_scree.set_ylabel("Explained variance (%)", fontsize=max(7.0, label_size - 2))
    ax_scree.tick_params(labelsize=max(5.5, tick_size - 2))
    ax_scree.set_ylim(0.0, 105.0)
    ax_scree.text(0.97, 0.06, f"cumulative PC1–2 = {cumulative[min(1, len(cumulative) - 1)]:.1f}%",
                  transform=ax_scree.transAxes, ha="right", va="bottom",
                  fontsize=max(5.5, tick_size - 2), color="#1A1A1A")
    title = "PCA: samples on PC1-PC2 with scree plot"

    variance_rows = [
        {"pc": f"PC{index + 1}", "eigenvalue": float(model.explained_variance_[index]),
         "variance_pct": float(ratio[index]), "cumulative_pct": float(cumulative[index])}
        for index in range(len(ratio))
    ]
    variance_table = C.table(
        "pca_variance", "主成分解释方差",
        [{"key": "pc", "label": "主成分", "type": "str"},
         {"key": "eigenvalue", "label": "特征值", "type": "num"},
         {"key": "variance_pct", "label": "解释方差 %", "type": "num"},
         {"key": "cumulative_pct", "label": "累计 %", "type": "num"}],
        variance_rows)
    loading_rows = []
    loading_matrix = np.asarray(model.components_, dtype=float)
    for index, feature in enumerate(features):
        if index >= loading_matrix.shape[1]:
            break
        loading_rows.append({"feature": feature,
                             "PC1": float(loading_matrix[0, index]),
                             "PC2": float(loading_matrix[1, index])})
    loading_rows.sort(key=lambda row: -(abs(row["PC1"]) + abs(row["PC2"])))
    loading_table = C.table(
        "pca_loadings", "主成分载荷（前 15 个变量）",
        [{"key": "feature", "label": "变量", "type": "str"},
         {"key": "PC1", "label": "PC1 载荷", "type": "num"},
         {"key": "PC2", "label": "PC2 载荷", "type": "num"}],
        loading_rows[:15])
    kruskal_p = None
    if group_labels is not None and len(group_levels) >= 2:
        grouped_scores = [scores[[index for index, label in enumerate(group_labels) if label == level], 0]
                          for level in group_levels]
        grouped_scores = [values for values in grouped_scores if values.size >= 3]
        if len(grouped_scores) >= 2:
            try:
                kruskal_p = float(sps.kruskal(*grouped_scores).pvalue)
            except Exception:  # noqa: BLE001
                kruskal_p = None
    bullets = [
        f"PCA 基于 {arr.shape[0]} 个样本 × {arr.shape[1]} 个特征"
        + ("（变量已标准化）" if do_scale else "（未标准化）"),
        f"PC1 解释 {ratio[0]:.1f}%，PC2 解释 {ratio[min(1, len(ratio) - 1)]:.1f}%，"
        f"前两个主成分累计 {cumulative[min(1, len(cumulative) - 1)]:.1f}%",
    ]
    if group_labels is not None:
        bullets.append(f"按 {group_name} 着色（{len(group_levels)} 组）")
        if kruskal_p is not None and math.isfinite(kruskal_p):
            bullets.append(f"PC1 得分组间差异（Kruskal-Wallis）p = {C.fmt_p(kruskal_p)}，"
                           f"{'分组在主成分上可区分' if kruskal_p < 0.05 else '分组在主成分上无明显差异'}")
    if loading_rows:
        bullets.append(f"PC1 载荷最大的变量：{loading_rows[0]['feature']}、"
                       f"{loading_rows[1]['feature'] if len(loading_rows) > 1 else '—'}")
    stats = {
        "method": "PCA",
        "scale": do_scale,
        "n_samples": int(arr.shape[0]),
        "n_features": int(arr.shape[1]),
        "explained_variance": {f"PC{index + 1}": float(value) for index, value in enumerate(ratio)},
        "pc1_variance": float(ratio[0]),
        "pc2_variance": float(ratio[min(1, len(ratio) - 1)]),
        "kruskal_p_pc1": kruskal_p,
    }
    return {
        "fig": fig,
        "title": title,
        "bullets": bullets,
        "key_numbers": {"n_samples": int(arr.shape[0]), "n_features": int(arr.shape[1]),
                        "pc1_variance_pct": float(ratio[0]),
                        "pc2_variance_pct": float(ratio[min(1, len(ratio) - 1)])},
        "tables": [variance_table, loading_table],
        "warnings": warnings,
        "stats": stats,
        "annotations": {"ellipse_confidence": 0.95} if show_ellipse else None,
    }


# ---------------------------------------------------------------------------
# 渲染调度
# ---------------------------------------------------------------------------

_RENDERERS = {
    "bar": _render_bar,
    "bar_dot": _render_bar_dot,
    "box": _render_box,
    "violin": _render_violin,
    "raincloud": _render_raincloud,
    "dotplot": _render_dotplot,
    "stack": _render_stack,
    "pie": _render_pie,
    "scatter": _render_scatter,
    "bubble": _render_bubble,
    "line": _render_line,
    "area": _render_area,
    "heatmap": _render_heatmap,
    "corr": _render_corr,
    "pairplot": _render_pairplot,
    "hist": _render_hist,
    "density": _render_density,
    "qq": _render_qq,
    "roc": _render_roc,
    "km": _render_km,
    "forest": _render_forest,
    "pca": _render_pca,
}


def _declared_independent(params: Mapping[str, Any]) -> bool:
    value = str(params.get("independent_samples") or "").strip().lower()
    return value in ("sample", "independent", "1", "true", "yes")


def render(chart_id: str, data: Mapping[str, Any], params: Mapping[str, Any],
           style: Mapping[str, Any], outdir: str) -> dict:
    """渲染单个图表并返回 DESIGN §5 的结果 dict。"""
    chart = str(chart_id or "").strip()
    meta = CHART_INDEX.get(chart)
    if meta is None:
        raise C.not_found(f"未知图表 {chart_id}；可用图表：{', '.join(CHART_INDEX)}")
    renderer = _RENDERERS.get(chart)
    if renderer is None:
        raise C.unsupported(f"图表 {chart} 暂不支持渲染")
    params_dict = _params_dict(params)
    resolved_style = _resolve_style(style)
    data = data if isinstance(data, Mapping) else {}
    outdir = str(outdir or ".")
    data_warnings = [str(note) for note in (data.get("notes") or [])]
    if data.get("melted") and not _declared_independent(params_dict):
        # 矩阵融化视图：同一生物样本内的多个特征值是技术重复，汇总图默认只做描述性
        data_warnings.append(
            "矩阵数据已融化为长表：本次汇总图只做描述性展示（描述性/技术重复），"
            "同一生物样本内的多个特征值属于技术重复，不能当作独立生物学重复做显著性检验；"
            "如需正式检验，请显式声明独立样本（params.independent_samples=\"sample\"）"
        )
        resolved_style = dict(resolved_style)
        resolved_style["stat_annot"] = False
    try:
        with C.style_context(resolved_style):
            out = renderer(data, params_dict, resolved_style)
            if not isinstance(out, Mapping) or out.get("fig") is None:
                raise C.EngineError("internal", f"图表 {chart} 未返回图形对象")
            fig = out["fig"]
            title = str(out.get("title") or meta["title"])
            aux_axes = list(out.get("aux") or [])
            aux_state = []
            for ax in aux_axes:
                try:
                    aux_state.append((ax, ax.get_xlim(), ax.get_ylim(),
                                      ax.get_xscale(), ax.get_yscale()))
                except Exception:  # noqa: BLE001
                    continue
            C.finalize_figure(fig, resolved_style, title=title)
            for ax, xlim, ylim, xscale, yscale in aux_state:
                _restore_aux(ax, resolved_style, xlim, ylim, xscale, yscale)
            figure = C.save_fig(fig, outdir, chart, title=title,
                                dpi=int(resolved_style.get("dpi") or 300))
    except C.EngineError:
        raise
    except Exception as exc:  # noqa: BLE001 - 兜底成引擎错误，避免 worker 崩溃
        import traceback

        trace = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise C.EngineError("internal", f"图表 {chart} 渲染失败：{trace}", chart=chart) from exc

    bullets = [str(item) for item in (out.get("bullets") or []) if str(item).strip()]
    if not bullets:
        bullets = [f"{meta['title']}已生成（{figure.get('width')}×{figure.get('height')} px）"]
    warnings: list[str] = []
    for warning in data_warnings + [str(item) for item in (out.get("warnings") or [])]:
        if warning and warning not in warnings:
            warnings.append(warning)
    stats = out.get("stats")
    annotations = out.get("annotations")
    tables = [table for table in (out.get("tables") or []) if table]
    return {
        "figures": [figure],
        "summary": {
            "bullets": bullets,
            "key_numbers": C.jsonable(out.get("key_numbers") or {}),
        },
        "tables": tables,
        "warnings": warnings,
        "stats": C.jsonable(stats) if stats else None,
        "annotations": C.jsonable(annotations) if annotations else None,
    }


# ===========================================================================
# 代码生成：可独立运行的 Python 片段
# ===========================================================================

_PY_FINISH = '''GRID = "{grid}"
FRAME = "{frame}"
TICK_DIR = "{tick_dir}"
X_ROTATE = {x_rotate}
X_LOG = {x_log}
Y_LOG = {y_log}
LEGEND_POS = "{legend_pos}"


def _frame_visible(side):
    if FRAME == "full":
        return True
    if FRAME == "none":
        return False
    return side in ("left", "bottom")


def _finish(ax, legend=True):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(_frame_visible(side))
    ax.tick_params(axis="both", which="major", direction=TICK_DIR, length=3.5, labelsize=TICK_SIZE)
    if GRID in ("both", "x"):
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#B0B0B0")
        ax.set_axisbelow(True)
    if GRID in ("both", "y"):
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#B0B0B0")
        ax.set_axisbelow(True)
    if X_ROTATE:
        plt.setp(ax.get_xticklabels(), rotation=X_ROTATE,
                 ha="right" if X_ROTATE >= 30 else "center")
    if X_LOG:
        ax.set_xscale("log")
    if Y_LOG:
        ax.set_yscale("log")
    ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_SIZE)
    ax.set_title(TITLE, fontsize=TITLE_SIZE, fontweight="bold" if TITLE_BOLD else "normal")
    if legend and LEGEND_POS != "none":
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if LEGEND_POS == "right":
                ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
                          fontsize=LEGEND_SIZE, frameon=False)
            elif LEGEND_POS == "top":
                ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                          ncol=min(4, len(labels)), fontsize=LEGEND_SIZE, frameon=False)
            else:
                ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                          ncol=min(4, len(labels)), fontsize=LEGEND_SIZE, frameon=False)
    plt.tight_layout()
'''

_PY_HELPERS = {
    "summary": '''def _summary(values, how="mean", error="se"):
    """聚合值 + 误差线（与工作台一致：mean/median/sum × sd/se/ci95/none）。"""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = len(values)
    if n == 0:
        return np.nan, 0.0, 0
    if how == "median":
        center = float(values.median())
    elif how == "sum":
        center = float(values.sum())
    else:
        center = float(values.mean())
    err = 0.0
    if error != "none" and n > 1:
        sd = float(values.std(ddof=1))
        if error == "sd":
            err = sd
        elif error == "se":
            err = sd / np.sqrt(n)
        else:
            err = 1.96 * sd / np.sqrt(n)
    return center, err, n
''',
    "kde": '''def _kde(values, grid, bw=1.0):
    """分组核密度，返回归一化到峰值 1 的密度曲线；数据太少/方差为 0 返回 None。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or float(np.std(values)) <= 1e-12:
        return None
    try:
        kde = stats.gaussian_kde(values, bw_method="scott")
        if abs(float(bw) - 1.0) > 1e-9:
            kde.set_bandwidth(float(kde.factor) * float(bw))
        density = np.asarray(kde(grid), dtype=float)
    except Exception:
        return None
    peak = float(np.max(density))
    if not np.isfinite(peak) or peak <= 0:
        return None
    return density / peak
''',
    "box": '''def _box(values):
    """箱线统计量（Tukey 1.5 IQR 须线）。"""
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    q1, med, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    low = values[values >= q1 - 1.5 * iqr]
    high = values[values <= q3 + 1.5 * iqr]
    return {"med": med, "q1": q1, "q3": q3,
            "whislo": float(low.min()) if len(low) else float(values.min()),
            "whishi": float(high.max()) if len(high) else float(values.max()),
            "fliers": list(values[(values < (low.min() if len(low) else values.min())) |
                                  (values > (high.max() if len(high) else values.max()))]),
            "n": int(len(values))}
''',
    "fit": '''def _fit_line(x, y):
    """最小二乘直线 + 均值响应的 95% 置信带。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    grid = np.linspace(float(x.min()), float(x.max()), 120)
    y_hat = slope * grid + intercept
    resid = y - (slope * x + intercept)
    dof = len(x) - 2
    band = None
    if dof > 0:
        s2 = float(np.sum(resid ** 2)) / dof
        xbar = float(np.mean(x))
        sxx = float(np.sum((x - xbar) ** 2))
        if s2 >= 0 and sxx > 1e-12:
            se = np.sqrt(s2 * (1.0 / len(x) + (grid - xbar) ** 2 / sxx))
            tcrit = float(stats.t.ppf(0.975, dof))
            band = (y_hat - tcrit * se, y_hat + tcrit * se)
    return slope, intercept, grid, y_hat, band
''',
    "brackets": '''def _draw_brackets(ax, pairs, levels):
    """在分类轴上画显著性括号（pairs = [(a, b, 标记), ...]）。"""
    if not pairs:
        return
    y0, y1 = ax.get_ylim()
    span = (y1 - y0) or 1.0
    for index, (a_label, b_label, text) in enumerate(pairs):
        xa, xb = levels.index(a_label) + 1, levels.index(b_label) + 1
        y = y1 + span * (0.05 + 0.085 * index)
        ax.plot([xa, xa, xb, xb], [y - span * 0.02, y, y, y - span * 0.02],
                color="#3A3A3A", linewidth=1.0, clip_on=False)
        ax.text((xa + xb) / 2.0, y + span * 0.008, text, ha="center", va="bottom",
                fontsize=max(6.0, TICK_SIZE - 2), clip_on=False)
    ax.set_ylim(y0, y1 + span * (0.10 + 0.085 * len(pairs)))
''',
}


def _cmap_name(style: Mapping[str, Any], requested: Any = "", diverging: bool = False) -> str:
    text = str(requested or "").strip()
    if text:
        mapped = C.PALETTE_CMAPS.get(text, text)
        try:
            matplotlib.colormaps[mapped]
            return mapped
        except Exception:  # noqa: BLE001
            pass
    palette = str(style.get("palette") or "")
    if palette in C.PALETTE_CMAPS:
        return C.PALETTE_CMAPS[palette]
    return "RdBu_r" if diverging else "viridis"


def _py_colors_repr(style: Mapping[str, Any], levels: Sequence[Any]) -> str:
    mapping = _colors(style, levels)
    items = ", ".join(f'"{name}": "{color}"' for name, color in mapping.items())
    return "{" + items + "}"


def _py_prelude(style: Mapping[str, Any], helpers: Sequence[str] = ()) -> str:
    family = C.font_family(style)
    serif = family in ("Times New Roman", "Georgia")
    fallback = [family] + [name for name in
                           ("Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Songti SC", "DejaVu Sans")
                           if name != family]
    palette = C.color_list(style, 10)
    lines = [
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib",
        "import matplotlib.pyplot as plt",
        "from scipy import stats",
        "",
        "# ===== 风格（与 PlotLab 工作台一致）=====",
        f"PALETTE = {palette!r}".replace("'", '"'),
        "plt.rcParams.update({",
        f'    "font.family": "{"serif" if serif else "sans-serif"}",',
        f'    "font.{"serif" if serif else "sans-serif"}": {fallback!r},'.replace("'", '"'),
        f'    "font.size": {float(style.get("text_size", 10))},',
        f'    "axes.titlesize": {float(style.get("title_size", 14))},',
        f'    "axes.labelsize": {float(style.get("label_size", 12))},',
        f'    "xtick.labelsize": {float(style.get("tick_size", 10))},',
        f'    "ytick.labelsize": {float(style.get("tick_size", 10))},',
        f'    "legend.fontsize": {float(style.get("legend_size", 10))},',
        '    "axes.unicode_minus": False,',
        '    "figure.facecolor": "white",',
        '    "savefig.facecolor": "white",',
        "})",
        "",
        f"FIG_W, FIG_H = {float(style.get('fig_width', 6.2))}, {float(style.get('fig_height', 4.4))}",
        f"TITLE_SIZE, LABEL_SIZE, TICK_SIZE = {float(style.get('title_size', 14))}, "
        f"{float(style.get('label_size', 12))}, {float(style.get('tick_size', 10))}",
        f"LEGEND_SIZE = {float(style.get('legend_size', 10))}",
        f"TITLE_BOLD = {bool(style.get('title_bold', True))}",
        f"ALPHA = {float(style.get('alpha', 0.85))}",
        f"LW = {float(style.get('line_width', 1.4))}",
        f"POINT = {float(style.get('point_size', 28))}",
        f"BAR_WIDTH = {float(style.get('bar_width', 0.72))}",
        f"EDGE = {float(style.get('edge_width', 0.8))}",
        f"JITTER_SIZE = {float(style.get('jitter_size', 2.2))}",
        f"JITTER_ALPHA = {float(style.get('jitter_alpha', 0.55))}",
        "TITLE, X_LABEL, Y_LABEL = \"\", \"\", \"\"",
        _PY_FINISH.format(
            grid=str(style.get("grid") or "none"),
            frame=str(style.get("frame") or "left_bottom"),
            tick_dir=str(style.get("tick_dir") or "out"),
            x_rotate=float(style.get("x_rotate") or 0),
            x_log=bool(style.get("x_log")),
            y_log=bool(style.get("y_log")),
            legend_pos=str(style.get("legend_pos") or "right"),
        ),
    ]
    for name in helpers:
        helper = _PY_HELPERS.get(name)
        if helper:
            lines.append(helper)
    return "\n".join(lines)


def _py_save() -> str:
    return ('plt.savefig("plotlab_figure.png", dpi=300, bbox_inches="tight")\n'
            'print("saved plotlab_figure.png")')


def _py_join(prelude: str, data_block: str, body: str) -> str:
    return "\n\n".join(part.strip("\n") for part in (prelude, data_block, body, _py_save()) if part)


def _py_literal(values: Any) -> str:
    series = pd.Series(list(values)) if not isinstance(values, pd.Series) else values
    if pd.api.types.is_datetime64_any_dtype(series):
        return "[" + ", ".join('"%s"' % str(value)[:19] for value in series.tolist()) + "]"
    if pd.api.types.is_numeric_dtype(series):
        parts = []
        for value in series.tolist():
            try:
                number = float(value)
            except (TypeError, ValueError):
                parts.append("np.nan")
                continue
            if not math.isfinite(number):
                parts.append("np.nan")
            else:
                parts.append(repr(round(number, 6)))
        return "[" + ", ".join(parts) + "]"
    parts = []
    for value in series.tolist():
        if value is None:
            parts.append("None")
            continue
        try:
            if isinstance(value, float) and math.isnan(value):
                parts.append("np.nan")
                continue
        except Exception:  # noqa: BLE001
            pass
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        parts.append('"%s"' % text[:120])
    return "[" + ", ".join(parts) + "]"


def _py_tidy_block(data: Mapping[str, Any], roles: Sequence[str]) -> tuple[str, dict[str, str]]:
    """生成 tidy 数据准备代码：返回 (code, {role: 列名})。"""
    tidy = data.get("tidy")
    if not isinstance(tidy, pd.DataFrame) or tidy.empty:
        raise C.prerequisite("没有可用的 tidy 数据，无法生成 Python 代码")
    labels = data.get("labels") or {}
    selected: list[tuple[str, str]] = []
    for role in roles:
        canonical = f"__{role}"
        if canonical in tidy.columns:
            selected.append((role, canonical))
    if not selected:
        raise C.prerequisite("tidy 数据缺少图表需要的角色列")
    names: dict[str, str] = {}
    used: set[str] = set()
    for role, _canonical in selected:
        base = str(labels.get(role) or role) or role
        name = base
        index = 2
        while name in used:
            name = f"{base}_{index}"
            index += 1
        used.add(name)
        names[role] = name
    frame = tidy[[canonical for _role, canonical in selected]].copy()
    frame.columns = [names[role] for role, _canonical in selected]
    lines: list[str] = []
    if len(frame) <= 400:
        lines.append("# ===== 数据（工作台清洗+角色映射后的观测；也可换成 pd.read_csv(\"your.csv\")）=====")
        lines.append("df = pd.DataFrame({")
        for column in frame.columns:
            lines.append(f'    "{column}": {_py_literal(frame[column])},')
        lines.append("})")
    else:
        columns = list(frame.columns)
        lines.append(f"# 数据有 {len(frame)} 行，不内联；把路径换成你的清洗后数据文件（列名与下方角色一致）")
        lines.append('df = pd.read_csv("your_data.csv")')
        lines.append("df = df[%r]" % columns)
    for role in ("y", "size", "time", "event"):
        if role in names:
            lines.append(f'df["{names[role]}"] = pd.to_numeric(df["{names[role]}"], errors="coerce")')
    if "x" in names and pd.api.types.is_datetime64_any_dtype(tidy["__x"]):
        lines.append(f'df["{names["x"]}"] = pd.to_datetime(df["{names["x"]}"], errors="coerce")')
    role_variables = {"x": "X", "y": "Y", "group": "G", "color": "C",
                      "size": "S", "time": "T", "event": "E"}
    lines.append("")
    lines.append("# ===== 角色映射（列名 → 变量）=====")
    for role, variable in role_variables.items():
        if role in names:
            lines.append(f'{variable} = "{names[role]}"')
    return "\n".join(lines), names


def _py_wide_block(data: Mapping[str, Any]) -> str:
    wide = data.get("wide")
    if not isinstance(wide, Mapping) or wide.get("matrix") is None:
        raise C.prerequisite("没有可用的矩阵/宽表数据，无法生成 Python 代码")
    matrix = np.asarray(wide.get("matrix"), dtype=float)
    features = [str(value) for value in (wide.get("features") or [])]
    samples = [str(value) for value in (wide.get("samples") or [])]
    groups = wide.get("sample_groups")
    group_list = ([str(value) for value in list(groups)]
                  if groups is not None and len(groups) == matrix.shape[0] else None)
    lines = ["# ===== 数据：矩阵（行=样本，列=特征）====="]
    if matrix.size <= 6000:
        lines.append(f"features = {features!r}".replace("'", '"'))
        lines.append(f"samples = {samples!r}".replace("'", '"'))
        matrix_text = np.array2string(matrix, separator=", ", threshold=1000000,
                                      max_line_width=140)
        lines.append(f"matrix = np.array({matrix_text}, dtype=float)")
        lines.append(f"sample_groups = {group_list!r}".replace("'", '"'))
    else:
        lines.append('raw = pd.read_csv("your_matrix.csv", index_col=0)  # 首列特征名，其余列样本')
        lines.append("features = [str(value) for value in raw.index]")
        lines.append("samples = [str(value) for value in raw.columns]")
        lines.append("matrix = raw.to_numpy(dtype=float).T  # 行=样本，列=特征")
        lines.append("sample_groups = None")
    return "\n".join(lines)


_SIG_LEVELS_CACHE: dict[str, Any] = {}


def _py_stat_annotation(data: Mapping[str, Any], style: Mapping[str, Any],
                        levels: Sequence[Any]) -> str:
    """把组间比较结果固化成可复现代码（同一套检验与校正，数值直接写出）。"""
    if not style.get("stat_annot"):
        return ""
    tidy = data.get("tidy")
    if not isinstance(tidy, pd.DataFrame) or "__x" not in tidy.columns:
        return ""
    result = _compare_groups(tidy, "__x", "__y", style, levels=list(levels))
    if not result or not result.get("pairs"):
        return ""
    pairs = [(pair["a"], pair["b"], pair["label"]) for pair in result["pairs"] if pair["label"]]
    if not pairs:
        return ""
    pair_text = ", ".join(f'("{a}", "{b}", "{label}")' for a, b, label in pairs)
    method = str(result.get("method") or "test")
    correction = str(result.get("p_adjust") or "none")
    lines = [
        f"# ===== 组间显著性（{method}，{correction} 校正；数值由工作台同一套统计计算得出）=====",
        f"PAIRS = [{pair_text}]",
        "LEVELS_X = [str(value) for value in LEVELS]",
        "_draw_brackets(ax, PAIRS, LEVELS_X)",
    ]
    return "\n".join(lines)


def _py_comparison_axes(style: Mapping[str, Any]) -> str:
    return "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))"


# ---------------------------------------------------------------------------
# 逐个图表的 Python 代码生成
# ---------------------------------------------------------------------------


def _py_bar(data: Mapping[str, Any], params: Mapping[str, Any], style: Mapping[str, Any]) -> str:
    agg = _pchoice(params, "agg", "mean", ["mean", "median", "sum"])
    error = _pchoice(params, "error", "auto", ["auto", "sd", "se", "ci95", "none"])
    if error == "auto":
        error = str(style.get("error") or "se")
    show_points = _pbool(params, "points", True)
    tidy = data.get("tidy")
    group_key = _pick_group_col(data, tidy, keys=("__group",)) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y"] + (["group"] if group_key else [])
    block, names = _py_tidy_block(data, roles)
    levels = _levels_of(data.get("tidy"), "__x", max_levels=24) if isinstance(data.get("tidy"), pd.DataFrame) else []
    levels = [str(_fmt_level(level)) for level in levels]
    frames = data.get("tidy")
    group_levels = ([str(_fmt_level(level)) for level in _levels_of(frames, group_key, max_levels=10)]
                    if group_key else [])
    color_levels = group_levels if group_levels else list(levels)
    x_var, y_var = names["x"], names["y"]
    g_var = names.get("group")
    filters = f"df = df.dropna(subset=[{x_var!r}, {y_var!r}]).copy()".replace("'", '"')
    labels = data.get("labels") or {}
    body = [
        filters,
        f"LEVELS = {levels!r}".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, color_levels)}",
        f"AGG, ERROR = {agg!r}, {error!r}".replace("'", '"'),
        f"SHOW_POINTS = {bool(show_points)}",
        _py_comparison_axes(style),
    ]
    body.append("if GROUP_LEVELS:")
    body.append("    bar_width = BAR_WIDTH / len(GROUP_LEVELS)")
    body.append("    offsets = {group: (index - (len(GROUP_LEVELS) - 1) / 2.0) * bar_width")
    body.append("               for index, group in enumerate(GROUP_LEVELS)}")
    body.append("    for group in GROUP_LEVELS:")
    body.append("        xs, heights, errs = [], [], []")
    body.append("        for level in LEVELS:")
    body.append("            values = df[(df[X] == level) & (df[G] == group)][Y]")
    body.append("            center, err, count = _summary(values, AGG, ERROR)")
    body.append("            xs.append(LEVELS.index(level) + 1 + offsets[group])")
    body.append("            heights.append(center)")
    body.append("            errs.append(err)")
    body.append("        ax.bar(xs, heights, width=bar_width, color=COLORS[group],")
    body.append("               edgecolor=\"white\", linewidth=EDGE, alpha=ALPHA, label=group)")
    body.append("        if ERROR != \"none\":")
    body.append("            ax.errorbar(xs, heights, yerr=errs, fmt=\"none\", ecolor=\"#2B2B2B\",")
    body.append("                        elinewidth=max(0.6, EDGE * 0.9), capsize=3)")
    body.append("        if SHOW_POINTS:")
    body.append("            rng = np.random.default_rng(11)")
    body.append("            for level, xpos in zip(LEVELS, xs):")
    body.append("                vals = pd.to_numeric(df[(df[X] == level) & (df[G] == group)][Y],")
    body.append("                                   errors=\"coerce\").dropna()")
    body.append("                ax.scatter(xpos + rng.normal(0, bar_width * 0.13, len(vals)), vals,")
    body.append("                           s=max(6.0, (JITTER_SIZE * 1.7) ** 2), color=\"#3A3A3A\",")
    body.append("                           alpha=JITTER_ALPHA, linewidths=0, zorder=4)")
    body.append("else:")
    body.append("    xs, heights, errs, colors = [], [], [], []")
    body.append("    for level in LEVELS:")
    body.append("        center, err, count = _summary(df[df[X] == level][Y], AGG, ERROR)")
    body.append("        xs.append(LEVELS.index(level) + 1)")
    body.append("        heights.append(center)")
    body.append("        errs.append(err)")
    body.append("        colors.append(COLORS.get(level, PALETTE[0]))")
    body.append("    ax.bar(xs, heights, width=BAR_WIDTH, color=colors, edgecolor=\"white\",")
    body.append("           linewidth=EDGE, alpha=ALPHA)")
    body.append("    if ERROR != \"none\":")
    body.append("        ax.errorbar(xs, heights, yerr=errs, fmt=\"none\", ecolor=\"#2B2B2B\",")
    body.append("                    elinewidth=max(0.6, EDGE * 0.9), capsize=3)")
    body.append("    if SHOW_POINTS:")
    body.append("        rng = np.random.default_rng(11)")
    body.append("        for level, xpos in zip(LEVELS, xs):")
    body.append("            vals = pd.to_numeric(df[df[X] == level][Y], errors=\"coerce\").dropna()")
    body.append("            ax.scatter(xpos + rng.normal(0, BAR_WIDTH * 0.12, len(vals)), vals,")
    body.append("                       s=max(6.0, (JITTER_SIZE * 1.7) ** 2), color=\"#3A3A3A\",")
    body.append("                       alpha=JITTER_ALPHA, linewidths=0, zorder=4)")
    body.append("ax.set_xticks([LEVELS.index(level) + 1 for level in LEVELS])")
    body.append("ax.set_xticklabels(LEVELS)")
    body.append("ax.set_xlim(0.3, len(LEVELS) + 0.7)")
    body.append(f"X_LABEL, Y_LABEL = {str(labels.get('x') or 'x')!r}, {str(labels.get('y') or 'y')!r}".replace("'", '"'))
    body.append(f"TITLE = \"Bar chart: {agg} {{Y_LABEL}} by {{X_LABEL}}\"")
    body.append("_finish(ax)")
    body.append(_py_stat_annotation(data, style, [_fmt_level(level) for level in levels]))
    return _py_join(_py_prelude(style, ("summary", "brackets")), block, "\n".join(body))


def _py_bar_dot(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> str:
    agg = _pchoice(params, "agg", "mean", ["mean", "median", "sum"])
    error = _pchoice(params, "error", "auto", ["auto", "sd", "se", "ci95", "none"])
    if error == "auto":
        error = str(style.get("error") or "se")
    tidy = data.get("tidy")
    group_key = _pick_group_col(data, tidy, keys=("__group",)) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y"] + (["group"] if group_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    levels = [str(_fmt_level(level)) for level in
              (_levels_of(data.get("tidy"), "__x", max_levels=24) if isinstance(data.get("tidy"), pd.DataFrame) else [])]
    group_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), group_key, max_levels=10)]
                    if group_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}]).copy()".replace("'", '"'),
        f"LEVELS = {levels!r}".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, group_levels if group_levels else levels)}",
        f"AGG, ERROR = {agg!r}, {error!r}".replace("'", '"'),
        _py_comparison_axes(style),
        "if GROUP_LEVELS:",
        "    bar_width = BAR_WIDTH / len(GROUP_LEVELS)",
        "    for group in GROUP_LEVELS:",
        "        offset = (GROUP_LEVELS.index(group) - (len(GROUP_LEVELS) - 1) / 2.0) * bar_width",
        "        xs, heights, errs = [], [], []",
        "        for level in LEVELS:",
        "            values = df[(df[X] == level) & (df[G] == group)][Y]",
        "            center, err, count = _summary(values, AGG, ERROR)",
        "            xs.append(LEVELS.index(level) + 1 + offset)",
        "            heights.append(center)",
        "            errs.append(err)",
        "        ax.bar(xs, heights, width=bar_width, color=COLORS[group], edgecolor=\"white\",",
        "               linewidth=EDGE, alpha=ALPHA * 0.45, label=group)",
        "        rng = np.random.default_rng(5)",
        "        for level, xpos in zip(LEVELS, xs):",
        "            vals = pd.to_numeric(df[(df[X] == level) & (df[G] == group)][Y],",
        "                               errors=\"coerce\").dropna()",
        "            ax.scatter(xpos + rng.normal(0, bar_width * 0.22, len(vals)), vals,",
        "                       s=max(6.0, (JITTER_SIZE * 1.7) ** 2), color=COLORS[group],",
        "                       alpha=max(0.35, JITTER_ALPHA), linewidths=0, zorder=4)",
        "        if ERROR != \"none\":",
        "            ax.errorbar(xs, heights, yerr=errs, fmt=\"none\", ecolor=\"#2B2B2B\", capsize=3)",
        "else:",
        "    xs, heights, errs = [], [], []",
        "    for level in LEVELS:",
        "        center, err, count = _summary(df[df[X] == level][Y], AGG, ERROR)",
        "        xpos = LEVELS.index(level) + 1",
        "        xs.append(xpos)",
        "        heights.append(center)",
        "        errs.append(err)",
        "        ax.bar(xpos, center, width=BAR_WIDTH, color=COLORS[level], edgecolor=\"white\",",
        "               linewidth=EDGE, alpha=ALPHA * 0.45)",
        "        vals = pd.to_numeric(df[df[X] == level][Y], errors=\"coerce\").dropna()",
        "        ax.scatter(xpos + np.random.default_rng(5).normal(0, BAR_WIDTH * 0.2, len(vals)),",
        "                   vals, s=max(6.0, (JITTER_SIZE * 1.7) ** 2), color=COLORS[level],",
        "                   alpha=max(0.35, JITTER_ALPHA), linewidths=0, zorder=4)",
        "    if ERROR != \"none\":",
        "        ax.errorbar(xs, heights, yerr=errs, fmt=\"none\", ecolor=\"#2B2B2B\", capsize=3)",
        "ax.set_xticks([LEVELS.index(level) + 1 for level in LEVELS])",
        "ax.set_xticklabels(LEVELS)",
        "ax.set_xlim(0.3, len(LEVELS) + 0.7)",
        f"X_LABEL, Y_LABEL = {str(labels.get('x') or 'x')!r}, {str(labels.get('y') or 'y')!r}".replace("'", '"'),
        f"TITLE = \"Bar + dot chart: {agg} {{Y_LABEL}} by {{X_LABEL}}\"",
        "_finish(ax)",
        _py_stat_annotation(data, style, [_fmt_level(level) for level in levels]),
    ]
    return _py_join(_py_prelude(style, ("summary", "brackets")), block, "\n".join(body))


def _py_box(data: Mapping[str, Any], params: Mapping[str, Any],
            style: Mapping[str, Any]) -> str:
    show_outliers = _pbool(params, "outliers", True)
    show_points = _pbool(params, "points", True)
    tidy = data.get("tidy")
    group_key = _pick_group_col(data, tidy, keys=("__group",)) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y"] + (["group"] if group_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    levels = [str(_fmt_level(level)) for level in
              (_levels_of(data.get("tidy"), "__x", max_levels=24) if isinstance(data.get("tidy"), pd.DataFrame) else [])]
    group_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), group_key, max_levels=10)]
                    if group_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}]).copy()".replace("'", '"'),
        f"LEVELS = {levels!r}".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, group_levels if group_levels else levels)}",
        f"SHOW_OUTLIERS, SHOW_POINTS = {bool(show_outliers)!r}, {bool(show_points)!r}".replace("'", '"'),
        _py_comparison_axes(style),
        "slot = 0.82",
        "box_width = min(0.66, slot * 0.9) if not GROUP_LEVELS else slot / len(GROUP_LEVELS) * 0.82",
        "rng = np.random.default_rng(19)",
        "for gi, group in enumerate(GROUP_LEVELS or [None]):",
        "    offset = ((gi - (len(GROUP_LEVELS) - 1) / 2.0) * (slot / len(GROUP_LEVELS))",
        "              if GROUP_LEVELS else 0.0)",
        "    for li, level in enumerate(LEVELS):",
        "        mask = df[X] == level",
        "        if group is not None:",
        "            mask = mask & (df[G] == group)",
        "        values = pd.to_numeric(df.loc[mask, Y], errors=\"coerce\").dropna().to_numpy()",
        "        if len(values) == 0:",
        "            continue",
        "        position = li + 1 + offset",
        "        color = COLORS.get(group, PALETTE[0]) if group is not None else COLORS[level]",
        "        artists = ax.bxp([{**_box(values), \"label\": \"\"}], positions=[position],",
        "                         widths=box_width, patch_artist=True, showfliers=SHOW_OUTLIERS,",
        "                         manage_ticks=False)",
        "        for patch in artists[\"boxes\"]:",
        "            patch.set_facecolor(color)",
        "            patch.set_alpha(ALPHA)",
        "            patch.set_edgecolor(\"#2B2B2B\")",
        "        for line in artists[\"medians\"]:",
        "            line.set_color(\"#1A1A1A\")",
        "        if SHOW_POINTS:",
        "            ax.scatter(position + rng.normal(0.0, box_width * 0.18, len(values)), values,",
        "                       s=max(6.0, (JITTER_SIZE * 1.7) ** 2), color=\"#2B2B2B\",",
        "                       alpha=JITTER_ALPHA, linewidths=0, zorder=3)",
        "ax.set_xticks([LEVELS.index(level) + 1 for level in LEVELS])",
        "ax.set_xticklabels(LEVELS)",
        "ax.set_xlim(0.35, len(LEVELS) + 0.65)",
        f"X_LABEL, Y_LABEL = {str(labels.get('x') or 'x')!r}, {str(labels.get('y') or 'y')!r}".replace("'", '"'),
        "TITLE = \"Box plot: \" + Y_LABEL + \" by \" + X_LABEL",
        "_finish(ax)",
        _py_stat_annotation(data, style, [_fmt_level(level) for level in levels]),
    ]
    return _py_join(_py_prelude(style, ("box", "brackets")), block, "\n".join(body))


def _py_violin(data: Mapping[str, Any], params: Mapping[str, Any],
               style: Mapping[str, Any]) -> str:
    inner = _pchoice(params, "inner", "box", ["box", "quartile", "none"])
    bw = _pfloat(params, "bw", 1.0, 0.2, 4.0)
    trim = _pbool(params, "trim", True)
    tidy = data.get("tidy")
    group_key = _pick_group_col(data, tidy, keys=("__group",)) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y"] + (["group"] if group_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    levels = [str(_fmt_level(level)) for level in
              (_levels_of(data.get("tidy"), "__x", max_levels=24) if isinstance(data.get("tidy"), pd.DataFrame) else [])]
    group_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), group_key, max_levels=10)]
                    if group_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}]).copy()".replace("'", '"'),
        f"LEVELS = {levels!r}".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, group_levels if group_levels else levels)}",
        f"INNER, BW, TRIM = {inner!r}, {bw}, {bool(trim)}".replace("'", '"'),
        _py_comparison_axes(style),
        "values = pd.to_numeric(df[Y], errors=\"coerce\").dropna().to_numpy()",
        "low, high = float(np.min(values)), float(np.max(values))",
        "pad = max((high - low) * 0.12, 1e-9)",
        "grid = np.linspace(low - pad, high + pad, 160)",
        "slot = 0.84",
        "cell_width = min(0.7, slot * 0.92) if not GROUP_LEVELS else slot / len(GROUP_LEVELS) * 0.9",
        "for gi, group in enumerate(GROUP_LEVELS or [None]):",
        "    offset = ((gi - (len(GROUP_LEVELS) - 1) / 2.0) * (slot / len(GROUP_LEVELS))",
        "              if GROUP_LEVELS else 0.0)",
        "    for li, level in enumerate(LEVELS):",
        "        mask = df[X] == level",
        "        if group is not None:",
        "            mask = mask & (df[G] == group)",
        "        vals = pd.to_numeric(df.loc[mask, Y], errors=\"coerce\").dropna().to_numpy()",
        "        density = _kde(vals, grid, BW)",
        "        if density is None:",
        "            continue",
        "        if TRIM:",
        "            density = np.where((grid >= vals.min()) & (grid <= vals.max()), density, 0.0)",
        "        center = li + 1 + offset",
        "        color = COLORS.get(group, PALETTE[0]) if group is not None else COLORS[level]",
        "        half = density * cell_width / 2.0",
        "        ax.fill_betweenx(grid, center - half, center + half, facecolor=color,",
        "                         alpha=ALPHA * 0.9, edgecolor=\"#2B2B2B\", linewidth=EDGE * 0.8, zorder=2)",
        "        box = _box(vals)",
        "        if INNER == \"box\":",
        "            inner_artists = ax.bxp([{**box, \"label\": \"\"}], positions=[center],",
        "                                   widths=cell_width * 0.16, patch_artist=True,",
        "                                   showfliers=False, manage_ticks=False,",
        "                                   medianprops={\"color\": \"#1A1A1A\", \"linewidth\": 1.2})",
        "            for patch in inner_artists[\"boxes\"]:",
        "                patch.set_facecolor(\"white\")",
        "                patch.set_alpha(0.9)",
        "        elif INNER == \"quartile\":",
        "            ax.plot([center, center], [box[\"q1\"], box[\"q3\"]], color=\"#2B2B2B\",",
        "                    linewidth=max(1.0, EDGE), zorder=4)",
        "            ax.plot([center], [box[\"med\"]], marker=\"o\", markersize=4,",
        "                    markerfacecolor=\"white\", markeredgecolor=\"#2B2B2B\", zorder=5)",
        "ax.set_xticks([LEVELS.index(level) + 1 for level in LEVELS])",
        "ax.set_xticklabels(LEVELS)",
        "ax.set_xlim(0.35, len(LEVELS) + 0.65)",
        "ax.set_ylim(low - pad, high + pad)",
        f"X_LABEL, Y_LABEL = {str(labels.get('x') or 'x')!r}, {str(labels.get('y') or 'y')!r}".replace("'", '"'),
        "TITLE = \"Violin plot: \" + Y_LABEL + \" by \" + X_LABEL",
        "_finish(ax)",
        _py_stat_annotation(data, style, [_fmt_level(level) for level in levels]),
    ]
    return _py_join(_py_prelude(style, ("kde", "box", "brackets")), block, "\n".join(body))


def _py_raincloud(data: Mapping[str, Any], params: Mapping[str, Any],
                  style: Mapping[str, Any]) -> str:
    show_box = _pbool(params, "box", True)
    show_points = _pbool(params, "points", True)
    tidy = data.get("tidy")
    group_key = _pick_group_col(data, tidy, keys=("__group",)) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y"] + (["group"] if group_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    levels = [str(_fmt_level(level)) for level in
              (_levels_of(data.get("tidy"), "__x", max_levels=24) if isinstance(data.get("tidy"), pd.DataFrame) else [])]
    group_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), group_key, max_levels=10)]
                    if group_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}]).copy()".replace("'", '"'),
        f"LEVELS = {levels!r}".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, group_levels if group_levels else levels)}",
        _py_comparison_axes(style),
        "values = pd.to_numeric(df[Y], errors=\"coerce\").dropna().to_numpy()",
        "low, high = float(np.min(values)), float(np.max(values))",
        "pad = max((high - low) * 0.12, 1e-9)",
        "grid = np.linspace(low - pad, high + pad, 160)",
        "slot = 0.86",
        "cell_width = slot * 0.92 if not GROUP_LEVELS else slot / len(GROUP_LEVELS) * 0.9",
        "rng = np.random.default_rng(23)",
        "for gi, group in enumerate(GROUP_LEVELS or [None]):",
        "    offset = ((gi - (len(GROUP_LEVELS) - 1) / 2.0) * (slot / len(GROUP_LEVELS))",
        "              if GROUP_LEVELS else 0.0)",
        "    for li, level in enumerate(LEVELS):",
        "        mask = df[X] == level",
        "        if group is not None:",
        "            mask = mask & (df[G] == group)",
        "        vals = pd.to_numeric(df.loc[mask, Y], errors=\"coerce\").dropna().to_numpy()",
        "        if len(vals) == 0:",
        "            continue",
        "        center = li + 1 + offset",
        "        color = COLORS.get(group, PALETTE[0]) if group is not None else COLORS[level]",
        "        density = _kde(vals, grid, 1.0)",
        "        if density is not None:",
        "            half = density * cell_width * 0.42",
        "            ax.fill_betweenx(grid, center - half, center, facecolor=color, alpha=ALPHA * 0.9,",
        "                             edgecolor=\"#2B2B2B\", linewidth=EDGE * 0.8, zorder=2)",
        "        box = _box(vals)",
        "        if SHOW_BOX:",
        "            box_center = center + cell_width * 0.16",
        "            box_width = cell_width * 0.16",
        "            ax.add_patch(plt.Rectangle((box_center - box_width / 2.0, box[\"q1\"]), box_width,",
        "                                       max(box[\"q3\"] - box[\"q1\"], 1e-9), facecolor=color,",
        "                                       edgecolor=\"#2B2B2B\", linewidth=EDGE * 0.9, zorder=3))",
        "            ax.plot([box_center, box_center], [box[\"whislo\"], box[\"q1\"]], color=\"#2B2B2B\",",
        "                    linewidth=EDGE * 0.9, zorder=3)",
        "            ax.plot([box_center, box_center], [box[\"q3\"], box[\"whishi\"]], color=\"#2B2B2B\",",
        "                    linewidth=EDGE * 0.9, zorder=3)",
        "            cap = box_width * 0.35",
        "            ax.plot([box_center - cap, box_center + cap], [box[\"whislo\"]] * 2,",
        "                    color=\"#2B2B2B\", linewidth=EDGE * 0.9, zorder=3)",
        "            ax.plot([box_center - cap, box_center + cap], [box[\"whishi\"]] * 2,",
        "                    color=\"#2B2B2B\", linewidth=EDGE * 0.9, zorder=3)",
        "            ax.plot([box_center - box_width / 2.0, box_center + box_width / 2.0],",
        "                    [box[\"med\"]] * 2, color=\"white\", linewidth=max(1.2, EDGE), zorder=4)",
        "        if SHOW_POINTS:",
        "            point_center = center + cell_width * 0.34",
        "            jitter = rng.normal(0.0, cell_width * 0.055, len(vals))",
        "            ax.scatter(point_center + jitter, vals, s=max(6.0, (JITTER_SIZE * 1.7) ** 2) * 0.8,",
        "                       color=color, alpha=JITTER_ALPHA, edgecolors=\"none\", zorder=5)",
        "ax.set_xticks([LEVELS.index(level) + 1 for level in LEVELS])",
        "ax.set_xticklabels(LEVELS)",
        "ax.set_xlim(0.3, len(LEVELS) + 0.7)",
        "ax.set_ylim(low - pad, high + pad)",
        f"X_LABEL, Y_LABEL = {str(labels.get('x') or 'x')!r}, {str(labels.get('y') or 'y')!r}".replace("'", '"'),
        "TITLE = \"Raincloud plot: \" + Y_LABEL + \" by \" + X_LABEL",
        "_finish(ax)",
        _py_stat_annotation(data, style, [_fmt_level(level) for level in levels]),
    ]
    # 生成代码里用到的开关常量
    body.insert(4, f"SHOW_BOX, SHOW_POINTS = {bool(show_box)}, {bool(show_points)}")
    return _py_join(_py_prelude(style, ("kde", "box", "brackets")), block, "\n".join(body))


def _py_dotplot(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> str:
    agg = _pchoice(params, "agg", "mean", ["mean", "median"])
    show_values = _pbool(params, "show_values", False)
    tidy = data.get("tidy")
    if not isinstance(tidy, pd.DataFrame):
        raise C.prerequisite("点图代码生成需要 tidy 数据")
    row_key = _pick_group_col(data, tidy, keys=("__group", "__color"))
    roles = ["x"] + (["group"] if "__group" in tidy.columns else []) \
        + (["color"] if "__color" in tidy.columns else []) \
        + (["y"] if "__y" in tidy.columns else []) \
        + (["size"] if "__size" in tidy.columns else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    body = [
        f"df = df.dropna(subset=[{names['x']!r}]).copy()".replace("'", '"'),
        f"ROW = {names.get(row_key[2:]) if row_key else None!r}".replace("'", '"'),
        f"VALUE = {names.get('y')!r}".replace("'", '"'),
        f"SIZE = {names.get('size')!r}".replace("'", '"'),
        f"AGG, SHOW_VALUES = {agg!r}, {bool(show_values)}".replace("'", '"'),
        f"CMAP = \"{_cmap_name(style, params.get('cmap'))}\"",
        _py_comparison_axes(style),
        "if VALUE is None or pd.to_numeric(df[VALUE], errors=\"coerce\").notna().sum() < 3:",
        "    VALUE = SIZE",
        "levels = list(dict.fromkeys(df[X].astype(str)))",
        "rows = list(dict.fromkeys(df[ROW].astype(str))) if ROW is not None else [\"All\"]",
        "matrix = np.full((len(rows), len(levels)), np.nan)",
        "for ri, row in enumerate(rows):",
        "    for ci, level in enumerate(levels):",
        "        cell = df[(df[X].astype(str) == level)]",
        "        if ROW is not None:",
        "            cell = cell[cell[ROW].astype(str) == row]",
        "        if VALUE is None:",
        "            matrix[ri, ci] = len(cell)",
        "        else:",
        "            values = pd.to_numeric(cell[VALUE], errors=\"coerce\").dropna()",
        "            matrix[ri, ci] = values.median() if AGG == \"median\" else values.mean()",
        "    if ROW is not None:",
        "        pass",
        "finite = matrix[np.isfinite(matrix)]",
        "vmin, vmax = float(finite.min()), float(finite.max())",
        "span = vmax - vmin if vmax > vmin else 1.0",
        "norm = np.clip(np.nan_to_num((matrix - vmin) / span, nan=0.0), 0.0, 1.0)",
        "areas = max(6.0, POINT * 0.45) + norm * (max(18.0, POINT * 6.0) - max(6.0, POINT * 0.45))",
        "xx, yy = np.meshgrid(np.arange(1, len(levels) + 1), np.arange(len(rows), 0, -1))",
        "scatter = ax.scatter(xx.ravel(), yy.ravel(), s=areas.ravel(), c=matrix.ravel(),",
        "                     cmap=CMAP, edgecolors=\"#2B2B2B\", linewidths=EDGE * 0.6, alpha=ALPHA)",
        "if SHOW_VALUES:",
        "    for ri in range(len(rows)):",
        "        for ci in range(len(levels)):",
        "            if np.isfinite(matrix[ri, ci]):",
        "                ax.text(ci + 1, len(rows) - ri, f\"{matrix[ri, ci]:.2f}\", ha=\"center\",",
        "                        va=\"center\", fontsize=max(5.5, TICK_SIZE - 2), zorder=5)",
        "ax.set_xticks(np.arange(1, len(levels) + 1))",
        "ax.set_xticklabels(levels)",
        "ax.set_yticks(np.arange(len(rows), 0, -1))",
        "ax.set_yticklabels(rows)",
        "ax.set_xlim(0.4, len(levels) + 0.6)",
        "ax.set_ylim(0.4, len(rows) + 0.6)",
        "plt.colorbar(scatter, ax=ax, shrink=0.86, pad=0.02, label=(VALUE or \"count\"))",
        "for value in (vmin, (vmin + vmax) / 2.0, vmax):",
        "    area = max(6.0, POINT * 0.45) + (0.0 if vmax == vmin else (value - vmin) / span) * (max(18.0, POINT * 6.0) - max(6.0, POINT * 0.45))",
        "    ax.plot([], [], marker=\"o\", linestyle=\"none\", markersize=np.sqrt(area) * 0.9,",
        "            markerfacecolor=\"#7F7F7F\", markeredgecolor=\"none\", label=f\"{value:.2f}\")",
        f"X_LABEL, Y_LABEL = {str(labels.get('x') or 'x')!r}, {str(labels.get(row_key[2:] if row_key else 'group') or 'Group')!r}".replace("'", '"'),
        "TITLE = \"Dot plot: \" + (VALUE or \"count\") + \" by \" + X_LABEL",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _py_stack(data: Mapping[str, Any], params: Mapping[str, Any],
              style: Mapping[str, Any]) -> str:
    normalize = _pbool(params, "normalize", False)
    agg = _pchoice(params, "agg", "sum", ["sum", "mean"])
    show_total = _pbool(params, "show_total", False)
    tidy = data.get("tidy")
    group_key = _pick_group_col(data, tidy, keys=("__group",)) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y"] + (["group"] if group_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    levels = [str(_fmt_level(level)) for level in
              (_levels_of(data.get("tidy"), "__x", max_levels=20) if isinstance(data.get("tidy"), pd.DataFrame) else [])]
    group_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), group_key, max_levels=12)]
                    if group_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}]).copy()".replace("'", '"'),
        f"LEVELS = {levels!r}".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, group_levels if group_levels else levels)}",
        f"NORMALIZE, AGG, SHOW_TOTAL = {bool(normalize)!r}, {agg!r}, {bool(show_total)!r}".replace("'", '"'),
        _py_comparison_axes(style),
        "if GROUP_LEVELS:",
        "    matrix = np.zeros((len(GROUP_LEVELS), len(LEVELS)))",
        "    for gi, group in enumerate(GROUP_LEVELS):",
        "        for ci, level in enumerate(LEVELS):",
        "            values = pd.to_numeric(df[(df[X] == level) & (df[G] == group)][Y],",
        "                                   errors=\"coerce\").dropna()",
        "            matrix[gi, ci] = values.sum() if AGG == \"sum\" else values.mean() if len(values) else 0.0",
        "else:",
        "    matrix = np.zeros((1, len(LEVELS)))",
        "    for ci, level in enumerate(LEVELS):",
        "        values = pd.to_numeric(df[df[X] == level][Y], errors=\"coerce\").dropna()",
        "        matrix[0, ci] = values.sum() if AGG == \"sum\" else values.mean() if len(values) else 0.0",
        "if NORMALIZE:",
        "    totals = np.nansum(matrix, axis=0)",
        "    matrix = matrix / np.where(totals > 1e-12, totals, 1.0) * 100.0",
        "bottom = np.zeros(len(LEVELS))",
        "for gi, group in enumerate(GROUP_LEVELS or [None]):",
        "    label = group if group is not None else None",
        "    color = COLORS[group] if group is not None else PALETTE[0]",
        "    ax.bar(np.arange(1, len(LEVELS) + 1), matrix[gi], bottom=bottom, width=BAR_WIDTH,",
        "           color=color, edgecolor=\"white\", linewidth=EDGE * 0.6, alpha=ALPHA, label=label)",
        "    bottom = bottom + np.nan_to_num(matrix[gi])",
        "if SHOW_TOTAL:",
        "    for index, total in enumerate(bottom):",
        "        ax.text(index + 1, total, f\"{total:.2f}\", ha=\"center\", va=\"bottom\",",
        "                fontsize=max(6.0, TICK_SIZE - 1))",
        "ax.set_xticks(np.arange(1, len(LEVELS) + 1))",
        "ax.set_xticklabels(LEVELS)",
        "ax.set_xlim(0.35, len(LEVELS) + 0.65)",
        f"X_LABEL = {str(labels.get('x') or 'x')!r}".replace("'", '"'),
        "Y_LABEL = \"Percentage (%)\" if NORMALIZE else %r" % str(labels.get("y") or "y"),
        "TITLE = \"Stacked bar: \" + Y_LABEL + \" by \" + X_LABEL",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _py_pie(data: Mapping[str, Any], params: Mapping[str, Any],
            style: Mapping[str, Any]) -> str:
    donut = _pbool(params, "donut", True)
    show_pct = _pbool(params, "show_pct", True)
    start_angle = _pfloat(params, "start_angle", 90.0, 0.0, 360.0)
    tidy = data.get("tidy")
    y_present = isinstance(tidy, pd.DataFrame) and "__y" in tidy.columns
    roles = ["x"] + (["y"] if y_present else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    levels = [str(_fmt_level(level)) for level in
              (_levels_of(data.get("tidy"), "__x", max_levels=16) if isinstance(data.get("tidy"), pd.DataFrame) else [])]
    body = [
        f"df = df.dropna(subset=[{names['x']!r}]).copy()".replace("'", '"'),
        f"LEVELS = {levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, levels)}",
        f"DONUT, SHOW_PCT, START_ANGLE = {bool(donut)!r}, {bool(show_pct)!r}, {float(start_angle)}".replace("'", '"'),
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        f"VALUE = {names.get('y')!r}".replace("'", '"'),
        "values = []",
        "for level in LEVELS:",
        "    cell = df[df[X] == level]",
        "    if VALUE is not None and pd.to_numeric(cell[VALUE], errors=\"coerce\").notna().sum() >= max(3, len(cell) // 2):",
        "        values.append(float(pd.to_numeric(cell[VALUE], errors=\"coerce\").dropna().sum()))",
        "    else:",
        "        values.append(float(len(cell)))",
        "if min(values) < 0:",
        "    values = [float(len(df[df[X] == level])) for level in LEVELS]",
        "autopct = (lambda pct: f\"{pct:.1f}%\") if SHOW_PCT else None",
        "pie_result = ax.pie(values, colors=[COLORS[level] for level in LEVELS],",
        "                        startangle=START_ANGLE, counterclock=False, autopct=autopct,",
        "                        pctdistance=0.78 if DONUT else 0.62,",
        "                        wedgeprops={\"width\": 0.42 if DONUT else 1.0, \"edgecolor\": \"white\",",
        "                                    \"linewidth\": max(0.5, EDGE)})",
        "ax.set_aspect(\"equal\")",
        "total = float(np.sum(values))",
        "for level, value in zip(LEVELS, values):",
        "    ax.plot([], [], marker=\"s\", linestyle=\"none\", markersize=9,",
        "            markerfacecolor=COLORS[level], markeredgecolor=\"none\",",
        "            label=f\"{level} ({value / total * 100:.1f}%)\")",
        f"X_LABEL = {str(labels.get('x') or 'x')!r}".replace("'", '"'),
        "Y_LABEL = \"\"",
        f"TITLE = \"{'Donut' if donut else 'Pie'} chart: composition of \" + X_LABEL",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _code_args_x_path(data: Mapping[str, Any], tidy: pd.DataFrame,
                     names: Mapping[str, str], x_label: str) -> list[str]:
    return [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}]).copy()".replace("'", '"'),
    ]


def _py_scatter(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> str:
    add_fit = _pbool(params, "add_fit", True)
    add_ci = _pbool(params, "add_ci", True)
    add_corr = _pbool(params, "add_corr", True)
    method = _pchoice(params, "method", "pearson", ["pearson", "spearman", "kendall"])
    x_bins = _pint(params, "x_bins", 0, 0, 30)
    tidy = data.get("tidy")
    color_key = _pick_group_col(data, tidy, keys=("__color", "__group")) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y"] + (["color"] if color_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    color_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), color_key, max_levels=10)]
                    if color_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}]).copy()".replace("'", '"'),
        f"COLOR_LEVELS = {color_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, color_levels or ['x'])}",
        f"ADD_FIT, ADD_CI, ADD_CORR = {bool(add_fit)!r}, {bool(add_ci)!r}, {bool(add_corr)!r}".replace("'", '"'),
        f"METHOD, X_BINS = {method!r}, {int(x_bins)}".replace("'", '"'),
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "if COLOR_LEVELS:",
        "    for level in COLOR_LEVELS:",
        "        cell = df[df[C] == level]",
        "        ax.scatter(cell[X], cell[Y], s=POINT, color=COLORS[level], alpha=ALPHA,",
        "                   edgecolors=\"none\", label=level)",
        "else:",
        "    ax.scatter(df[X], df[Y], s=POINT, color=PALETTE[0], alpha=ALPHA, edgecolors=\"none\")",
        "if ADD_FIT:",
        "    if COLOR_LEVELS and len(COLOR_LEVELS) <= 4:",
        "        for level in COLOR_LEVELS:",
        "            cell = df[df[C] == level]",
        "            slope, intercept, grid, y_hat, band = _fit_line(cell[X], cell[Y])",
        "            ax.plot(grid, y_hat, color=COLORS[level], linewidth=LW)",
        "    else:",
        "        slope, intercept, grid, y_hat, band = _fit_line(df[X], df[Y])",
        "        if ADD_CI and band is not None:",
        "            ax.fill_between(grid, band[0], band[1], color=\"#333333\", alpha=0.15, linewidth=0)",
        "        ax.plot(grid, y_hat, color=\"#333333\", linewidth=LW, label=\"Linear fit\")",
        "if ADD_CORR and len(df) >= 3:",
        "    if METHOD == \"spearman\":",
        "        r, p = stats.spearmanr(df[X], df[Y])",
        "    elif METHOD == \"kendall\":",
        "        r, p = stats.kendalltau(df[X], df[Y])",
        "    else:",
        "        r, p = stats.pearsonr(df[X], df[Y])",
        "    label = {\"pearson\": \"Pearson r\", \"spearman\": \"Spearman rho\", \"kendall\": \"Kendall tau\"}[METHOD]",
        "    ax.text(0.02, 0.98, f\"{label} = {r:.3f}, p = {p:.2g}, n = {len(df)}\",",
        "            transform=ax.transAxes, ha=\"left\", va=\"top\", fontsize=max(6.5, TICK_SIZE - 1),",
        "            bbox={\"boxstyle\": \"round,pad=0.28\", \"facecolor\": \"white\",",
        "                  \"edgecolor\": \"#CCCCCC\", \"alpha\": 0.85, \"linewidth\": 0.6})",
        "if X_BINS >= 2:",
        "    edges = np.linspace(float(df[X].min()), float(df[X].max()), X_BINS + 1)",
        "    centers, means, errors = [], [], []",
        "    for index in range(X_BINS):",
        "        mask = (df[X] >= edges[index]) & (df[X] <= edges[index + 1])",
        "        if index < X_BINS - 1:",
        "            mask = mask & (df[X] < edges[index + 1])",
        "        values = pd.to_numeric(df.loc[mask, Y], errors=\"coerce\").dropna()",
        "        if len(values) >= 2:",
        "            centers.append((edges[index] + edges[index + 1]) / 2.0)",
        "            means.append(float(values.mean()))",
        "            errors.append(float(values.std(ddof=1) / np.sqrt(len(values))))",
        "    if centers:",
        "        ax.errorbar(centers, means, yerr=errors, fmt=\"s\", color=\"#B24745\",",
        "                    markersize=5, capsize=2.5, linewidth=0.8, label=\"Binned mean\")",
        f"X_LABEL, Y_LABEL = {str(labels.get('x') or 'x')!r}, {str(labels.get('y') or 'y')!r}".replace("'", '"'),
        "TITLE = \"Scatter plot: \" + Y_LABEL + \" vs \" + X_LABEL",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style, ("fit",)), block, "\n".join(body))


def _py_bubble(data: Mapping[str, Any], params: Mapping[str, Any],
               style: Mapping[str, Any]) -> str:
    add_fit = _pbool(params, "add_fit", False)
    size_min = _pfloat(params, "size_min", 20.0, 4.0, 1000.0)
    size_max = _pfloat(params, "size_max", 420.0, 20.0, 5000.0)
    tidy = data.get("tidy")
    color_key = _pick_group_col(data, tidy, keys=("__color", "__group")) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y", "size"] + (["color"] if color_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    color_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), color_key, max_levels=10)]
                    if color_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}, {names['size']!r}]).copy()".replace("'", '"'),
        f"COLOR_LEVELS = {color_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, color_levels or ['x'])}",
        f"ADD_FIT, SIZE_MIN, SIZE_MAX = {bool(add_fit)!r}, {float(size_min)}, {float(size_max)}".replace("'", '"'),
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "sizes = pd.to_numeric(df[S], errors=\"coerce\")",
        "span = float(sizes.max() - sizes.min()) or 1.0",
        "areas = SIZE_MIN + (sizes - sizes.min()) / span * (SIZE_MAX - SIZE_MIN)",
        "if COLOR_LEVELS:",
        "    for level in COLOR_LEVELS:",
        "        mask = df[C] == level",
        "        ax.scatter(df.loc[mask, X], df.loc[mask, Y], s=areas[mask], color=COLORS[level],",
        "                   alpha=ALPHA, edgecolors=\"#2B2B2B\", linewidths=EDGE * 0.7, label=level)",
        "else:",
        "    ax.scatter(df[X], df[Y], s=areas, color=PALETTE[0], alpha=ALPHA,",
        "               edgecolors=\"#2B2B2B\", linewidths=EDGE * 0.7)",
        "if ADD_FIT:",
        "    slope, intercept, grid, y_hat, band = _fit_line(df[X], df[Y])",
        "    ax.plot(grid, y_hat, color=\"#333333\", linewidth=LW, label=\"Linear fit\")",
        f"for quantile, name in ((0.0, \"min: \"), (0.5, \"median: \"), (1.0, \"max: \")):"
        if False else "for quantile in (0.0, 0.5, 1.0):",
        "    value = float(sizes.quantile(quantile))",
        "    area = SIZE_MIN + (value - float(sizes.min())) / span * (SIZE_MAX - SIZE_MIN)",
        "    ax.plot([], [], marker=\"o\", linestyle=\"none\", markersize=np.sqrt(area) * 0.9,",
        "            markerfacecolor=\"#7F7F7F\", markeredgecolor=\"none\", label=f\"{value:.3g}\")",
        f"X_LABEL, Y_LABEL = {str(labels.get('x') or 'x')!r}, {str(labels.get('y') or 'y')!r}".replace("'", '"'),
        "TITLE = \"Bubble plot: \" + Y_LABEL + \" vs \" + X_LABEL",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style, ("fit",)), block, "\n".join(body))


def _py_line(data: Mapping[str, Any], params: Mapping[str, Any],
             style: Mapping[str, Any]) -> str:
    agg = _pchoice(params, "agg", "mean", ["mean", "median", "sum"])
    error = _pchoice(params, "error", "auto", ["auto", "sd", "se", "ci95", "none"])
    if error == "auto":
        error = str(style.get("error") or "se")
    marker = _pbool(params, "marker", True)
    tidy = data.get("tidy")
    group_key = _pick_group_col(data, tidy, keys=("__group",)) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y"] + (["group"] if group_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    group_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), group_key, max_levels=12)]
                    if group_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}]).copy()".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, group_levels or ['x'])}",
        f"AGG, ERROR, MARKER = {agg!r}, {error!r}, {bool(marker)!r}".replace("'", '"'),
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "x_ticks = None",
        "for gi, group in enumerate(GROUP_LEVELS or [None]):",
        "    cell = df if group is None else df[df[G] == group]",
        "    grouped = cell.groupby(X, observed=True)[Y]",
        "    center = grouped.agg(AGG)",
        "    count = grouped.count()",
        "    sd = grouped.std(ddof=1).fillna(0.0)",
        "    if ERROR == \"none\":",
        "        err = count * 0.0",
        "    elif ERROR == \"sd\":",
        "        err = sd",
        "    elif ERROR == \"se\":",
        "        err = sd / np.sqrt(count)",
        "    else:",
        "        err = 1.96 * sd / np.sqrt(count)",
        "    x_values = np.arange(len(center))",
        "    x_ticks = [str(value) for value in center.index]",
        "    color = COLORS[group] if group is not None else PALETTE[0]",
        "    label = group if group is not None else str(Y)",
        "    ax.plot(x_values, center.values, color=color, linewidth=LW, alpha=ALPHA, label=label,",
        "            marker=\"o\" if MARKER else None, markersize=max(3.0, np.sqrt(POINT) * 0.7),",
        "            markeredgecolor=\"white\")",
        "    if ERROR != \"none\":",
        "        ax.fill_between(x_values, center.values - err.values, center.values + err.values,",
        "                        color=color, alpha=ALPHA * 0.18, linewidth=0)",
        "if x_ticks:",
        "    ax.set_xticks(range(len(x_ticks)))",
        "    ax.set_xticklabels(x_ticks)",
        f"X_LABEL, Y_LABEL = {str(labels.get('x') or 'x')!r}, {str(labels.get('y') or 'y')!r}".replace("'", '"'),
        "TITLE = \"Line chart: \" + AGG + \" \" + Y_LABEL + \" vs \" + X_LABEL",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _py_area(data: Mapping[str, Any], params: Mapping[str, Any],
             style: Mapping[str, Any]) -> str:
    agg = _pchoice(params, "agg", "mean", ["mean", "median", "sum"])
    stack = _pbool(params, "stack", True)
    normalize = _pbool(params, "normalize", False)
    tidy = data.get("tidy")
    group_key = _pick_group_col(data, tidy, keys=("__group",)) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x", "y"] + (["group"] if group_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    group_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), group_key, max_levels=10)]
                    if group_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}, {names['y']!r}]).copy()".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, group_levels or ['x'])}",
        f"STACK, NORMALIZE, AGG = {bool(stack)!r}, {bool(normalize)!r}, {agg!r}".replace("'", '"'),
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "if GROUP_LEVELS:",
        "    pivot = df.pivot_table(index=X, columns=G, values=Y, aggfunc=AGG)",
        "    pivot = pivot.sort_index()",
        "    x_values = np.arange(len(pivot.index))",
        "    matrix = pivot.to_numpy(dtype=float).T",
        "else:",
        "    grouped = df.groupby(X, observed=True)[Y].agg(AGG).sort_index()",
        "    x_values = np.arange(len(grouped.index))",
        "    matrix = grouped.to_numpy(dtype=float).reshape(1, -1)",
        "matrix = np.nan_to_num(matrix, nan=0.0)",
        "if NORMALIZE:",
        "    totals = matrix.sum(axis=0)",
        "    matrix = matrix / np.where(totals > 1e-12, totals, 1.0) * 100.0",
        "names = GROUP_LEVELS or [Y]",
        "colors = [COLORS.get(name, PALETTE[index % len(PALETTE)]) for index, name in enumerate(names)]",
        "if STACK and len(names) > 1:",
        "    ax.stackplot(x_values, matrix, colors=colors, alpha=ALPHA, labels=names,",
        "                 edgecolor=\"white\", linewidth=0.4)",
        "elif STACK:",
        "    ax.fill_between(x_values, 0, matrix[0], color=colors[0], alpha=ALPHA * 0.7,",
        "                    linewidth=0, label=names[0])",
        "    ax.plot(x_values, matrix[0], color=colors[0], linewidth=LW)",
        "else:",
        "    for index, name in enumerate(names):",
        "        ax.fill_between(x_values, 0, matrix[index], color=colors[index],",
        "                        alpha=ALPHA * 0.7, linewidth=0, label=name)",
        "        ax.plot(x_values, matrix[index], color=colors[index], linewidth=LW)",
        "x_index = pivot.index if GROUP_LEVELS else grouped.index",
        "x_labels = [str(value) for value in x_index]",
        "ax.set_xticks(x_values)",
        "ax.set_xticklabels(x_labels)",
        "ax.set_ylim(bottom=0)",
        f"X_LABEL = {str(labels.get('x') or 'x')!r}".replace("'", '"'),
        "Y_LABEL = \"Percentage (%)\" if NORMALIZE else %r" % str(labels.get('y') or 'y'),
        "TITLE = \"Area chart: \" + Y_LABEL + \" vs \" + X_LABEL",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _py_heatmap(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> str:
    zscore = _pchoice(params, "zscore", "row", ["none", "row", "col"])
    cluster_rows = _pbool(params, "cluster_rows", True)
    cluster_cols = _pbool(params, "cluster_cols", True)
    annotate = _pbool(params, "annotate", False)
    block = _py_wide_block(data)
    wide = data.get("wide") or {}
    groups = wide.get("sample_groups")
    group_labels = [str(value) for value in list(groups)] if groups is not None else []
    group_levels = list(dict.fromkeys(group_labels))
    cmap = _cmap_name(style, params.get("cmap"), diverging=(zscore != "none"))
    body = [
        "# Z 的每一行=一个特征、每一列=一个样本",
        "Z = np.asarray(matrix, dtype=float).T.copy()",
        f"ZSCORE = {zscore!r}".replace("'", '"'),
        f"ANNOTATE = {bool(annotate)!r}",
        "if ZSCORE == \"row\":",
        "    mean = np.nanmean(Z, axis=1, keepdims=True)",
        "    sd = np.nanstd(Z, axis=1, ddof=0, keepdims=True)",
        "    Z = (Z - mean) / np.where(sd > 1e-12, sd, 1.0)",
        "elif ZSCORE == \"col\":",
        "    mean = np.nanmean(Z, axis=0, keepdims=True)",
        "    sd = np.nanstd(Z, axis=0, ddof=0, keepdims=True)",
        "    Z = (Z - mean) / np.where(sd > 1e-12, sd, 1.0)",
        f"if {bool(cluster_rows)} and Z.shape[0] >= 3:",
        "    from scipy.cluster.hierarchy import leaves_list, linkage",
        "    order = leaves_list(linkage(Z, method=\"average\"))",
        "    Z = Z[order]",
        "    features = [features[index] for index in order]",
        f"if {bool(cluster_cols)} and Z.shape[1] >= 3:",
        "    from scipy.cluster.hierarchy import leaves_list, linkage",
        "    order = leaves_list(linkage(Z.T, method=\"average\"))",
        "    Z = Z[:, order]",
        "    samples = [samples[index] for index in order]",
        "    if sample_groups is not None:",
        "        sample_groups = [sample_groups[index] for index in order]",
        "finite = Z[np.isfinite(Z)]",
        "if ZSCORE == \"none\":",
        "    vmin, vmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))",
        "    color_label = \"value\"",
        "else:",
        "    limit = max(1.5, min(3.0, float(np.nanpercentile(np.abs(Z), 98))))",
        "    vmin, vmax = -limit, limit",
        "    color_label = \"z-score\"",
        f"CMAP = matplotlib.colormaps[\"{cmap}\"]",
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"GROUP_COLORS = {_py_colors_repr(style, group_levels) if group_levels else '{}'}",
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "image = ax.imshow(Z, aspect=\"auto\", cmap=CMAP, vmin=vmin, vmax=vmax,",
        "                  interpolation=\"nearest\")",
        "ax.set_xticks(np.arange(len(samples)))",
        "ax.set_xticklabels(samples, fontsize=max(5.0, TICK_SIZE - 2),",
        "                   rotation=90 if len(samples) > 8 else 0)",
        "if len(samples) > 8:",
        "    plt.setp(ax.get_xticklabels(), ha=\"right\")",
        "ax.set_yticks(np.arange(len(features)))",
        "ax.set_yticklabels(features, fontsize=max(5.0, TICK_SIZE - 1))",
        "if sample_groups is not None and len(sample_groups) == len(samples):",
        "    for label, group in zip(ax.get_xticklabels(), sample_groups):",
        "        label.set_color(GROUP_COLORS.get(str(group), \"#333333\"))",
        "for group in GROUP_LEVELS:",
        "    ax.plot([], [], marker=\"s\", linestyle=\"none\", markersize=9,",
        "            markerfacecolor=GROUP_COLORS[group], markeredgecolor=\"none\", label=group)",
        "if ANNOTATE and Z.size <= 400:",
        "    norm = plt.Normalize(vmin, vmax)",
        "    font = max(4.5, min(8.0, 9.0 - 0.06 * max(Z.shape)))",
        "    for ri in range(Z.shape[0]):",
        "        for ci in range(Z.shape[1]):",
        "            if np.isfinite(Z[ri, ci]):",
        "                rgba = CMAP(norm(Z[ri, ci]))",
        "                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]",
        "                ax.text(ci, ri, f\"{Z[ri, ci]:.2f}\", ha=\"center\", va=\"center\",",
        "                        fontsize=font, color=\"white\" if lum < 0.55 else \"#1A1A1A\")",
        "plt.colorbar(image, ax=ax, shrink=0.86, pad=0.02, label=color_label)",
        f"X_LABEL, Y_LABEL = \"Samples\", \"Features\"".replace("'", '"'),
        "TITLE = f\"Heatmap: {len(features)} features x {len(samples)} samples\"",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _py_corr(data: Mapping[str, Any], params: Mapping[str, Any],
             style: Mapping[str, Any]) -> str:
    method = _pchoice(params, "method", "pearson", ["pearson", "spearman", "kendall"])
    show_star = _pbool(params, "star", True)
    annotate = _pbool(params, "annotate", True)
    block = _py_wide_block(data)
    body = [
        f"METHOD, SHOW_STAR, ANNOTATE = {method!r}, {bool(show_star)!r}, {bool(annotate)!r}".replace("'", '"'),
        "corr = pd.DataFrame(matrix, columns=features).corr(method=METHOD).to_numpy()",
        "mask = np.isfinite(matrix)",
        "counts = mask.T.astype(float) @ mask.astype(float)",
        "pmatrix = np.ones_like(corr)",
        "for i in range(len(features)):",
        "    for j in range(len(features)):",
        "        r = corr[i, j]",
        "        n = counts[i, j]",
        "        if i == j or not np.isfinite(r) or n < 3:",
        "            pmatrix[i, j] = 0.0 if i == j else np.nan",
        "            continue",
        "        if abs(r) >= 1:",
        "            pmatrix[i, j] = 0.0",
        "        else:",
        "            t_value = abs(r) * np.sqrt((n - 2) / (1 - r * r))",
        "            pmatrix[i, j] = 2 * stats.t.sf(t_value, n - 2)",
        f"CMAP = matplotlib.colormaps[\"{_cmap_name(style, '', diverging=True)}\"]",
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "image = ax.imshow(corr, vmin=-1, vmax=1, cmap=CMAP, aspect=\"equal\")",
        "rotation = 45 if len(features) > 6 else 0",
        "ax.set_xticks(np.arange(len(features)))",
        "ax.set_yticks(np.arange(len(features)))",
        "ax.set_xticklabels(features, fontsize=max(5.0, TICK_SIZE - 2), rotation=rotation,",
        "                   ha=\"right\" if rotation else \"center\")",
        "ax.set_yticklabels(features, fontsize=max(5.0, TICK_SIZE - 1))",
        "if ANNOTATE and len(features) <= 25:",
        "    norm = plt.Normalize(-1, 1)",
        "    font = max(4.5, min(8.5, 10.5 - 0.28 * len(features)))",
        "    for i in range(len(features)):",
        "        for j in range(len(features)):",
        "            if not np.isfinite(corr[i, j]):",
        "                continue",
        "            text = f\"{corr[i, j]:.2f}\"",
        "            if SHOW_STAR and i != j and np.isfinite(pmatrix[i, j]):",
        "                p = pmatrix[i, j]",
        "                text += \"***\" if p < 0.001 else \"**\" if p < 0.01 else \"*\" if p < 0.05 else \"\"",
        "            rgba = CMAP(norm(corr[i, j]))",
        "            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]",
        "            ax.text(j, i, text, ha=\"center\", va=\"center\", fontsize=font,",
        "                    color=\"white\" if lum < 0.55 else \"#1A1A1A\")",
        "plt.colorbar(image, ax=ax, shrink=0.86, pad=0.02, label=METHOD.capitalize() + \" correlation\")",
        f"X_LABEL, Y_LABEL = \"\", \"\"".replace("'", '"'),
        "TITLE = f\"Correlation matrix: {len(features)} variables ({METHOD})\"",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _py_pairplot(data: Mapping[str, Any], params: Mapping[str, Any],
                 style: Mapping[str, Any]) -> str:
    block = _py_wide_block(data)
    requested = _split_terms(params.get("vars") or "")
    wide = data.get("wide") or {}
    features = [str(value) for value in (wide.get("features") or [])]
    keep = [index for index, name in enumerate(features) if name in requested] if requested else []
    selected = [features[index] for index in keep] if keep else features[:6]
    body = [
        f"SELECTED = {selected!r}".replace("'", '"'),
        "index = [features.index(name) for name in SELECTED]",
        "matrix = matrix[:, index]",
        "features = SELECTED",
        f"DIAG, SHOW_R = {_pchoice(params, 'diag', 'hist', ['hist', 'kde'])!r}, {bool(_pbool(params, 'show_r', False))!r}".replace("'", '"'),
        "df = pd.DataFrame(matrix, columns=features)",
        "if sample_groups is not None:",
        "    df[\"__group__\"] = [str(value) for value in sample_groups]",
        "group_levels = list(dict.fromkeys(df[\"__group__\"])) if \"__group__\" in df else []",
        f"GROUP_COLORS = {_py_colors_repr(style, list(dict.fromkeys([str(v) for v in (wide.get('sample_groups') or [])])))}",
        "n = len(features)",
        "side = max(FIG_W, 1.5 * n + 1.8)",
        "fig, axes = plt.subplots(n, n, figsize=(side, side), layout=\"constrained\")",
        "for i in range(n):",
        "    for j in range(n):",
        "        ax = axes[i, j]",
        "        if i == j:",
        "            if DIAG == \"kde\" and group_levels:",
        "                for group in group_levels:",
        "                    values = df.loc[df[\"__group__\"] == group, features[i]].dropna().to_numpy(float)",
        "                    if len(values) < 3:",
        "                        continue",
        "                    pad = max((values.max() - values.min()) * 0.1, 1e-9)",
        "                    grid = np.linspace(values.min() - pad, values.max() + pad, 160)",
        "                    density = _kde(values, grid)",
        "                    if density is not None:",
        "                        ax.plot(grid, density, color=GROUP_COLORS.get(group, PALETTE[0]),",
        "                                linewidth=LW, alpha=ALPHA)",
        "            elif group_levels:",
        "                for group in group_levels:",
        "                    values = df.loc[df[\"__group__\"] == group, features[i]].dropna()",
        "                    ax.hist(values, bins=14, alpha=0.55,",
        "                            color=GROUP_COLORS.get(group, PALETTE[0]))",
        "            else:",
        "                values = df[features[i]].dropna()",
        "                if DIAG == \"kde\" and len(values) >= 3:",
        "                    pad = max((values.max() - values.min()) * 0.1, 1e-9)",
        "                    grid = np.linspace(values.min() - pad, values.max() + pad, 160)",
        "                    density = _kde(values.to_numpy(float), grid)",
        "                    if density is not None:",
        "                        ax.plot(grid, density, color=PALETTE[0], linewidth=LW)",
        "                    else:",
        "                        ax.hist(values, bins=14, color=PALETTE[0], alpha=ALPHA)",
        "                else:",
        "                    ax.hist(values, bins=14, color=PALETTE[0], alpha=ALPHA)",
        "        else:",
        "            if group_levels:",
        "                for group in group_levels:",
        "                    cell = df[df[\"__group__\"] == group]",
        "                    ax.scatter(cell[features[j]], cell[features[i]], s=POINT * 0.45,",
        "                               color=GROUP_COLORS.get(group, PALETTE[0]), alpha=ALPHA,",
        "                               edgecolors=\"none\")",
        "            else:",
        "                ax.scatter(df[features[j]], df[features[i]], s=POINT * 0.45,",
        "                           color=PALETTE[0], alpha=ALPHA, edgecolors=\"none\")",
        "            if SHOW_R:",
        "                r = df[features[j]].corr(df[features[i]])",
        "                ax.text(0.96, 0.06, f\"r = {r:.2f}\", transform=ax.transAxes,",
        "                        ha=\"right\", va=\"bottom\", fontsize=max(5.0, TICK_SIZE - 3))",
        "        ax.tick_params(labelsize=max(5.0, TICK_SIZE - 2), length=2.5)",
        "        if j == 0 and i != 0:",
        "            ax.set_ylabel(features[i], fontsize=max(6.0, LABEL_SIZE - 2))",
        "        if i == n - 1:",
        "            ax.set_xlabel(features[j], fontsize=max(6.0, LABEL_SIZE - 2))",
        "for group in group_levels:",
        "    axes[0, 0].plot([], [], marker=\"s\", linestyle=\"none\", markersize=9,",
        "                    markerfacecolor=GROUP_COLORS.get(group, PALETTE[0]),",
        "                    markeredgecolor=\"none\", label=group)",
        "handles, labels_ = axes[0, 0].get_legend_handles_labels()",
        "if handles:",
        "    axes[0, 0].legend(handles, labels_, loc=\"upper right\", fontsize=max(6.0, LEGEND_SIZE - 2),",
        "                      frameon=False)",
        "fig.suptitle(f\"Scatter matrix: {n} variables\", fontsize=TITLE_SIZE,",
        "             fontweight=\"bold\" if TITLE_BOLD else \"normal\")",
    ]
    return _py_join(_py_prelude(style, ("kde",)), block, "\n".join(body))


def _py_hist(data: Mapping[str, Any], params: Mapping[str, Any],
             style: Mapping[str, Any]) -> str:
    bins = _pint(params, "bins", 20, 0, 200)
    density = _pbool(params, "density", False)
    stacked = _pbool(params, "stack", False)
    show_stats = _pbool(params, "show_stats", True)
    tidy = data.get("tidy")
    color_key = _pick_group_col(data, tidy, keys=("__color", "__group")) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x"] + (["color"] if color_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    color_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), color_key, max_levels=10)]
                    if color_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}]).copy()".replace("'", '"'),
        f"COLOR_LEVELS = {color_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, color_levels or ['x'])}",
        f"BINS = {(int(bins) if bins > 0 else 'auto')!r}".replace("'", '"'),
        f"DENSITY, STACKED, SHOW_STATS = {bool(density)!r}, {bool(stacked)!r}, {bool(show_stats)!r}".replace("'", '"'),
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "if COLOR_LEVELS:",
        "    arrays = [pd.to_numeric(df[df[C] == level][X], errors=\"coerce\").dropna().to_numpy()",
        "              for level in COLOR_LEVELS]",
        "    colors = [COLORS[level] for level in COLOR_LEVELS]",
        "    if STACKED:",
        "        ax.hist(arrays, bins=BINS, stacked=True, density=DENSITY, color=colors,",
        "                alpha=ALPHA, edgecolor=\"white\", linewidth=EDGE * 0.6, label=COLOR_LEVELS)",
        "    else:",
        "        for level, values, color in zip(COLOR_LEVELS, arrays, colors):",
        "            ax.hist(values, bins=BINS, density=DENSITY, histtype=\"stepfilled\", color=color,",
        "                    alpha=ALPHA * 0.62, edgecolor=\"white\", linewidth=EDGE * 0.6, label=level)",
        "else:",
        "    ax.hist(pd.to_numeric(df[X], errors=\"coerce\").dropna(), bins=BINS, density=DENSITY,",
        "            color=PALETTE[0], alpha=ALPHA, edgecolor=\"white\", linewidth=EDGE * 0.6)",
        "values = pd.to_numeric(df[X], errors=\"coerce\").dropna().to_numpy()",
        "if SHOW_STATS and len(values):",
        "    mean_value, median_value = float(np.mean(values)), float(np.median(values))",
        "    ax.axvline(mean_value, color=\"#B24745\", linestyle=\"--\", linewidth=LW * 0.8)",
        "    ax.axvline(median_value, color=\"#0072B2\", linestyle=\":\", linewidth=LW * 0.9)",
        "    ax.text(0.98, 0.96, f\"mean = {mean_value:.3g}\\nmedian = {median_value:.3g}\\n\"",
        "                        f\"SD = {np.std(values, ddof=1):.3g}\\nn = {len(values)}\",",
        "            transform=ax.transAxes, ha=\"right\", va=\"top\", fontsize=max(6.0, TICK_SIZE - 1),",
        "            bbox={\"boxstyle\": \"round,pad=0.25\", \"facecolor\": \"white\",",
        "                  \"edgecolor\": \"#CCCCCC\", \"alpha\": 0.85, \"linewidth\": 0.6})",
        f"X_LABEL = {str(labels.get('x') or 'x')!r}".replace("'", '"'),
        "Y_LABEL = \"Density\" if DENSITY else \"Count\"",
        "TITLE = \"Histogram: \" + X_LABEL",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _py_density(data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> str:
    fill = _pbool(params, "fill", True)
    bw = _pfloat(params, "bw", 1.0, 0.2, 4.0)
    rug = _pbool(params, "rug", False)
    tidy = data.get("tidy")
    color_key = _pick_group_col(data, tidy, keys=("__color", "__group")) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x"] + (["color"] if color_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    color_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), color_key, max_levels=10)]
                    if color_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}]).copy()".replace("'", '"'),
        f"COLOR_LEVELS = {color_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, color_levels or ['x'])}",
        f"FILL, BW, RUG = {bool(fill)!r}, {float(bw)}, {bool(rug)!r}".replace("'", '"'),
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "all_values = pd.to_numeric(df[X], errors=\"coerce\").dropna().to_numpy()",
        "pad = max((all_values.max() - all_values.min()) * 0.12, 1e-9)",
        "grid = np.linspace(all_values.min() - pad, all_values.max() + pad, 200)",
        "for gi, group in enumerate(COLOR_LEVELS or [None]):",
        "    cell = df if group is None else df[df[C] == group]",
        "    values = pd.to_numeric(cell[X], errors=\"coerce\").dropna().to_numpy()",
        "    density = _kde(values, grid, BW)",
        "    if density is None:",
        "        continue",
        "    color = COLORS[group] if group is not None else PALETTE[0]",
        "    label = group if group is not None else str(X)",
        "    if FILL:",
        "        ax.fill_between(grid, 0, density, color=color, alpha=ALPHA * 0.35, linewidth=0)",
        "    ax.plot(grid, density, color=color, linewidth=LW, label=label)",
        "    if RUG:",
        "        ax.plot(values, np.zeros(len(values)), marker=\"|\", linestyle=\"none\",",
        "                markersize=5, color=color, alpha=0.5)",
        f"X_LABEL = {str(labels.get('x') or 'x')!r}".replace("'", '"'),
        "Y_LABEL = \"Density\"",
        "TITLE = \"Density plot: \" + X_LABEL",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style, ("kde",)), block, "\n".join(body))


def _py_qq(data: Mapping[str, Any], params: Mapping[str, Any],
           style: Mapping[str, Any]) -> str:
    dist = _pchoice(params, "dist", "normal", ["normal", "uniform", "exponential"])
    scipy_dist = {"normal": "norm", "uniform": "uniform", "exponential": "expon"}[dist]
    tidy = data.get("tidy")
    color_key = _pick_group_col(data, tidy, keys=("__color", "__group")) if isinstance(tidy, pd.DataFrame) else None
    roles = ["x"] + (["color"] if color_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    color_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), color_key, max_levels=10)]
                    if color_key else [])
    body = [
        f"df = df.dropna(subset=[{names['x']!r}]).copy()".replace("'", '"'),
        f"COLOR_LEVELS = {color_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, color_levels or ['x'])}",
        f"DIST = \"{scipy_dist}\"",
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "for gi, group in enumerate(COLOR_LEVELS or [None]):",
        "    cell = df if group is None else df[df[C] == group]",
        "    values = pd.to_numeric(cell[X], errors=\"coerce\").dropna().to_numpy()",
        "    if len(values) < 3:",
        "        continue",
        "    color = COLORS[group] if group is not None else PALETTE[0]",
        "    label = group if group is not None else str(X)",
        "    (osm, osr), (slope, intercept, r_value) = stats.probplot(values, dist=DIST, fit=True)",
        "    ax.scatter(osm, osr, s=POINT * 0.8, color=color, alpha=ALPHA, edgecolors=\"none\",",
        "               label=label)",
        "    ax.plot(osm, slope * np.asarray(osm) + intercept, color=color, linewidth=LW)",
        "if DIST == \"norm\":",
        "    lines = [f\"Shapiro-Wilk p = {stats.shapiro(values)[1]:.3g}\"]",
        "    ax.text(0.02, 0.98, \"\\n\".join(lines), transform=ax.transAxes, ha=\"left\", va=\"top\",",
        "            fontsize=max(6.0, TICK_SIZE - 1),",
        "            bbox={\"boxstyle\": \"round,pad=0.28\", \"facecolor\": \"white\",",
        "                  \"edgecolor\": \"#CCCCCC\", \"alpha\": 0.85, \"linewidth\": 0.6})",
        "X_LABEL = f\"Theoretical quantiles ({DIST})\"",
        "Y_LABEL = \"Sample quantiles\"",
        f"TITLE = \"Q-Q plot: {str(labels.get('x') or 'x')} vs {dist} distribution\"".replace("'", '"'),
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _py_roc(data: Mapping[str, Any], params: Mapping[str, Any],
            style: Mapping[str, Any]) -> str:
    import warnings as _warnings
    warnings: list[str] = []
    frame = P.require_tidy(data, ["__y"])
    scores = pd.to_numeric(frame["__y"], errors="coerce").to_numpy(dtype=float)
    label_col = None
    for key in ("__group", "__color"):
        if key in frame.columns and frame[key].notna().any():
            label_col = key
            break
    if label_col is None:
        raise C.prerequisite("ROC 代码生成需要 group 角色（0/1 标签）")
    labels_raw = [_fmt_level(value) for value in frame[label_col].tolist()]
    work = pd.DataFrame({"score": scores, "label": labels_raw})
    work = work[np.isfinite(work["score"])].reset_index(drop=True)
    levels = list(dict.fromkeys(work["label"].tolist()))
    if len(levels) != 2:
        raise C.bad_request("ROC 需要恰好 2 个标签水平")
    positive = _ptext(params, "positive", "").strip() or levels[-1]
    y_true = (work["label"] == positive).astype(int).to_numpy()
    score_values = work["score"].to_numpy(dtype=float)
    add_ci = _pbool(params, "add_ci", True)
    ci = None
    if add_ci:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            ci = _bootstrap_auc(y_true, score_values, 200)
    labels = data.get("labels") or {}
    score_literal = _py_literal(pd.Series(score_values))
    true_literal = _py_literal(pd.Series(y_true))
    body = [
        f"SCORES = np.array({score_literal}, dtype=float)",
        f"Y_TRUE = np.array({true_literal}, dtype=int)  # 阳性水平：{positive}",
        f"CI = {ci!r}",
        f"POSITIVE = {positive!r}".replace("'", '"'),
        "order = np.argsort(-SCORES)",
        "y_sorted = Y_TRUE[order]",
        "score_sorted = SCORES[order]",
        "tps = np.cumsum(y_sorted)",
        "fps = np.cumsum(1 - y_sorted)",
        "tpr = np.concatenate([[0.0], tps / tps[-1]])",
        "fpr = np.concatenate([[0.0], fps / fps[-1]])",
        "auc_value = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))",
        "if auc_value < 0.5:",
        "    SCORES = -SCORES",
        "    order = np.argsort(-SCORES)",
        "    y_sorted = Y_TRUE[order]",
        "    tps = np.cumsum(y_sorted)",
        "    fps = np.cumsum(1 - y_sorted)",
        "    tpr = np.concatenate([[0.0], tps / tps[-1]])",
        "    fpr = np.concatenate([[0.0], fps / fps[-1]])",
        "    auc_value = float(np.trapz(tpr, fpr))",
        "youden = tpr - fpr",
        "best = int(np.argmax(youden))",
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "label = f\"ROC (AUC = {auc_value:.3f})\"",
        "if CI is not None:",
        "    label += f\", 95% CI {CI[0]:.3f}-{CI[1]:.3f}\"",
        "ax.plot(fpr, tpr, color=PALETTE[0], linewidth=LW + 0.4, label=label)",
        "ax.plot([0, 1], [0, 1], linestyle=\"--\", color=\"#9A9A9A\", linewidth=1.0)",
        "ax.scatter([fpr[best]], [tpr[best]], s=POINT * 1.4, color=\"#B24745\",",
        "           edgecolors=\"white\", linewidths=0.8, zorder=6)",
        "ax.set_xlim(-0.01, 1.01)",
        "ax.set_ylim(-0.01, 1.01)",
        "lines = [f\"AUC = {auc_value:.3f}\"]",
        "if CI is not None:",
        "    lines.append(f\"95% CI {CI[0]:.3f}-{CI[1]:.3f}\")",
        "lines.append(f\"n = {len(Y_TRUE)} (positive {int(Y_TRUE.sum())} / negative {int((1 - Y_TRUE).sum())})\")",
        "ax.text(0.98, 0.04, \"\\n\".join(lines), transform=ax.transAxes, ha=\"right\", va=\"bottom\",",
        "        fontsize=max(6.0, TICK_SIZE - 1),",
        "        bbox={\"boxstyle\": \"round,pad=0.3\", \"facecolor\": \"white\",",
        "              \"edgecolor\": \"#CCCCCC\", \"alpha\": 0.9, \"linewidth\": 0.6})",
        "X_LABEL = \"False positive rate (1 - specificity)\"",
        "Y_LABEL = \"True positive rate (sensitivity)\"",
        f"TITLE = \"ROC curve: {str(labels.get('y') or 'score')}\"",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), "df = None  # ROC 直接使用内联数组", "\n".join(body))


def _py_km(data: Mapping[str, Any], params: Mapping[str, Any],
           style: Mapping[str, Any]) -> str:
    show_ci = _pbool(params, "ci", True)
    show_censors = _pbool(params, "censor", True)
    risk_table = _pbool(params, "risk_table", False)
    tidy = data.get("tidy")
    group_key = _pick_group_col(data, tidy, keys=("__group", "__color")) if isinstance(tidy, pd.DataFrame) else None
    roles = ["time", "event"] + (["group"] if group_key else [])
    block, names = _py_tidy_block(data, roles)
    labels = data.get("labels") or {}
    group_levels = ([str(_fmt_level(level)) for level in _levels_of(data.get("tidy"), group_key, max_levels=10)]
                    if group_key else [])
    body = [
        f"df = df.dropna(subset=[{names['time']!r}, {names['event']!r}]).copy()".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"COLORS = {_py_colors_repr(style, group_levels or ['x'])}",
        f"SHOW_CI, SHOW_CENSORS, RISK_TABLE = {bool(show_ci)!r}, {bool(show_censors)!r}, {bool(risk_table)!r}".replace("'", '"'),
        "from lifelines import KaplanMeierFitter",
        "df[T] = pd.to_numeric(df[T], errors=\"coerce\")",
        "df[E] = (pd.to_numeric(df[E], errors=\"coerce\") > 0).astype(int)",
        "df = df[df[T] >= 0]",
        "if RISK_TABLE:",
        "    fig, (ax, ax_risk) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H * 1.32),",
        "                                     sharex=True, layout=\"constrained\",",
        "                                     gridspec_kw={\"height_ratios\": [3.4, 1.0]})",
        "else:",
        "    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "    ax_risk = None",
        "curves = []",
        "for gi, group in enumerate(GROUP_LEVELS or [\"All\"]):",
        "    cell = df if not GROUP_LEVELS else df[df[G] == group]",
        "    color = COLORS[group] if GROUP_LEVELS else PALETTE[0]",
        "    label = f\"{group} (n={len(cell)})\"",
        "    kmf = KaplanMeierFitter()",
        "    kmf.fit(cell[T].to_numpy(float), cell[E].to_numpy(int), label=label)",
        "    kmf.plot_survival_function(ax=ax, ci_show=SHOW_CI, show_censors=SHOW_CENSORS,",
        "                               censor_styles={\"marker\": \"+\", \"ms\": 4.5},",
        "                               color=color, linewidth=LW)",
        "    curves.append((group, cell, color))",
        "if len(curves) >= 2:",
        "    from lifelines.statistics import logrank_test, multivariate_logrank_test",
        "    if len(curves) == 2:",
        "        (_, first, _), (_, second, _) = curves[0], curves[1]",
        "        result = logrank_test(first[T], second[T], first[E], second[E])",
        "    else:",
        "        result = multivariate_logrank_test(df[T], df[G], df[E])",
        "    p_value = float(result.p_value)",
        "    ax.text(0.02, 0.04, f\"Log-rank p = {p_value:.3g}\", transform=ax.transAxes,",
        "            ha=\"left\", va=\"bottom\", fontsize=max(6.0, TICK_SIZE - 1),",
        "            bbox={\"boxstyle\": \"round,pad=0.28\", \"facecolor\": \"white\",",
        "                  \"edgecolor\": \"#CCCCCC\", \"alpha\": 0.85, \"linewidth\": 0.6})",
        "ax.set_ylim(0.0, 1.02)",
        "ax.set_xlim(left=0.0)",
        "if RISK_TABLE and ax_risk is not None:",
        "    times = np.linspace(0.0, float(df[T].max()), 5)",
        "    for index, (group, cell, _color) in enumerate(curves):",
        "        y_position = len(curves) - index",
        "        for time_point in times:",
        "            ax_risk.text(time_point, y_position, str(int((cell[T] >= time_point).sum())),",
        "                         ha=\"center\", va=\"center\", fontsize=max(5.5, TICK_SIZE - 2))",
        "    ax_risk.set_yticks([len(curves) - index for index in range(len(curves))])",
        "    ax_risk.set_yticklabels([group for group, _cell, _color in curves],",
        "                            fontsize=max(5.5, TICK_SIZE - 2))",
        "    ax_risk.set_ylim(0.4, len(curves) + 0.6)",
        "    ax_risk.set_ylabel(\"At risk\", fontsize=max(6.0, LABEL_SIZE - 2))",
        "    ax_risk.set_xlabel(T, fontsize=LABEL_SIZE)",
        f"X_LABEL, Y_LABEL = {str(labels.get('time') or 'Time')!r}, \"Survival probability\"".replace("'", '"'),
        "TITLE = \"Kaplan-Meier survival curves\"",
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


def _py_forest(data: Mapping[str, Any], params: Mapping[str, Any],
               style: Mapping[str, Any]) -> str:
    warnings: list[str] = []
    rows = _parse_estimates(params.get("estimates"))
    source = "user"
    if not rows:
        rows = _forest_rows_from_cox(data, params, warnings)
        source = "cox"
    if not rows:
        raise C.prerequisite("森林图代码生成需要 estimates 参数或 time/event 角色")
    ref_line = _pfloat(params, "ref_line", 1.0, -1000.0, 1000.0)
    log_scale = _pbool(params, "log_scale", True) and all(row["lower"] > 0 for row in rows) and ref_line > 0
    show_text = _pbool(params, "show_text", True)
    rows_literal = "[\n" + "".join(
        f'    {{"label": {row["label"]!r}, "estimate": {float(row["estimate"])!r}, '
        f'"lower": {float(row["lower"])!r}, "upper": {float(row["upper"])!r}, '
        f'"p": {float(row["p"]) if row.get("p") is not None and math.isfinite(float(row["p"])) else None!r}}},\n'
        for row in rows) + "]"
    body = [
        f"ROWS = {rows_literal}",
        f"REF_LINE, USE_LOG, SHOW_TEXT = {float(ref_line)!r}, {bool(log_scale)!r}, {bool(show_text)!r}".replace("'", '"'),
        "COLOR_SIG, COLOR_NS = \"#B24745\", \"#4E79A7\"".replace("'", '"'),
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))",
        "for index, row in enumerate(ROWS):",
        "    y = len(ROWS) - index",
        "    significant = row[\"p\"] is not None and row[\"p\"] < 0.05",
        "    color = COLOR_SIG if significant else COLOR_NS",
        "    ax.plot([row[\"lower\"], row[\"upper\"]], [y, y], color=\"#4D4D4D\", linewidth=LW)",
        "    ax.plot([row[\"estimate\"]], [y], marker=\"s\", markersize=max(5.0, np.sqrt(POINT) * 1.1),",
        "            markerfacecolor=color, markeredgecolor=\"white\", markeredgewidth=0.8)",
        "    if SHOW_TEXT:",
        "        text = f\"{row['estimate']:.2f} ({row['lower']:.2f}-{row['upper']:.2f})\"",
        "        p_value = row[\"p\"]",
        "        if p_value is not None:",
        "            text += \"***\" if p_value < 0.001 else \"**\" if p_value < 0.01 else \"*\" if p_value < 0.05 else \"\"",
        "        text_x = row[\"upper\"] * 1.12 if USE_LOG else row[\"upper\"] + abs(row[\"upper\"]) * 0.05 + 0.05",
        "        ax.text(text_x, y, text, ha=\"left\", va=\"center\", fontsize=max(5.5, TICK_SIZE - 2))",
        "ax.axvline(REF_LINE, color=\"#6A6A6A\", linestyle=\"--\", linewidth=1.0)",
        "ax.set_yticks([len(ROWS) - index for index in range(len(ROWS))])",
        "ax.set_yticklabels([row[\"label\"] for row in ROWS], fontsize=max(5.5, TICK_SIZE - 1))",
        "ax.set_ylim(0.4, len(ROWS) + 0.6)",
        "if USE_LOG:",
        "    ax.set_xscale(\"log\")",
        "ax.plot([], [], marker=\"s\", linestyle=\"none\", markersize=8, markerfacecolor=COLOR_SIG,",
        "        markeredgecolor=\"white\", label=\"p < 0.05\")",
        "ax.plot([], [], marker=\"s\", linestyle=\"none\", markersize=8, markerfacecolor=COLOR_NS,",
        "        markeredgecolor=\"white\", label=\"p >= 0.05\")",
        f"X_LABEL, Y_LABEL = {('Hazard ratio (per SD)' if source == 'cox' else 'Effect estimate (HR / OR / beta)')!r}, \"\"".replace("'", '"'),
        f"TITLE = {('Forest plot: hazard ratios (univariate Cox)' if source == 'cox' else 'Forest plot: effect estimates')!r}".replace("'", '"'),
        "_finish(ax)",
    ]
    return _py_join(_py_prelude(style), "df = None  # 森林图使用内联效应量表", "\n".join(body))


def _py_pca(data: Mapping[str, Any], params: Mapping[str, Any],
            style: Mapping[str, Any]) -> str:
    do_scale = _pbool(params, "scale", True)
    show_ellipse = _pbool(params, "ellipse", True)
    show_loadings = _pbool(params, "loadings", False)
    block = _py_wide_block(data)
    wide = data.get("wide") or {}
    matrix = np.asarray(wide.get("matrix"), dtype=float)
    group_labels, _group_name = _pair_colors(data, wide, matrix.shape[0])
    group_levels = list(dict.fromkeys(group_labels)) if group_labels else []
    body = [
        f"SCALE, SHOW_ELLIPSE, SHOW_LOADINGS = {bool(do_scale)!r}, {bool(show_ellipse)!r}, {bool(show_loadings)!r}".replace("'", '"'),
        f"GROUP_LABELS = {group_labels!r}".replace("'", '"'),
        f"GROUP_LEVELS = {group_levels!r}".replace("'", '"'),
        f"GROUP_COLORS = {_py_colors_repr(style, group_levels) if group_levels else '{}'}",
        "X = np.asarray(matrix, dtype=float).copy()",
        "col_mean = np.nanmean(X, axis=0)",
        "col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)",
        "invalid = ~np.isfinite(X)",
        "if invalid.any():",
        "    X[invalid] = np.take(col_mean, np.where(invalid)[1])",
        "if SCALE:",
        "    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0, ddof=0) > 1e-12, X.std(axis=0, ddof=0), 1.0)",
        "U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)",
        "scores = U * S",
        "ratio = S ** 2 / np.sum(S ** 2) * 100.0",
        "cumulative = np.cumsum(ratio)",
        "loadings = Vt.T",
        "fig, (ax, ax_scree) = plt.subplots(1, 2, figsize=(FIG_W * 1.62, FIG_H), layout=\"constrained\",",
        "                                   gridspec_kw={\"width_ratios\": [2.45, 1.0]})",
        "if GROUP_LEVELS:",
        "    for level in GROUP_LEVELS:",
        "        index = [i for i, label in enumerate(GROUP_LABELS) if label == level]",
        "        if not index:",
        "            continue",
        "        ax.scatter(scores[index, 0], scores[index, 1], s=POINT, color=GROUP_COLORS[level],",
        "                   alpha=ALPHA, edgecolors=\"white\", linewidths=0.5, label=level)",
        "        if SHOW_ELLIPSE and len(index) >= 3:",
        "            cov = np.cov(scores[index, :2].T)",
        "            eigenvalues, eigenvectors = np.linalg.eigh(cov)",
        "            order = np.argsort(eigenvalues)[::-1]",
        "            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]",
        "            chi2 = float(stats.chi2.ppf(0.95, df=2))",
        "            width = 2.0 * np.sqrt(max(eigenvalues[0], 0.0) * chi2)",
        "            height = 2.0 * np.sqrt(max(eigenvalues[1], 0.0) * chi2)",
        "            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))",
        "            ax.add_patch(plt.matplotlib.patches.Ellipse(",
        "                xy=(float(np.mean(scores[index, 0])), float(np.mean(scores[index, 1]))),",
        "                width=width, height=height, angle=angle, facecolor=GROUP_COLORS[level],",
        "                edgecolor=GROUP_COLORS[level], alpha=0.12, linewidth=1.0))",
        "else:",
        "    ax.scatter(scores[:, 0], scores[:, 1], s=POINT, color=PALETTE[0], alpha=ALPHA,",
        "               edgecolors=\"white\", linewidths=0.5)",
        "if SHOW_LOADINGS:",
        "    norms = np.linalg.norm(loadings[:, :2], axis=1)",
        "    top = np.argsort(-norms)[: min(8, len(features))]",
        "    scale_factor = 0.85 * float(np.max(np.abs(scores[:, :2]))) / max(float(np.max(norms[top])), 1e-12)",
        "    for index in top:",
        "        dx, dy = loadings[index, 0] * scale_factor, loadings[index, 1] * scale_factor",
        "        ax.annotate(\"\", xy=(dx, dy), xytext=(0, 0),",
        "                    arrowprops={\"arrowstyle\": \"->\", \"color\": \"#555555\", \"linewidth\": max(0.6, LW * 0.6)})",
        "        ax.text(dx * 1.06, dy * 1.06, features[int(index)], fontsize=max(5.0, TICK_SIZE - 3),",
        "                color=\"#333333\")",
        "ax.axhline(0, color=\"#D0D0D0\", linewidth=0.7)",
        "ax.axvline(0, color=\"#D0D0D0\", linewidth=0.7)",
        "ax.set_xlabel(f\"PC1 ({ratio[0]:.1f}% variance)\", fontsize=LABEL_SIZE)",
        "ax.set_ylabel(f\"PC2 ({ratio[1]:.1f}% variance)\", fontsize=LABEL_SIZE)",
        "ax.tick_params(labelsize=TICK_SIZE)",
        "positions = np.arange(1, len(ratio) + 1)",
        "ax_scree.bar(positions, ratio, color=PALETTE[: len(ratio)], alpha=ALPHA,",
        "             edgecolor=\"white\", linewidth=0.5)",
        "ax_scree.plot(positions, cumulative, color=\"#B24745\", marker=\"o\", markersize=3.2,",
        "              linewidth=LW)",
        "ax_scree.set_xticks(positions)",
        "ax_scree.set_xticklabels([str(int(value)) for value in positions],",
        "                         fontsize=max(5.5, TICK_SIZE - 2))",
        "ax_scree.set_xlabel(\"Principal component\", fontsize=max(7.0, LABEL_SIZE - 2))",
        "ax_scree.set_ylabel(\"Explained variance (%)\", fontsize=max(7.0, LABEL_SIZE - 2))",
        "ax_scree.tick_params(labelsize=max(5.5, TICK_SIZE - 2))",
        "ax_scree.set_ylim(0, 105)",
        "fig.suptitle(\"PCA: samples on PC1-PC2 with scree plot\", fontsize=TITLE_SIZE,",
        "             fontweight=\"bold\" if TITLE_BOLD else \"normal\")",
        "plt.tight_layout()",
    ]
    return _py_join(_py_prelude(style), block, "\n".join(body))


_PY_BUILDERS = {
    "bar": _py_bar,
    "bar_dot": _py_bar_dot,
    "box": _py_box,
    "violin": _py_violin,
    "raincloud": _py_raincloud,
    "dotplot": _py_dotplot,
    "stack": _py_stack,
    "pie": _py_pie,
    "scatter": _py_scatter,
    "bubble": _py_bubble,
    "line": _py_line,
    "area": _py_area,
    "heatmap": _py_heatmap,
    "corr": _py_corr,
    "pairplot": _py_pairplot,
    "hist": _py_hist,
    "density": _py_density,
    "qq": _py_qq,
    "roc": _py_roc,
    "km": _py_km,
    "forest": _py_forest,
    "pca": _py_pca,
}


def _py_fallback(chart: str, style: Mapping[str, Any], error: Exception) -> str:
    meta = CHART_INDEX.get(chart) or {"title": chart}
    prelude = _py_prelude(style)
    body = (
        f"# 说明：{meta.get('title', chart)} 的数据不足以生成完整可复现代码（{type(error).__name__}: {error}）。\n"
        "# 请在页面里补齐角色映射后重新生成；下面是画布与风格骨架。\n"
        "df = pd.DataFrame()  # TODO: 换成你的清洗后数据\n"
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))\n"
        f"ax.set_title({str(meta.get('title', chart))!r})\n"
        "_finish(ax)\n"
    )
    return _py_join(prelude, "", body)


def python_code(chart_id: str, data: Mapping[str, Any], params: Mapping[str, Any],
                style: Mapping[str, Any]) -> str:
    """返回「数据准备 + 作图」的可独立运行 Python 片段（含 savefig）。"""
    chart = str(chart_id or "").strip()
    if chart not in CHART_INDEX:
        raise C.not_found(f"未知图表 {chart_id}")
    builder = _PY_BUILDERS.get(chart)
    if builder is None:
        raise C.unsupported(f"图表 {chart} 暂不支持生成 Python 代码")
    params_dict = _params_dict(params)
    resolved = _resolve_style(style)
    data = data if isinstance(data, Mapping) else {}
    try:
        return builder(data, params_dict, resolved)
    except C.EngineError:
        raise
    except Exception as exc:  # noqa: BLE001
        return _py_fallback(chart, resolved, exc)


# ===========================================================================
# 代码生成：等价的 R / ggplot2 片段
# ===========================================================================

_R_THEME = '''theme_plotlab <- function(base_size = {text_size}) {{
  theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(size = {title_size}, face = "{title_face}"),
      axis.title = element_text(size = {label_size}),
      axis.text = element_text(size = {tick_size}),
      legend.text = element_text(size = {legend_size}),
      legend.position = "{legend_pos}",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
}}'''


def _r_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return repr(round(number, 6))


def _r_string(text: Any) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"')[:120] + '"'


def _r_vector(values: Any) -> str:
    series = pd.Series(list(values)) if not isinstance(values, pd.Series) else values
    if pd.api.types.is_numeric_dtype(series):
        items = [_r_number(value) for value in series.tolist()]
    else:
        items = []
        for value in series.tolist():
            if value is None:
                items.append("NA")
            else:
                try:
                    if isinstance(value, float) and math.isnan(value):
                        items.append("NA")
                        continue
                except Exception:  # noqa: BLE001
                    pass
                items.append(_r_string(value))
    return "c(" + ", ".join(items) + ")"


def _r_palette(style: Mapping[str, Any], levels: Sequence[Any]) -> str:
    mapping = _colors(style, levels)
    if not mapping:
        colors = C.color_list(style, 1)
        return "c(" + ", ".join(_r_string(color) for color in colors) + ")"
    return "c(" + ", ".join(f"{_r_string(name)} = {_r_string(color)}"
                            for name, color in mapping.items()) + ")"


def _r_tidy_block(data: Mapping[str, Any], roles: Sequence[str],
                  style: Mapping[str, Any]) -> str:
    tidy = data.get("tidy")
    if not isinstance(tidy, pd.DataFrame) or tidy.empty:
        raise C.prerequisite("没有可用的 tidy 数据，无法生成 R 代码")
    labels = data.get("labels") or {}
    present = [role for role in roles if f"__{role}" in tidy.columns]
    if not present:
        raise C.prerequisite("tidy 数据缺少图表需要的角色列")
    lines = ["# ===== 数据（把路径换成你的清洗后 CSV；角色映射见注释）====="]
    mapping_text = ", ".join(f"{role} <- {labels.get(role) or role}" for role in present)
    lines.append(f"# 角色映射：{mapping_text}")
    if len(tidy) <= 400:
        lines.append("df <- data.frame(")
        for role in present:
            lines.append(f"  {role} = {_r_vector(tidy[f'__{role}'])},  # {labels.get(role) or role}")
        lines.append("  stringsAsFactors = FALSE")
        lines.append(")")
    else:
        lines.append('df_all <- read.csv("your_data.csv", stringsAsFactors = FALSE)')
        original = [str(labels.get(role) or role) for role in present]
        lines.append("df <- df_all[, " + _r_vector(original) + ", drop = FALSE]")
        lines.append("names(df) <- " + _r_vector(present))
    for role in ("y", "size", "time", "event"):
        if role in present:
            lines.append(f"df${role} <- as.numeric(df${role})")
    required = [role for role in present if role in ("x", "y", "time", "event")]
    lines.append("df <- df[complete.cases(df[, " + _r_vector(required) + ", drop = FALSE]), ]")
    return "\n".join(lines)


def _r_wide_block(data: Mapping[str, Any]) -> str:
    wide = data.get("wide")
    if not isinstance(wide, Mapping) or wide.get("matrix") is None:
        raise C.prerequisite("没有可用的矩阵/宽表数据，无法生成 R 代码")
    matrix = np.asarray(wide.get("matrix"), dtype=float)
    features = [str(value) for value in (wide.get("features") or [])]
    samples = [str(value) for value in (wide.get("samples") or [])]
    groups = wide.get("sample_groups")
    group_vector = (_r_vector([str(value) for value in list(groups)])
                    if groups is not None and len(groups) == matrix.shape[0] else "NULL")
    lines = ["# ===== 数据：矩阵（行=样本，列=特征）=====",
             "# 原始矩阵一般从 CSV 读入（首列特征名、其余列样本）：",
             '# raw <- read.csv("your_matrix.csv", row.names = 1, check.names = FALSE)',
             '# mat <- as.matrix(raw)']
    lines.append("features <- " + _r_vector(features))
    lines.append("samples <- " + _r_vector(samples))
    matrix_text = np.array2string(matrix, separator=", ", threshold=1000000, max_line_width=140)
    lines.append("mat <- matrix(" + matrix_text + ", nrow = " +
                 str(matrix.shape[0]) + ", byrow = TRUE)  # 行=样本，列=特征")
    lines.append("sample_groups <- " + group_vector)
    return "\n".join(lines)


def _r_stat_layer(data: Mapping[str, Any], style: Mapping[str, Any]) -> str:
    """ggpubr 显著性标注片段。"""
    if not style.get("stat_annot"):
        return ""
    tidy = data.get("tidy")
    if not isinstance(tidy, pd.DataFrame) or "__x" not in tidy.columns:
        return ""
    result = _compare_groups(tidy, "__x", "__y", style, levels=_levels_of(tidy, "__x", max_levels=8))
    if not result:
        return ""
    levels = [str(_fmt_level(level)) for level in _levels_of(tidy, "__x", max_levels=8)]
    method = "t.test" if "t-test" in str(result.get("method")) else (
        "wilcox.test" if "Wilcoxon" in str(result.get("method")) else (
            "anova" if "ANOVA" in str(result.get("method")) else "kruskal.test"))
    pairs = [(pair["a"], pair["b"]) for pair in (result.get("pairs") or [])][:6]
    correction = {"bh": "BH", "bonferroni": "bonferroni", "holm": "holm", "none": "none"}.get(
        str(result.get("p_adjust") or "bh"), "BH")
    lines = [f"# 组间显著性（与工作台一致：{result.get('method')}，{correction} 校正）"]
    if len(levels) == 2:
        lines.append('  stat_compare_means(method = "%s", label = "p.signif", size = 3.5) +'
                     % ("t.test" if method == "t.test" else "wilcox.test"))
    else:
        lines.append(f'  stat_compare_means(method = "{method}", label = "p.format", '
                     f'p.adjust.method = "{correction}", size = 3.2) +')
        comparison_r = ", ".join(f"c({_r_string(a)}, {_r_string(b)})" for a, b in pairs)
        if comparison_r:
            lines.append(f'  stat_compare_means(comparisons = list({comparison_r}), '
                         f'method = "{method}", label = "p.signif", p.adjust.method = "{correction}") +')
    return "\n".join(lines)


def _r_labs(x_label: str, y_label: str, title: str) -> str:
    return (f"  labs(title = {_r_string(title)}, x = {_r_string(x_label)}, "
            f"y = {_r_string(y_label)}) +")


def _r_tail(width: float, height: float) -> str:
    return (f'\nggsave("plotlab_figure.png", p, width = {float(width)}, '
            f'height = {float(height)}, dpi = 300)\nprint(p)\n')


def _r_size_params(params: Mapping[str, Any], key_size: str, key_min: str, key_max: str) -> str:
    if key_size in params:
        return f"scale_size_continuous(range = c(2, 10))"
    return ""


def _r_stat_layer(data: Mapping[str, Any], style: Mapping[str, Any]) -> str:
    """ggpubr 显著性标注片段（每行以 + 结尾，方便拼进 ggplot 链）。"""
    if not style.get("stat_annot"):
        return ""
    tidy = data.get("tidy")
    if not isinstance(tidy, pd.DataFrame) or "__x" not in tidy.columns:
        return ""
    levels_raw = _levels_of(tidy, "__x", max_levels=8)
    result = _compare_groups(tidy, "__x", "__y", style, levels=levels_raw)
    if not result:
        return ""
    levels = [str(_fmt_level(level)) for level in levels_raw]
    method_text = str(result.get("method") or "")
    if "Wilcoxon" in method_text or "Kruskal" in method_text:
        method = "wilcox.test" if len(levels) == 2 else "kruskal.test"
    elif "ANOVA" in method_text:
        method = "anova"
    else:
        method = "t.test" if len(levels) == 2 else "anova"
    correction = {"bh": "BH", "bonferroni": "bonferroni", "holm": "holm", "none": "none"}.get(
        str(result.get("p_adjust") or "bh"), "BH")
    lines = [f"  # 组间显著性（与工作台一致：{method_text}，{correction} 校正）"]
    if len(levels) == 2:
        lines.append(f'  stat_compare_means(method = "{method}", label = "p.signif", '
                     f'p.adjust.method = "{correction}", size = 3.5) +')
    else:
        lines.append(f'  stat_compare_means(method = "{method}", label = "p.format", '
                     f'p.adjust.method = "{correction}", size = 3.2) +')
        pairs = [(pair["a"], pair["b"]) for pair in (result.get("pairs") or [])][:6]
        if pairs:
            comparison_r = ", ".join(f"c({_r_string(a)}, {_r_string(b)})" for a, b in pairs)
            lines.append(f'  stat_compare_means(comparisons = list({comparison_r}), '
                         f'method = "{method}", label = "p.signif", '
                         f'p.adjust.method = "{correction}") +')
    return "\n".join(line for line in lines if line.strip())


def _r_forest_context(data: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    warnings: list[str] = []
    rows = _parse_estimates(params.get("estimates"))
    if not rows:
        rows = _forest_rows_from_cox(data, params, warnings)
    if not rows:
        raise C.prerequisite("森林图 R 代码需要 estimates 参数或 time/event 角色")
    return {
        "forest_labels": ", ".join(_r_string(row["label"]) for row in rows),
        "forest_estimates": ", ".join(_r_number(row["estimate"]) for row in rows),
        "forest_lower": ", ".join(_r_number(row["lower"]) for row in rows),
        "forest_upper": ", ".join(_r_number(row["upper"]) for row in rows),
        "forest_p": ", ".join(_r_number(row.get("p")) for row in rows),
    }


_R_COMPARISON_ROLES = {
    "bar": ("x", "y"), "bar_dot": ("x", "y"), "box": ("x", "y"),
    "violin": ("x", "y"), "raincloud": ("x", "y"),
}


def _r_code_impl(chart: str, data: Mapping[str, Any], params: Mapping[str, Any],
                 style: Mapping[str, Any]) -> str:
    labels = data.get("labels") or {}
    width = float(style.get("fig_width", 6.2))
    height = float(style.get("fig_height", 4.4))
    tidy = data.get("tidy")
    theme = _r_theme(style)
    if chart in ("heatmap", "corr", "pca", "pairplot"):
        block = _r_wide_block(data)
        body = _R_WIDE_BODIES[chart].format(width=width, height=height)
        return _R_HEADER_MATRIX.format(theme=theme) + "\n\n" + block + "\n\n" + body
    group_key = (_pick_group_col(data, tidy, keys=("__group",))
                 if isinstance(tidy, pd.DataFrame) else None)
    color_key = (_pick_group_col(data, tidy, keys=("__color",))
                 if isinstance(tidy, pd.DataFrame) else None)
    columns = set(tidy.columns) if isinstance(tidy, pd.DataFrame) else set()
    roles: list[str] = []
    if chart in _R_COMPARISON_ROLES:
        roles = ["x", "y"] + (["group"] if group_key else [])
    elif chart == "dotplot":
        roles = ["x"] + (["group"] if "__group" in columns else []) \
            + (["color"] if "__color" in columns else []) \
            + (["y"] if "__y" in columns else []) + (["size"] if "__size" in columns else [])
    elif chart == "stack":
        roles = ["x", "y"] + (["group"] if group_key else [])
    elif chart == "pie":
        roles = ["x"] + (["y"] if "__y" in columns else [])
    elif chart in ("scatter", "bubble"):
        roles = ["x", "y"] + (["color"] if color_key else []) \
            + (["size"] if "__size" in columns else [])
    elif chart in ("line", "area"):
        roles = ["x", "y"] + (["group"] if group_key else [])
    elif chart in ("hist", "density", "qq"):
        roles = ["x"] + (["color"] if color_key else [])
    elif chart == "roc":
        roles = ["y"] + (["group"] if group_key else [])
    elif chart == "km":
        roles = ["time", "event"] + (["group"] if group_key else [])
    else:
        roles = []
    block = (_r_tidy_block(data, roles, style) if chart != "forest"
             else "# 森林图：使用下面的效应量表（也可从回归结果表读入）\n"
                  "# est <- read.csv(\"your_estimates.csv\", stringsAsFactors = FALSE)")
    level_source = group_key or color_key
    levels = ([str(_fmt_level(level)) for level in _levels_of(tidy, level_source, max_levels=12)]
              if level_source and isinstance(tidy, pd.DataFrame)
              else [str(_fmt_level(level)) for level in
                    (_levels_of(tidy, "__x", max_levels=16) if isinstance(tidy, pd.DataFrame) else [])])
    palette = _r_palette(style, levels) if levels else "NULL"
    x_label = str(labels.get("x") or "x")
    y_label = str(labels.get("y") or "y")
    fill_aes = ", fill = group" if group_key else ", fill = x"
    colour_aes = ", colour = color" if color_key else ""
    if chart in ("line", "area", "stack") and group_key:
        fill_aes = ", fill = group"
        colour_aes = ", colour = group" if chart == "line" else ""
    context = {
        "width": width,
        "height": height,
        "x_label": x_label,
        "y_label": y_label,
        "bar_width": float(style.get("bar_width", 0.72)),
        "alpha": float(style.get("alpha", 0.85)),
        "jitter_alpha": float(style.get("jitter_alpha", 0.55)),
        "line_width": float(style.get("line_width", 1.4)),
        "point_size": float(style.get("point_size", 28)),
        "fill_aes": fill_aes,
        "colour_aes": colour_aes,
        "scale_fill_layer": "  scale_fill_manual(values = PLOTLAB_PALETTE) +\n",
        "scale_colour_layer": ("  scale_colour_manual(values = PLOTLAB_PALETTE) +\n"
                               if colour_aes else ""),
        "stat": _r_stat_layer(data, style) if style.get("stat_annot") else "",
        "method": _pchoice(params, "method", "pearson", ["pearson", "spearman", "kendall"]),
        "dist": _pchoice(params, "dist", "normal", ["normal", "uniform", "exponential"]),
        "time_label": str(labels.get("time") or "Time"),
        "palette": palette,
        "roc_response": "group" if group_key else "color",
        "km_formula": "Surv(time, event) ~ group" if group_key else "Surv(time, event) ~ 1",
    }
    context.update(_r_forest_context(data, params) if chart == "forest" else {})
    template = _R_BODIES.get(chart)
    if template is None:
        raise C.unsupported(f"图表 {chart} 暂不支持生成 R 代码")
    body = template.format(**context)
    prefix = f"PLOTLAB_PALETTE <- {palette}\n\n" if palette != "NULL" else ""
    return _R_HEADER.format(theme=theme) + "\n\n" + block + "\n\n" + prefix + body


def _r_theme(style: Mapping[str, Any]) -> str:
    legend_pos = str(style.get("legend_pos") or "right")
    return _R_THEME.format(
        text_size=float(style.get("text_size", 10)),
        title_size=float(style.get("title_size", 14)),
        label_size=float(style.get("label_size", 12)),
        tick_size=float(style.get("tick_size", 10)),
        legend_size=float(style.get("legend_size", 10)),
        title_face="bold" if style.get("title_bold", True) else "plain",
        legend_pos=legend_pos if legend_pos in ("right", "top", "bottom", "none") else "right",
    )


_R_HEADER = """library(ggplot2)
library(ggpubr)   # stat_compare_means：组间显著性；不需要可删掉

# ===== 风格（与 PlotLab 工作台一致）=====
{theme}"""

_R_HEADER_MATRIX = """library(ggplot2)
library(reshape2)   # 热图/相关矩阵需要把宽表 melt 成长表

# ===== 风格（与 PlotLab 工作台一致）=====
{theme}"""


_R_BODIES = {
    "bar": """p <- ggplot(df, aes(x = x, y = y{fill_aes})) +
  stat_summary(fun = "mean", geom = "bar", width = {bar_width}, colour = "white",
               linewidth = 0.4) +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.2, linewidth = 0.7) +
  geom_jitter(width = 0.12, size = 1.2, alpha = {jitter_alpha}, colour = "#3A3A3A") +
{scale_fill_layer}{stat}
  labs(title = "Bar chart: mean {y_label} by {x_label}", x = "{x_label}", y = "{y_label}") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "bar_dot": """p <- ggplot(df, aes(x = x, y = y{fill_aes})) +
  stat_summary(fun = "mean", geom = "bar", width = {bar_width}, colour = "white",
               linewidth = 0.4, alpha = 0.45) +
  geom_jitter(width = 0.2, size = 1.4, alpha = {jitter_alpha}, aes(colour = x)) +
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.2, linewidth = 0.7) +
{scale_fill_layer}{stat}
  labs(title = "Bar + dot chart: mean {y_label} by {x_label}", x = "{x_label}", y = "{y_label}") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "box": """p <- ggplot(df, aes(x = x, y = y{fill_aes})) +
  geom_boxplot(width = {bar_width}, outlier.shape = 19, outlier.size = 1.6,
               alpha = {alpha}, colour = "#2B2B2B") +
  geom_jitter(width = 0.12, size = 1.2, alpha = {jitter_alpha}, colour = "#2B2B2B") +
{scale_fill_layer}{stat}
  labs(title = "Box plot: {y_label} by {x_label}", x = "{x_label}", y = "{y_label}") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "violin": """p <- ggplot(df, aes(x = x, y = y{fill_aes})) +
  geom_violin(trim = TRUE, alpha = {alpha}, colour = "#2B2B2B", linewidth = 0.5) +
  geom_boxplot(width = 0.12, fill = "white", outlier.shape = NA, linewidth = 0.5) +
{scale_fill_layer}{stat}
  labs(title = "Violin plot: {y_label} by {x_label}", x = "{x_label}", y = "{y_label}") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "raincloud": """# 说明：R 的雨云图通常用 gghalves::geom_half_violin 实现；这里用 geom_violin
# + geom_boxplot + geom_jitter 给出最接近的 ggplot2 版本（装 gghalves 后可替换第一层）。
p <- ggplot(df, aes(x = x, y = y{fill_aes})) +
  geom_violin(trim = TRUE, alpha = {alpha}, colour = "#2B2B2B", linewidth = 0.5) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, linewidth = 0.5) +
  geom_jitter(width = 0.06, size = 1.2, alpha = {jitter_alpha}, colour = "#2B2B2B") +
{scale_fill_layer}{stat}
  labs(title = "Raincloud plot: {y_label} by {x_label}", x = "{x_label}", y = "{y_label}") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "dotplot": """# 点图：先把长表聚合成 行×列 矩阵，再用点大小/颜色表示数值
agg <- aggregate(y ~ x + group, data = df, FUN = mean)
p <- ggplot(agg, aes(x = x, y = group, size = y, colour = y)) +
  geom_point() +
  scale_colour_gradient(low = "#56B4E9", high = "#B24745") +
  scale_size_continuous(range = c(1, 9)) +
  labs(title = "Dot plot: mean {y_label} by {x_label}", x = "{x_label}",
       y = "Group", size = "{y_label}", colour = "{y_label}") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "stack": """p <- ggplot(df, aes(x = x, y = y{fill_aes})) +
  geom_bar(stat = "summary", fun = "sum", width = {bar_width}, colour = "white",
           linewidth = 0.4) +
{scale_fill_layer}
  labs(title = "Stacked bar: {y_label} by {x_label}", x = "{x_label}", y = "{y_label}") +
  theme_plotlab()

# 百分比堆叠：把 geom_bar 改成 position = "fill"

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "pie": """# 饼图：先按 x 汇总，再用 coord_polar 变饼图（环图加 xlim(c(0.4, 1.6))）
pie_df <- aggregate(y ~ x, data = df, FUN = sum)
pie_df$pct <- round(pie_df$y / sum(pie_df$y) * 100, 1)
p <- ggplot(pie_df, aes(x = "", y = y, fill = x)) +
  geom_bar(stat = "identity", width = 1, colour = "white", linewidth = 0.5) +
  coord_polar(theta = "y", start = 0) +
{scale_fill_layer}
  labs(title = "Pie chart: composition of {x_label}", x = NULL, y = NULL, fill = "{x_label}") +
  theme_plotlab() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        panel.grid = element_blank())

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "scatter": """p <- ggplot(df, aes(x = x, y = y{colour_aes})) +
  geom_point(size = 2.2, alpha = {alpha}) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE, alpha = 0.15,
              colour = "#333333") +
  stat_cor(method = "{method}", label.x.npc = "left", label.y.npc = "top", size = 3.2) +
{scale_colour_layer}{stat}
  labs(title = "Scatter plot: {y_label} vs {x_label}", x = "{x_label}", y = "{y_label}") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "bubble": """p <- ggplot(df, aes(x = x, y = y, size = size{colour_aes})) +
  geom_point(alpha = {alpha}) +
  scale_size_continuous(range = c(1.5, 12)) +
{scale_colour_layer}
  labs(title = "Bubble plot: {y_label} vs {x_label}", x = "{x_label}", y = "{y_label}",
       size = "size") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "line": """# 折线：按 x 与分组聚合后连线，误差带用 mean_se 的 ribbon
p <- ggplot(df, aes(x = x, y = y{colour_aes})) +
  stat_summary(fun = "mean", geom = "line", linewidth = {line_width},
               aes(group = group)) +
  stat_summary(fun = "mean", geom = "point", size = 2, aes(group = group)) +
  stat_summary(fun.data = "mean_se", geom = "ribbon", alpha = 0.18, colour = NA,
               aes(group = group, fill = group)) +
{scale_fill_layer}{scale_colour_layer}
  labs(title = "Line chart: mean {y_label} vs {x_label}", x = "{x_label}", y = "{y_label}") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "area": """p <- ggplot(df, aes(x = x, y = y{fill_aes})) +
  geom_area(stat = "summary", fun = "mean", alpha = {alpha}, colour = "white",
            linewidth = 0.3) +
{scale_fill_layer}
  labs(title = "Area chart: mean {y_label} vs {x_label}", x = "{x_label}", y = "{y_label}") +
  theme_plotlab()

# 百分比堆叠：geom_area(stat = "summary", fun = "mean", position = "fill")

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "hist": """p <- ggplot(df, aes(x = x{fill_aes})) +
  geom_histogram(bins = 20, alpha = {alpha}, colour = "white", linewidth = 0.3,
                 position = "identity") +
{scale_fill_layer}
  labs(title = "Histogram: {x_label}", x = "{x_label}", y = "Count") +
  theme_plotlab()

# 密度归一化：geom_histogram(aes(y = after_stat(density)))
# 堆叠分组：geom_histogram(position = "stack")

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "density": """p <- ggplot(df, aes(x = x{fill_aes})) +
  geom_density(linewidth = {line_width}, alpha = {alpha}, colour = "#2B2B2B") +
{scale_fill_layer}
  labs(title = "Density plot: {x_label}", x = "{x_label}", y = "Density") +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "qq": """p <- ggplot(df, aes(sample = x)) +
  stat_qq(size = 1.6, alpha = {alpha}, colour = "#4E79A7") +
  stat_qq_line(colour = "#333333", linewidth = {line_width}) +
  labs(title = "Q-Q plot: {x_label}", x = "Theoretical quantiles ({dist})",
       y = "Sample quantiles") +
  theme_plotlab()

# 正态性检验：shapiro.test(df$x)

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "roc": """# ROC 曲线：pROC + ggroc（等价 ggplot2 版本）
library(pROC)
roc_obj <- roc(response = df${roc_response}, predictor = df$y, quiet = TRUE)
p <- ggroc(roc_obj, colour = "#4E79A7", linewidth = {line_width}) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", colour = "#9A9A9A") +
  labs(title = sprintf("ROC curve (AUC = %.3f)", auc(roc_obj)),
       x = "False positive rate (1 - specificity)",
       y = "True positive rate (sensitivity)") +
  theme_plotlab()

# 最佳阈值（Youden）：coords(roc_obj, "best", ret = c("threshold", "sensitivity",
#                                                     "specificity"), transpose = FALSE)

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "km": """# 生存曲线：survminer::ggsurvplot（ggplot2 语法）
library(survival)
library(survminer)
fit <- survfit({km_formula}, data = df)
p <- ggsurvplot(fit, data = df, risk.table = TRUE, conf.int = TRUE,
                censor = TRUE, palette = {palette},
                xlab = "{time_label}", ylab = "Survival probability",
                title = "Kaplan-Meier survival curves",
                ggtheme = theme_plotlab())

# log-rank 检验：survdiff({km_formula}, data = df)

ggsave("plotlab_figure.png", p$plot, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "forest": """# 森林图：效应量表 → geom_pointrange（estimate ± 95% CI）
est <- data.frame(
  label = c({forest_labels}),
  estimate = c({forest_estimates}),
  lower = c({forest_lower}),
  upper = c({forest_upper}),
  p = c({forest_p})
)
est$significant <- ifelse(est$p < 0.05, "p < 0.05", "p >= 0.05")
est$label <- factor(est$label, levels = rev(est$label))
p <- ggplot(est, aes(x = estimate, y = label, colour = significant)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "#6A6A6A") +
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.18, linewidth = 0.7,
                 colour = "#4D4D4D") +
  geom_point(size = 2.6) +
  scale_colour_manual(values = c("p < 0.05" = "#B24745", "p >= 0.05" = "#4E79A7")) +
  labs(title = "Forest plot", x = "Hazard ratio (per SD)", y = NULL, colour = NULL) +
  theme_plotlab()

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
}


_R_WIDE_BODIES = {
    "heatmap": """# 热图：pheatmap 支持行/列聚类（推荐）；下面是 ggplot2 的等价版本（只排序不聚类）
# pheatmap::pheatmap(mat, scale = "row", cluster_rows = TRUE, cluster_cols = TRUE)
long <- as.data.frame(t(mat))
long$feature <- features
long <- melt(long, id.vars = "feature", variable.name = "sample", value.name = "value")
long$value_z <- ave(long$value, long$feature, FUN = function(v) (v - mean(v)) / sd(v))
p <- ggplot(long, aes(x = sample, y = feature, fill = value_z)) +
  geom_tile() +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, name = "z-score") +
  theme_bw(base_size = 10) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 7),
        axis.text.y = element_text(size = 7),
        panel.grid = element_blank()) +
  labs(title = "Heatmap: features x samples", x = "Samples", y = "Features")

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "corr": """# 相关矩阵：ggcorrplot 最接近（含显著性星号）；下面是 ggplot2 geom_tile 版本
# ggcorrplot::ggcorrplot(cor(mat), method = "square", type = "lower", lab = TRUE)
cor_mat <- cor(mat, method = "pearson")
cor_long <- melt(cor_mat)
p <- ggplot(cor_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", value)), size = 2.6) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limits = c(-1, 1), name = "r") +
  theme_bw(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        panel.grid = element_blank()) +
  labs(title = "Correlation matrix (Pearson)", x = NULL, y = NULL)

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "pairplot": """# 散点矩阵：GGally::ggpairs（推荐；也可用 base R 的 pairs(mat)）
library(GGally)
plot_df <- as.data.frame(mat)
names(plot_df) <- features
if (!is.null(sample_groups)) plot_df$group <- sample_groups
p <- ggpairs(plot_df, columns = seq_along(features)) +
  theme_bw(base_size = 10)

# 按分组着色：ggpairs(plot_df, columns = seq_along(features),
#                     ggplot2::aes(colour = group, alpha = 0.6))

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
    "pca": """# PCA：prcomp + ggplot2 得分图（碎石图用 base barplot；PCA 本身不是 ggplot2 图层）
pca_fit <- prcomp(mat, scale. = TRUE, center = TRUE)
scores_df <- as.data.frame(pca_fit$x[, 1:2])
scores_df$group <- if (is.null(sample_groups)) "All" else sample_groups
var_pct <- round(100 * pca_fit$sdev^2 / sum(pca_fit$sdev^2), 1)
p <- ggplot(scores_df, aes(x = PC1, y = PC2, colour = group, fill = group)) +
  geom_point(size = 2.4, alpha = 0.85) +
  stat_ellipse(geom = "polygon", alpha = 0.12, level = 0.95, show.legend = FALSE) +
  labs(title = "PCA: samples on PC1-PC2",
       x = sprintf("PC1 (%.1f%% variance)", var_pct[1]),
       y = sprintf("PC2 (%.1f%% variance)", var_pct[2])) +
  theme_plotlab()

# 碎石图：barplot(var_pct, xlab = "Principal component", ylab = "Explained variance (%)")

ggsave("plotlab_figure.png", p, width = {width}, height = {height}, dpi = 300)
print(p)
""",
}


def r_code(chart_id: str, data: Mapping[str, Any], params: Mapping[str, Any],
           style: Mapping[str, Any]) -> str:
    """返回等价的 ggplot2 片段；ggplot2 做不出的（聚类热图/PCA 碎石图等）在注释里说明。"""
    chart = str(chart_id or "").strip()
    if chart not in CHART_INDEX:
        raise C.not_found(f"未知图表 {chart_id}")
    params_dict = _params_dict(params)
    resolved = _resolve_style(style)
    data = data if isinstance(data, Mapping) else {}
    try:
        code = _r_code_impl(chart, data, params_dict, resolved)
    except C.EngineError:
        raise
    except Exception as exc:  # noqa: BLE001
        meta = CHART_INDEX.get(chart) or {"title": chart}
        return (
            f"# 说明：{meta.get('title', chart)} 的数据不足以生成完整 R 代码"
            f"（{type(exc).__name__}: {exc}）。\n"
            "# 请补齐角色映射后重新生成；下面是画布与风格骨架。\n"
            "library(ggplot2)\n\n"
            "df <- read.csv(\"your_data.csv\", stringsAsFactors = FALSE)\n\n"
            + _r_theme(resolved) + "\n\n"
            "p <- ggplot(df, aes(x = 1, y = 1)) +\n"
            f"  labs(title = {_r_string(meta.get('title', chart))}) +\n"
            "  theme_plotlab()\n\n"
            f"ggsave(\"plotlab_figure.png\", p, width = {float(resolved.get('fig_width', 6.2))}, "
            f"height = {float(resolved.get('fig_height', 4.4))}, dpi = 300)\n"
            "print(p)\n"
        )
    return code

