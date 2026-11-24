"""PlotLab 数据集容器与内置示例数据。

Dataset 结构见 DESIGN.md §2；本模块只负责「从 DataFrame 生成合法 dataset」与 demo 数据，
不持有全局状态（全局状态在 pl_registry）。
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Mapping

import numpy as np
import pandas as pd

import pl_common as C
import pl_prep as P


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def build_dataset(
    df: pd.DataFrame,
    name: str,
    kind: str | None = None,
    source: str | None = None,
    feature_col: str | None = None,
    sample_info: pd.DataFrame | None = None,
    id_col: str | None = None,
    notes: list[str] | None = None,
) -> dict:
    """把 DataFrame 规整成 dataset dict：列类型、tidy/matrix 判定、角色推断。"""
    if df is None or df.empty:
        raise C.bad_request("数据为空（0 行），无法导入")
    frame = df.copy()
    # 列名去重/去空格
    seen: dict[str, int] = {}
    renamed = []
    for column in frame.columns:
        key = str(column).strip()
        if key in seen:
            seen[key] += 1
            key = f"{key}.{seen[key]}"
        else:
            seen[key] = 1
        renamed.append(key)
    frame.columns = renamed
    if frame.columns.duplicated().any():
        frame = frame.loc[:, ~frame.columns.duplicated()]

    detected_kind, detected_feature = P.infer_kind_and_feature_col(frame, kind)
    feature = feature_col if (feature_col and feature_col in frame.columns) else detected_feature
    if id_col and id_col not in frame.columns:
        id_col = None
    if not id_col:
        for key in frame.columns:
            if P._matches(str(key), P.ID_HINTS):
                id_col = str(key)
                break

    columns = P.column_summaries(frame)
    dataset = {
        "id": "",
        "name": str(name or "dataset"),
        "source": source,
        "kind": detected_kind,
        "df": frame,
        "raw_df": frame.copy(),
        "columns": columns,
        "roles": {},
        "group_col": None,
        "groups": None,
        "cleaning": [],
        "history": [],
        "need": "",
        "notes": list(notes or []),
        "warnings": [],
        "created_at": now_iso(),
        "steps": [],
        "feature_col": feature if detected_kind == "matrix" else None,
        "sample_info": sample_info if isinstance(sample_info, pd.DataFrame) else None,
        "id_col": id_col,
    }
    roles = P.infer_roles(dataset)
    dataset["roles"] = roles
    group = roles.get("group")
    if group:
        levels = [str(v) for v in pd.unique(frame[group].dropna().astype(str))][:60]
        dataset["group_col"] = str(group)
        dataset["groups"] = levels
    return dataset


def refresh_columns(dataset: Mapping[str, Any]) -> None:
    """清洗后重算 columns / groups / 角色有效性。"""
    df = dataset.get("df")
    dataset["columns"] = P.column_summaries(df)
    existing = {str(c["key"]) for c in dataset["columns"]}
    roles = dict(dataset.get("roles") or {})
    dataset["roles"] = {k: (v if v in existing else None) for k, v in roles.items()}
    group = dataset["roles"].get("group")
    if group and group in df.columns:
        dataset["group_col"] = str(group)
        dataset["groups"] = [str(v) for v in pd.unique(df[group].dropna().astype(str))][:60]
    else:
        dataset["group_col"] = None
        dataset["groups"] = None


# ---------------------------------------------------------------------------
# 内置示例数据
# ---------------------------------------------------------------------------


def _clinical_demo() -> pd.DataFrame:
    rng = np.random.default_rng(20261124)
    n_per = 48
    groups = ["Control"] * n_per + ["Treatment"] * n_per
    sex = rng.choice(["Female", "Male"], size=n_per * 2)
    age = rng.normal(58, 9, size=n_per * 2).round(1)
    crp = np.concatenate([
        rng.lognormal(mean=1.4, sigma=0.55, size=n_per),
        rng.lognormal(mean=1.05, sigma=0.5, size=n_per),
    ]).round(2)
    score = np.concatenate([
        rng.normal(62, 12, n_per),
        rng.normal(74, 11, n_per),
    ]).round(1)
    bmi = rng.normal(24.6, 3.4, size=n_per * 2).round(1)
    response = []
    for group in groups:
        if group == "Control":
            response.append(rng.choice(["No", "Yes"], p=[0.72, 0.28]))
        else:
            response.append(rng.choice(["No", "Yes"], p=[0.38, 0.62]))
    base = np.where(np.array(groups) == "Treatment", 14.5, 10.2)
    survival = np.clip(rng.gamma(shape=4.0, scale=base / 4.0), 1.0, 60.0).round(1)
    event = rng.binomial(1, np.where(np.array(groups) == "Treatment", 0.42, 0.68))
    frame = pd.DataFrame(
        {
            "sample_id": [f"S{index + 1:03d}" for index in range(n_per * 2)],
            "group": groups,
            "sex": sex,
            "age": age,
            "bmi": bmi,
            "crp": crp,
            "score": score,
            "response": response,
            "survival_months": survival,
            "event": event,
        }
    )
    return frame


def _expression_demo() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    genes = [
        "IL6", "TNF", "CD8A", "PDCD1", "CXCL9", "STAT1", "FOXP3", "GZMB",
        "MKI67", "VEGFA", "COL1A1", "ACTB",
    ]
    samples = [f"T{index + 1:02d}" for index in range(8)] + [f"C{index + 1:02d}" for index in range(8)]
    groups = ["Tumor"] * 8 + ["Normal"] * 8
    base = rng.normal(7.5, 1.1, size=(len(genes), 1))
    effect = rng.normal(0, 1.0, size=(len(genes), 1))
    is_tumor = np.array([1.0 if g == "Tumor" else 0.0 for g in groups]).reshape(1, -1)
    values = base + effect * is_tumor + rng.normal(0, 0.45, size=(len(genes), len(samples)))
    values = np.clip(values, 0, None).round(2)
    frame = pd.DataFrame(values, columns=samples)
    frame.insert(0, "gene", genes)
    sample_info = pd.DataFrame({"sample": samples, "group": groups})
    return frame, sample_info


def _dose_demo() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for dose in [0, 1, 3, 10, 30]:
        for group in ["WT", "KO"]:
            for replicate in range(4):
                effect = 1 + 0.18 * dose if group == "WT" else 1 + 0.05 * dose
                value = effect * rng.lognormal(0, 0.12) * 100
                rows.append(
                    {
                        "dose": dose,
                        "genotype": group,
                        "replicate": f"R{replicate + 1}",
                        "viability": round(value, 1),
                    }
                )
    return pd.DataFrame(rows)


DEMOS: dict[str, dict] = {
    "clinical": {
        "title": "临床队列（96 例，分组/生存/指标）",
        "description": "tidy 数据：分组、性别、年龄、BMI、CRP、评分、缓解、生存时间与事件。适合柱状图/箱线图/散点/ROC/生存/回归。",
        "kind": "tidy",
    },
    "expression": {
        "title": "表达矩阵（12 基因 × 16 样本）",
        "description": "matrix 数据：首列基因，其余为样本，附带样本分组（Tumor/Normal）。适合热图/相关矩阵/PCA/箱线图。",
        "kind": "matrix",
    },
    "dose": {
        "title": "剂量反应（5 剂量 × 2 基因型 × 4 重复）",
        "description": "tidy 数据：剂量、基因型、重复、活力值。适合折线图/散点/分组柱状图。",
        "kind": "tidy",
    },
}


def create_demo(name: str = "clinical") -> dict:
    key = str(name or "clinical").strip().lower()
    if key in ("", "default", "clinical", "队列", "示例"):
        key = "clinical"
    if key not in DEMOS:
        raise C.bad_request(f"未知示例 {name}，可选：{('、'.join(DEMOS))}")
    if key == "clinical":
        frame = _clinical_demo()
        dataset = build_dataset(frame, "示例-临床队列", kind="tidy")
    elif key == "expression":
        frame, info = _expression_demo()
        dataset = build_dataset(frame, "示例-表达矩阵", kind="matrix", sample_info=info, feature_col="gene")
        dataset["roles"]["group"] = "group"  # 样本分组（来自 sample_info）
        dataset["group_col"] = "group"
        dataset["groups"] = [str(v) for v in pd.unique(info["group"].astype(str))]
    else:
        frame = _dose_demo()
        dataset = build_dataset(frame, "示例-剂量反应", kind="tidy")
    dataset["source"] = f"demo:{key}"
    dataset["notes"].append(f"内置示例数据：{DEMOS[key]['description']}")
    return dataset
