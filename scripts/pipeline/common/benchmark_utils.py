from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "AGENTS.md").exists():
            return path
    raise RuntimeError(f"Could not locate repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
DATA_FEATURES = REPO_ROOT / "data" / "features"
KG_EXPORT = DATA_INTERIM / "KG_Export"
OUTPUTS = REPO_ROOT / "outputs"
TABLES = OUTPUTS / "tables"
CHECKPOINTS = OUTPUTS / "checkpoints"
LOGS = OUTPUTS / "logs"
FIGURES = OUTPUTS / "figures"

TABULAR_INTERNAL_TABLES = TABLES / "tabular_internal"

GNN_INTERNAL_TABLES = TABLES / "gnn_internal"
GNN_INTERNAL_CHECKPOINTS = CHECKPOINTS / "gnn_internal"
GNN_INTERNAL_LOGS = LOGS / "gnn_internal"

FPMTB_EXTERNAL_TABLES = TABLES / "fpmtb_external"
FPMTB_EXTERNAL_CHECKPOINTS = CHECKPOINTS / "fpmtb_external"
FPMTB_EXTERNAL_LOGS = LOGS / "fpmtb_external"

LEGACY_GNN_TABLES = TABLES / "legacy_gnn_leaky"
LEGACY_GNN_CHECKPOINTS = CHECKPOINTS / "legacy_gnn_leaky"
LEGACY_GNN_LOGS = LOGS / "legacy_gnn_leaky"
LEGACY_GNN_FIGURES = FIGURES / "legacy_gnn_leaky"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def output_dir_for_run(base_path: Path, run_name: str | None) -> Path:
    if not run_name:
        return base_path
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name)).strip("._")
    if not safe_name:
        raise ValueError("run_name must contain at least one alphanumeric character")
    return base_path / safe_name


def normalize_drug_name(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def parse_rule_cell(value: object) -> tuple[str, str] | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or ">" not in text:
        return None
    left, right = [part.strip() for part in text.split(">", 1)]
    if not left.startswith("ENSG") or not right.startswith("ENSG"):
        return None
    return left, right


def load_ktsp_rules(path: Path) -> dict[str, list[tuple[str, str]]]:
    df = pd.read_excel(path)
    rules: dict[str, list[tuple[str, str]]] = {}
    for drug_name in df.columns:
        drug_rules = []
        for value in df[drug_name].dropna():
            rule = parse_rule_cell(value)
            if rule is not None:
                drug_rules.append(rule)
        rules[drug_name] = drug_rules
    return rules


def safe_roc_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~(pd.isna(y_true) | pd.isna(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true.astype(int), y_score))


def safe_average_precision(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~(pd.isna(y_true) | pd.isna(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true.astype(int), y_score))


def safe_corr(x, y, method: str = "pearson") -> float:
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.corr(y, method=method))


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    valid = ~(pd.isna(y_true) | pd.isna(y_score))
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]
    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
        }
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred))
            if len(np.unique(y_true)) > 1
            else np.nan
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity) if not np.isnan(sensitivity) else np.nan,
        "specificity": float(specificity) if not np.isnan(specificity) else np.nan,
    }


def count_classes(values) -> tuple[int, int]:
    series = pd.Series(values).dropna().astype(int)
    return int((series == 0).sum()), int((series == 1).sum())


def select_threshold_by_bacc(y_true, y_score) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        metrics = binary_metrics(y_true, y_score, threshold)
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "distance_to_0_5": abs(threshold - 0.5),
            }
        )
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["balanced_accuracy"]).copy()
    if valid.empty:
        return 0.5, df
    valid = valid.sort_values(
        ["balanced_accuracy", "distance_to_0_5"], ascending=[False, True]
    )
    return float(valid.iloc[0]["threshold"]), df
