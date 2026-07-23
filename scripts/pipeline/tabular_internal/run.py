from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from scripts.pipeline.common.benchmark_utils import (
    DATA_INTERIM,
    TABULAR_INTERNAL_TABLES,
    binary_metrics,
    ensure_dirs,
    safe_average_precision,
    safe_roc_auc,
)


DATASET = DATA_INTERIM / "beataml_phase1_dataset_v3.csv"
DROP_COLS = {
    "dbgap_subject_id",
    "inhibitor",
    "auc",
    "response",
    "label_auc100",
    "cohort",
}


def load_protocol_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATASET)
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    df["label_auc100"] = (pd.to_numeric(df["auc"], errors="coerce") < 100).astype(int)

    train_val = df[df["cohort"].astype(str).eq("Waves1+2")].copy()
    test = df[df["cohort"].astype(str).eq("Waves3+4")].copy()
    if train_val.empty or test.empty:
        raise ValueError("Expected non-empty Waves1+2 train/val and Waves3+4 test rows.")
    return train_val.reset_index(drop=True), test.reset_index(drop=True)


def make_features(
    train_val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    encoder = LabelEncoder()
    encoder.fit(pd.concat([train_val["inhibitor"], test["inhibitor"]]).astype(str))

    train_work = train_val.copy()
    test_work = test.copy()
    train_work["inhibitor_encoded"] = encoder.transform(train_work["inhibitor"].astype(str))
    test_work["inhibitor_encoded"] = encoder.transform(test_work["inhibitor"].astype(str))

    x_train = train_work.drop(columns=[col for col in DROP_COLS if col in train_work.columns])
    x_test = test_work.drop(columns=[col for col in DROP_COLS if col in test_work.columns])
    x_train = x_train.select_dtypes(include=[np.number])
    x_test = x_test[x_train.columns]

    fill_values = x_train.median(numeric_only=True).fillna(0)
    x_train = x_train.fillna(fill_values).fillna(0)
    x_test = x_test.fillna(fill_values).fillna(0)

    return x_train, x_test, train_work["label_auc100"], test_work["label_auc100"]


def fit_models(x_train: pd.DataFrame, y_train: pd.Series, seed: int, xgb_iter: int):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = {}

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    models["rf"] = {
        "model": rf.fit(x_train, y_train),
        "cv_auroc_mean": float(
            cross_val_score(rf, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        ),
    }

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    xgb = XGBClassifier(
        random_state=seed,
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=neg / max(pos, 1),
    )
    param_dist = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2],
    }
    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=xgb_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    models["xgb"] = {
        "model": search.best_estimator_,
        "cv_auroc_mean": float(search.best_score_),
        "best_params": search.best_params_,
    }

    return models


def build_predictions(
    model_name: str,
    model,
    x_test: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    scores = model.predict_proba(x_test)[:, 1]
    return pd.DataFrame(
        {
            "model": model_name,
            "patient_id": test["dbgap_subject_id"].astype(str),
            "drug_name": test["inhibitor"].astype(str),
            "cohort": test["cohort"].astype(str),
            "auc_true": pd.to_numeric(test["auc"], errors="coerce"),
            "label_auc100": test["label_auc100"].astype(int),
            "score_sensitive": scores,
            "pred_label_auc100": (scores >= 0.5).astype(int),
            "threshold_used": 0.5,
        }
    )


def per_drug_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, drug_name), group in predictions.groupby(["model", "drug_name"]):
        y_true = group["label_auc100"].astype(int).to_numpy()
        y_score = group["score_sensitive"].astype(float).to_numpy()
        metrics = binary_metrics(y_true, y_score, threshold=0.5)
        rows.append(
            {
                "model": model_name,
                "drug": drug_name,
                "n_test": int(len(group)),
                "n_pos": int((y_true == 1).sum()),
                "n_neg": int((y_true == 0).sum()),
                "pos_rate": float(np.mean(y_true)),
                "auroc": safe_roc_auc(y_true, y_score),
                "ap": safe_average_precision(y_true, y_score),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "drug"]).reset_index(drop=True)


def ktsp_compatible_metrics(
    predictions: pd.DataFrame,
    per_drug: pd.DataFrame,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model_name in sorted(predictions["model"].unique()):
        model_metrics = per_drug[per_drug["model"].eq(model_name)]
        for drug_name in sorted(set(train_val["inhibitor"]).union(set(test["inhibitor"]))):
            tr = train_val[train_val["inhibitor"].eq(drug_name)]
            te = test[test["inhibitor"].eq(drug_name)]
            train_pos = int((tr["label_auc100"] == 1).sum())
            train_neg = int((tr["label_auc100"] == 0).sum())
            test_pos = int((te["label_auc100"] == 1).sum())
            test_neg = int((te["label_auc100"] == 0).sum())
            metric_row = model_metrics[model_metrics["drug"].eq(drug_name)]
            metric_values = metric_row.iloc[0].to_dict() if len(metric_row) else {}
            rows.append(
                {
                    "model": model_name,
                    "drug": drug_name,
                    "ktsp_eligible": bool(
                        train_pos >= 20
                        and train_neg >= 20
                        and test_pos >= 10
                        and test_neg >= 10
                    ),
                    "train_sensitive_auc100": train_pos,
                    "train_resistant_auc100": train_neg,
                    "test_sensitive_auc100": test_pos,
                    "test_resistant_auc100": test_neg,
                    "n_test": metric_values.get("n_test", 0),
                    "auroc": metric_values.get("auroc", np.nan),
                    "ap": metric_values.get("ap", np.nan),
                    "balanced_accuracy": metric_values.get("balanced_accuracy", np.nan),
                    "f1": metric_values.get("f1", np.nan),
                    "sensitivity": metric_values.get("sensitivity", np.nan),
                    "specificity": metric_values.get("specificity", np.nan),
                }
            )
    return pd.DataFrame(rows)


def summary_table(
    predictions: pd.DataFrame,
    per_drug: pd.DataFrame,
    ktsp_compatible: pd.DataFrame,
    cv_scores: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model"):
        y_true = group["label_auc100"].astype(int).to_numpy()
        y_score = group["score_sensitive"].astype(float).to_numpy()
        global_metrics = binary_metrics(y_true, y_score, threshold=0.5)
        model_per_drug = per_drug[per_drug["model"].eq(model_name)]
        ktsp_subset = ktsp_compatible[
            ktsp_compatible["model"].eq(model_name) & ktsp_compatible["ktsp_eligible"]
        ]
        rows.append(
            {
                "model": model_name,
                "cv_auroc_mean": cv_scores.get(model_name, np.nan),
                "threshold": 0.5,
                "n_test_edges": int(len(group)),
                "n_test_drugs": int(model_per_drug["drug"].nunique()),
                "test_auroc": safe_roc_auc(y_true, y_score),
                "test_ap": safe_average_precision(y_true, y_score),
                "test_balanced_accuracy": global_metrics["balanced_accuracy"],
                "test_f1": global_metrics["f1"],
                "test_accuracy": global_metrics["accuracy"],
                "macro_per_drug_auroc": model_per_drug["auroc"].mean(skipna=True),
                "macro_per_drug_bacc": model_per_drug["balanced_accuracy"].mean(
                    skipna=True
                ),
                "n_ktsp_eligible_drugs": int(ktsp_subset["drug"].nunique()),
                "ktsp_subset_macro_auroc": ktsp_subset["auroc"].mean(skipna=True),
                "ktsp_subset_macro_bacc": ktsp_subset["balanced_accuracy"].mean(
                    skipna=True
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb-iter", type=int, default=20)
    args = parser.parse_args(argv)

    ensure_dirs(TABULAR_INTERNAL_TABLES)
    train_val, test = load_protocol_data()
    x_train, x_test, y_train, _ = make_features(train_val, test)
    models = fit_models(x_train, y_train, args.seed, args.xgb_iter)

    prediction_frames = [
        build_predictions(name, values["model"], x_test, test)
        for name, values in models.items()
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    per_drug = per_drug_metrics(predictions)
    ktsp_compatible = ktsp_compatible_metrics(predictions, per_drug, train_val, test)
    cv_scores = {name: values["cv_auroc_mean"] for name, values in models.items()}
    summary = summary_table(predictions, per_drug, ktsp_compatible, cv_scores)

    predictions.to_csv(TABULAR_INTERNAL_TABLES / "predictions.csv", index=False)
    per_drug.to_csv(TABULAR_INTERNAL_TABLES / "per_drug_metrics.csv", index=False)
    ktsp_compatible.to_csv(
        TABULAR_INTERNAL_TABLES / "ktsp_compatible_metrics.csv", index=False
    )
    summary.to_csv(TABULAR_INTERNAL_TABLES / "summary.csv", index=False)

    print("Internal BeatAML tabular baseline finished")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
