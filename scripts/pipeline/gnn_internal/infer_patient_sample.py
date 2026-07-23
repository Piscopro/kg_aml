from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from scripts.pipeline.common.benchmark_utils import (
    GNN_INTERNAL_CHECKPOINTS,
    GNN_INTERNAL_TABLES,
    KG_EXPORT,
)
from scripts.pipeline.gnn_internal.train import build_data, make_model, predict

KG_EXPORT_DIR = KG_EXPORT
CHECKPOINT_PATH = GNN_INTERNAL_CHECKPOINTS / "best_gat_beataml_internal.pt"
SUMMARY_PATH = GNN_INTERNAL_TABLES / "summary.csv"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--kg-export-dir", default=str(KG_EXPORT_DIR))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def matching_summary_path(checkpoint_path: Path) -> Path:
    run_name = checkpoint_path.parent.name
    if checkpoint_path.parent == GNN_INTERNAL_CHECKPOINTS:
        return SUMMARY_PATH
    return GNN_INTERNAL_TABLES / run_name / "summary.csv"


def main() -> None:
    args = parse_args()
    kg_export_dir = repo_path(args.kg_export_dir)
    checkpoint_path = repo_path(args.checkpoint)
    if not checkpoint_path.exists() and checkpoint_path.name == "best_model.pt":
        checkpoint_path = checkpoint_path.with_name("best_gat_beataml_internal.pt")
    summary_path = matching_summary_path(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        data,
        train_val_edges,
        test_edges,
        _train_val_pred_edges,
        _test_pred_edges,
        _train_val_pred_attrs,
        _test_pred_attrs,
        _train_val_labels,
        _test_labels,
        gene_dim,
        num_pathways,
        num_diseases,
        drug_dim,
        _target_edge_count,
        ppi_edge_count,
        gene_pathway_edge_count,
        disease_gene_edge_count,
        _primekg_feature_counts,
        _train_idx,
        _val_idx,
        _split_strategy,
    ) = build_data(device, val_fraction=0.2, seed=42, kg_export_dir=kg_export_dir)

    patient_nodes = pd.read_csv(kg_export_dir / "patient_nodes.csv")
    drug_nodes = pd.read_csv(kg_export_dir / "drug_nodes.csv")
    patient_map = {
        str(pid): idx for idx, pid in enumerate(patient_nodes["dbgap_subject_id"])
    }
    drug_map = {str(drug): idx for idx, drug in enumerate(drug_nodes["inhibitor"])}

    patient_id = str(args.patient_id or test_edges.iloc[0]["patient_id"])
    edges = pd.concat([train_val_edges, test_edges], ignore_index=True)
    sample_edges = edges[edges["patient_id"].astype(str).eq(patient_id)].copy()
    if sample_edges.empty:
        raise ValueError(f"No treatment rows found for patient {patient_id}")

    pred_edge_index = torch.tensor(
        [
            [patient_map[patient_id] for _ in range(len(sample_edges))],
            [drug_map[str(drug)] for drug in sample_edges["drug_name"]],
        ],
        dtype=torch.long,
        device=device,
    )
    pred_edge_attr = torch.tensor(
        sample_edges[["target_match", "target_rel_exp"]].values,
        dtype=torch.float32,
        device=device,
    )

    model = make_model(
        gene_dim=gene_dim,
        num_pathways=num_pathways,
        num_diseases=num_diseases,
        drug_dim=drug_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        heads=4,
        use_ppi=ppi_edge_count > 0,
        use_pathway=gene_pathway_edge_count > 0,
        use_disease=disease_gene_edge_count > 0,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    threshold = 0.5
    if summary_path.exists():
        threshold = float(pd.read_csv(summary_path).iloc[0].get("threshold", threshold))

    sample_edges["score_sensitive"] = predict(
        model, data, pred_edge_index, pred_edge_attr, batch_size=65536
    )
    sample_edges["pred_sensitive"] = (
        sample_edges["score_sensitive"].ge(threshold).astype(int)
    )

    cols = [
        "patient_id",
        "drug_name",
        "score_sensitive",
        "pred_sensitive",
        "auc",
        "label_auc100",
        "target_match",
        "target_rel_exp",
    ]
    print(
        sample_edges.sort_values("score_sensitive", ascending=False)
        .head(args.top_k)[cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
