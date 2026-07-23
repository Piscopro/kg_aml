from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pipeline.common.benchmark_utils import (
    DATA_FEATURES,
    KG_EXPORT,
    ensure_dirs,
    resolve_repo_path,
)


DEFAULT_BASE_EXPORT = KG_EXPORT / "beataml_primekg_pathway_disease"
DEFAULT_OUTPUT_EXPORT = KG_EXPORT / "beataml_primekg_pathway_disease_embeddings"
DEFAULT_FEATURE_DIR = DATA_FEATURES / "primekg" / "beataml_primekg_pathway_disease_embeddings"


def copy_base_export(base_dir: Path, output_dir: Path) -> None:
    ensure_dirs(output_dir)
    for path in base_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def add_edge(records: list[tuple[str, str]], left: str, right: str) -> None:
    if left and right and left != right:
        records.append((left, right))


def build_primekg_context_edges(base_dir: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []

    drug_gene_path = base_dir / "drug_gene_edges.csv"
    if drug_gene_path.exists():
        edges = pd.read_csv(drug_gene_path)
        for row in edges.dropna(subset=["inhibitor", "gene_name"]).itertuples(index=False):
            add_edge(records, f"drug::{row.inhibitor}", f"gene::{row.gene_name}")

    ppi_path = base_dir / "ppi_edges.csv"
    if ppi_path.exists():
        edges = pd.read_csv(ppi_path)
        for row in edges.dropna(subset=["gene_a", "gene_b"]).itertuples(index=False):
            add_edge(records, f"gene::{row.gene_a}", f"gene::{row.gene_b}")

    gene_pathway_path = base_dir / "gene_pathway_edges.csv"
    if gene_pathway_path.exists():
        edges = pd.read_csv(gene_pathway_path)
        for row in edges.dropna(subset=["gene_name", "pathway_id"]).itertuples(index=False):
            add_edge(records, f"gene::{row.gene_name}", f"pathway::{row.pathway_id}")

    disease_gene_path = base_dir / "disease_gene_edges.csv"
    if disease_gene_path.exists():
        edges = pd.read_csv(disease_gene_path)
        for row in edges.dropna(subset=["disease_id", "gene_name"]).itertuples(index=False):
            add_edge(records, f"disease::{row.disease_id}", f"gene::{row.gene_name}")

    return records


def build_normalized_adjacency(seed_nodes: list[str], edges: list[tuple[str, str]]) -> tuple[np.ndarray, list[str]]:
    node_order = list(dict.fromkeys(seed_nodes + [node for edge in edges for node in edge]))
    node_index = {node: idx for idx, node in enumerate(node_order)}
    adjacency = np.zeros((len(node_order), len(node_order)), dtype=np.float32)
    for left, right in edges:
        left_idx = node_index[left]
        right_idx = node_index[right]
        adjacency[left_idx, right_idx] = 1.0
        adjacency[right_idx, left_idx] = 1.0

    adjacency += np.eye(len(node_order), dtype=np.float32)
    degree = adjacency.sum(axis=1)
    degree[degree == 0] = 1.0
    scale = 1.0 / np.sqrt(degree)
    normalized = adjacency * scale[:, None] * scale[None, :]
    return normalized, node_order


def svd_embeddings(matrix: np.ndarray, dim: int) -> np.ndarray:
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    use_dim = min(dim, len(singular_values))
    embedding = u[:, :use_dim] * np.sqrt(singular_values[:use_dim])
    if use_dim < dim:
        padding = np.zeros((embedding.shape[0], dim - use_dim), dtype=embedding.dtype)
        embedding = np.hstack([embedding, padding])
    return embedding.astype(np.float32)


def align_rows(
    node_names: list[str],
    embedding: np.ndarray,
    row_ids: list[str],
    id_column: str,
    dim: int,
) -> pd.DataFrame:
    index = {node: idx for idx, node in enumerate(node_names)}
    columns = [f"primekg_emb_{idx:03d}" for idx in range(dim)]
    rows = []
    for row_id in row_ids:
        values = np.zeros(dim, dtype=np.float32)
        node = f"{id_column.split('_')[0]}::{row_id}"
        if node in index:
            values = embedding[index[node]]
        rows.append([row_id, *values.tolist()])
    return pd.DataFrame(rows, columns=[id_column, *columns])


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(KG_EXPORT.parent.parent.parent).as_posix()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-kg-export-dir", type=str, default=str(DEFAULT_BASE_EXPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_EXPORT))
    parser.add_argument("--feature-dir", type=str, default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument("--embedding-dim", type=int, default=32)
    args = parser.parse_args(argv)
    base_dir = resolve_repo_path(args.base_kg_export_dir)
    output_dir = resolve_repo_path(args.output_dir)
    feature_dir = resolve_repo_path(args.feature_dir)

    print("1. Copying base KG variant files...")
    copy_base_export(base_dir, output_dir)
    ensure_dirs(feature_dir)

    print("2. Building filtered PrimeKG context adjacency...")
    drug_nodes = pd.read_csv(base_dir / "drug_nodes.csv")
    gene_nodes = pd.read_csv(base_dir / "gene_nodes.csv")
    drug_ids = drug_nodes["inhibitor"].dropna().astype(str).tolist()
    gene_ids = gene_nodes["gene_name"].dropna().astype(str).tolist()
    seed_nodes = [f"drug::{drug}" for drug in drug_ids] + [
        f"gene::{gene}" for gene in gene_ids
    ]
    context_edges = build_primekg_context_edges(base_dir)
    matrix, node_names = build_normalized_adjacency(seed_nodes, context_edges)

    print("3. Computing deterministic SVD embeddings...")
    embedding = svd_embeddings(matrix, args.embedding_dim)
    drug_embeddings = align_rows(
        node_names, embedding, drug_ids, "drug_id", args.embedding_dim
    ).rename(columns={"drug_id": "inhibitor"})
    gene_embeddings = align_rows(
        node_names, embedding, gene_ids, "gene_id", args.embedding_dim
    ).rename(columns={"gene_id": "gene_name"})

    drug_path = feature_dir / "drug_primekg_graph_embeddings.csv"
    gene_path = feature_dir / "gene_primekg_graph_embeddings.csv"
    drug_embeddings.to_csv(drug_path, index=False)
    gene_embeddings.to_csv(gene_path, index=False)

    print("4. Writing embedding feature config...")
    config = {
        "feature_variant": output_dir.name,
        "base_export": base_dir.relative_to(KG_EXPORT.parent).as_posix(),
        "method": "normalized_adjacency_svd",
        "embedding_dim": int(args.embedding_dim),
        "context_edges": int(len(context_edges)),
        "context_nodes": int(len(node_names)),
        "drug_embedding_path": repo_relative(drug_path),
        "gene_embedding_path": repo_relative(gene_path),
        "drug_rows": int(len(drug_embeddings)),
        "gene_rows": int(len(gene_embeddings)),
    }
    with (output_dir / "primekg_feature_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    metadata_path = output_dir / "variant_metadata.json"
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    metadata.update(
        {
            "export_variant": output_dir.name,
            "base_export": base_dir.relative_to(KG_EXPORT.parent).as_posix(),
            "primekg_embedding_features": config,
        }
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("PrimeKG embedding feature KG export built")
    print(f"  output: {output_dir}")
    print(f"  feature dir: {feature_dir}")
    print(f"  context nodes: {len(node_names)}")
    print(f"  context edges: {len(context_edges)}")
    print(f"  embedding dim: {args.embedding_dim}")
    print("  default KG export was not modified")


if __name__ == "__main__":
    main()
