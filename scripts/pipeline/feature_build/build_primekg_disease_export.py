from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.pipeline.common.benchmark_utils import (
    DATA_RAW,
    KG_EXPORT,
    ensure_dirs,
    resolve_repo_path,
)


DEFAULT_BASE_EXPORT = KG_EXPORT / "beataml_primekg_pathway"
DEFAULT_OUTPUT_EXPORT = KG_EXPORT / "beataml_primekg_pathway_disease"
PRIMEKG_PATH = DATA_RAW / "primekg" / "kg.csv"
AML_DISEASE_PATTERN = (
    r"acute myeloid|myeloid leukemia|leukemia, myeloid|myeloid leukaemia"
)
DISEASE_NODE_COLUMNS = [
    "disease_id",
    "disease_name",
    "node_type",
    "primekg_id",
    "primekg_source",
    "n_genes",
]
DISEASE_GENE_COLUMNS = [
    "disease_id",
    "disease_name",
    "gene_name",
    "relationship",
    "source",
    "primekg_relation",
    "primekg_display_relation",
    "primekg_disease_id",
    "primekg_disease_source",
    "primekg_gene_id",
    "primekg_gene_source",
]


def copy_base_export(base_dir: Path, output_dir: Path) -> None:
    ensure_dirs(output_dir)
    for path in base_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def gene_lookup(gene_nodes: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for gene_name in gene_nodes["gene_name"].dropna().astype(str):
        lookup.setdefault(gene_name.upper(), gene_name)
    return lookup


def add_disease_gene_records(
    records: dict[tuple[str, str], dict[str, object]],
    frame: pd.DataFrame,
    genes: dict[str, str],
    disease_prefix: str,
    gene_prefix: str,
) -> None:
    for row in frame.itertuples(index=False):
        gene_key = str(getattr(row, f"{gene_prefix}_name")).upper()
        gene_name = genes.get(gene_key)
        if gene_name is None:
            continue

        disease_id = f"disease:{getattr(row, f'{disease_prefix}_id')}"
        key = (disease_id, gene_name)
        records.setdefault(
            key,
            {
                "disease_id": disease_id,
                "disease_name": getattr(row, f"{disease_prefix}_name"),
                "gene_name": gene_name,
                "relationship": "ASSOCIATED_WITH",
                "source": "PrimeKG",
                "primekg_relation": row.relation,
                "primekg_display_relation": row.display_relation,
                "primekg_disease_id": str(getattr(row, f"{disease_prefix}_id")),
                "primekg_disease_source": getattr(row, f"{disease_prefix}_source"),
                "primekg_gene_id": str(getattr(row, f"{gene_prefix}_id")),
                "primekg_gene_source": getattr(row, f"{gene_prefix}_source"),
            },
        )


def collect_disease_gene_edges(
    primekg_path: Path,
    genes: dict[str, str],
    chunksize: int,
) -> pd.DataFrame:
    if not primekg_path.exists():
        raise FileNotFoundError(f"Missing PrimeKG file: {primekg_path}")

    usecols = [
        "relation",
        "display_relation",
        "x_id",
        "x_type",
        "x_name",
        "x_source",
        "y_id",
        "y_type",
        "y_name",
        "y_source",
    ]
    gene_keys = set(genes)
    records: dict[tuple[str, str], dict[str, object]] = {}

    for chunk in pd.read_csv(
        primekg_path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        edges = chunk[
            chunk["relation"].eq("disease_protein")
            & chunk["display_relation"].eq("associated with")
        ].copy()
        if edges.empty:
            continue

        edges["x_gene_key"] = edges["x_name"].astype(str).str.upper()
        edges["y_gene_key"] = edges["y_name"].astype(str).str.upper()
        x_disease = edges[
            edges["x_type"].eq("disease")
            & edges["x_name"].astype(str).str.contains(
                AML_DISEASE_PATTERN, case=False, regex=True, na=False
            )
            & edges["y_type"].eq("gene/protein")
            & edges["y_gene_key"].isin(gene_keys)
        ]
        add_disease_gene_records(records, x_disease, genes, "x", "y")

        y_disease = edges[
            edges["y_type"].eq("disease")
            & edges["y_name"].astype(str).str.contains(
                AML_DISEASE_PATTERN, case=False, regex=True, na=False
            )
            & edges["x_type"].eq("gene/protein")
            & edges["x_gene_key"].isin(gene_keys)
        ]
        add_disease_gene_records(records, y_disease, genes, "y", "x")

    if not records:
        return pd.DataFrame(columns=DISEASE_GENE_COLUMNS)

    return (
        pd.DataFrame(records.values(), columns=DISEASE_GENE_COLUMNS)
        .sort_values(["disease_name", "gene_name"])
        .reset_index(drop=True)
    )


def filter_disease_edges(
    edges: pd.DataFrame, min_genes: int, max_diseases: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty:
        return (
            pd.DataFrame(columns=DISEASE_NODE_COLUMNS),
            pd.DataFrame(columns=DISEASE_GENE_COLUMNS),
        )

    summary = (
        edges.groupby(["disease_id", "disease_name"], as_index=False)
        .agg(
            n_genes=("gene_name", "nunique"),
            primekg_id=("primekg_disease_id", "first"),
            primekg_source=("primekg_disease_source", "first"),
        )
    )
    summary = summary[summary["n_genes"] >= min_genes].copy()
    summary = summary.sort_values(
        ["n_genes", "disease_name"], ascending=[False, True]
    ).head(max_diseases)

    keep_ids = set(summary["disease_id"].astype(str))
    kept_edges = edges[edges["disease_id"].astype(str).isin(keep_ids)].copy()
    disease_nodes = summary.copy()
    disease_nodes["node_type"] = "Disease"
    disease_nodes = disease_nodes[DISEASE_NODE_COLUMNS]
    return disease_nodes.reset_index(drop=True), kept_edges.reset_index(drop=True)


def merge_metadata(output_dir: Path, payload: dict[str, object]) -> None:
    path = output_dir / "variant_metadata.json"
    existing: dict[str, object] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    existing.update(payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-kg-export-dir", type=str, default=str(DEFAULT_BASE_EXPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_EXPORT))
    parser.add_argument("--min-genes-per-disease", type=int, default=2)
    parser.add_argument("--max-diseases", type=int, default=25)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args(argv)
    base_dir = resolve_repo_path(args.base_kg_export_dir)
    output_dir = resolve_repo_path(args.output_dir)

    print("1. Copying base KG variant files...")
    copy_base_export(base_dir, output_dir)

    print("2. Collecting PrimeKG AML disease-gene edges...")
    gene_nodes = pd.read_csv(base_dir / "gene_nodes.csv")
    genes = gene_lookup(gene_nodes)
    raw_edges = collect_disease_gene_edges(PRIMEKG_PATH, genes, args.chunksize)

    print("3. Filtering disease nodes...")
    disease_nodes, disease_gene_edges = filter_disease_edges(
        raw_edges,
        min_genes=args.min_genes_per_disease,
        max_diseases=args.max_diseases,
    )

    print("4. Writing disease KG variant...")
    disease_nodes.to_csv(output_dir / "disease_nodes.csv", index=False)
    disease_gene_edges.to_csv(output_dir / "disease_gene_edges.csv", index=False)
    merge_metadata(
        output_dir,
        {
            "export_variant": output_dir.name,
            "base_export": base_dir.relative_to(KG_EXPORT.parent).as_posix(),
            "disease_filter": {
                "relation": "disease_protein",
                "display_relation": "associated with",
                "disease_name_pattern": AML_DISEASE_PATTERN,
                "min_genes_per_disease": args.min_genes_per_disease,
                "max_diseases": args.max_diseases,
            },
            "disease_nodes": int(len(disease_nodes)),
            "disease_gene_edges": int(len(disease_gene_edges)),
        },
    )

    print("PrimeKG AML disease KG export built")
    print(f"  output: {output_dir}")
    print(f"  genes: {len(gene_nodes)}")
    print(f"  disease nodes retained: {len(disease_nodes)}")
    print(f"  disease-gene edges retained: {len(disease_gene_edges)}")
    print("  default KG export was not modified")


if __name__ == "__main__":
    main()
