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
    normalize_drug_name,
    resolve_repo_path,
)


DEFAULT_BASE_EXPORT = KG_EXPORT / "beataml_primekg_ppi"
DEFAULT_OUTPUT_EXPORT = KG_EXPORT / "beataml_primekg_ppi_targets"
PRIMEKG_PATH = DATA_RAW / "primekg" / "kg.csv"
BASE_EXPORT_FILES = (
    "patient_nodes.csv",
    "drug_nodes.csv",
    "gene_nodes.csv",
    "mutation_edges.csv",
    "treatment_edges.csv",
    "ppi_edges.csv",
)
TARGET_METADATA_COLUMNS = [
    "inhibitor",
    "gene_name",
    "sources",
    "primekg_relation",
    "primekg_display_relation",
    "primekg_drug_name",
    "primekg_drug_id",
    "primekg_drug_source",
    "primekg_gene_name",
    "primekg_gene_id",
    "primekg_gene_source",
]


def copy_base_files(base_dir: Path, output_dir: Path) -> None:
    ensure_dirs(output_dir)
    for file_name in BASE_EXPORT_FILES:
        source = base_dir / file_name
        if source.exists():
            shutil.copy2(source, output_dir / file_name)


def canonical_gene_lookup(gene_nodes: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for gene_name in gene_nodes["gene_name"].dropna().astype(str):
        lookup.setdefault(gene_name.upper(), gene_name)
    return lookup


def unique_normalized_drug_lookup(drug_nodes: pd.DataFrame) -> dict[str, str]:
    drugs = drug_nodes["inhibitor"].dropna().astype(str).drop_duplicates()
    normalized = drugs.map(normalize_drug_name)
    counts = normalized.value_counts()
    lookup: dict[str, str] = {}
    for drug_name, key in zip(drugs, normalized, strict=False):
        if counts.get(key, 0) == 1:
            lookup[key] = drug_name
    return lookup


def add_target_records(
    records: dict[tuple[str, str], dict[str, object]],
    frame: pd.DataFrame,
    drug_lookup: dict[str, str],
    gene_lookup: dict[str, str],
    drug_prefix: str,
    gene_prefix: str,
) -> None:
    drug_norm_col = f"{drug_prefix}_norm"
    gene_key_col = f"{gene_prefix}_gene_key"
    frame = frame.copy()
    frame["inhibitor"] = frame[drug_norm_col].map(drug_lookup)
    frame["gene_name"] = frame[gene_key_col].map(gene_lookup)
    frame = frame.dropna(subset=["inhibitor", "gene_name"])

    for row in frame.itertuples(index=False):
        inhibitor = str(row.inhibitor)
        gene_name = str(row.gene_name)
        key = (inhibitor, gene_name)
        records.setdefault(
            key,
            {
                "inhibitor": inhibitor,
                "gene_name": gene_name,
                "sources": "PrimeKG",
                "primekg_relation": row.relation,
                "primekg_display_relation": row.display_relation,
                "primekg_drug_name": getattr(row, f"{drug_prefix}_name"),
                "primekg_drug_id": str(getattr(row, f"{drug_prefix}_id")),
                "primekg_drug_source": getattr(row, f"{drug_prefix}_source"),
                "primekg_gene_name": getattr(row, f"{gene_prefix}_name"),
                "primekg_gene_id": str(getattr(row, f"{gene_prefix}_id")),
                "primekg_gene_source": getattr(row, f"{gene_prefix}_source"),
            },
        )


def build_primekg_target_edges(
    primekg_path: Path,
    drug_lookup: dict[str, str],
    gene_lookup: dict[str, str],
    chunksize: int = 500_000,
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
    records: dict[tuple[str, str], dict[str, object]] = {}
    gene_keys = set(gene_lookup)

    for chunk in pd.read_csv(
        primekg_path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        targets = chunk[
            chunk["relation"].eq("drug_protein")
            & chunk["display_relation"].eq("target")
        ].copy()
        if targets.empty:
            continue

        targets["x_norm"] = targets["x_name"].map(normalize_drug_name)
        targets["y_norm"] = targets["y_name"].map(normalize_drug_name)
        targets["x_gene_key"] = targets["x_name"].astype(str).str.upper()
        targets["y_gene_key"] = targets["y_name"].astype(str).str.upper()

        x_drug = targets[
            targets["x_type"].eq("drug")
            & targets["x_norm"].isin(drug_lookup)
            & targets["y_type"].eq("gene/protein")
            & targets["y_gene_key"].isin(gene_keys)
        ]
        add_target_records(
            records, x_drug, drug_lookup, gene_lookup, "x", "y"
        )

        y_drug = targets[
            targets["y_type"].eq("drug")
            & targets["y_norm"].isin(drug_lookup)
            & targets["x_type"].eq("gene/protein")
            & targets["x_gene_key"].isin(gene_keys)
        ]
        add_target_records(
            records, y_drug, drug_lookup, gene_lookup, "y", "x"
        )

    if not records:
        return pd.DataFrame(columns=TARGET_METADATA_COLUMNS)

    return (
        pd.DataFrame(records.values(), columns=TARGET_METADATA_COLUMNS)
        .sort_values(["inhibitor", "gene_name"])
        .reset_index(drop=True)
    )


def build_target_metadata(
    base_edges: pd.DataFrame, primekg_edges: pd.DataFrame
) -> pd.DataFrame:
    base_meta = base_edges[["inhibitor", "gene_name"]].drop_duplicates().copy()
    base_meta["sources"] = "BeatAML"
    for column in TARGET_METADATA_COLUMNS:
        if column not in base_meta.columns:
            base_meta[column] = ""

    combined = pd.concat(
        [base_meta[TARGET_METADATA_COLUMNS], primekg_edges[TARGET_METADATA_COLUMNS]],
        ignore_index=True,
    )

    rows = []
    for (inhibitor, gene_name), group in combined.groupby(["inhibitor", "gene_name"]):
        sources = sorted(
            source
            for value in group["sources"].dropna().astype(str)
            for source in value.split(";")
            if source
        )
        prime = group[group["sources"].astype(str).str.contains("PrimeKG", na=False)]
        row = group.iloc[0].to_dict()
        if not prime.empty:
            row.update(prime.iloc[0].to_dict())
        row["inhibitor"] = inhibitor
        row["gene_name"] = gene_name
        row["sources"] = ";".join(dict.fromkeys(sources))
        rows.append(row)

    return (
        pd.DataFrame(rows, columns=TARGET_METADATA_COLUMNS)
        .sort_values(["inhibitor", "gene_name"])
        .reset_index(drop=True)
    )


def write_metadata(
    output_dir: Path,
    base_dir: Path,
    base_edges: pd.DataFrame,
    primekg_edges: pd.DataFrame,
    metadata_edges: pd.DataFrame,
) -> None:
    overlap = metadata_edges["sources"].astype(str).str.contains(
        "BeatAML;PrimeKG", regex=False
    )
    payload = {
        "export_variant": output_dir.name,
        "base_export": base_dir.relative_to(KG_EXPORT.parent).as_posix(),
        "primekg_source": (
            "data/raw/primekg/kg.csv relation=drug_protein display_relation=target"
        ),
        "base_target_edges": int(len(base_edges.drop_duplicates())),
        "primekg_target_edges": int(len(primekg_edges)),
        "overlap_target_edges": int(overlap.sum()),
        "union_target_edges": int(len(metadata_edges)),
    }
    with (output_dir / "variant_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-kg-export-dir", type=str, default=str(DEFAULT_BASE_EXPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_EXPORT))
    args = parser.parse_args(argv)
    base_dir = resolve_repo_path(args.base_kg_export_dir)
    output_dir = resolve_repo_path(args.output_dir)

    print("1. Copying base KG variant files...")
    copy_base_files(base_dir, output_dir)

    print("2. Loading base nodes and target edges...")
    drug_nodes = pd.read_csv(base_dir / "drug_nodes.csv")
    gene_nodes = pd.read_csv(base_dir / "gene_nodes.csv")
    base_edges = (
        pd.read_csv(base_dir / "drug_gene_edges.csv")[["inhibitor", "gene_name"]]
        .dropna()
        .drop_duplicates()
    )
    drug_lookup = unique_normalized_drug_lookup(drug_nodes)
    gene_lookup = canonical_gene_lookup(gene_nodes)

    print("3. Filtering PrimeKG drug-protein target edges...")
    primekg_edges = build_primekg_target_edges(PRIMEKG_PATH, drug_lookup, gene_lookup)
    metadata_edges = build_target_metadata(base_edges, primekg_edges)
    union_edges = metadata_edges[["inhibitor", "gene_name"]].copy()

    print("4. Writing target-supplement KG variant...")
    union_edges.to_csv(output_dir / "drug_gene_edges.csv", index=False)
    primekg_edges.to_csv(output_dir / "primekg_drug_gene_edges.csv", index=False)
    metadata_edges.to_csv(output_dir / "drug_gene_edges_metadata.csv", index=False)
    write_metadata(output_dir, base_dir, base_edges, primekg_edges, metadata_edges)

    print("PrimeKG target-supplement KG export built")
    print(f"  output: {output_dir}")
    print(f"  base target edges: {len(base_edges)}")
    print(f"  PrimeKG target edges retained: {len(primekg_edges)}")
    print(f"  union target edges: {len(union_edges)}")
    print("  default KG export was not modified")


if __name__ == "__main__":
    main()
