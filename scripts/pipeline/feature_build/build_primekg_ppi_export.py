from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.pipeline.common.benchmark_utils import DATA_RAW, KG_EXPORT, ensure_dirs


BASE_EXPORT_FILES = (
    "patient_nodes.csv",
    "drug_nodes.csv",
    "gene_nodes.csv",
    "mutation_edges.csv",
    "treatment_edges.csv",
    "drug_gene_edges.csv",
)
OUTPUT_DIR = KG_EXPORT / "beataml_primekg_ppi"
PRIMEKG_PATH = DATA_RAW / "primekg" / "kg.csv"
PPI_COLUMNS = [
    "gene_a",
    "gene_b",
    "relationship",
    "source",
    "primekg_relation",
    "primekg_display_relation",
    "gene_a_primekg_id",
    "gene_b_primekg_id",
    "gene_a_primekg_source",
    "gene_b_primekg_source",
]


def copy_base_export(output_dir: Path) -> None:
    ensure_dirs(output_dir)
    for file_name in BASE_EXPORT_FILES:
        source = KG_EXPORT / file_name
        if not source.exists():
            raise FileNotFoundError(f"Missing base KG export file: {source}")
        shutil.copy2(source, output_dir / file_name)


def gene_lookup(gene_nodes: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for gene_name in gene_nodes["gene_name"].dropna().astype(str):
        lookup.setdefault(gene_name.upper(), gene_name)
    return lookup


def build_ppi_edges(
    primekg_path: Path, genes: dict[str, str], chunksize: int = 500_000
) -> pd.DataFrame:
    if not primekg_path.exists():
        raise FileNotFoundError(f"Missing PrimeKG file: {primekg_path}")

    usecols = [
        "relation",
        "display_relation",
        "x_id",
        "x_name",
        "x_source",
        "y_id",
        "y_name",
        "y_source",
    ]
    edges: dict[tuple[str, str], dict[str, object]] = {}
    gene_keys = set(genes)

    for chunk in pd.read_csv(
        primekg_path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        ppi = chunk[
            chunk["relation"].eq("protein_protein")
            & chunk["display_relation"].eq("ppi")
        ].copy()
        if ppi.empty:
            continue

        ppi["x_gene_key"] = ppi["x_name"].astype(str).str.upper()
        ppi["y_gene_key"] = ppi["y_name"].astype(str).str.upper()
        ppi = ppi[
            ppi["x_gene_key"].isin(gene_keys) & ppi["y_gene_key"].isin(gene_keys)
        ]
        if ppi.empty:
            continue

        for row in ppi.itertuples(index=False):
            x_gene = genes[row.x_gene_key]
            y_gene = genes[row.y_gene_key]
            if x_gene == y_gene:
                continue

            gene_a, gene_b = sorted((x_gene, y_gene))
            if gene_a == x_gene:
                gene_a_id, gene_b_id = row.x_id, row.y_id
                gene_a_source, gene_b_source = row.x_source, row.y_source
            else:
                gene_a_id, gene_b_id = row.y_id, row.x_id
                gene_a_source, gene_b_source = row.y_source, row.x_source

            edges.setdefault(
                (gene_a, gene_b),
                {
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "relationship": "PPI",
                    "source": "PrimeKG",
                    "primekg_relation": row.relation,
                    "primekg_display_relation": row.display_relation,
                    "gene_a_primekg_id": str(gene_a_id),
                    "gene_b_primekg_id": str(gene_b_id),
                    "gene_a_primekg_source": gene_a_source,
                    "gene_b_primekg_source": gene_b_source,
                },
            )

    if not edges:
        return pd.DataFrame(columns=PPI_COLUMNS)

    return (
        pd.DataFrame(edges.values(), columns=PPI_COLUMNS)
        .sort_values(["gene_a", "gene_b"])
        .reset_index(drop=True)
    )


def write_variant_json(output_dir: Path, ppi_edges: pd.DataFrame) -> None:
    source_json = KG_EXPORT / "beataml_kg.json"
    if not source_json.exists():
        raise FileNotFoundError(f"Missing base KG JSON export: {source_json}")

    with source_json.open("r", encoding="utf-8") as handle:
        kg = json.load(handle)

    kg.setdefault("edges", {})["ppi"] = ppi_edges.to_dict(orient="records")
    metadata = kg.setdefault("metadata", {})
    metadata["export_variant"] = "beataml_primekg_ppi"
    metadata["base_export"] = "data/interim/KG_Export"
    metadata["ppi_source"] = (
        "data/raw/primekg/kg.csv relation=protein_protein display_relation=ppi"
    )
    metadata["total_ppi_edges"] = int(len(ppi_edges))

    out_path = output_dir / "beataml_primekg_ppi_kg.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(kg, handle, indent=2)


def main() -> None:
    gene_nodes = pd.read_csv(KG_EXPORT / "gene_nodes.csv")
    genes = gene_lookup(gene_nodes)

    print("1. Copying base BeatAML KG export...")
    copy_base_export(OUTPUT_DIR)

    print("2. Filtering PrimeKG PPI edges to current BeatAML gene nodes...")
    ppi_edges = build_ppi_edges(PRIMEKG_PATH, genes)
    ppi_edges.to_csv(OUTPUT_DIR / "ppi_edges.csv", index=False)

    print("3. Writing variant JSON export...")
    write_variant_json(OUTPUT_DIR, ppi_edges)

    print("PrimeKG PPI KG export built")
    print(f"  output: {OUTPUT_DIR}")
    print(f"  genes retained from base KG: {len(genes)}")
    print(f"  PPI edges retained: {len(ppi_edges)}")
    print("  base KG export was not modified")


if __name__ == "__main__":
    main()
