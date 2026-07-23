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


DEFAULT_BASE_EXPORT = KG_EXPORT / "beataml_primekg_ppi_targets"
DEFAULT_OUTPUT_EXPORT = KG_EXPORT / "beataml_primekg_pathway"
PRIMEKG_PATH = DATA_RAW / "primekg" / "kg.csv"
PATHWAY_KEYWORD_PATTERN = (
    r"apoptosis|cell death|proliferation|cell cycle|dna repair|dna damage|kinase|"
    r"phosphorylation|hematopo|myeloid|leukocyte|immune|differentiation|"
    r"transcription|p53|tp53|jak|stat|ras|mapk|flt3|chromatin|methylation|splicing"
)
PATHWAY_NODE_COLUMNS = [
    "pathway_id",
    "pathway_name",
    "node_type",
    "primekg_type",
    "primekg_id",
    "primekg_source",
    "n_genes",
    "score",
]
GENE_PATHWAY_COLUMNS = [
    "gene_name",
    "pathway_id",
    "pathway_name",
    "relationship",
    "source",
    "primekg_relation",
    "primekg_display_relation",
    "primekg_gene_id",
    "primekg_gene_source",
    "primekg_pathway_id",
    "primekg_pathway_source",
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


def add_gene_pathway_records(
    records: dict[tuple[str, str], dict[str, object]],
    frame: pd.DataFrame,
    genes: dict[str, str],
    gene_prefix: str,
    pathway_prefix: str,
) -> None:
    for row in frame.itertuples(index=False):
        gene_key = str(getattr(row, f"{gene_prefix}_name")).upper()
        gene_name = genes.get(gene_key)
        if gene_name is None:
            continue

        pathway_id = f"{getattr(row, f'{pathway_prefix}_type')}:{getattr(row, f'{pathway_prefix}_id')}"
        key = (gene_name, pathway_id)
        records.setdefault(
            key,
            {
                "gene_name": gene_name,
                "pathway_id": pathway_id,
                "pathway_name": getattr(row, f"{pathway_prefix}_name"),
                "relationship": "PARTICIPATES_IN",
                "source": "PrimeKG",
                "primekg_relation": row.relation,
                "primekg_display_relation": row.display_relation,
                "primekg_gene_id": str(getattr(row, f"{gene_prefix}_id")),
                "primekg_gene_source": getattr(row, f"{gene_prefix}_source"),
                "primekg_pathway_id": str(getattr(row, f"{pathway_prefix}_id")),
                "primekg_pathway_source": getattr(row, f"{pathway_prefix}_source"),
                "primekg_type": getattr(row, f"{pathway_prefix}_type"),
            },
        )


def collect_gene_pathway_edges(
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
            (
                chunk["relation"].eq("pathway_protein")
                | chunk["relation"].eq("bioprocess_protein")
            )
            & chunk["display_relation"].eq("interacts with")
        ].copy()
        if edges.empty:
            continue

        edges["x_gene_key"] = edges["x_name"].astype(str).str.upper()
        edges["y_gene_key"] = edges["y_name"].astype(str).str.upper()

        x_gene = edges[
            edges["x_type"].eq("gene/protein")
            & edges["x_gene_key"].isin(gene_keys)
            & edges["y_type"].isin(["pathway", "biological_process"])
        ]
        add_gene_pathway_records(records, x_gene, genes, "x", "y")

        y_gene = edges[
            edges["y_type"].eq("gene/protein")
            & edges["y_gene_key"].isin(gene_keys)
            & edges["x_type"].isin(["pathway", "biological_process"])
        ]
        add_gene_pathway_records(records, y_gene, genes, "y", "x")

    if not records:
        return pd.DataFrame(columns=[*GENE_PATHWAY_COLUMNS, "primekg_type"])

    return (
        pd.DataFrame(records.values())
        .sort_values(["pathway_name", "gene_name"])
        .reset_index(drop=True)
    )


def filter_pathway_edges(
    edges: pd.DataFrame,
    min_genes: int,
    max_pathways: int,
    keyword_pattern: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty:
        return (
            pd.DataFrame(columns=PATHWAY_NODE_COLUMNS),
            pd.DataFrame(columns=GENE_PATHWAY_COLUMNS),
        )

    summary = (
        edges.groupby(["pathway_id", "pathway_name", "primekg_type"], as_index=False)
        .agg(
            n_genes=("gene_name", "nunique"),
            primekg_pathway_id=("primekg_pathway_id", "first"),
            primekg_pathway_source=("primekg_pathway_source", "first"),
        )
    )
    summary = summary[summary["n_genes"] >= min_genes].copy()
    if keyword_pattern:
        summary["keyword_match"] = summary["pathway_name"].astype(str).str.contains(
            keyword_pattern, case=False, regex=True, na=False
        )
    else:
        summary["keyword_match"] = True
    summary = summary[summary["keyword_match"]].copy()
    summary["score"] = summary["n_genes"] + summary["keyword_match"].astype(int)
    summary = summary.sort_values(
        ["score", "n_genes", "pathway_name"], ascending=[False, False, True]
    ).head(max_pathways)

    keep_ids = set(summary["pathway_id"].astype(str))
    kept_edges = edges[edges["pathway_id"].astype(str).isin(keep_ids)].copy()
    pathway_nodes = summary.rename(
        columns={
            "primekg_pathway_id": "primekg_id",
            "primekg_pathway_source": "primekg_source",
        }
    )
    pathway_nodes["node_type"] = "Pathway"
    pathway_nodes = pathway_nodes[PATHWAY_NODE_COLUMNS]
    kept_edges = kept_edges[GENE_PATHWAY_COLUMNS]
    return pathway_nodes.reset_index(drop=True), kept_edges.reset_index(drop=True)


def write_variant_metadata(
    output_dir: Path,
    base_dir: Path,
    gene_count: int,
    pathway_count: int,
    gene_pathway_edge_count: int,
    min_genes: int,
    max_pathways: int,
) -> None:
    payload = {
        "export_variant": output_dir.name,
        "base_export": base_dir.relative_to(KG_EXPORT.parent).as_posix(),
        "primekg_source": "data/raw/primekg/kg.csv",
        "filter": {
            "relations": ["pathway_protein", "bioprocess_protein"],
            "display_relation": "interacts with",
            "min_genes_per_pathway": min_genes,
            "max_pathways": max_pathways,
            "keyword_pattern": PATHWAY_KEYWORD_PATTERN,
        },
        "genes": int(gene_count),
        "pathway_nodes": int(pathway_count),
        "gene_pathway_edges": int(gene_pathway_edge_count),
    }
    with (output_dir / "variant_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-kg-export-dir", type=str, default=str(DEFAULT_BASE_EXPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_EXPORT))
    parser.add_argument("--min-genes-per-pathway", type=int, default=2)
    parser.add_argument("--max-pathways", type=int, default=200)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args(argv)
    base_dir = resolve_repo_path(args.base_kg_export_dir)
    output_dir = resolve_repo_path(args.output_dir)

    print("1. Copying base KG variant files...")
    copy_base_export(base_dir, output_dir)

    print("2. Collecting PrimeKG pathway/bioprocess edges...")
    gene_nodes = pd.read_csv(base_dir / "gene_nodes.csv")
    genes = gene_lookup(gene_nodes)
    raw_edges = collect_gene_pathway_edges(PRIMEKG_PATH, genes, args.chunksize)

    print("3. Filtering pathway nodes...")
    pathway_nodes, gene_pathway_edges = filter_pathway_edges(
        raw_edges,
        min_genes=args.min_genes_per_pathway,
        max_pathways=args.max_pathways,
        keyword_pattern=PATHWAY_KEYWORD_PATTERN,
    )

    print("4. Writing pathway KG variant...")
    pathway_nodes.to_csv(output_dir / "pathway_nodes.csv", index=False)
    gene_pathway_edges.to_csv(output_dir / "gene_pathway_edges.csv", index=False)
    write_variant_metadata(
        output_dir,
        base_dir,
        gene_count=len(gene_nodes),
        pathway_count=len(pathway_nodes),
        gene_pathway_edge_count=len(gene_pathway_edges),
        min_genes=args.min_genes_per_pathway,
        max_pathways=args.max_pathways,
    )

    print("PrimeKG pathway KG export built")
    print(f"  output: {output_dir}")
    print(f"  genes: {len(gene_nodes)}")
    print(f"  pathway/process nodes retained: {len(pathway_nodes)}")
    print(f"  gene-pathway edges retained: {len(gene_pathway_edges)}")
    print("  default KG export was not modified")


if __name__ == "__main__":
    main()
