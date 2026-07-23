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
DEFAULT_OUTPUT_EXPORT = KG_EXPORT / "beataml_primekg_ppi_targets_1hop_filtered"
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
GENE_METADATA_COLUMNS = [
    "gene_name",
    "node_type",
    "source",
    "primekg_id",
    "primekg_source",
    "ppi_degree",
    "seed_ppi_links",
    "score",
    "flag_aml_disease",
    "flag_cancer_disease",
    "flag_pathway_keyword",
    "flag_bioprocess_keyword",
]
AML_DISEASE_PATTERN = r"acute myeloid|myeloid leukemia|aml"
CANCER_DISEASE_PATTERN = (
    r"leukemia|cancer|carcinoma|neoplasm|tumor|lymphoma|myeloma|sarcoma|malignant"
)
BIOLOGY_KEYWORD_PATTERN = (
    r"apoptosis|cell death|proliferation|cell cycle|dna repair|dna damage|kinase|"
    r"phosphorylation|hematopo|myeloid|leukocyte|immune|differentiation|"
    r"transcription|p53|tp53|jak|stat|ras|mapk|flt3|chromatin|methylation|splicing"
)


def copy_base_export(base_dir: Path, output_dir: Path) -> None:
    ensure_dirs(output_dir)
    for path in base_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def base_gene_lookup(gene_nodes: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for gene_name in gene_nodes["gene_name"].dropna().astype(str):
        lookup.setdefault(gene_name.upper(), gene_name)
    return lookup


def collect_ppi_seed_neighbors(
    primekg_path: Path, seed_genes: set[str], chunksize: int
) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, str]]]:
    degree: dict[str, int] = {}
    seed_links: dict[str, int] = {}
    gene_info: dict[str, dict[str, str]] = {}
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

    for chunk in pd.read_csv(
        primekg_path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        ppi = chunk[
            chunk["relation"].eq("protein_protein")
            & chunk["display_relation"].eq("ppi")
            & chunk["x_type"].eq("gene/protein")
            & chunk["y_type"].eq("gene/protein")
        ].copy()
        if ppi.empty:
            continue

        ppi["x_key"] = ppi["x_name"].astype(str).str.upper()
        ppi["y_key"] = ppi["y_name"].astype(str).str.upper()
        for row in ppi.itertuples(index=False):
            x_key = row.x_key
            y_key = row.y_key
            if x_key == y_key:
                continue

            degree[x_key] = degree.get(x_key, 0) + 1
            degree[y_key] = degree.get(y_key, 0) + 1
            gene_info.setdefault(
                x_key,
                {
                    "gene_name": str(row.x_name),
                    "primekg_id": str(row.x_id),
                    "primekg_source": str(row.x_source),
                },
            )
            gene_info.setdefault(
                y_key,
                {
                    "gene_name": str(row.y_name),
                    "primekg_id": str(row.y_id),
                    "primekg_source": str(row.y_source),
                },
            )

            if x_key in seed_genes and y_key not in seed_genes:
                seed_links[y_key] = seed_links.get(y_key, 0) + 1
            if y_key in seed_genes and x_key not in seed_genes:
                seed_links[x_key] = seed_links.get(x_key, 0) + 1

    return degree, seed_links, gene_info


def collect_context_flags(
    primekg_path: Path, candidate_genes: set[str], chunksize: int
) -> dict[str, dict[str, int]]:
    flags = {
        gene: {
            "aml_disease": 0,
            "cancer_disease": 0,
            "pathway_keyword": 0,
            "bioprocess_keyword": 0,
        }
        for gene in candidate_genes
    }
    usecols = ["relation", "x_type", "x_name", "y_type", "y_name"]

    for chunk in pd.read_csv(
        primekg_path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        x_gene = chunk["x_type"].eq("gene/protein") & chunk["x_name"].astype(
            str
        ).str.upper().isin(candidate_genes)
        y_gene = chunk["y_type"].eq("gene/protein") & chunk["y_name"].astype(
            str
        ).str.upper().isin(candidate_genes)
        one_gene = chunk[x_gene | y_gene]
        if one_gene.empty:
            continue

        for row in one_gene.itertuples(index=False):
            if row.x_type == "gene/protein":
                gene_key = str(row.x_name).upper()
                other_type = row.y_type
                other_name = str(row.y_name)
            else:
                gene_key = str(row.y_name).upper()
                other_type = row.x_type
                other_name = str(row.x_name)

            if gene_key not in flags:
                continue

            other = pd.Series([other_name])
            if row.relation == "disease_protein" and other_type == "disease":
                if other.str.contains(
                    AML_DISEASE_PATTERN, case=False, regex=True, na=False
                ).iloc[0]:
                    flags[gene_key]["aml_disease"] = 1
                if other.str.contains(
                    CANCER_DISEASE_PATTERN, case=False, regex=True, na=False
                ).iloc[0]:
                    flags[gene_key]["cancer_disease"] = 1
            if other_type == "pathway" and other.str.contains(
                BIOLOGY_KEYWORD_PATTERN, case=False, regex=True, na=False
            ).iloc[0]:
                flags[gene_key]["pathway_keyword"] = 1
            if other_type == "biological_process" and other.str.contains(
                BIOLOGY_KEYWORD_PATTERN, case=False, regex=True, na=False
            ).iloc[0]:
                flags[gene_key]["bioprocess_keyword"] = 1

    return flags


def score_candidates(
    seed_links: dict[str, int],
    degree: dict[str, int],
    flags: dict[str, dict[str, int]],
    gene_info: dict[str, dict[str, str]],
    max_degree: int,
    max_new_genes: int,
) -> pd.DataFrame:
    rows = []
    for gene_key, links in seed_links.items():
        ppi_degree = degree.get(gene_key, 0)
        gene_flags = flags.get(gene_key, {})
        if ppi_degree > max_degree or not any(gene_flags.values()):
            continue

        score = (
            10 * gene_flags.get("aml_disease", 0)
            + 4 * gene_flags.get("cancer_disease", 0)
            + 3 * gene_flags.get("pathway_keyword", 0)
            + 2 * gene_flags.get("bioprocess_keyword", 0)
            + 2 * links
            - (ppi_degree / max_degree)
        )
        info = gene_info[gene_key]
        rows.append(
            {
                "gene_key": gene_key,
                "gene_name": info["gene_name"],
                "node_type": "Gene",
                "source": "PrimeKG_1hop_filtered",
                "primekg_id": info["primekg_id"],
                "primekg_source": info["primekg_source"],
                "ppi_degree": ppi_degree,
                "seed_ppi_links": links,
                "score": score,
                "flag_aml_disease": gene_flags.get("aml_disease", 0),
                "flag_cancer_disease": gene_flags.get("cancer_disease", 0),
                "flag_pathway_keyword": gene_flags.get("pathway_keyword", 0),
                "flag_bioprocess_keyword": gene_flags.get("bioprocess_keyword", 0),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["gene_key", *GENE_METADATA_COLUMNS])

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["score", "seed_ppi_links", "ppi_degree", "gene_name"],
            ascending=[False, False, True, True],
        )
        .head(max_new_genes)
        .reset_index(drop=True)
    )


def build_expanded_gene_nodes(
    base_genes: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    new_genes = selected[["gene_name", "node_type"]].copy()
    return pd.concat(
        [base_genes[["gene_name", "node_type"]], new_genes], ignore_index=True
    ).drop_duplicates("gene_name")


def collect_expanded_ppi_edges(
    primekg_path: Path, genes: dict[str, str], chunksize: int
) -> pd.DataFrame:
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
        ppi = chunk[
            chunk["relation"].eq("protein_protein")
            & chunk["display_relation"].eq("ppi")
            & chunk["x_type"].eq("gene/protein")
            & chunk["y_type"].eq("gene/protein")
        ].copy()
        if ppi.empty:
            continue

        ppi["x_key"] = ppi["x_name"].astype(str).str.upper()
        ppi["y_key"] = ppi["y_name"].astype(str).str.upper()
        ppi = ppi[ppi["x_key"].isin(gene_keys) & ppi["y_key"].isin(gene_keys)]
        if ppi.empty:
            continue

        for row in ppi.itertuples(index=False):
            x_gene = genes[row.x_key]
            y_gene = genes[row.y_key]
            if x_gene == y_gene:
                continue

            gene_a, gene_b = sorted((x_gene, y_gene))
            if gene_a == x_gene:
                gene_a_id, gene_b_id = row.x_id, row.y_id
                gene_a_source, gene_b_source = row.x_source, row.y_source
            else:
                gene_a_id, gene_b_id = row.y_id, row.x_id
                gene_a_source, gene_b_source = row.y_source, row.x_source

            records.setdefault(
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

    if not records:
        return pd.DataFrame(columns=PPI_COLUMNS)

    return (
        pd.DataFrame(records.values(), columns=PPI_COLUMNS)
        .sort_values(["gene_a", "gene_b"])
        .reset_index(drop=True)
    )


def write_variant_metadata(
    output_dir: Path,
    base_dir: Path,
    base_gene_count: int,
    selected_count: int,
    ppi_edge_count: int,
    max_degree: int,
    max_new_genes: int,
) -> None:
    payload = {
        "export_variant": output_dir.name,
        "base_export": base_dir.relative_to(KG_EXPORT.parent).as_posix(),
        "primekg_source": "data/raw/primekg/kg.csv",
        "filter": {
            "max_ppi_degree": max_degree,
            "max_new_genes": max_new_genes,
            "required_context": [
                "AML/cancer disease association",
                "pathway keyword",
                "biological process keyword",
            ],
        },
        "base_genes": int(base_gene_count),
        "new_1hop_genes": int(selected_count),
        "total_genes": int(base_gene_count + selected_count),
        "ppi_edges": int(ppi_edge_count),
    }
    with (output_dir / "variant_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-kg-export-dir", type=str, default=str(DEFAULT_BASE_EXPORT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_EXPORT))
    parser.add_argument("--max-degree", type=int, default=200)
    parser.add_argument("--max-new-genes", type=int, default=300)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args(argv)
    base_dir = resolve_repo_path(args.base_kg_export_dir)
    output_dir = resolve_repo_path(args.output_dir)

    print("1. Copying base KG variant files...")
    copy_base_export(base_dir, output_dir)

    print("2. Collecting PrimeKG 1-hop PPI candidates...")
    base_genes = pd.read_csv(base_dir / "gene_nodes.csv")
    seed_lookup = base_gene_lookup(base_genes)
    degree, seed_links, gene_info = collect_ppi_seed_neighbors(
        PRIMEKG_PATH, set(seed_lookup), args.chunksize
    )
    candidate_genes = set(seed_links)

    print("3. Scoring candidates by AML/cancer/pathway context...")
    flags = collect_context_flags(PRIMEKG_PATH, candidate_genes, args.chunksize)
    selected = score_candidates(
        seed_links,
        degree,
        flags,
        gene_info,
        max_degree=args.max_degree,
        max_new_genes=args.max_new_genes,
    )

    print("4. Writing expanded gene nodes and PPI edges...")
    expanded_genes = build_expanded_gene_nodes(base_genes, selected)
    expanded_genes.to_csv(output_dir / "gene_nodes.csv", index=False)
    selected[GENE_METADATA_COLUMNS].to_csv(
        output_dir / "primekg_1hop_gene_metadata.csv", index=False
    )

    expanded_lookup = base_gene_lookup(expanded_genes)
    ppi_edges = collect_expanded_ppi_edges(
        PRIMEKG_PATH, expanded_lookup, args.chunksize
    )
    ppi_edges.to_csv(output_dir / "ppi_edges.csv", index=False)
    write_variant_metadata(
        output_dir,
        base_dir,
        base_gene_count=len(base_genes),
        selected_count=len(selected),
        ppi_edge_count=len(ppi_edges),
        max_degree=args.max_degree,
        max_new_genes=args.max_new_genes,
    )

    print("PrimeKG filtered 1-hop PPI KG export built")
    print(f"  output: {output_dir}")
    print(f"  base genes: {len(base_genes)}")
    print(f"  new 1-hop genes: {len(selected)}")
    print(f"  total genes: {len(expanded_genes)}")
    print(f"  PPI edges retained: {len(ppi_edges)}")
    print("  default KG export was not modified")


if __name__ == "__main__":
    main()
