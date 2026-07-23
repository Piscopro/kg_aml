from __future__ import annotations

import pandas as pd

from scripts.pipeline.common.benchmark_utils import DATA_RAW, KG_EXPORT


def main() -> None:
    raw_path = DATA_RAW / "beataml" / "beataml_drug_families.xlsx"
    kg_export_dir = KG_EXPORT

    print("1. Loading raw Drug-Gene mapping...")
    if not raw_path.exists():
        print(f"Error: {raw_path} not found.")
        return

    drug_gene_df = pd.read_excel(raw_path, sheet_name="drug_gene")
    drug_gene_df = drug_gene_df[["inhibitor", "Symbol"]].dropna().drop_duplicates()

    print("2. Loading existing KG nodes...")
    gene_nodes = pd.read_csv(kg_export_dir / "gene_nodes.csv")
    valid_genes = set(gene_nodes["gene_name"])

    drug_nodes = pd.read_csv(kg_export_dir / "drug_nodes.csv")
    valid_drugs = set(drug_nodes["inhibitor"])

    print("3. Filtering valid target edges...")
    edges = drug_gene_df[
        drug_gene_df["Symbol"].isin(valid_genes)
        & drug_gene_df["inhibitor"].isin(valid_drugs)
    ].rename(columns={"Symbol": "gene_name"})

    kg_export_dir.mkdir(parents=True, exist_ok=True)
    out_path = kg_export_dir / "drug_gene_edges.csv"
    edges.to_csv(out_path, index=False)
    print(f"Saved {len(edges)} TARGETS edges to {out_path}")


if __name__ == "__main__":
    main()
