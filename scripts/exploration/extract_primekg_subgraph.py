from pathlib import Path

import pandas as pd

# Load the PrimeKG graph
project_root = Path(__file__).resolve().parents[2]
kg_path = project_root / 'data' / 'raw' / 'primekg' / 'kg.csv'
output_dir = project_root / 'data' / 'interim' / 'primekg'
output_dir.mkdir(parents=True, exist_ok=True)

kg = pd.read_csv(kg_path, low_memory=True)

# Get all nodes
def get_nodes(kg):
    """Extract unique nodes from the knowledge graph"""
    x_nodes = kg[['x_id', 'x_name', 'x_type', 'x_source']].rename(
        columns={'x_id': 'node_id', 'x_name': 'node_name', 
                 'x_type': 'node_type', 'x_source': 'node_source'})
    y_nodes = kg[['y_id', 'y_name', 'y_type', 'y_source']].rename(
        columns={'y_id': 'node_id', 'y_name': 'node_name', 
                 'y_type': 'node_type', 'y_source': 'node_source'})
    nodes = pd.concat([x_nodes, y_nodes]).drop_duplicates().reset_index(drop=True)
    return nodes

nodes = get_nodes(kg)

# Filter for AML disease node(s)
# You might need to search for the exact disease name or ID
aml_disease = nodes.query('node_type == "disease" and node_name.str.contains("acute myeloid", case=False)')
print(aml_disease)

# Get the AML disease ID(s)
aml_ids = aml_disease['node_id'].values

# Filter edges connected to AML
aml_kg = kg[(kg['x_id'].isin(aml_ids)) | (kg['y_id'].isin(aml_ids))]

# Save the filtered graph
aml_kg.to_csv(output_dir / 'aml_subgraph.csv', index=False)
