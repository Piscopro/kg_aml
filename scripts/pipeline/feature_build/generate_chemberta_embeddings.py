import pandas as pd
import numpy as np
import pubchempy as pcp
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from scripts.pipeline.common.benchmark_utils import DATA_FEATURES, KG_EXPORT

def get_smiles(drug_name):
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            return compounds[0].isomeric_smiles
    except Exception as e:
        pass
    return None

def main():
    kg_export_dir = KG_EXPORT
    out_dir = DATA_FEATURES / 'drug'

    print("1. Loading drug nodes...")
    drug_nodes_path = kg_export_dir / 'drug_nodes.csv'
    if not drug_nodes_path.exists():
        print(f"Error: {drug_nodes_path} not found.")
        return
    
    df_drugs = pd.read_csv(drug_nodes_path)
    unique_drugs = df_drugs['inhibitor'].unique()
    
    print(f"Loaded {len(unique_drugs)} unique drugs. Fetching SMILES...")
    smiles_dict = {}
    for drug in tqdm(unique_drugs):
        smiles = get_smiles(drug)
        smiles_dict[drug] = smiles
        
    # Model init
    print("2. Loading ChemBERTa model...")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    embeddings = {}
    valid_embeddings = []
    
    print("3. Generating embeddings...")
    for drug in tqdm(unique_drugs):
        smiles = smiles_dict[drug]
        if smiles:
            inputs = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding (index 0)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                embeddings[drug] = cls_embedding
                valid_embeddings.append(cls_embedding)
        else:
            embeddings[drug] = None
            
    # Impute missing with mean
    if len(valid_embeddings) > 0:
        mean_embedding = np.mean(valid_embeddings, axis=0)
    else:
        mean_embedding = np.zeros(384) # ChemBERTa-77M hidden size
        
    print("4. Handling missing & saving...")
    final_embeddings = []
    for drug in unique_drugs:
        if embeddings[drug] is None:
            final_embeddings.append(mean_embedding)
        else:
            final_embeddings.append(embeddings[drug])
            
    final_embeddings = np.array(final_embeddings)
    
    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(final_embeddings)
    df_out.insert(0, 'inhibitor', unique_drugs)
    out_path = out_dir / 'drug_smiles_embedding_chemberta.csv'
    df_out.to_csv(out_path, index=False)
    
    print(f"Done! Embeddings saved to {out_path} with shape {final_embeddings.shape}")

if __name__ == "__main__":
    main()
