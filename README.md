# KG AML

Repository containing notebooks and scripts for AML data processing, Knowledge Graph construction, and model training.

## Folder structure

```text
├── data/
│   ├── raw/          # Driginal data from external sources
│   ├── interim/      # Processed data and KG exports
│   └── features/     # Embeddings and feature matrices
├── notebooks/        # Main research workflow notebooks
├── scripts/
│   ├── pipeline/     # Scripts supporting the main pipeline
│   └── exploration/  # Exploratory analysis and experiments
├── outputs/
│   ├── checkpoints/  # Trained model files
│   ├── figures/      # Figures and plots
│   ├── tables/       # Metrics and result tables
│   └── logs/         # Run logs
└── requirements.txt  # Required Python packages
```

## Main notebook order

1. `01_Data_Preparation.ipynb`: prepare BeatAML data.
2. `02_KG_build.ipynb`: build the Knowledge Graph.
3. `03_Tabular_BeatAML_Baseline.ipynb`: run tabular baseline models.
4. `04_GNN_training_fix.ipynb`: train and evaluate the GNN model.

Generated files should be saved under `data/` or `outputs/`, not in the repository root.