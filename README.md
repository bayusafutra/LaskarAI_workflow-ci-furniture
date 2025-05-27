# Workflow CI - Furniture Price Prediction

Repositori ini berisi workflow CI menggunakan MLflow Project untuk model prediksi harga furniture.

## Struktur Direktori
Workflow-CI/
├── .workflow/
├── MLProject/
│ ├── modelling.py
│ ├── conda.yaml
│ ├── MLProject
│ ├── furniture_preprocessed.csv


## Cara Menjalankan
```bash
conda env create -f MLProject/conda.yaml
conda activate mlflow-env
cd MLProject
python -m mlflow run . --entry-point main
