# Scripts

## Notebooks

Located in `notebooks/`, these are the main analysis pipeline. Run in order:

| Notebook | Description |
|----------|-------------|
| `01_preprocessing_and_eda.ipynb` | Data cleaning, exploratory data analysis, feature engineering |
| `02_tabular_models.ipynb` | Leakage detection, binary classification (XGBoost, Logistic Regression) |
| `03_survival_analysis.ipynb` | Cox PH, Random Survival Forest, Gradient Boosted Survival Analysis |
| `04_radiomics.ipynb` | Pre-extracted imaging features, clinical vs imaging model comparison |
| `05_growth_rate_prediction.ipynb` | Tumour volume change regression from baseline features |
| `06_pyradiomics_feature_extraction.ipynb` | Comprehensive PyRadiomics feature extraction and analysis |

Notebooks run on Google Colab with data stored on Google Drive.

## Standalone Scripts

| Script | Description |
|--------|-------------|
| `06_pyradiomics_extraction_csf3.py` | PyRadiomics extraction adapted for the University of Manchester CSF3 cluster. Generates `06_full_radiomic_features.csv` used by notebook 06. |
| `compile_project.sh` | Compiles `Report.tex` using the figures in the external build directory. |
