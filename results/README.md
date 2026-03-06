# Results

Output data and figures from each analysis notebook, organised by pipeline stage.

| Directory | Source Notebook | Contents |
|-----------|----------------|----------|
| `01_eda/` | `01_preprocessing_and_eda.ipynb` | EDA figures (demographics, tumour characteristics, molecular biomarkers, clinical outcomes) |
| `02_tabular_models/` | `02_tabular_models.ipynb` | Binary classification figures (correlation analysis, ROC curves, confusion matrices, feature importance) |
| `03_survival/` | `03_survival_analysis.ipynb` | Survival analysis figures (KM curves, Cox coefficients, model comparison, survival distributions) |
| `04_radiomics/` | `04_radiomics.ipynb` | Imaging analysis figures (volume analysis, imaging distributions, combined feature importance) |
| `05_growth_rate/` | `05_growth_rate_prediction.ipynb` | Growth prediction figures (target distribution, R2 comparison, feature importance, mortality analysis) |
| `06_pyradiomics_features/` | `06_pyradiomics_feature_extraction.ipynb` | PyRadiomics figures (correlation heatmap, selected features, KM stratification, model comparison, demo overlay) |

Each subdirectory contains a `figures/` folder with the generated plots (PNG format).
