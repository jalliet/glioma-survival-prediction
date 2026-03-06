# glioma-survival-prediction
Final-year University Project on Glioma Survival Prediction

## Repository Structure

```
├── scripts/
│   ├── notebooks/          # Jupyter notebooks (01–06), run on Google Colab
│   ├── 06_pyradiomics_extraction_csf3.py   # PyRadiomics extraction for CSF3 cluster
│   └── compile_project.sh  # Compiles Report.tex via external build directory
├── results/
│   ├── 01_eda/             # EDA figures
│   ├── 02_tabular_models/  # Binary classification figures
│   ├── 03_survival/        # Survival analysis figures
│   ├── 04_radiomics/       # Imaging feature analysis figures
│   ├── 05_growth_rate/     # Tumour growth prediction figures
│   └── 06_pyradiomics_features/  # Comprehensive radiomics figures
├── 06 data/                # PyRadiomics extraction output (CSF3)
├── 06 figures/             # PyRadiomics analysis figures
└── Report.tex              # Project report (LaTeX)
```

See `scripts/README.md` and `results/README.md` for details.
