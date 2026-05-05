# An Honest Benchmark for Glioma Survival Prediction

Quantifying Data Leakage on the MU-Glioma-Post Dataset.

Joshua Alliet. Final-year project, University of Manchester. Supervisor: Dr Fumie Costen.

This project develops a leakage-aware survival prediction pipeline for adult glioma. A four-step detection methodology removes seven progression-related features that encode outcomes unavailable at diagnosis, dropping XGBoost AUC from 0.730 to 0.674. After cleaning, Random Survival Forest reaches a concordance index of 0.706, near the EORTC nomogram (0.66). Imaging features add nothing on top of clinical and molecular markers. The honest performance ceiling for tabular glioma prediction sits at AUC 0.67 to 0.72, well below the 0.85+ commonly reported in the literature.

This repo is a showcase. It holds the analysis code, derived results, the final report, and the MIMUC submission. The MU-Glioma-Post imaging data is not included.

## Layout

```
report/        Final dissertation PDF
mimuc/         3rd-place MIMUC certificate and presentation
results/       Figures and derived data from notebooks 01-06
scripts/       Notebooks (01-06) and the CSF3 PyRadiomics extraction script
requirements.txt
LICENSE
```

See `results/README.md` and `scripts/README.md` for per-directory detail.

## Data

MU-Glioma-Post is hosted on The Cancer Imaging Archive: <https://www.cancerimagingarchive.net/collection/mu-glioma-post/>. Access requires registration. 203 patients, 596 post-treatment MRI time points, four sequences (T1, T1+contrast, T2, FLAIR), with neuroradiologist-reviewed segmentations and linked clinical, pathology, and survival records.

## Notebooks

Six pipelines, run in order on Google Colab with data on Google Drive:

1. `01_preprocessing_and_eda` — cleaning, EDA, feature engineering
2. `02_tabular_models` — leakage detection and binary classification
3. `03_survival_analysis` — Cox PH, RSF, GBSA
4. `04_radiomics` — pre-extracted imaging features vs clinical
5. `05_growth_rate_prediction` — tumour volume change regression
6. `06_pyradiomics_feature_extraction` — full PyRadiomics extraction and analysis

The CSF3 batch script (`scripts/06_pyradiomics_extraction_csf3.py`) runs notebook 06's extraction step on the University of Manchester CSF3 cluster.

## MIMUC

3rd place at the [Manchester Interdisciplinary Mathematics Undergraduate Conference](https://www.mimuc.org.uk/). Certificate and presentation in `mimuc/`.

## License

Code, text, and figures: [CC-BY-NC 4.0](LICENSE).
