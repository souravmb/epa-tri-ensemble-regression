# Stratified Differential Evolution Ensemble Regression with Leak-Guarded Bayesian Optimization

**Regulatory-Grade Prediction of Industrial Toxic Chemical Releases — A SHAP-Interpreted Analysis of EPA Toxics Release Inventory 2022**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Marimo](https://img.shields.io/badge/Notebook-Marimo%200.21.1-EE4B2B)](https://marimo.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22863a)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-FF6600)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20HPO-3B4EFF)](https://optuna.org)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretability-FF0066)](https://shap.readthedocs.io)
[![R²](https://img.shields.io/badge/R%C2%B2-0.9966-brightgreen)](#results)
[![RMSE](https://img.shields.io/badge/RMSE-0.2341-brightgreen)](#results)
[![Institution](https://img.shields.io/badge/Amrita%20Vishwa%20Vidyapeetham-Coimbatore-800000)](https://www.amrita.edu)

---

## Authors

| Name | Roll No. | Program | Institution |
|---|---|---|---|
| **Sourav M B** | CB.PS.P2ASD25023 | M.Sc Applied Statistics & Data Analytics | Amrita Vishwa Vidyapeetham, Coimbatore |
| **Sruti S Kumar** | CB.PS.P2ASD25025 | M.Sc Applied Statistics & Data Analytics | Amrita Vishwa Vidyapeetham, Coimbatore |

**ORCID (Sourav M B):** [0009-0006-8642-3339](https://orcid.org/0009-0006-8642-3339)

---

## Abstract

The United States Environmental Protection Agency's Toxics Release Inventory (TRI) program mandates annual reporting of chemical releases from over 21,000 industrial facilities. This repository presents a rigorously **leak-guarded machine learning pipeline** to predict total toxic releases (in pounds) using the **2022 TRI Basic Data File** comprising **80,040 facility-chemical records across 122 variables**.

All preprocessing transformations — imputation, encoding, scaling, and feature selection — are fitted **exclusively on training data**. Arithmetic components of the target are systematically identified and excluded via **17 leakage-pattern rules**. Five regression models are benchmarked with Bayesian hyperparameter optimisation via **Optuna (TPE sampler)**. A two-level stacking strategy combining **Differential Evolution-optimised weighted blending** with a **Linear Regression meta-learner** achieves the following on the held-out test set:

| Metric | Value |
|---|---|
| RMSE (log₁p scale) | **0.2341** |
| R² | **0.9966** |
| Nested CV R² — 95% CI | **[0.9954, 0.9965]** |
| Weighted F1 (5-class severity) | **0.93** |

SHAP TreeExplainer analysis confirms that on-site and off-site waste-management quantities dominate prediction, aligning with domain expectations from environmental engineering.

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Pipeline Architecture](#pipeline-architecture)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Key Design Decisions](#key-design-decisions)
- [Limitations and Future Work](#limitations-and-future-work)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Background

The **Toxics Release Inventory (TRI)** is established under the Emergency Planning and Community Right-to-Know Act (EPCRA) of 1986. Predicting total release quantities from facility and chemical characteristics enables regulators to prioritise inspections, identify anomalous reporting, and allocate environmental monitoring capacity proactively.

The TRI dataset presents three core modelling challenges:

1. **Extreme target skew** — spans nine orders of magnitude (0 to 327 million lbs), skewness ≈ 146
2. **Mixed feature types** — numeric operational metrics and high-cardinality categoricals (city, chemical name, NAICS code)
3. **Severe leakage risk** — columns such as `FUGITIVE_AIR`, `STACK_AIR`, and `LANDFILLS` are arithmetic components of the target; including them naively yields R² > 0.999, a trivially misleading result

This work explicitly and systematically addresses all three challenges.

---

## Dataset

**Source:** [EPA TRI Basic Data Files — Calendar Years 1987–Present](https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files-calendar-years-1987-present)

**File used:** `2022_us.csv`

| Property | Value |
|---|---|
| Records | 80,040 |
| Columns | 122 |
| Target variable | `107. TOTAL RELEASES` (pounds) |
| Target skewness | ≈ 146 |
| Zero-release records | ≈ 23% |
| Missing values | 1,449,185 nulls across 25 columns |
| Fully null columns | 7 (all SIC codes + `HORIZONTAL_DATUM`) |

> **Note:** `2022_us.csv` is not included in this repository (~150 MB). Download it from the EPA link above and place it at `data/raw/2022_us.csv`. Then update the path in Cell 1 of the notebook to match your local path.

---

## Methodology

### 1. Exploratory Data Analysis

- Log₁p-transformed target distribution characterisation
- Feature skewness profiling (67 features with |skew| > 2)
- Pearson correlation with target — top-20 ranked
- Random Forest feature importance — top-20 ranked
- PCA intrinsic dimensionality: 90% variance at 53 PCs, 95% at 59 PCs
- Inter-feature multicollinearity heatmap (top-15 features)

### 2. Leak-Guarded Preprocessing

- **Column removal:** 16 identifier columns + ~40 leakage columns via 17 regex patterns + near-constant features via `VarianceThreshold(1e-6)`, all fitted on training data
- **Whitelisting:** 24 legitimate M-code operational transfer columns explicitly retained
- **Imputation:** Median for numerics, mode for categoricals — fitted on training set only
- **Encoding:** One-hot (≤10 categories) / frequency (11–50) / Bayesian smoothed target encoding (>50) — all fitted on training labels only
- **Skew correction:** Log₁p transform for features with |skew| > 3
- **Winsorisation:** [1st, 99th] percentile clipping — fitted on training data
- **Deduplication:** Feature pairs with |r| > 0.95 removed
- **Adaptive scaling:** StandardScaler for approximately normal features, RobustScaler for |skew| > 1.5, no scaling for binary features

### 3. Feature Selection

Dual-criterion conservative filter — a feature is removed only if it simultaneously satisfies **both** conditions:

- Bottom 10th percentile of Random Forest importance (mean decrease in impurity)
- Absolute Pearson r < 0.02 with the log₁p-transformed target

Features with high non-linear predictive power but low Pearson r are retained. **Final feature set: 72 variables.**

### 4. Model Training

| Model | Configuration |
|---|---|
| Linear Regression | OLS baseline |
| Decision Tree (CART) | max_depth=15, min_samples_leaf=5 |
| Random Forest | 200 trees, max_depth=15, min_samples_leaf=5 |
| Gradient Boosting | 200 trees, max_depth=5, lr=0.1, subsample=0.8 |
| XGBoost | 300 trees, max_depth=6, lr=0.1, colsample_bytree=0.8, reg_alpha=0.1 |

### 5. Bayesian Hyperparameter Optimisation

Optuna TPE sampler — 15 trials × 3-fold cross-validation on training data only — applied to XGBoost, RandomForest, and GradientBoosting. Minimises cross-validated RMSE on the log₁p-transformed target.

### 6. Ensemble Strategies

**Method 1 — Differential Evolution Weighted Blending:**
Out-of-fold predictions generated via 3-fold `cross_val_predict` on training data. Differential Evolution optimises ensemble weights on the probability simplex (w ≥ 0, Σw = 1) to minimise OOF RMSE.

**Method 2 — Linear Regression Stacking:**
A Linear Regression meta-learner is trained on the n × 3 OOF prediction matrix, allowing negative weights and an intercept. Applied to test-set predictions from models retrained on the full training set.

### 7. Evaluation

- Regression metrics: RMSE, MAE, R² on log₁p scale
- Classification metrics: Weighted precision, recall, F1 across five regulatory severity bins
- Residual diagnostics: predicted vs. actual, residual distribution, homoscedasticity check
- Nested cross-validation: 5-fold × 3 repeats = 15 estimates with 95% confidence intervals
- Error stratification: per-bin RMSE and bias analysis
- SHAP TreeExplainer: global feature importance bar plot and beeswarm plot for GradientBoosting

---

## Pipeline Architecture

```
Raw EPA TRI 2022 CSV  (80,040 records x 122 columns)
            |
            v
  Stratified Sample  (10,000 records, 10 decile bins)
            |
            v
  log1p(TOTAL_RELEASES)  -- target transformation
            |
    80% Train | 20% Test  <-- split BEFORE any preprocessing
            |
            v
  +--------------------------------------------------+
  |           Leak-Guarded Preprocessing             |
  |  (1) Drop identifiers + 17-pattern leakage cols  |
  |  (2) Impute median/mode          [train-fitted]  |
  |  (3) Encode OHE/freq/target-enc  [train-fitted]  |
  |  (4) VarianceThreshold           [train-fitted]  |
  |  (5) Log1p highly skewed features               |
  |  (6) Winsorise [1st, 99th] pct   [train-fitted]  |
  |  (7) Dedup pairs with |r| > 0.95 [train-fitted]  |
  |  (8) Adaptive scale Std/Robust   [train-fitted]  |
  +--------------------------------------------------+
            |
            v
  Dual-criterion feature selection  -->  72 features
            |
            v
  +--------------------------------------------------+
  |               Base Models (5)                    |
  |  LinearReg | CART | RandomForest | GBM | XGBoost |
  +--------------------------------------------------+
            |
            v
  Bayesian HPO via Optuna TPE  (RF, GBM, XGB tuned)
            |
            v
  +--------------------------------------------------+
  |              Ensemble Layer                      |
  |  OOF predictions via 3-fold cross_val_predict    |
  |  (1) Differential Evolution Weighted Blending    |
  |  (2) Linear Regression Stacking (meta-learner)   |
  +--------------------------------------------------+
            |
            v
  Final Predictions  -->  inverse transform  -->  exp(y*) - 1
            |
            v
  +--------------------------------------------------+
  |             Evaluation Suite                     |
  |  RMSE / MAE / R2  (log1p scale)                 |
  |  5-class severity classification F1              |
  |  Nested CV  (5 folds x 3 repeats = 15 estimates) |
  |  Error stratification by severity bin            |
  |  SHAP TreeExplainer (bar + beeswarm)             |
  |  Learning curves + overfitting diagnosis         |
  +--------------------------------------------------+
```

---

## Results

### Base Model Performance — Test Set (log₁p scale)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **GradientBoosting** | **0.2376** | **0.0884** | **0.9965** |
| XGBoost | 0.2567 | 0.1165 | 0.9959 |
| RandomForest | 0.3383 | 0.0885 | 0.9928 |
| DecisionTree (CART) | 0.3807 | 0.1157 | 0.9909 |
| LinearRegression | 1.1793 | 0.8151 | 0.9126 |

### Hyperparameter Tuning Gains

| Model | Base RMSE | Tuned RMSE | Improvement |
|---|---|---|---|
| XGBoost | 0.2567 | 0.2463 | −4.1% |
| RandomForest | 0.3383 | 0.2665 | −21.2% |
| GradientBoosting | 0.2376 | 0.2453 | near-optimal default |

### Ensemble Performance

| Method | RMSE | R² |
|---|---|---|
| Best Base (GradientBoosting) | 0.2376 | 0.9965 |
| DE Weighted Blending | 0.2344 | 0.9965 |
| **LR Stacking** | **0.2341** | **0.9966** |

### Statistical Validation — Nested Cross-Validation (GradientBoosting)

| Metric | Mean | 95% CI |
|---|---|---|
| RMSE | 0.2482 ± 0.0154 | [0.2328, 0.2636] |
| MAE | 0.0798 ± 0.0024 | [0.0774, 0.0822] |
| R² | 0.9960 ± 0.0006 | **[0.9954, 0.9965]** |

### Error Stratification by Severity Bin

| Bin | n | RMSE | MAE | Bias |
|---|---|---|---|---|
| Zero (≈0 lbs) | 462 | 0.2564 | 0.0275 | −0.027 |
| Low (< 20 lbs) | 370 | 0.0847 | 0.0447 | +0.003 |
| Medium (20–1,096 lbs) | 523 | 0.0808 | 0.0561 | −0.006 |
| High (1,096–163K lbs) | 598 | 0.2358 | 0.0929 | −0.011 |
| **Very High (> 163K lbs)** | 47 | **0.9210** | 0.5822 | **+0.250** |

### Top SHAP Features — GradientBoosting

| Rank | Feature | Mean SHAP | Interpretation |
|---|---|---|---|
| 1 | 8.1B On-Site Other | 2.80 | On-site non-contained releases — dominates by 8x |
| 2 | 8.1D Off-Site Other Releases | 0.35 | Off-site non-contained disposal |
| 3 | 8.1C Off-Site Contained | 0.30 | Off-site contained disposal |
| 4 | 8.1A On-Site Contained | 0.25 | On-site contained releases |
| 5 | Production Waste (8.1–8.7) | 0.15 | Total production-related waste |
| 6–15 | CITY_TENC, COUNTY_TENC, CHEMICAL_TENC | — | Geographic and chemical release profiles |

---

## Repository Structure

```
epa-tri-ensemble-regression/
|
├── EPATRIML_1_.py         Marimo reactive notebook (primary — run this)
├── EPATRIML-2.ipynb       Jupyter/IPython export
├── SOURAV_ML.pdf          Full research paper (IEEE-style)
|
├── data/
|   └── raw/
|       └── .gitkeep       Place 2022_us.csv here (not tracked by git)
|
├── .github/
|   ├── ISSUE_TEMPLATE/
|   |   ├── bug_report.md           Bug report form with pipeline-stage checklist
|   |   ├── methodology_proposal.md Proposal form tied to paper's limitations
|   |   └── config.yml              Disables blank issues, redirects to Discussions
|   └── PULL_REQUEST_TEMPLATE.md    PR form with 7-point leakage audit checklist
|
├── environment.yml        Conda environment (Python 3.11, tri-ml)
├── requirements.txt       pip requirements
├── CITATION.cff           Machine-readable citation metadata
├── CODE_OF_CONDUCT.md     Project conduct standards
├── CONTRIBUTING.md        Contribution guidelines
├── SECURITY.md            Vulnerability reporting and data handling policy
├── LICENSE                MIT License
└── README.md              This file
```

---

## Installation

### Option A — Conda (Recommended)

```bash
git clone https://github.com/souravmb/epa-tri-ensemble-regression.git
cd epa-tri-ensemble-regression
conda env create -f environment.yml
conda activate tri-ml
```

### Option B — pip

```bash
git clone https://github.com/souravmb/epa-tri-ensemble-regression.git
cd epa-tri-ensemble-regression
pip install -r requirements.txt
```

### Dataset Setup

1. Download `2022_us.csv` from the [EPA TRI Basic Data Files page](https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files-calendar-years-1987-present)
2. Place it at `data/raw/2022_us.csv`
3. Update the file path in Cell 1 of the notebook if your local path differs

---

## Running the Notebook

### Marimo — Recommended

```bash
pip install marimo==0.21.1

# View as a read-only app
marimo run EPATRIML_1_.py

# Open in edit mode (full reactive notebook)
marimo edit EPATRIML_1_.py
```

Marimo notebooks are pure Python `.py` files — no hidden kernel state, no JSON, meaningful git diffs. Cells re-execute automatically when their dependencies change.

### Jupyter

```bash
jupyter notebook EPATRIML-2.ipynb
```

### Notebook Cell Sequence

| Cell | Stage | Description |
|---|---|---|
| 1 | Data Loading | Load `2022_us.csv`, inspect shape and dtypes |
| 1b | Quality Check | Null counts, duplicates, target distribution stats |
| 2 | Sampling & Transform | Stratified 10K sample, log₁p target, 80/20 split |
| 3a | Leakage Removal | Drop identifiers and 17-pattern leakage columns |
| 3b | Imputation | Median/mode, train-fitted — verify zero nulls |
| 3c | Encoding | One-hot / frequency / Bayesian target encoding |
| 3d | Memory + Variance | float32 downcast, VarianceThreshold |
| 4 | EDA | Target dist, skewness, correlations, RF importance, PCA, heatmap |
| 5a–5d | Feature Engineering | Skew transform, winsorise, dedup, adaptive scaling |
| 6 | Base Models | 5 regressors trained and benchmarked |
| 6b | Classification Metrics | F1, confusion matrices, ROC/AUC — all 5 models |
| 7 | Bayesian HPO | Optuna TPE, 15 trials, 3-fold CV |
| 8 | Ensemble | DE blending + LR stacking, OOF-based |
| 9 | Results Table | Base → tuned → ensemble comparison |
| 10 | Final Evaluation | Original-scale metrics, binned classification report |
| 11 | Residual + SHAP | Diagnostics, beeswarm plot, global bar plot |
| 12a | Learning Curves | Train vs. validation RMSE, data-hunger diagnosis |
| 12b | Nested CV | 5 x 3 = 15 estimates, 95% CI for RMSE/MAE/R² |
| 12c | Error Stratification | Per-bin RMSE, bias, residual boxplots |
| 13 | Discussion | Principal findings, limitations, future work |

---

## Key Design Decisions

### Why Marimo?

[Marimo](https://marimo.io) was chosen over Jupyter for three reasons:

1. **Reactive execution** — cells re-run automatically when upstream values change, eliminating out-of-order execution bugs
2. **Git-friendly** — notebooks are plain Python files, not JSON, enabling meaningful diffs and code review
3. **No hidden state** — the notebook is always in a consistent, reproducible state

### Why Differential Evolution for Ensemble Weights?

Standard gradient descent assumes a differentiable loss landscape. The ensemble weight optimisation surface over the probability simplex (w ≥ 0, Σw = 1) is non-convex and non-smooth. Differential Evolution — a population-based global optimiser — explores this space without gradient information, providing a principled alternative to grid or random search over weights.

### Why Bayesian Smoothed Target Encoding?

High-cardinality categoricals (city, chemical name) with standard mean target encoding suffer from high variance on rare categories. Bayesian smoothing shrinks rare-category estimates toward the global prior:

$$\text{enc}(c) = \frac{n_c \bar{y}_c + m \bar{y}_g}{n_c + m}$$

where $n_c$ is the category count, $\bar{y}_c$ is the category target mean, $\bar{y}_g$ is the global training mean, and $m = 10$ is the smoothing factor. All statistics are computed exclusively on training labels.

### Why Dual-Criterion Feature Selection?

A feature can have high non-linear predictive power (high RF importance) but low linear correlation with the target. A Pearson-r threshold alone would incorrectly discard such features. The dual criterion — RF importance AND Pearson |r| must both be low — ensures only genuinely uninformative features are removed.

---

## Limitations and Future Work

| Limitation | Proposed Extension |
|---|---|
| 10K stratified sample (12.5% of available data) | Scale to all 80,040 records — learning curves confirm continued improvement |
| 2022 data only | Temporal validation across reporting years 2010–2022 |
| Very High bin: RMSE = 0.921, bias = +0.250 | Two-stage model: binary zero/non-zero classifier followed by regression |
| Zero↔Low confusion: 105 of 462 records misclassified | Zero-inflated regression (Hurdle model or ZIML) |
| Section 8 feature dominance | Full EPA codebook review to confirm no residual leakage in features 8.1A–8.1D |
| Single reporting year | Multi-year panel model with facility fixed effects |

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for scope, the leakage-safety requirements that apply to all PRs, and the branch naming convention.

**Maintainers:**

| GitHub | Role |
|---|---|
| [@souravmb](https://github.com/souravmb) | Primary author |
| [@Srutiskumar](https://github.com/Srutiskumar) | Co-author |

**Issue templates** are available for [bug reports](.github/ISSUE_TEMPLATE/bug_report.md) and [methodology proposals](.github/ISSUE_TEMPLATE/methodology_proposal.md). All pull requests use the [PR template](.github/PULL_REQUEST_TEMPLATE.md), which includes a mandatory leakage audit checklist.

General questions about the methodology or dataset belong in [Discussions](https://github.com/souravmb/epa-tri-ensemble-regression/discussions) rather than Issues.

This project follows the standards described in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). To report a security concern or a subtle data leakage path, see [`SECURITY.md`](SECURITY.md).

---

## Citation

If you use this codebase, methodology, or results in your research, please cite:

```bibtex
@misc{sourav2026tri,
  author       = {Sourav, M B and Kumar, Sruti S},
  title        = {Stratified Differential Evolution Ensemble Regression with
                  Leak-Guarded Bayesian Optimization for Regulatory-Grade
                  Prediction of Industrial Toxic Chemical Releases},
  year         = {2026},
  institution  = {Amrita Vishwa Vidyapeetham, Coimbatore},
  note         = {M.Sc Applied Statistics and Data Analytics,
                  Department of Mathematics},
  url          = {https://github.com/souravmb/epa-tri-ensemble-regression}
}
```

Machine-readable citation metadata is also available in [`CITATION.cff`](CITATION.cff) — GitHub renders a "Cite this repository" button automatically from this file.

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for full terms.

Copyright 2026 Sourav M B and Sruti S Kumar, Amrita Vishwa Vidyapeetham

---

*Department of Mathematics, Amrita Vishwa Vidyapeetham, Coimbatore*
*Deemed to be University under Section 3 of the UGC Act, 1956*
