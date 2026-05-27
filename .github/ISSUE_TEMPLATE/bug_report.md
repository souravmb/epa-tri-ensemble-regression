---
name: Bug Report
about: Report a reproducibility failure, incorrect output, or pipeline error
title: "[BUG] "
labels: bug
assignees: souravmb, Srutiskumar
---

## Description

A clear description of what the bug is.

## Cell / Function Affected

Which notebook cell, function, or stage of the pipeline does this occur in?

- [ ] Data loading (Cell 1)
- [ ] Sampling and target transform (Cell 2)
- [ ] Leakage removal (Cell 3a)
- [ ] Imputation (Cell 3b)
- [ ] Encoding (Cell 3c)
- [ ] Feature engineering / scaling (Cells 5a–5d)
- [ ] Base model training (Cell 6)
- [ ] Bayesian HPO (Cell 7)
- [ ] Ensemble (Cell 8)
- [ ] Evaluation / SHAP (Cells 10–12)
- [ ] Other (describe below)

## Steps to Reproduce

1. 
2. 
3. 

## Expected Behaviour

What should have happened?

## Actual Behaviour

What actually happened? Include the full error traceback if applicable.

```
Paste traceback here
```

## Environment

| Item | Value |
|---|---|
| OS | |
| Python version | |
| Marimo version | |
| scikit-learn version | |
| XGBoost version | |
| Optuna version | |
| SHAP version | |

## Reproduces Without Modification?

- [ ] Yes — the bug occurs on a clean clone with no local changes
- [ ] No — only under specific conditions (describe below)

## Additional Context

Any other information, screenshots, or context that may help diagnose the issue.
