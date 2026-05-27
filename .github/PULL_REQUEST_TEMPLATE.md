## Summary

What does this PR do? One paragraph maximum.

## Related Issue

Closes # (issue number, if applicable)

## Type of Change

- [ ] Bug fix — corrects incorrect output, error, or reproducibility failure
- [ ] Methodology extension — adds a new model, feature engineering step, or evaluation
- [ ] Scaling — applies the pipeline to more data (full 80K records, multi-year, etc.)
- [ ] Documentation — README, docstrings, comments
- [ ] Dependency update — version bump or replacement
- [ ] Other (describe below)

---

## Leakage Audit Checklist

This is the most critical checklist in this repository. Every PR that touches the preprocessing, feature selection, or modelling cells must confirm the following:

- [ ] No imputer, scaler, encoder, or threshold was fitted on any data that includes test observations
- [ ] No new feature derived from `TOTAL_RELEASES` or its arithmetic components has been added without updating the leakage-pattern rules
- [ ] Target encoding statistics (`y_bar_c`, `y_bar_g`) are computed exclusively on training labels
- [ ] Winsorisation bounds are computed on training data and applied (not re-fitted) on test data
- [ ] Correlation-based deduplication uses only training-set feature pairs
- [ ] If a new column from the TRI dataset is included, it has been verified against the EPA codebook as a non-component of `TOTAL_RELEASES`

---

## Testing

Describe how the change was validated:

- [ ] Ran the full notebook end-to-end on a clean environment
- [ ] Final RMSE is within expected range (< 0.30 on test set, log₁p scale)
- [ ] Final R² is within expected range (> 0.995 on test set)
- [ ] No null values remain after imputation step (Cell 3b verification)
- [ ] SHAP values were inspected and top features have physical interpretations

---

## Performance Impact

If this PR changes model results, fill in the table. Otherwise delete this section.

| Metric | Before | After |
|---|---|---|
| RMSE (test, log₁p) | | |
| R² (test) | | |
| Weighted F1 (5-class) | | |
| Nested CV R² mean | | |

---

## Environment

| Item | Value |
|---|---|
| OS | |
| Python version | |
| Marimo version | |

---

## Notes for Reviewers

Anything specific you want the maintainers to pay attention to during review.
