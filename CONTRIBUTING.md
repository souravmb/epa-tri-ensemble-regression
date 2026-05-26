# Contributing Guidelines

Thank you for your interest in this project. Contributions that improve reproducibility, extend the methodology, or address identified limitations are welcome.

## Scope of Contributions

Contributions aligned with the paper's research agenda are prioritised:

- Scaling the pipeline to the full 80,040-record dataset
- Temporal validation across reporting years (2010–2022)
- Two-stage zero-inflated modelling for the Zero↔Low classification problem
- Additional ensemble strategies (stochastic gradient boosting, neural meta-learner)
- Codebook-level verification of Section 8 features for residual leakage

## How to Contribute

1. **Fork** the repository and create a branch from `main`
2. Make your changes in a well-named branch (e.g., `feature/temporal-validation`)
3. Ensure all preprocessing remains **strictly train-fitted** — this is the core methodological invariant
4. Update the README if you add new pipeline stages or results
5. Open a **Pull Request** with a clear description of what was changed and why

## Code Style

- Python 3.11+
- Notebooks: Marimo `.py` format (preferred) or Jupyter `.ipynb`
- Variable names: descriptive, snake_case
- All random operations seeded with `SEED = 42`

## Reporting Issues

If you identify a leakage risk, a reproducibility failure, or a methodological concern, please open an Issue with:

- A clear description of the problem
- The cell(s) or function(s) affected
- Proposed fix or investigation steps

## Academic Integrity

This repository is associated with a research submission. All contributions must be original and must not incorporate third-party code without proper attribution and compatible licensing.
