# Security Policy

## Scope

This repository contains a machine learning research pipeline and does not operate as a web service, API, or deployed application. There is no network-facing component, no user authentication, and no production infrastructure.

Security concerns relevant to this project are:

- **Data leakage in the ML pipeline** — the most consequential "security" issue in this codebase is the introduction of target-leaking features that inflate reported performance metrics. This is treated as a first-class correctness concern.
- **Dependency vulnerabilities** — third-party packages (scikit-learn, XGBoost, Optuna, SHAP) may have CVEs in older versions.
- **Notebook code execution** — running untrusted `.py` or `.ipynb` files carries the same risks as running any Python script.

## Supported Versions

| File | Status |
|---|---|
| `EPATRIML_1_.py` (Marimo notebook) | Actively maintained |
| `EPATRIML-2.ipynb` (Jupyter export) | Maintained in sync with the Marimo source |

## Reporting a Vulnerability

**Do not open a public issue for security-sensitive reports.**

If you identify a concern — such as a dependency with a known CVE, a subtle data leakage path not covered by the existing 17-pattern rules, or a reproducibility failure that could mislead downstream research — please report it privately:

1. Email **Sourav M B** at cb.ps.p2asd25023@cb.students.amrita.edu with the subject line `[SECURITY] epa-tri-ensemble-regression`
2. Describe the issue clearly: which file, which cell or function, what the impact is, and a minimal reproduction if possible
3. We will acknowledge the report within 5 business days and aim to resolve confirmed issues within 30 days

For dependency vulnerabilities, opening a GitHub Dependabot alert or a private advisory via **Security → Advisories → New draft advisory** on this repository is also acceptable.

## Data Handling Note

The EPA TRI dataset used in this project is publicly available and contains no personally identifiable information (PII). The dataset covers facility-level chemical release reporting aggregated at the facility-chemical combination level. No individual-level data is processed.
