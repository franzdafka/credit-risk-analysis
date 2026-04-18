# 🏦 Deployable Fintech Credit Risk Demo

This project now includes a **data foundation + benchmark + explainability stack** for credit risk modeling.

## Level 1 — Data Foundation

- Dataset: German Credit dataset (`GermanCredit.csv`) with local-cache-first loading.
- Feature engineering added:
  - `dti_ratio`
  - `credit_history_length`
  - `number_of_delinquencies`
  - `employment_length`
  - `loan_purpose`
- Professional EDA script: `eda.py` (exports figures + missing-value strategy report).

### Run EDA

```bash
python eda.py
```

Outputs:
- `reports/figures/target_imbalance.png`
- `reports/figures/feature_distributions.png`
- `reports/figures/correlation_matrix.png`
- `reports/missing_value_report.csv`

## Level 2 — Model Quality & Explainability

`benchmark_models.py` performs model comparison and explainability:

- Logistic Regression (baseline)
- Random Forest
- XGBoost (if installed) or gradient-boosting fallback

Metrics:
- AUC-ROC
- Precision-Recall AUC + PR curves
- Gini coefficient

Imbalance handling:
- `class_weight='balanced'` in baseline/tree models

SHAP explainability:
- Global importance beeswarm: `reports/figures/shap_summary_beeswarm.png`
- Individual waterfall explanation: `reports/figures/shap_waterfall_first_prediction.png`

### Run benchmarking

```bash
python benchmark_models.py
```

Outputs:
- `reports/benchmark_results.csv`
- `reports/figures/pr_curve_*.png`
- SHAP plots under `reports/figures/`

## API + UI

- **FastAPI scoring service** (`/predict`, `/health`, `/metrics`)
- **Streamlit underwriting UI** with API-first scoring and local fallback
- **Docker Compose** deployment

### Quick Start

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
# in another terminal
streamlit run app.py
```

Run tests:

```bash
pytest
```
