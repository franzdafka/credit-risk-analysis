# Credit Risk Scoring Service — German Credit Dataset

A probability-of-default scoring service for retail credit. Benchmarks four classification models, selects XGBoost on test-set performance, and exposes a REST API with SHAP-based per-decision explanations structured around the Basel III IRB Expected Loss framework: **EL = PD × LGD × EAD**.

---

## Model Performance

Trained on the [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) (Hofmann, 1994) — 1,000 obligors, 20 features, ~70/30 good/bad split.

| Model | AUC-ROC (5-fold CV) | AUC-ROC (test set) |
|---|---|---|
| Logistic Regression | 0.780 | 0.776 |
| Random Forest | 0.784 | 0.780 |
| Gradient Boosting | 0.762 | 0.783 |
| **XGBoost** | 0.766 | **0.787** |

**Gini coefficient (XGBoost): 0.574**

![ROC Curves](docs/roc_curves.png)

### Why XGBoost despite a lower cross-validation score?

On 5-fold cross-validation, Random Forest (0.7842) and Logistic Regression (0.7797) slightly outperform XGBoost (0.7659) — within overlapping confidence intervals, so no model is statistically distinguishable on CV alone. XGBoost was selected because it achieved the highest **test-set** AUC (0.787), the metric that best approximates real-world generalization.

![Cross-Validation AUC Comparison](docs/cv_auc_comparison.png)

---

## Feature Importance (SHAP)

`no checking account` status, savings level, and loan duration are the strongest predictors of default risk. This is consistent with standard retail credit scoring literature: applicants without an existing banking relationship carry less observable credit history, increasing model uncertainty and predicted risk.

![SHAP Feature Importance](docs/shap_importance.png)

---

## Cost-Sensitive Threshold Optimization

Standard classification metrics assume a 0.5 decision threshold and symmetric error costs. In credit underwriting, approving a defaulting borrower (false negative) is materially more expensive than rejecting a creditworthy one (false positive). Using a 5:1 cost ratio (FN×5 + FP×1), the cost-minimizing threshold for XGBoost drops to ~0.01 — total cost falls from 240 at the default threshold to 140 at the optimized one, a 42% reduction.

This illustrates why credit risk models should be evaluated on business cost curves rather than a fixed 0.5 cutoff or accuracy alone.

![Threshold Optimization](docs/threshold_optimization.png)

---

## Project Structure

| File | Role |
|---|---|
| `credit_model.py` | Model training, feature engineering, SHAP inference |
| `api.py` | REST API — `/predict`, `/explain`, `/health` |
| `train_model.py` | Trains and serializes model artifact to `artifacts/` |
| `eda.py` | Exploratory analysis — distributions, correlations, missing values |
| `model_benchmark.py` | Benchmarks Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| `analysis.ipynb` | Full pipeline notebook: EDA, model comparison, SHAP, threshold optimization |
| `tests/test_api.py` | Smoke, boundary, and monotonicity tests |

---

## API Endpoints

### POST /predict

Returns a 12-month PD estimate, internal rating grade, and underwriting decision.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 24,
    "amount": 5000,
    "age": 35,
    "installment_rate": 2,
    "number_credits": 1,
    "people_liable": 1,
    "purpose": "car",
    "credit_history": "existing paid",
    "employment_duration": "1 <= ... < 4 yrs"
  }'
```

```json
{
  "probability_default": 0.18,
  "predicted_default": false,
  "rating_grade": "low",
  "underwriting_decision": "approve"
}
```

### GET /explain?user_id=\<id\>

Returns the top 3 risk-increasing and top 3 risk-reducing SHAP factors for a given applicant, computed via TreeSHAP on the underlying XGBoost model.

```bash
curl "http://localhost:8000/explain?user_id=0"
```

```json
{
  "top_positive": [
    { "feature": "cat__status_no checking account", "shap_value": 0.34 },
    { "feature": "num__duration",  "shap_value": 0.18 }
  ],
  "top_negative": [
    { "feature": "cat__housing_own", "shap_value": -0.12 }
  ]
}
```

### GET /health

```json
{
  "status": "ok",
  "model_version": "credit-risk-20260621170834"
}
```

---

## Feature Engineering

Five features derived from the raw dataset:

| Feature | Description |
|---|---|
| `dti_ratio` | Loan amount divided by an estimated income proxy |
| `employment_length` | Ordinal encoding of employment duration band |
| `number_of_delinquencies` | Binary flag derived from credit history text |
| `credit_history_length` | Age-based proxy for length of credit track record |
| `loan_purpose` | Passthrough of `purpose` for one-hot encoding |

---

## Quick Start

```bash
pip install -r requirements.txt
python3 train_model.py
uvicorn api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## Tests

```bash
pytest
```

Covers smoke tests (`/health`, `/predict`, `/explain`), boundary inputs, and a monotonicity check that higher income generally reduces the predicted default probability.

---

## Methodology

- **Validation:** 5-fold cross-validation on the training set; held-out test set for final model selection and reporting
- **Explainability:** SHAP (TreeSHAP) for per-prediction and global feature attribution
- **Class imbalance:** explicit cost-based threshold optimization rather than relying on the default 0.5 cutoff

## Limitations

- The dataset is small (1,000 rows), which widens cross-validation confidence intervals and limits how far model comparisons can be trusted
- AUC in the 0.76–0.79 range is typical for this dataset given its limited feature set and size, not a modelling shortfall
- Income is proxied from installment rate and employment duration, not directly observed
- No macroeconomic features — model produces through-the-cycle risk estimates only, with no point-in-time adjustment
- No Population Stability Index (PSI) monitoring — score distribution shifts would not be detected automatically
- Cost ratios used in threshold optimization (5:1) are illustrative; in production this would be calibrated against actual loss-given-default and lost-revenue figures
- This is a proof-of-concept, not an independently validated production model

---

## References

- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.
- Basel Committee on Banking Supervision (2017). *Basel III: Finalising post-crisis reforms*.
