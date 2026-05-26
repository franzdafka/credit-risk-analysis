# Credit Risk Analysis

![CI](https://github.com/franzdafka/credit-risk-analysis/actions/workflows/ci.yml/badge.svg)

A credit scoring project I built to explore how machine learning can support loan approval decisions. The idea was to go beyond a basic model and build something closer to how banks actually think about default risk -- with proper evaluation metrics, explainability, and a deployable API.

## What it does

The service takes applicant information (age, loan amount, employment duration, credit history, etc.) and returns a default probability plus a risk band (low / medium / high). There is also a SHAP-based explanation endpoint that shows which features drove the score for a specific applicant.

Three endpoints:

- `POST /predict` -- default probability and underwriting decision
- `GET /explain?user_id=<id>` -- top positive and negative SHAP factors for a given profile
- `GET /health` -- service status and model version

## Dataset

German Credit Data (Statlog) -- 1,000 loan applications, 20 features, binary target (1 = good payer, 0 = default). The dataset has a 70/30 class imbalance, which required using balanced class weights during training. It is a well-known benchmark dataset but has real limitations: it is small, collected in 1990s Germany, and contains no bureau or behavioural signals.

Source: UCI Machine Learning Repository

## Modelling approach

### Feature engineering

The raw dataset has limited features so I built several derived ones:

- `dti_ratio` -- debt-to-income proxy (loan amount divided by estimated income from installment rate and employment duration)
- `credit_history_length` -- age minus 18, adjusted for loan duration
- `has_delinquency` -- binary flag extracted from credit history category
- `employment_length` -- numeric encoding of employment duration categories
- `amount_x_duration` -- interaction term to capture large long-term loans

### Model comparison

I compared four classifiers using 5-fold stratified cross-validation. The full results with ROC curves, lift curves, calibration plots and SHAP analysis are in `analysis.ipynb`.

| Model | Description |
|---|---|
| Logistic Regression | Primary production model |
| Random Forest | Ensemble baseline |
| Gradient Boosting | Ensemble baseline |
| XGBoost | Ensemble baseline |

### Why Logistic Regression for the API

It is not always the highest AUC model, but it has three practical advantages here: the probabilities are well-calibrated by construction, it is interpretable in the way regulated lending environments require (GDPR Art. 22, Basel II IRB), and SHAP values for a linear model are mathematically exact rather than approximate. Any move to an ensemble should come with a clear AUC improvement to justify the interpretability trade-off.

### Evaluation metrics

I used metrics that are standard in credit scoring rather than just accuracy:

- AUC-ROC and Gini coefficient (2 x AUC - 1), where Gini > 0.40 is generally acceptable for a single-bureau scorecard
- KS statistic -- maximum separation between TPR and FPR curves
- Cumulative lift curve -- how many defaults are captured in the top X% of scored applicants
- Calibration curve -- whether predicted probabilities match actual default rates

### Decision threshold

The default 0.5 threshold is not optimal when approving a defaulter costs more than rejecting a good applicant. The notebook includes a threshold optimization section using an asymmetric cost matrix.

## Project structure

```
credit_model.py        feature engineering, model training, SHAP explanations
train_model.py         trains and serializes the model artifact
api.py                 FastAPI service
app.py                 Streamlit frontend for local exploration
analysis.ipynb         EDA, model comparison, evaluation plots
benchmark_models.py    standalone model benchmarking script
eda.py                 standalone EDA script
tests/test_api.py      API tests
docker-compose.yml     runs API and frontend together
```

## Running locally

```bash
pip install -r requirements.txt
python train_model.py
uvicorn api:app --reload --port 8000
```

In a separate terminal:

```bash
streamlit run app.py
```

API docs: http://localhost:8000/docs

With Docker:

```bash
docker compose up --build
```

Tests:

```bash
pytest tests/ -v
```

## Example requests

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 24,
    "amount": 5000,
    "income": 4200,
    "age": 35,
    "installment_rate": 2,
    "number_credits": 1,
    "people_liable": 1,
    "purpose": "car",
    "credit_history": "existing paid",
    "employment_duration": "1 <= ... < 4 yrs"
  }'
```

```bash
curl "http://127.0.0.1:8000/explain?user_id=0"
```

## Known limitations

- No macroeconomic features -- the model does not use unemployment rates, inflation, or regional data
- The income variable is estimated from installment rate and employment duration rather than observed directly, which adds noise to the DTI ratio
- No data drift monitoring -- in a real production setting this would be necessary
- Fairness and disparate impact analysis is not yet implemented
- Testing on a larger and more recent dataset (Lending Club, Home Credit) would be a natural next step

## References

- Baesens et al. (2003). Benchmarking State-of-the-Art Classification Algorithms for Credit Scoring. Journal of the Operational Research Society.
- Hand & Henley (1997). Statistical Classification Methods in Consumer Credit Scoring. Journal of the Royal Statistical Society.
- Basel Committee on Banking Supervision. Basel II: International Convergence of Capital Measurement and Capital Standards, 2004.
