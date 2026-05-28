# Credit Risk Scoring Model

![CI](https://github.com/franzdafka/credit-risk-analysis/actions/workflows/ci.yml/badge.svg)

A credit scoring system built around the regulatory framework banks operate under -- specifically the Internal Ratings-Based (IRB) approach under Basel II/III, which requires institutions to estimate Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD) for each credit exposure.

This project focuses on the PD component: estimating the likelihood that a borrower will default within a 12-month horizon, and providing auditable, feature-level explanations for each decision as required under GDPR Article 22 and EBA guidelines on internal models.

---

## Regulatory context

Under the Basel III IRB approach, a credit institution must produce three risk parameters for each exposure:

- **PD (Probability of Default)** -- the likelihood of default within one year. This is what the model estimates.
- **LGD (Loss Given Default)** -- the fraction of exposure lost if default occurs. Not modelled here; typically estimated separately from recovery data.
- **EAD (Exposure at Default)** -- the outstanding balance at the time of default. Determined by the loan terms, not modelled.

The model output (`probability_default`) maps directly to the PD parameter. The risk band (low / medium / high) is a derived classification based on PD thresholds, analogous to the rating grades used in internal rating systems.

Logistic regression was chosen as the primary classifier because its outputs are calibrated probabilities by construction -- a requirement for PD estimation under IRB -- and because SHAP values for linear models are mathematically exact, producing explanations that are auditable rather than approximate. This satisfies the explainability requirements imposed by the ECB, PRA, and GDPR Art. 22 on automated credit decisions.

---

## Model output and decision mapping

| Output field | Meaning | IRB mapping |
|---|---|---|
| `probability_default` | Estimated 12-month PD | PD input to RWA calculation |
| `risk_band: low` | PD < 0.30 | Investment grade equivalent |
| `risk_band: medium` | 0.30 <= PD < 0.60 | Sub-investment grade |
| `risk_band: high` | PD >= 0.60 | Watch list / decline |

---

## Explainability

The `/explain` endpoint returns SHAP values for each applicant -- the top factors increasing and decreasing default risk. For a linear model, SHAP values are equivalent to the product of feature coefficients and mean-centred inputs, making them mathematically exact and legally defensible under GDPR Art. 22 (right to explanation for automated decisions).

This is the centrepiece of the project. In a production banking environment, every automated credit decision must be explainable to the applicant and auditable by the regulator. The `/explain` endpoint is designed with that requirement in mind.

```bash
curl "http://127.0.0.1:8000/explain?user_id=0"
```

Example response:

```json
{
  "top_positive": [
    {"feature": "num__dti_ratio", "shap_value": 0.312},
    {"feature": "num__duration", "shap_value": 0.198}
  ],
  "top_negative": [
    {"feature": "num__credit_history_length", "shap_value": -0.241},
    {"feature": "cat__credit_history_existing paid", "shap_value": -0.189}
  ]
}
```

---

## Dataset

German Credit Data (Statlog) -- 1,000 loan applications, 20 features, binary target (1 = good payer, 0 = default). The dataset has a 70/30 class imbalance, handled via balanced class weights during training.

Source: UCI Machine Learning Repository / Hofmann (1994)

**Known limitations of this dataset:** it is small by modern standards, collected in 1990s Germany, and contains no bureau data, behavioural signals, or macroeconomic features. A production IRB model would require a minimum of several years of through-the-cycle data and would be subject to annual model validation. Migration to Home Credit Default Risk (307k applications) is a planned next step.

---

## Modelling approach

### Feature engineering

The raw dataset has limited features so several derived ones were constructed:

- `dti_ratio` -- debt-to-income proxy: loan amount divided by estimated income from installment rate and employment duration
- `credit_history_length` -- age minus 18, adjusted for loan duration; proxy for credit file length
- `has_delinquency` -- binary flag for past delinquency extracted from credit history category
- `employment_length` -- numeric encoding of employment duration categories
- `amount_x_duration` -- interaction term capturing large long-term exposures

### Model selection

Four classifiers were compared using 5-fold stratified cross-validation. Full results including ROC curves, lift curves, calibration plots, and SHAP analysis are in `analysis.ipynb`.

| Model | Role |
|---|---|
| Logistic Regression | Primary model (deployed) |
| Random Forest | Ensemble benchmark |
| Gradient Boosting | Ensemble benchmark |
| XGBoost | Ensemble benchmark |

Logistic Regression was selected for deployment on three grounds: calibrated probability outputs (required for PD estimation), mathematical exactness of SHAP explanations (required for auditability), and regulatory preference for interpretable models in credit decisions (EBA/GL/2020/06).

### Evaluation metrics

Standard credit scoring metrics were used rather than accuracy:

- AUC-ROC and Gini coefficient (2 x AUC - 1). Gini > 0.40 is generally acceptable for a single-bureau scorecard; > 0.60 is considered strong.
- KS statistic -- maximum separation between the cumulative default and non-default distributions
- Cumulative lift curve -- defaults captured in the top X% of the scored population
- Calibration curve -- alignment between predicted PD and observed default rates

### Decision threshold

The default 0.5 classification threshold is not optimal when false negatives (approving a defaulter) carry higher cost than false positives (declining a good applicant). The `analysis.ipynb` notebook includes threshold optimization under an asymmetric cost matrix, which is the standard approach in retail credit decisioning.

---

## Model validation

A production credit model at a regulated institution would go through independent model validation covering at minimum:

- **Population Stability Index (PSI)** -- monitoring whether the score distribution has shifted between development and current population
- **Gini stability** -- tracking discriminatory power over time
- **Stress testing** -- estimating PD uplift under adverse macroeconomic scenarios (e.g. unemployment +3pp, GDP -2%)
- **Backtesting** -- comparing predicted PD against observed default rates by rating grade

PSI monitoring and stress testing are identified as next steps for this project. The current implementation includes Kupiec backtesting logic in the notebook as a starting point.

---

## API endpoints

- `POST /predict` -- returns PD estimate, classification, and risk band
- `GET /explain?user_id=<id>` -- returns top SHAP factors for a given applicant
- `GET /health` -- returns service status and deployed model version

---

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

---

## Running locally

```bash
pip install -r requirements.txt
python train_model.py
uvicorn api:app --reload --port 8000
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

---

## Example request

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

---

## Known limitations

- No macroeconomic features -- PD estimates are through-the-cycle rather than point-in-time
- Income is estimated from installment rate and employment duration rather than observed directly
- No population stability monitoring or data drift detection
- Dataset size (1,000 observations) is insufficient for a production IRB model; minimum regulatory expectation is several years of through-the-cycle data
- Fairness and disparate impact analysis not yet implemented

---

## References

- Baesens et al. (2003). Benchmarking State-of-the-Art Classification Algorithms for Credit Scoring. Journal of the Operational Research Society.
- Hand & Henley (1997). Statistical Classification Methods in Consumer Credit Scoring. Journal of the Royal Statistical Society.
- Basel Committee on Banking Supervision. Basel II: International Convergence of Capital Measurement and Capital Standards, 2004.
- EBA/GL/2020/06. Guidelines on loan origination and monitoring. European Banking Authority, 2020.
- ECB. Guide to internal models -- Credit risk. European Central Bank, 2019.
