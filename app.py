import os
import requests
import streamlit as st
from credit_model import get_model_metrics, predict_risk

API_URL = os.getenv("CREDIT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦", layout="wide")
st.title("🏦 Fintech Credit Risk Demo")
st.caption("German Credit (real-world dataset) + Logistic Regression")

metrics = get_model_metrics()
st.sidebar.markdown("### Model Benchmark")
st.sidebar.write(f"AUC-ROC: **{metrics.auc_roc:.3f}**")
st.sidebar.write(f"Gini: **{metrics.gini_coefficient:.3f}**")

st.sidebar.header("Client Information")
duration = st.sidebar.slider("Loan Duration (months)", 4, 72, 24)
amount = st.sidebar.number_input("Loan Amount (DM)", 250, 20000, 5000, step=100)
age = st.sidebar.slider("Age", 18, 75, 35)
installment_rate = st.sidebar.slider("Installment Rate (1-4)", 1, 4, 2)
number_credits = st.sidebar.slider("Number of Existing Credits", 1, 4, 1)
people_liable = st.sidebar.slider("People Liable", 1, 2, 1)

payload = {
    "duration": duration,
    "amount": amount,
    "age": age,
    "installment_rate": installment_rate,
    "number_credits": number_credits,
    "people_liable": people_liable,
}


def score_client() -> dict:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=2)
        response.raise_for_status()
        scored = response.json()
        scored["source"] = "API"
        return scored
    except requests.RequestException:
        local = predict_risk(payload)
        return {
            "probability_default": local.probability_default,
            "predicted_default": local.predicted_default,
            "decision": "reject" if local.predicted_default else "approve",
            "risk_band": local.risk_band,
            "source": "Local fallback",
        }


if st.button("Evaluate Application", type="primary"):
    scored = score_client()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Default Probability", f"{scored['probability_default'] * 100:.1f}%")
    with col2:
        st.metric("Decision", scored["decision"].upper())
    with col3:
        st.metric("Risk Band", scored["risk_band"].upper())
    if scored["decision"] == "reject":
        st.error("⚠️ HIGH RISK — Loan not recommended")
    else:
        st.success("✅ ACCEPTABLE RISK — Loan recommended")
    st.write("Scoring source:", scored["source"])
    st.json(payload)
else:
    st.info("Set borrower details and click **Evaluate Application**.")
