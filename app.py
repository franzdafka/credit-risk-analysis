import os

import requests
import streamlit as st

from credit_model import predict_risk

API_URL = os.getenv("CREDIT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦", layout="wide")

st.title("🏦 Fintech Credit Risk Demo")
st.caption("Production-style demo with Streamlit UI + REST scoring API")

st.sidebar.header("Client Information")
age = st.sidebar.slider("Age", 18, 80, 35)
annual_income = st.sidebar.number_input("Annual Income (€)", 10000, 200000, 40000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount (€)", 1000, 50000, 10000, step=500)

payload = {"age": age, "annual_income": annual_income, "loan_amount": loan_amount}


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
