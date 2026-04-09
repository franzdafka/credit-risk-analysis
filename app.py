import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Train model
data = {
    "age": [25, 45, 35, 50, 23, 40, 60, 28, 33, 55,
            29, 48, 37, 52, 24, 41, 62, 30, 34, 57],
    "annual_income": [25000, 60000, 40000, 80000, 20000,
                      55000, 90000, 30000, 45000, 75000,
                      27000, 63000, 42000, 82000, 22000,
                      58000, 95000, 32000, 47000, 78000],
    "loan_amount": [5000, 15000, 8000, 20000, 3000,
                    12000, 25000, 6000, 9000, 18000,
                    5500, 16000, 8500, 21000, 3500,
                    13000, 27000, 6500, 9500, 19000],
    "default": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
}

df = pd.DataFrame(data)
X = df[["age", "annual_income", "loan_amount"]]
y = df["default"]

model = LogisticRegression()
model.fit(X, y)

# ============================================
# STREAMLIT APP
# ============================================
st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦")

st.title("🏦 Credit Risk Analyzer")
st.markdown("Enter client information to predict default risk.")

st.sidebar.header("Client Information")

age = st.sidebar.slider("Age", 18, 80, 35)
annual_income = st.sidebar.number_input("Annual Income (€)", 10000, 200000, 40000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount (€)", 1000, 50000, 10000, step=500)

client = pd.DataFrame([{
    "age": age,
    "annual_income": annual_income,
    "loan_amount": loan_amount
}])

prediction = model.predict(client)[0]
probability = model.predict_proba(client)[0][1]

st.subheader("📊 Risk Assessment")

col1, col2 = st.columns(2)

with col1:
    st.metric("Default Probability", f"{probability * 100:.1f}%")

with col2:
    if prediction == 1:
        st.error("⚠️ HIGH RISK — Loan not recommended")
    else:
        st.success("✅ LOW RISK — Loan recommended")

st.subheader("👤 Client Profile")
st.dataframe(client)

risk_level = "🔴 High" if probability > 0.6 else "🟡 Medium" if probability > 0.3 else "🟢 Low"
st.markdown(f"**Risk Level:** {risk_level}")
