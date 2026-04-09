import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shap

# ============================================
# 1. DATA
# ============================================
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

# ============================================
# 2. EDA
# ============================================
print("=== General Statistics ===")
print(df.describe())
print(f"\nDefault Rate: {df['default'].mean() * 100:.1f}%\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Customer Credit Risk Analysis", fontsize=14)

axes[0, 0].hist(df["age"], bins=8, color="steelblue", edgecolor="black")
axes[0, 0].set_title("Age Distribution")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Number of Clients")

df.groupby("default")["annual_income"].mean().plot(
    kind="bar", ax=axes[0, 1], color=["green", "red"], edgecolor="black"
)
axes[0, 1].set_title("Average Income by Default Status")
axes[0, 1].set_xlabel("Default (0=No, 1=Yes)")
axes[0, 1].set_ylabel("Average Income (€)")
axes[0, 1].tick_params(rotation=0)

colors = ["red" if d == 1 else "green" for d in df["default"]]
axes[1, 0].scatter(df["annual_income"], df["loan_amount"], c=colors)
axes[1, 0].set_title("Loan Amount vs Annual Income")
axes[1, 0].set_xlabel("Annual Income (€)")
axes[1, 0].set_ylabel("Loan Amount (€)")

axes[1, 1].pie(
    [df["default"].sum(), len(df) - df["default"].sum()],
    labels=["Default", "No Default"],
    colors=["red", "green"],
    autopct="%1.1f%%"
)
axes[1, 1].set_title("Default Distribution")

plt.tight_layout()
plt.savefig("visualizations.png")
plt.show()

# ============================================
# 3. MODEL
# ============================================
X = df[["age", "annual_income", "loan_amount"]]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Model Performance ===")
print(classification_report(y_test, y_pred))

# ============================================
# 4. CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ============================================
# 5. COST OF ERROR
# ============================================
# False Negative = bank gives loan to someone who defaults → loses loan amount
# False Positive = bank refuses a good client → loses potential interest (10%)

avg_loan = df["loan_amount"].mean()
interest_rate = 0.10

FN = cm[1][0]  # missed defaults
FP = cm[0][1]  # refused good clients

cost_FN = FN * avg_loan
cost_FP = FP * avg_loan * interest_rate
total_cost = cost_FN + cost_FP

print("=== Cost of Error ===")
print(f"False Negatives (missed defaults): {FN} → Cost: €{cost_FN:,.0f}")
print(f"False Positives (refused good clients): {FP} → Lost revenue: €{cost_FP:,.0f}")
print(f"Total estimated cost of errors: €{total_cost:,.0f}")

# ============================================
# 6. PROFIT SIMULATION
# ============================================
correct_defaults_caught = cm[1][1]
good_clients_approved = cm[0][0]

profit = (good_clients_approved * avg_loan * interest_rate) - cost_FN
print(f"\n=== Profit Simulation ===")
print(f"Revenue from good clients: €{good_clients_approved * avg_loan * interest_rate:,.0f}")
print(f"Losses from missed defaults: €{cost_FN:,.0f}")
print(f"Net estimated profit: €{profit:,.0f}")

# ============================================
# 7. SHAP
# ============================================
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.show()

print("\nDone! Check visualizations.png, confusion_matrix.png, shap_summary.png")
