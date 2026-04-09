import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================
# 1. DATA CREATION
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
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

print("=== General Statistics ===")
print(df.describe())
print()

default_rate = df["default"].mean() * 100
print(f"Default Rate: {default_rate:.1f}%")
print()

# ============================================
# 3. VISUALIZATIONS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Customer Credit Risk Analysis", fontsize=14)

# Chart 1 — Age distribution
axes[0, 0].hist(df["age"], bins=8, color="steelblue", edgecolor="black")
axes[0, 0].set_title("Age Distribution")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Number of Clients")

# Chart 2 — Average income by default
df.groupby("default")["annual_income"].mean().plot(
    kind="bar", ax=axes[0, 1], color=["green", "red"], edgecolor="black"
)
axes[0, 1].set_title("Average Income by Default Status")
axes[0, 1].set_xlabel("Default (0=No, 1=Yes)")
axes[0, 1].set_ylabel("Average Income (€)")
axes[0, 1].tick_params(rotation=0)

# Chart 3 — Loan amount vs Income
colors = ["red" if d == 1 else "green" for d in df["default"]]
axes[1, 0].scatter(df["annual_income"], df["loan_amount"], c=colors)
axes[1, 0].set_title("Loan Amount vs Annual Income")
axes[1, 0].set_xlabel("Annual Income (€)")
axes[1, 0].set_ylabel("Loan Amount (€)")

# Chart 4 — Default rate pie chart
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
# 4. PREDICTION MODEL
# ============================================

X = df[["age", "annual_income", "loan_amount"]]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

print("=== Model Performance ===")
print(classification_report(y_test, model.predict(X_test)))

# New client prediction example
new_client = pd.DataFrame([{
    "age": 30,
    "annual_income": 35000,
    "loan_amount": 10000
}])

prediction = model.predict(new_client)[0]
print(f"New client prediction: {'High default risk ⚠️' if prediction == 1 else 'Reliable client ✅'}")
