from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from credit_model import FEATURE_COLUMNS, load_modeling_frame

REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"


def main() -> None:
    sns.set_theme(style="whitegrid")
    REPORT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_modeling_frame()

    # 1) Target imbalance
    plt.figure(figsize=(6, 4))
    sns.countplot(x="credit_risk", data=df)
    plt.title("Target Distribution (credit_risk)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_imbalance.png", dpi=140)
    plt.close()

    # 2) Feature distributions (numeric)
    numeric_features = [
        c for c in FEATURE_COLUMNS if pd.api.types.is_numeric_dtype(df[c]) and c != "credit_risk"
    ]
    dist_df = df[numeric_features].melt(var_name="feature", value_name="value")
    g = sns.FacetGrid(dist_df, col="feature", col_wrap=3, sharex=False, sharey=False, height=3)
    g.map_dataframe(sns.histplot, x="value", bins=25)
    g.fig.suptitle("Numeric Feature Distributions", y=1.02)
    g.savefig(FIG_DIR / "feature_distributions.png", dpi=140)
    plt.close("all")

    # 3) Correlation matrix
    corr = df[numeric_features + ["credit_risk"]].corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=140)
    plt.close()

    # 4) Missing value analysis + imputation plan
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_report = pd.DataFrame(
        {
            "missing_pct": missing_pct,
            "imputation_strategy": [
                "median" if pd.api.types.is_numeric_dtype(df[c]) else "most_frequent" for c in missing_pct.index
            ],
        }
    )
    missing_report.to_csv(REPORT_DIR / "missing_value_report.csv", index=True)

    print("EDA artifacts generated in reports/ and reports/figures/.")


if __name__ == "__main__":
    main()
