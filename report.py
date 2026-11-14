"""Figures and metrics for the README, generated from one training run."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import credit_model as cm

DOCS = Path(__file__).parent / "docs"
REPORTS = Path(__file__).parent / "reports"

BLUE, ORANGE, GREY = "#2a6fb0", "#d9822b", "#8a8f98"


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_cv(cv_table: pd.DataFrame) -> None:
    t = cv_table.sort_values("cv_auc_mean")
    fig, ax = plt.subplots(figsize=(6.5, 3))
    colors = [BLUE if i == len(t) - 1 else GREY for i in range(len(t))]
    ax.barh(t["model"], t["cv_auc_mean"], xerr=t["cv_auc_std"], color=colors, capsize=4)
    for y, (m, s) in enumerate(zip(t["cv_auc_mean"], t["cv_auc_std"])):
        ax.text(m + s + 0.004, y, f"{m:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.65, 0.88)
    ax.set_xlabel("AUC, repeated 5-fold CV on training set (mean ± 1 sd)")
    _style(ax, "Model selection by cross-validation")
    fig.tight_layout()
    fig.savefig(DOCS / "cv_auc_comparison.png", dpi=150)
    plt.close(fig)


def plot_roc(r: cm.TrainingResult) -> None:
    pd_test = r.model.predict_proba(r.X_test)[:, 1]
    fpr, tpr, _ = roc_curve(r.y_test, pd_test)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"{r.selected_model} (AUC {r.test_metrics['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=9)
    _style(ax, "ROC on held-out test set")
    fig.tight_layout()
    fig.savefig(DOCS / "roc_curve.png", dpi=150)
    plt.close(fig)


def plot_calibration(r: cm.TrainingResult) -> None:
    base = cm.candidate_models()[r.selected_model]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cm.RANDOM_STATE + 1)
    raw_oof = cross_val_predict(clone(base), r.X_train, r.y_train, cv=cv, method="predict_proba")[:, 1]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Perfect calibration")
    for probs, label, color in [(raw_oof, "Uncalibrated", ORANGE), (r.oof_pd, f"Calibrated ({r.calibration})", BLUE)]:
        frac, mean = calibration_curve(r.y_train, probs, n_bins=8, strategy="quantile")
        ax.plot(mean, frac, "o-", color=color, label=label)
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=9, loc="upper left")
    _style(ax, "Calibration (out-of-fold, training set)")
    fig.tight_layout()
    fig.savefig(DOCS / "calibration.png", dpi=150)
    plt.close(fig)


def plot_threshold(r: cm.TrainingResult) -> None:
    grid = np.round(np.arange(0.02, 0.98, 0.01), 2)
    cost = [cm.expected_cost(r.y_train, r.oof_pd, t) for t in grid]
    approval = [(r.oof_pd < t).mean() for t in grid]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(grid, cost, color=BLUE, lw=2, label="Model: cost per applicant")
    ax.axhline(cm.expected_cost(r.y_train, r.oof_pd, -1), color=ORANGE, ls="--", lw=1.2, label="Decline everyone")
    ax.axhline(cm.expected_cost(r.y_train, r.oof_pd, 2), color=GREY, ls=":", lw=1.2, label="Approve everyone")
    ax.axvline(r.threshold, color="black", lw=1)
    ax.text(r.threshold + 0.01, ax.get_ylim()[1] * 0.92, f"chosen {r.threshold:.2f}", fontsize=9)
    ax.set_xlabel("Decline if PD ≥ threshold")
    ax.set_ylabel("Expected cost per applicant")
    ax2 = ax.twinx()
    ax2.plot(grid, approval, color=GREY, lw=1, alpha=0.6)
    ax2.set_ylabel("Approval rate", color=GREY)
    ax2.spines[["top"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")
    _style(ax, "Threshold choice under a 5:1 cost matrix (out-of-fold)")
    fig.tight_layout()
    fig.savefig(DOCS / "threshold_optimization.png", dpi=150)
    plt.close(fig)


def plot_shap(r: cm.TrainingResult) -> pd.DataFrame:
    values, names = cm._shap_values(r.X_test)
    imp = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
    )
    t = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(t["feature"], t["mean_abs_shap"], color=BLUE)
    ax.set_xlabel("Mean |SHAP value| on test set (contribution to PD)")
    _style(ax, "Top 10 drivers of predicted default")
    fig.tight_layout()
    fig.savefig(DOCS / "shap_importance.png", dpi=150)
    plt.close(fig)
    return imp


def cost_ratio_table(r: cm.TrainingResult) -> pd.DataFrame:
    """How the decision changes if the bad:good cost ratio is not 5:1."""
    pd_test = r.model.predict_proba(r.X_test)[:, 1]
    rows = []
    saved = (cm.COST_BAD_APPROVED, cm.COST_GOOD_DECLINED)
    try:
        for ratio in (2, 3, 5, 8):
            cm.COST_BAD_APPROVED, cm.COST_GOOD_DECLINED = float(ratio), 1.0
            t = cm.choose_threshold(r.y_train, r.oof_pd)
            model_cost = cm.expected_cost(r.y_test, pd_test, t)
            best_trivial = min(cm.expected_cost(r.y_test, pd_test, -1), cm.expected_cost(r.y_test, pd_test, 2))
            rows.append(
                {
                    "cost_ratio": f"{ratio}:1",
                    "threshold": t,
                    "approval_rate": float((pd_test < t).mean()),
                    "cost_model": model_cost,
                    "cost_best_trivial": best_trivial,
                    "saving_vs_trivial": 1 - model_cost / best_trivial,
                }
            )
    finally:
        cm.COST_BAD_APPROVED, cm.COST_GOOD_DECLINED = saved
    return pd.DataFrame(rows)


def write_report(r: cm.TrainingResult) -> dict:
    DOCS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    plot_cv(r.cv_table)
    plot_roc(r)
    plot_calibration(r)
    plot_threshold(r)
    shap_top = plot_shap(r)
    ratios = cost_ratio_table(r)
    out = {
        "selected_model": r.selected_model,
        "calibration": r.calibration,
        "calibration_oof_brier": r.calibration_briers,
        "cv": r.cv_table.to_dict(orient="records"),
        "test": r.test_metrics,
        "cost_ratio_sensitivity": ratios.to_dict(orient="records"),
        "shap_top10": shap_top.to_dict(orient="records"),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))
    return out
