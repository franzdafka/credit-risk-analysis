"""
KS (Kolmogorov-Smirnov) Statistic for Credit Scoring Model Evaluation.

In credit scoring, the KS statistic measures the maximum separation between the
cumulative distribution functions (CDFs) of the predicted probabilities (or scores)
for the "good" (non-default) and "bad" (default) classes. A higher KS indicates
better discriminatory power: the model assigns systematically different scores
to good vs. bad applicants.

Industry benchmarks (typical for retail credit PD models):
- KS < 0.20 : weak separation
- 0.20 – 0.40 : acceptable
- 0.40 – 0.60 : good
- > 0.60 : excellent (rare on real data)

The statistic is computed using the two-sample KS test (scipy.stats.ks_2samp).
We also report the probability threshold at which the maximum CDF gap occurs.
"""

import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute the Kolmogorov-Smirnov (KS) statistic between the predicted
    probability distributions of good and bad classes.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (1 = good / non-default, 0 = bad / default).
    y_proba : np.ndarray
        Predicted probabilities of being good (class 1).

    Returns
    -------
    dict
        {
            "ks_statistic": float,   # max |CDF_good - CDF_bad|
            "ks_pvalue": float,      # p-value from ks_2samp
            "threshold": float       # proba value at which max separation occurs
        }
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    good = y_proba[y_true == 1]
    bad = y_proba[y_true == 0]

    if len(good) == 0 or len(bad) == 0:
        raise ValueError("Both classes must have samples to compute KS statistic.")

    # Two-sample KS test on the probability distributions
    ks_stat, pvalue = ks_2samp(good, bad, alternative="two-sided", mode="auto")

    # Find the score threshold corresponding to the maximum CDF gap
    all_scores = np.sort(np.unique(y_proba))
    cdf_bad = np.searchsorted(np.sort(bad), all_scores, side="right") / len(bad)
    cdf_good = np.searchsorted(np.sort(good), all_scores, side="right") / len(good)

    gaps = np.abs(cdf_good - cdf_bad)
    max_idx = int(np.argmax(gaps))
    threshold = float(all_scores[max_idx])

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(pvalue),
        "threshold": threshold,
    }


def plot_ks_curve(ks_result: dict, model_name: str, save_path: str,
                    y_true: np.ndarray = None, y_proba: np.ndarray = None) -> None:
    """
    Plot the KS curve: empirical CDF of predicted probabilities for good vs. bad classes.

    The point of maximum separation (the KS statistic) is annotated with an arrow.

    Uses a dark GitHub-style theme (dark background #0d1117 / #161b22, GitHub-dark
    accent colors) to match the style of docs/roc_curves.png and docs/shap_importance.png.

    If y_true and y_proba are provided, exact empirical CDFs are plotted.
    Otherwise a representative illustration based on the ks_result is used.

    Parameters
    ----------
    ks_result : dict
        Output from compute_ks_statistic (must contain ks_statistic and threshold).
    model_name : str
        Name of the model for the title (e.g. "XGBoost").
    save_path : str
        Destination PNG path (e.g. "docs/ks_curve.png").
    y_true, y_proba : np.ndarray, optional
        Original labels and probabilities for accurate CDFs.
    """
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    color_good = "#3fb950"
    color_bad = "#f85149"
    color_sep = "#d29922"
    text_color = "#c9d1d9"
    grid_color = "#30363d"

    ks = ks_result["ks_statistic"]
    thresh = ks_result.get("threshold", 0.5)

    if y_true is not None and y_proba is not None:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        good = y_proba[y_true == 1]
        bad = y_proba[y_true == 0]

        x = np.linspace(0, 1, 501)
        cdf_good = np.searchsorted(np.sort(good), x, side="right") / len(good)
        cdf_bad = np.searchsorted(np.sort(bad), x, side="right") / len(bad)
    else:
        # Fallback illustrative curves
        x = np.linspace(0, 1, 500)
        cdf_good = 1 / (1 + np.exp(-12 * (x - (thresh - 0.08))))
        cdf_bad = 1 / (1 + np.exp(-12 * (x - (thresh + 0.08))))
        gap = np.max(np.abs(cdf_good - cdf_bad))
        if gap > 0:
            scale = ks / gap
            cdf_good = 0.5 * (1 - scale) + scale * cdf_good
            cdf_bad = 0.5 * (1 - scale) + scale * cdf_bad

    ax.plot(x, cdf_good, color=color_good, lw=2.5, label="Good (y=1)")
    ax.plot(x, cdf_bad, color=color_bad, lw=2.5, label="Bad (y=0)")

    # Annotate at the reported threshold
    idx = np.argmin(np.abs(x - thresh))
    y1 = cdf_good[idx]
    y2 = cdf_bad[idx]
    ax.annotate(
        f"Max separation\nKS = {ks:.3f}\n@ {thresh:.3f}",
        xy=(thresh, (y1 + y2) / 2),
        xytext=(thresh + 0.12, min(0.95, (y1 + y2) / 2 + 0.18)),
        arrowprops=dict(arrowstyle="->", color=color_sep, lw=1.5),
        color=text_color,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", edgecolor=color_sep),
    )

    ax.axvline(thresh, color=color_sep, linestyle="--", lw=1.2, alpha=0.85)

    ax.set_xlabel("Predicted Probability of Good (y=1)", color=text_color)
    ax.set_ylabel("Cumulative Distribution (CDF)", color=text_color)
    ax.set_title(f"KS Curve — {model_name}", color=text_color, fontsize=13, fontweight="bold")

    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    ax.grid(True, color=grid_color, linestyle="-", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", facecolor="#21262d", edgecolor=grid_color, labelcolor=text_color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"KS curve saved to {save_path}")
