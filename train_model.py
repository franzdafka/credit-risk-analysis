"""Train the PD model, save the artifact and (optionally) regenerate README figures.

    python train_model.py            # train + save artifact + figures and reports/metrics.json
    python train_model.py --no-report  # train + save artifact only (used in CI)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import credit_model as cm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    result = cm.train(verbose=True)
    version = datetime.now(timezone.utc).strftime("credit-risk-%Y%m%d%H%M%S")
    path = cm.save_artifact(result, version)
    m = result.test_metrics
    print(
        f"Saved {path} | model={result.selected_model} calibration={result.calibration} "
        f"threshold={result.threshold:.2f} | test AUC={m['auc_roc']:.3f} Brier={m['brier']:.3f} "
        f"approval={m['approval_rate']:.1%}"
    )
    if not args.no_report:
        from report import write_report

        write_report(result)
        print("Wrote docs/*.png and reports/metrics.json")


if __name__ == "__main__":
    main()
