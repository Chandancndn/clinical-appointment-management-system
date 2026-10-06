"""E4: which feature groups add the most?  python -m ml.src.e4_ablation

Uses the model E3 chose by cross-validation, with its tuned hyper-parameters, and re-estimates it with groups of
features removed (drop_<group>) or used alone (only_<group>). Evaluated by PATIENT-GROUPED cross-validation on the
training part only: the holdout is not touched here. Neighbourhood is target-encoded inside each training fold.

Also the SMS check (CLAUDE.md): reminders may have been sent selectively, so no_sms_received compares the model with
and without the column, and e4_sms_crosstab.csv cross-tabulates SMS against lead time. Any effect is an ASSOCIATION.

Writes results/e4_ablation.csv and results/e4_sms_crosstab.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate, experiment, features, train
from .data import DEFAULT_SEED, RESULTS_DIR, ROOT
from .e1_profile import LEAD_BANDS, wilson_interval
from .evaluate import N_BOOT
from .experiment import ResultWriter

SCRIPT = "ml/src/e4_ablation.py"

DESCRIPTIONS = {
    "all_features": "every Kaggle feature (the reference)",
    "no_sms_received": "all features except sms_received",
}


def variants() -> dict[str, list[str]]:
    groups = features.KAGGLE_GROUPS
    everything = features.all_columns("kaggle")
    result = {"all_features": everything, "no_sms_received": [c for c in everything if c != "sms_received"]}
    for g in groups:
        result[f"drop_{g}"] = [c for c in everything if c not in groups[g]]
    for g in groups:
        result[f"only_{g}"] = list(groups[g])
        DESCRIPTIONS[f"drop_{g}"] = f"all features except the {g} group"
        DESCRIPTIONS[f"only_{g}"] = f"the {g} group alone"
    return result


def sms_crosstab(prep) -> pd.DataFrame:
    """No-show rate by lead-time band and SMS received, on the training part (the holdout is not explored)."""
    X, y = prep.X_train, prep.y_train
    band = pd.cut(X["lead_days"], bins=LEAD_BANDS[0], labels=LEAD_BANDS[1]).astype(str).to_numpy()
    sms = X["sms_received"].to_numpy()
    rows = []
    for label in LEAD_BANDS[1]:
        in_band = band == label
        if not in_band.any():
            continue
        rates = {}
        for received in (0, 1):
            mask = in_band & (sms == received)
            if not mask.any():
                continue
            low, high = wilson_interval(int(y[mask].sum()), int(mask.sum()))
            rates[received] = y[mask].mean()
            rows.append({"lead_time_band": label, "sms_received": received, "n": int(mask.sum()),
                         "no_show_n": int(y[mask].sum()), "rate": y[mask].mean(), "ci_low": low, "ci_high": high,
                         "share_of_band_with_sms": float(sms[in_band].mean())})
        difference = rates[1] - rates[0] if len(rates) == 2 else np.nan
        for row in rows:
            if row["lead_time_band"] == label:
                row["rate_difference_sms_minus_none"] = difference
    return pd.DataFrame(rows)


def run(prepared=None, results_dir=RESULTS_DIR, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED, grids=None,
        repo_root=ROOT) -> dict:
    prep = prepared or experiment.prepare_kaggle(seed=seed)
    selected = experiment.load_selected(results_dir)
    model, params = selected["model"], selected["params"]
    rows, reference = [], None
    for name, columns in variants().items():
        oof = train.cross_val_predict_proba(lambda c=columns: train.build_pipeline(model, c, params, seed),
                                            prep.X_train[columns], prep.y_train, prep.folds)
        scores = train.fold_scores(prep.y_train, oof, prep.folds)
        for auc in scores["auc"]:
            evaluate.check_auc_plausible(auc, f"a validation fold of the '{name}' ablation")
        if reference is None:  # all_features is first
            reference = scores
        rows.append({
            "variant": name, "description": DESCRIPTIONS[name], "model": model, "n_features": len(columns),
            "columns": ",".join(columns),
            "cv_auc_mean": scores["auc"].mean(), "cv_auc_std": scores["auc"].std(ddof=1),
            "cv_ap_mean": scores["ap"].mean(), "cv_ap_std": scores["ap"].std(ddof=1),
            "cv_brier_oof": float(np.mean((oof - prep.y_train) ** 2)),
            "delta_auc_vs_all": scores["auc"].mean() - reference["auc"].mean(),
            "delta_auc_fold_std": (scores["auc"] - reference["auc"]).std(ddof=1),
            "delta_ap_vs_all": scores["ap"].mean() - reference["ap"].mean(),
            "delta_ap_fold_std": (scores["ap"] - reference["ap"]).std(ddof=1),
        })
    ablation = pd.DataFrame(rows)
    crosstab = sms_crosstab(prep)

    writer = ResultWriter(results_dir, SCRIPT, seed, prep.source_files, repo_root)
    writer.table(ablation, "e4_ablation.csv")
    writer.table(crosstab, "e4_sms_crosstab.csv")
    return {"ablation": ablation, "sms_crosstab": crosstab}


def main() -> None:
    result = run()
    show = ["variant", "n_features", "cv_auc_mean", "cv_auc_std", "cv_ap_mean", "delta_auc_vs_all", "delta_ap_vs_all"]
    print(result["ablation"][show].round(4).to_string(index=False))
    print("\nSMS x lead time:"); print(result["sms_crosstab"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
