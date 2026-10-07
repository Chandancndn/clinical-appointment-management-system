"""E9: the deployable model.  python -m ml.src.e9_deployable

Trained on Kaggle with ONLY the six features the app can collect at booking time (age, sex, lead time, weekday,
earlier appointments, earlier no-show rate). The matrix is built by calling features.deployable_features() for every
booking, the function the app will import, and is cross-checked against the vectorised history of E3 to E8.

The model family is the one E3 chose by cross-validation; its hyper-parameters are tuned again on the six features by
patient-grouped CV on the training part. Calibration is the method E8 chose; the same three-way comparison (uncalibrated,
Platt, isotonic by out-of-fold Brier) is re-run for this model as a check and recorded, not acted on. The final
model is scored ONCE on the locked holdout with the usual metrics and 95% patient-level bootstrap intervals, and
compared with the all-features research model (paired bootstrap).

Writes results/e9_deployable.csv, ml/artifacts/risk_model.joblib and ml/artifacts/model_card.json (thresholds are
added to the card by E10).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import calibrate, evaluate, experiment, features, train
from .data import DEFAULT_SEED, RESULTS_DIR, ROOT
from .evaluate import METRIC_NAMES, N_BOOT, HoldoutGuard, paired_differences
from .experiment import ARTIFACTS_DIR, ResultWriter, ordered
from .manifest import file_sha256, git_state

SCRIPT = "ml/src/e9_deployable.py"
VERSION = "1.0.0"

FEATURE_DESCRIPTIONS = {
    "age": ("Age in completed years on the appointment date", "years; the app derives it from date_of_birth"),
    "is_female": ("Sex coded 1 for F, 0 for M", "from patient_profiles.sex"),
    "lead_days": ("Whole days from the booking date to the appointment date", "appointment date minus the date the booking was made"),
    "appt_weekday": ("Weekday of the appointment, Monday = 0", "Python date.weekday()"),
    "prev_appointments": ("Number of the patient's earlier appointments whose outcome was known at booking time",
                          "appointments dated strictly before the booking date"),
    "prev_no_show_rate": ("Share of those earlier appointments that were no-shows; missing when there are none",
                          "no-shows divided by prev_appointments; NaN with no history"),
}
INTENDED_USE = ("An ADVISORY flag shown to doctors and admins next to a booking (Low, Medium or High) so staff can decide "
                "whether to send a reminder or prepare a standby. It never decides who gets a slot: the slot rule is "
                "enforced by the database and is never relaxed, no patient is refused, moved or deprioritised because of "
                "the flag, and booking works unchanged if this model is missing or fails.")


def limitations(prep, X_train, X_holdout, result) -> list[str]:
    history_train = float((X_train["prev_appointments"] > 0).mean())
    history_holdout = float((X_holdout["prev_appointments"] > 0).mean())
    return [
        "Uses age and sex as inputs. The flag is advisory only and must never be a reason to refuse, move or "
        "deprioritise a booking.",
        "Trained on one public clinic dataset from Vitória, Brazil (Kaggle, appointments 29 April to 8 June 2016). It has "
        "not been validated on any other clinic. E6 shows that models using these same kinds of features lose a lot of "
        "AUC when moved between two datasets.",
        f"Patient history only exists from 29 April 2016, so it is sparse: {history_train:.1%} of training bookings had any "
        f"earlier appointment, against {history_holdout:.1%} of holdout bookings. The app's history will be a different mix.",
        f"Probabilities are calibrated to this dataset's base rate (no-show rate {float(prep.y_train.mean()):.1%} in the training "
        f"part, {float(prep.y_holdout.mean()):.1%} in the holdout). A clinic with a different base rate needs recalibrating "
        "before the probabilities are read as risks.",
        "The ranking is modest (see metrics.holdout.auc_roc), so even the High band catches only a small share of all "
        "no-shows (see thresholds). Treat it as a nudge for a reminder call, not a prediction about one person.",
        "No-shows in the training data are missed appointments only: the data has no cancellations, so the app's "
        "cancelled bookings are not modelled and are excluded from history.",
        "No SMS, health-flag or neighbourhood information is used, because the app cannot collect it in version 1; the "
        "research models (E3, E4) show these add only a little AUC.",
        "Scored on one time-based holdout (the latest 20% of appointment dates); intervals reflect sampling of "
        "patients, not a change of clinic or season.",
    ]


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def metric_block(result: dict) -> dict:
    block = {m: {"value": result[m], "lo": result[f"{m}_lo"], "hi": result[f"{m}_hi"]} for m in METRIC_NAMES}
    block.update({"n": result["n"], "n_positive": result["n_positive"], "n_boot": result["n_boot"]})
    return block


def run(prepared=None, results_dir=RESULTS_DIR, artifacts_dir=ARTIFACTS_DIR, n_boot: int = N_BOOT,
        seed: int = DEFAULT_SEED, grids=None, repo_root=ROOT) -> dict:
    prep = prepared or experiment.prepare_kaggle(seed=seed)
    selected = experiment.load_selected(results_dir)
    family = selected["model"]
    columns = features.DEPLOYABLE_FEATURES
    X_train, X_holdout = experiment.deployable_data(prep)
    y, groups = prep.y_train, prep.groups_train

    grid = (grids or train.GRIDS).get(family, train.GRIDS[family])
    tuned = train.tune(family, X_train, y, prep.folds, columns, grid=grid, seed=seed)
    params = tuned.best_params
    tuned_row = tuned.results[tuned.results["selected"]].iloc[0]

    def make():
        return train.build_pipeline(family, columns, params, seed)

    briers, check_rows = {}, []
    for method in calibrate.METHODS:  # the E8 comparison, repeated for the six-feature model as a check
        oof = calibrate.oof_predictions(make, X_train, y, groups, prep.folds, method, seed=seed)
        auc = roc_auc_score(y, oof)
        evaluate.check_auc_plausible(auc, f"out-of-fold predictions of the deployable model ({method})")
        briers[method] = brier_score_loss(y, oof)
        check_rows.append({"stage": "grouped_cv_calibration_check", "method": method, "model": family,
                           "brier_cv": briers[method], "ece_cv": calibrate.expected_calibration_error(y, oof),
                           "auc_cv": auc, "ap_cv": average_precision_score(y, oof)})
    method = experiment.load_calibration_choice(results_dir)  # E8's choice, as the plan says
    best_by_check = calibrate.choose_method(briers)

    final = make().fit(X_train, y) if method == "uncalibrated" else calibrate.fit_calibrated(make, X_train, y, groups, method, seed=seed)
    probabilities = final.predict_proba(X_holdout)[:, 1]
    guard = HoldoutGuard(n_boot, seed)
    result = guard.score("deployable", prep.y_holdout, probabilities, prep.groups_holdout)  # the one holdout look

    # the research model (E3's choice, all features) for the comparison; its row is read from E3, not re-scored
    research = pd.read_csv(Path(results_dir) / "e3_kaggle_models.csv").set_index("model").loc[family]
    research_fit = train.fit_final(family, prep.X_train, prep.y_train, selected["columns"], selected["params"], seed)
    research_scores = experiment.holdout_probabilities(research_fit, prep, selected["columns"])
    deltas = paired_differences(prep.y_holdout, probabilities, research_scores, prep.groups_holdout,
                                ["auc_roc", "avg_precision"], n_boot, seed)  # deployable minus research

    holdout_row = {"stage": "holdout", "model": "deployable", "features": ",".join(columns), "params": json.dumps(params, sort_keys=True),
                   "calibration_method": method, "cv_ap_mean": tuned_row["cv_ap_mean"], "cv_auc_mean": tuned_row["cv_auc_mean"], **result}
    for metric, short in (("auc_roc", "auc_roc"), ("avg_precision", "avg_precision")):
        diff, low, high = deltas[metric]
        holdout_row.update({f"delta_{short}_vs_research": diff, f"delta_{short}_vs_research_lo": low,
                            f"delta_{short}_vs_research_hi": high})
    reference_row = {"stage": "holdout", "model": "research_all_features", "features": ",".join(selected["columns"]),
                     "params": json.dumps(selected["params"], sort_keys=True), "calibration_method": "uncalibrated",
                     "note": "read from e3_kaggle_models.csv (scored once there); not re-scored here",
                     **{c: research[c] for c in ("n", "n_positive", "n_boot", *[x for m in METRIC_NAMES for x in (m, f"{m}_lo", f"{m}_hi")])}}
    table = ordered(pd.concat([pd.DataFrame([holdout_row, reference_row]), pd.DataFrame(check_rows)], ignore_index=True),
                    ["stage", "model", "features", "params", "calibration_method"])

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path, card_path = artifacts_dir / "risk_model.joblib", artifacts_dir / "model_card.json"
    joblib.dump(final, model_path)

    source = Path(prep.source_files[0]) if prep.source_files else None
    commit, dirty = git_state(repo_root)
    train_dates, holdout_dates = prep.train_frame["appointment_date"], prep.holdout_frame["appointment_date"]
    card = {
        "name": "CAMS no-show risk model", "version": VERSION, "created": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        "seed": seed, "target": "no_show = 1 when the patient did not attend the appointment (Kaggle 'No-show' = Yes)",
        "features": [{"name": n, "description": FEATURE_DESCRIPTIONS[n][0], "how_computed": FEATURE_DESCRIPTIONS[n][1]} for n in columns],
        "feature_function": "ml/src/features.py: deployable_features() and age_in_years(); training calls the same function",
        "training_data": {
            "dataset": "Kaggle 'Medical Appointment No Shows' (joniarroba/noshowappointments), Vitória, Brazil",
            "file": source.name if source else None, "sha256": file_sha256(source) if source and source.exists() else None,
            "rows_cleaned": int(len(y) + len(prep.y_holdout)), "rows_train": int(len(y)), "rows_holdout": int(len(prep.y_holdout)),
            "training_appointment_dates": {"first": str(train_dates.min().date()), "last": str(train_dates.max().date())},
            "holdout_appointment_dates": {"first": str(holdout_dates.min().date()), "last": str(holdout_dates.max().date())},
            "no_show_rate_train": float(y.mean()), "no_show_rate_holdout": float(prep.y_holdout.mean()),
            "cleaning": "results/cleaning_log.csv",
        },
        "model": {"family": family, "params": params, "tuned_by": "patient-grouped 5-fold cross-validation on the training "
                  "part, by average precision; family chosen in E3", "hyperparameter_search": grid,
                  "cv_average_precision": float(tuned_row["cv_ap_mean"]), "cv_auc_roc": float(tuned_row["cv_auc_mean"])},
        "calibration": {"method": method, "chosen_in": "E8", "cv_brier_check": briers, "best_in_this_check": best_by_check,
                        "note": "E8 found no calibration needed for the all-features model, because the out-of-fold Brier "
                                "scores were the same to four decimals; the same comparison on this model is recorded in "
                                "cv_brier_check" if method == "uncalibrated" else
                                f"{method} calibration, fitted with an inner patient-grouped cross-fit"},
        "metrics": {"holdout": metric_block(result),
                    "comparison_with_research_model": {
                        "research_model": f"{family} on all {len(selected['columns'])} Kaggle features (E3)",
                        "research_auc_roc": {"value": float(research["auc_roc"]), "lo": float(research["auc_roc_lo"]), "hi": float(research["auc_roc_hi"])},
                        "delta_auc_roc": {"value": deltas["auc_roc"][0], "lo": deltas["auc_roc"][1], "hi": deltas["auc_roc"][2]},
                        "delta_avg_precision": {"value": deltas["avg_precision"][0], "lo": deltas["avg_precision"][1], "hi": deltas["avg_precision"][2]},
                        "note": "paired bootstrap, deployable minus research, same resampled patients"}},
        "thresholds": None,
        "intended_use": INTENDED_USE,
        "limitations": limitations(prep, X_train, X_holdout, result),
        "environment": experiment.environment_info(),
        "code": {"script": SCRIPT, "git_commit": commit, "git_dirty": dirty},
        "artifact_sha256": file_sha256(model_path),
    }
    card_path.write_text(json.dumps(json_safe(card), indent=2) + "\n")

    writer = ResultWriter(results_dir, SCRIPT, seed, prep.source_files, repo_root)
    writer.table(table, "e9_deployable.csv")
    writer.record(model_path)
    writer.record(card_path)
    return {"table": table, "card": card, "scored": guard.scored, "model_path": model_path, "card_path": card_path}


def main() -> None:
    result = run()
    t = result["table"]
    holdout = t[t["stage"] == "holdout"]
    for _, r in holdout.iterrows():
        print(f"{r['model']:24s} AUC {r['auc_roc']:.3f} [{r['auc_roc_lo']:.3f}, {r['auc_roc_hi']:.3f}]  AP {r['avg_precision']:.3f}  Brier {r['brier']:.4f}")
    d = holdout.iloc[0]
    print(f"deployable minus research: AUC {d['delta_auc_roc_vs_research']:+.4f} [{d['delta_auc_roc_vs_research_lo']:+.4f}, {d['delta_auc_roc_vs_research_hi']:+.4f}]")
    print("calibration method (from E8):", result["card"]["calibration"]["method"], "| check:", {k: round(v, 5) for k, v in result["card"]["calibration"]["cv_brier_check"].items()})


if __name__ == "__main__":
    main()
