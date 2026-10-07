"""E10: what does each risk threshold give?  python -m ml.src.e10_thresholds

For the deployable model saved by E9, a table of fixed thresholds (0.20 to 0.70) and of the 50th, 70th, 80th, 90th and
95th percentiles of predicted risk, each with precision, sensitivity and the share of bookings flagged, on the locked
holdout (95% patient-level bootstrap intervals) and, for comparison, on the training part's out-of-fold predictions.

The Medium and High thresholds for the app are CHOSEN HERE, and not from the holdout: they are the lowest cut-offs at
which the out-of-fold precision on the training part stays at or above 30% (Medium) and 40% (High), the targets in
PLAN section 6. The holdout then reports what they deliver, so it stays an honest test (a test flips every holdout label
and checks the thresholds do not move). Percentile thresholds are percentiles of the same out-of-fold predictions.

Writes results/e10_thresholds.csv and fills the `thresholds` block of ml/artifacts/model_card.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import calibrate, evaluate, experiment, features, train
from .data import DEFAULT_SEED, RESULTS_DIR, ROOT
from .e9_deployable import json_safe
from .evaluate import N_BOOT
from .experiment import ARTIFACTS_DIR, ResultWriter

SCRIPT = "ml/src/e10_thresholds.py"
FIXED = [round(t, 2) for t in np.arange(0.20, 0.7001, 0.05)]
PERCENTILES = [50, 70, 80, 90, 95]
TARGETS = {"medium": 0.30, "high": 0.40}  # PLAN section 6: the precision each band should reach
MIN_SHARE = 0.01  # a cut-off must flag at least 1% of bookings, so a lucky handful of top scores cannot set it


def threshold_for_precision(y, p, target: float, min_share: float = MIN_SHARE, step: float = 0.01):
    """The lowest threshold (on a `step` grid) at which precision stays >= `target` for every higher threshold that
    still flags at least `min_share` of the bookings. None if even the highest such threshold misses the target."""
    y, p = np.asarray(y), np.asarray(p)
    grid = np.round(np.arange(step, 1.0, step), 2)
    flagged = np.array([(p >= t).sum() for t in grid])
    true_positives = np.array([y[p >= t].sum() for t in grid])
    precision = np.where(flagged > 0, true_positives / np.maximum(flagged, 1), np.nan)
    eligible = flagged / len(y) >= min_share
    chosen = None
    for i in range(len(grid) - 1, -1, -1):  # from the highest threshold down
        if not eligible[i]:
            continue
        if precision[i] >= target:
            chosen = float(grid[i])
        else:
            break
    return chosen


def membership(p, lows, highs) -> np.ndarray:
    """(rows, ranges) boolean: is row i flagged by range j, meaning low_j <= p_i < high_j. A threshold is (t, inf)."""
    p = np.asarray(p)
    return (p[:, None] >= np.asarray(lows)[None, :]) & (p[:, None] < np.asarray(highs)[None, :])


def counts_at(y, p, lows, highs) -> dict:
    """flagged count, share of bookings, precision (no-show rate among the flagged) and sensitivity (share of all
    no-shows that are flagged) for each range."""
    y = np.asarray(y, dtype=float)
    flagged_matrix = membership(p, lows, highs)
    flagged = flagged_matrix.sum(axis=0)
    true_positives = (y[:, None] * flagged_matrix).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"flagged_n": flagged, "share_flagged": flagged / len(y),
                "precision": np.where(flagged > 0, true_positives / np.maximum(flagged, 1), np.nan),
                "sensitivity": true_positives / y.sum() if y.sum() else np.full(len(flagged), np.nan)}


def holdout_intervals(y, p, lows, highs, groups, n_boot: int, seed: int) -> dict:
    """95% patient-level bootstrap intervals for share flagged, precision and sensitivity of each range."""
    y = np.asarray(y, dtype=float)
    flagged_matrix = membership(p, lows, highs).astype(np.float64)
    draws = {"share_flagged": [], "precision": [], "sensitivity": []}
    for weights in evaluate.weight_stream(groups, n_boot, seed, len(y)):
        w = weights.astype(np.float64)
        flagged, true_positives = w @ flagged_matrix, (w * y) @ flagged_matrix
        with np.errstate(invalid="ignore", divide="ignore"):
            draws["share_flagged"].append(flagged / w.sum())
            draws["precision"].append(np.where(flagged > 0, true_positives / np.maximum(flagged, 1e-12), np.nan))
            draws["sensitivity"].append(true_positives / (w * y).sum())
    return {k: np.nanpercentile(np.vstack(v), [2.5, 97.5], axis=0) for k, v in draws.items()}


def run(prepared=None, results_dir=RESULTS_DIR, artifacts_dir=ARTIFACTS_DIR, n_boot: int = N_BOOT,
        seed: int = DEFAULT_SEED, grids=None, repo_root=ROOT) -> dict:
    prep = prepared or experiment.prepare_kaggle(seed=seed)
    artifacts_dir = Path(artifacts_dir)
    card_path = artifacts_dir / "model_card.json"
    card = json.loads(card_path.read_text())
    model = joblib.load(artifacts_dir / "risk_model.joblib")
    columns = features.DEPLOYABLE_FEATURES
    family, params, method = card["model"]["family"], card["model"]["params"], card["calibration"]["method"]
    X_train, X_holdout = experiment.deployable_data(prep)

    # 1. thresholds come from out-of-fold predictions on the TRAINING part only
    def make():
        return train.build_pipeline(family, columns, params, seed)

    oof = calibrate.oof_predictions(make, X_train, prep.y_train, prep.groups_train, prep.folds, method, seed=seed)
    evaluate.check_auc_plausible(roc_auc_score(prep.y_train, oof), "out-of-fold predictions of the deployable model")
    percentile_values = [float(np.percentile(oof, q)) for q in PERCENTILES]
    chosen = {}
    for label, target in TARGETS.items():
        threshold = threshold_for_precision(prep.y_train, oof, target)
        if threshold is None:
            raise RuntimeError(f"no threshold reaches {target:.0%} out-of-fold precision on the training part")
        chosen[label] = threshold
    if not chosen["medium"] < chosen["high"]:
        raise RuntimeError(f"Medium ({chosen['medium']}) must be below High ({chosen['high']})")

    # 2. the holdout reports what those thresholds deliver
    probabilities = model.predict_proba(X_holdout)[:, 1]
    evaluate.check_auc_plausible(roc_auc_score(prep.y_holdout, probabilities), "holdout predictions of the deployable model")
    medium, high = chosen["medium"], chosen["high"]
    rows = ([("fixed", f"{t:.2f}", np.nan, np.nan, t, np.inf) for t in FIXED]
            + [("percentile", f"p{q}", q, np.nan, v, np.inf) for q, v in zip(PERCENTILES, percentile_values)]
            + [("chosen", label, np.nan, TARGETS[label], chosen[label], np.inf) for label in ("medium", "high")]
            + [("band", "low", np.nan, np.nan, 0.0, medium), ("band", "medium", np.nan, np.nan, medium, high),
               ("band", "high", np.nan, np.nan, high, np.inf)])
    lows, highs = [r[4] for r in rows], [r[5] for r in rows]
    holdout = counts_at(prep.y_holdout, probabilities, lows, highs)
    training = counts_at(prep.y_train, oof, lows, highs)
    intervals = holdout_intervals(prep.y_holdout, probabilities, lows, highs, prep.groups_holdout, n_boot, seed)

    table = pd.DataFrame({
        "kind": [r[0] for r in rows], "label": [r[1] for r in rows], "percentile": [r[2] for r in rows],
        "target_precision": [r[3] for r in rows], "threshold": lows,
        "upper_threshold": [np.nan if np.isinf(h) else h for h in highs],
        "flagged_n": holdout["flagged_n"], "share_flagged": holdout["share_flagged"],
        "share_flagged_lo": intervals["share_flagged"][0], "share_flagged_hi": intervals["share_flagged"][1],
        "precision": holdout["precision"], "precision_lo": intervals["precision"][0], "precision_hi": intervals["precision"][1],
        "sensitivity": holdout["sensitivity"], "sensitivity_lo": intervals["sensitivity"][0], "sensitivity_hi": intervals["sensitivity"][1],
        "oof_flagged_n": training["flagged_n"], "oof_share_flagged": training["share_flagged"],
        "oof_precision": training["precision"], "oof_sensitivity": training["sensitivity"],
        "n": len(prep.y_holdout), "n_positive": int(prep.y_holdout.sum()), "n_boot": n_boot,
    })

    # 3. record Medium and High, with the reasoning, in the model card
    base_rate = float(prep.y_train.mean())
    auc = float(card["metrics"]["holdout"]["auc_roc"]["value"])
    levels = {}
    for label in ("medium", "high"):
        i = table.index[(table["kind"] == "chosen") & (table["label"] == label)][0]
        levels[label] = {
            "threshold": chosen[label], "target_precision": TARGETS[label],
            "oof_precision": float(table.loc[i, "oof_precision"]), "oof_share_flagged": float(table.loc[i, "oof_share_flagged"]),
            "holdout_precision": float(table.loc[i, "precision"]), "holdout_sensitivity": float(table.loc[i, "sensitivity"]),
            "holdout_share_flagged": float(table.loc[i, "share_flagged"]),
            "holdout_precision_95ci": [float(table.loc[i, "precision_lo"]), float(table.loc[i, "precision_hi"])],
            "holdout_sensitivity_95ci": [float(table.loc[i, "sensitivity_lo"]), float(table.loc[i, "sensitivity_hi"])],
        }
    bands = {}
    for label in ("low", "medium", "high"):
        i = table.index[(table["kind"] == "band") & (table["label"] == label)][0]
        bands[label] = {
            "lower": float(table.loc[i, "threshold"]),
            "upper": None if np.isnan(table.loc[i, "upper_threshold"]) else float(table.loc[i, "upper_threshold"]),
            "holdout_share_of_bookings": float(table.loc[i, "share_flagged"]),
            "holdout_no_show_rate": float(table.loc[i, "precision"]),
            "holdout_no_show_rate_95ci": [float(table.loc[i, "precision_lo"]), float(table.loc[i, "precision_hi"])],
            "holdout_share_of_all_no_shows": float(table.loc[i, "sensitivity"]),
            "training_oof_share_of_bookings": float(table.loc[i, "oof_share_flagged"]),
            "training_oof_no_show_rate": float(table.loc[i, "oof_precision"]),
        }
    lo_b, md_b, hi_b = bands["low"], bands["medium"], bands["high"]
    m, h = levels["medium"], levels["high"]
    reasoning = (
        f"The targets are the plan's: High is the lowest cut-off at which at least 40% of flagged bookings were no-shows "
        f"(about twice the {base_rate:.1%} training base rate) and Medium the same for 30%, both measured on out-of-fold "
        f"predictions of the training part, never on the holdout, so the holdout stays an honest test. A cut-off must flag "
        f"at least {MIN_SHARE:.0%} of bookings so a lucky handful of top scores cannot set it, and precision must stay at "
        f"or above the target at every higher cut-off. On the holdout, High flags {h['holdout_share_flagged']:.1%} of bookings, "
        f"{h['holdout_precision']:.1%} of them no-shows, catching {h['holdout_sensitivity']:.1%} of all no-shows; Medium flags "
        f"{m['holdout_share_flagged']:.1%}, {m['holdout_precision']:.1%} precise, catching {m['holdout_sensitivity']:.1%}. "
        f"Holdout precision sits below the training figure because the holdout's base rate is lower and it has more "
        f"patient history (see limitations). Judged as three bands, which is how staff see them, the thresholds split the "
        f"holdout into Low ({lo_b['holdout_share_of_bookings']:.1%} of bookings, {lo_b['holdout_no_show_rate']:.1%} no-show), "
        f"Medium ({md_b['holdout_share_of_bookings']:.1%}, {md_b['holdout_no_show_rate']:.1%}) and High "
        f"({hi_b['holdout_share_of_bookings']:.1%}, {hi_b['holdout_no_show_rate']:.1%}): Low is genuinely safe and holds "
        f"only {lo_b['holdout_share_of_all_no_shows']:.1%} of all no-shows, High is a short list for a reminder call, and "
        f"Medium is the broad middle, so a Medium badge alone says little. The model's ranking is modest (holdout AUC "
        f"{auc:.3f}). The flag is advisory: staff see Low, Medium or High next to a booking, the slot rule never changes, "
        f"and no patient is refused or moved because of it."
    )
    card["thresholds"] = {
        "bands": {"low": "predicted risk below the Medium threshold", "medium": "risk at or above Medium and below High",
                  "high": "risk at or above the High threshold"},
        "chosen_on": "out-of-fold predictions of the training part (patient-grouped CV); the holdout only reports the result",
        "min_flagged_share": MIN_SHARE, "medium": m, "high": h, "bands": bands, "reasoning": reasoning, "table": "results/e10_thresholds.csv",
        "percentile_thresholds_on_training_oof": dict(zip([f"p{q}" for q in PERCENTILES], percentile_values)),
    }
    card_path.write_text(json.dumps(json_safe(card), indent=2) + "\n")

    writer = ResultWriter(results_dir, SCRIPT, seed, prep.source_files, repo_root)
    writer.table(table, "e10_thresholds.csv")
    writer.record(card_path)
    return {"table": table, "chosen": chosen, "levels": levels, "card": card}


def main() -> None:
    result = run()
    show = ["kind", "label", "threshold", "upper_threshold", "share_flagged", "precision", "sensitivity", "oof_share_flagged", "oof_precision", "oof_sensitivity"]
    print(result["table"][show].round(4).to_string(index=False))
    print("\nchosen:", result["chosen"])
    print(result["card"]["thresholds"]["reasoning"])


if __name__ == "__main__":
    main()
