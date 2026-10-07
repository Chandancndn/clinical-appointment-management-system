"""What the "Models and simulation" admin page shows: the saved model card, the risk bands and the policy simulation.

Everything is READ from files the experiments already wrote (the model card, results/*.csv, the saved figures). Nothing is
computed here and nothing in the booking code imports this module. A missing file becomes a note on the page, never an
error.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

# slug used in the URL -> (file name, caption, where it lives). A fixed list: nothing else can be served.
# "assets" is docs/report_assets/figures (built from results/); "results" is results/figures (written by an experiment).
FIGURES = {
    "policy-tradeoffs": ("f11_s2_tradeoffs.png",
                         "What each booking policy trades away: more patients served always costs longer waits and more overtime.", "assets"),
    "value-of-prediction": ("f16_s6_paired_differences.png",
                            "At the same number of double-booked slots: does choosing by predicted risk beat spacing them evenly or at random?", "assets"),
    "learned-policy": ("s7_learned_policy.png",
                       "Where the policies learned by reinforcement learning land among the hand-made rules.", "results"),
    "thresholds": ("f10_e10_thresholds.png",
                   "What each risk threshold catches, with the Medium and High cuts the app uses.", "assets"),
    "reliability": ("f09_e8_reliability.png",
                    "Predicted risk against the observed no-show rate on the held-out data.", "assets"),
}


FAMILIES = {"hist_gradient_boosting": "gradient-boosted tree", "random_forest": "random forest", "logistic_regression": "logistic regression"}


def figure_path(slug: str, figures_dir, results_dir=None) -> Optional[Path]:
    entry = FIGURES.get(slug)
    if entry is None:
        return None
    name, _, source = entry
    if source == "results":
        if results_dir is None:
            return None
        path = Path(results_dir) / "figures" / name
    else:
        path = Path(figures_dir) / name
    return path if path.is_file() else None


def _table(path: Path) -> list:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value, digits=3) -> str:
    return f"{float(value):.{digits}f}"


def _percent(value) -> str:
    return f"{float(value) * 100:.1f}%"


def load(results_dir, artifacts_dir, figures_dir) -> dict:
    """The page's content. Each part is read on its own, so one missing file only removes that part."""
    results, artifacts = Path(results_dir), Path(artifacts_dir)
    page = {"card": None, "metrics": None, "bands": None, "policies": None, "learned": None, "figures": [], "missing": []}

    def attempt(label, build):
        try:
            return build()
        except (OSError, KeyError, ValueError, StopIteration, json.JSONDecodeError):
            page["missing"].append(label)
            return None

    def card():
        data = json.loads((artifacts / "model_card.json").read_text())
        return {"name": data["name"], "version": data["version"], "family": FAMILIES.get(data["model"]["family"], data["model"]["family"].replace("_", " ")),
                "dataset": data["training_data"]["dataset"], "rows_train": f"{data['training_data']['rows_train']:,}",
                "rows_holdout": f"{data['training_data']['rows_holdout']:,}",
                "calibration": data["calibration"]["method"], "intended_use": data["intended_use"],
                "features": [(f["name"], f["description"]) for f in data["features"]], "limitations": list(data["limitations"])}

    def metrics():
        row = next(r for r in _table(results / "e9_deployable.csv") if r["stage"] == "holdout" and r["model"] == "deployable")
        return {"auc": _number(row["auc_roc"]), "auc_lo": _number(row["auc_roc_lo"]), "auc_hi": _number(row["auc_roc_hi"]),
                "ap": _number(row["avg_precision"]), "brier": _number(row["brier"])}

    def bands():
        found = {r["label"]: r for r in _table(results / "e10_thresholds.csv") if r["kind"] == "band"}
        return [{"band": name, "from": _number(found[name]["threshold"], 2),
                 "to": _number(found[name]["upper_threshold"], 2) if found[name]["upper_threshold"] else None,
                 "share": _percent(found[name]["share_flagged"]), "rate": _percent(found[name]["precision"]),
                 "caught": _percent(found[name]["sensitivity"])} for name in ("low", "medium", "high")]

    def policies():
        return [{"policy": r["policy"], "double": _number(r["double_slots_mean"], 2), "served": _number(r["patients_served"], 2),
                 "wait": _number(r["mean_wait_min"], 1), "overtime": _number(r["overtime_min"], 1), "idle": _number(r["idle_min"], 1),
                 "both": _percent(r["share_sessions_both_attend"]), "family": r["family"]} for r in _table(results / "s2_policy_tradeoffs.csv")]

    def learned():
        return [{"policy": r["policy"], "wait_cost": f"{float(r['wait_cost_per_hour']):g}", "overtime_cost": f"{float(r['overtime_cost_per_hour']):g}",
                 "double": _number(r["double_slots_mean"], 2), "served": _number(r["patients_served"], 2),
                 "wait": _number(r["mean_wait_min"], 1), "overtime": _number(r["overtime_min"], 1), "idle": _number(r["idle_min"], 1),
                 "best": r["best_standard_policy"], "gain": f"{float(r['reward_vs_best_standard']):+.2f}",
                 "gain_lo": f"{float(r['reward_vs_best_standard_lo']):+.2f}", "gain_hi": f"{float(r['reward_vs_best_standard_hi']):+.2f}",
                 "n_reps": f"{int(float(r['n_reps'])):,}"}
                for r in _table(results / "s7_learned_policy.csv")]

    page["card"] = attempt("the model card (ml/artifacts/model_card.json)", card)
    page["learned"] = attempt("results/s7_learned_policy.csv", learned)
    page["metrics"] = attempt("results/e9_deployable.csv", metrics)
    page["bands"] = attempt("results/e10_thresholds.csv", bands)
    page["policies"] = attempt("results/s2_policy_tradeoffs.csv", policies)
    page["figures"] = [(slug, caption, figure_path(slug, figures_dir, results_dir) is not None) for slug, (_, caption, _) in FIGURES.items()]
    if not any(available for _, _, available in page["figures"]):
        page["missing"].append("the saved figures (docs/report_assets/figures)")
    return page
