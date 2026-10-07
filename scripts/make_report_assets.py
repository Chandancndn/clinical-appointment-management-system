"""Build every table and figure the report needs, the results summary and the numbers audit, from results/ only.

    python -m scripts.make_report_assets            # writes docs/report_assets/, docs/RESULTS_SUMMARY.md, docs/numbers_audit.csv

Nothing here trains a model or runs a simulation: it reads the CSV files and PNG figures in results/ (the test suite
checks that no model, simulation or app code is imported). Every number in a table or in the summary goes through
scripts/reportlib.Registry, so it lands in docs/numbers_audit.csv, and scripts/check_numbers.py can re-check it.
Numbers written into figure images come from the same files at build time.

The ROC, precision-recall and QQ figures were drawn by the experiments from data that was not saved in results/, so
their images are reused as they are, with only the footnote strip (which was cut off at the right edge) redrawn.
"""
from __future__ import annotations

import argparse
import re
import shutil
import textwrap
from pathlib import Path

import jinja2

from .reportlib import ROOT, Registry

ASSETS = "report_assets"
FORMATS = {"int": "{:,.0f}", "pct": "{:.1%}", "pct0": "{:.0%}", "pts": "{pp}", "str": "{}", "s3": "{:+.3f}", "s4": "{:+.4f}", "aic": "{:,.1f}"}

GROUP = {"no_sms_received": "SMS reminder", "drop_history": "Patient history", "drop_lead_calendar": "Lead time and weekday", "drop_demographics": "Age and sex",
         "drop_flags": "Health flags", "drop_neighbourhood": "Neighbourhood", "only_history": "Patient history", "only_lead_calendar": "Lead time and weekday",
         "only_demographics": "Age and sex", "only_flags": "Health flags", "only_neighbourhood": "Neighbourhood"}
MODEL = {
    "always_show": "Always predict show", "earlier_no_show_rate_rule": "Earlier no-show rate rule",
    "logistic_regression": "Logistic regression", "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting", "control_shuffled_labels": "Control: shuffled labels",
    "deployable": "Deployable model (six features)", "research_all_features": "Research model (all features)",
}
DATASET = {"kaggle": "Kaggle (Vitória, Brazil)", "openml": "OpenML Medical-Appointment", "hangu": "Hangu (consultation times)"}
SELECTOR = {
    "p1_uniform_k": "Evenly spaced (every k-th slot)", "uniform": "Evenly spaced", "risk_top_m": "Highest-risk slots",
    "p2_risk_threshold": "Risk threshold (P2)", "random": "Random slots", "oracle": "Oracle (knows who will not come)",
}
VARIANT = {
    "all_features": "All features (reference)", "no_sms_received": "Without SMS reminder", "drop_history": "Without patient history",
    "drop_lead_calendar": "Without lead time and weekday", "drop_demographics": "Without age and sex",
    "drop_flags": "Without health flags", "drop_neighbourhood": "Without neighbourhood", "only_history": "Patient history only",
    "only_lead_calendar": "Lead time and weekday only", "only_demographics": "Age and sex only", "only_flags": "Health flags only",
    "only_neighbourhood": "Neighbourhood only",
}


# ---- tables ---------------------------------------------------------------------------------------------------
class Tables:
    """Collects tables; each is written as CSV and as Markdown with a numbered caption, a source line and a note."""

    def __init__(self, reg: Registry, out: Path):
        self.reg, self.out, self.index = reg, out / ASSETS / "tables", []
        self.out.mkdir(parents=True, exist_ok=True)

    def used_in(self, name: str) -> str:
        return f"docs/{ASSETS}/tables/{name}.md"

    def quoters(self, name: str):
        used = self.used_in(name)

        def c(file, where, col, fmt="{:.3f}"):
            if self.reg.raw(f"results/{file}.csv", where, col) == "" and fmt != "{}":
                return "n/a"
            return self.reg.cell(f"results/{file}.csv", where, col, fmt, used)

        def ci(file, where, col, fmt="{:.3f}", lo=None, hi=None):
            if self.reg.raw(f"results/{file}.csv", where, col) == "":
                return "n/a"
            return self.reg.ci(f"results/{file}.csv", where, col, fmt, used, lo, hi)

        return c, ci

    def write(self, name, experiment, title, header, rows, sources, note=""):
        import csv

        number = int(name[1:3])
        with open(self.out / f"{name}.csv", "w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        esc = lambda text: str(text).replace("|", "/")
        lines = [f"**Table {number} ({experiment}). {title}**", "",
                 "| " + " | ".join(header) + " |", "|" + "|".join("---:" if rows and all(re.match(r"^[+\-]?\d|^n/a", str(r[i])) for r in rows) else "---" for i in range(len(header))) + "|"]
        lines += ["| " + " | ".join(esc(cell) for cell in row) + " |" for row in rows]
        lines += ["", "Source: " + ", ".join(f"`{s}`" for s in sources) + "." + (f" {note}" if note else ""), ""]
        (self.out / f"{name}.md").write_text("\n".join(lines))
        self.index.append((name, experiment, title, sources))

    def keys(self, file, *columns):
        return [{c: r[c] for c in columns} for r in self.reg.table(f"results/{file}.csv")]


def build_tables(reg: Registry, out: Path) -> list:
    T = Tables(reg, out)
    yes_no = lambda raw: {"True": "Yes", "False": "No"}.get(raw, raw)

    # T01 dataset summary
    c, ci = T.quoters("t01_dataset_summary")
    plan = [
        ("kaggle", "rows_raw", "Appointments in the raw file", "int"), ("kaggle", "rows_clean", "Appointments after cleaning", "int"),
        ("kaggle", "rows_removed", "Rows removed by cleaning", "int"), ("kaggle", "no_show_rate_clean", "No-show rate", "pct"),
        ("kaggle", "unique_patients_clean", "Distinct patients", "int"), ("kaggle", "appointment_date_min", "First appointment date", "str"),
        ("kaggle", "appointment_date_max", "Last appointment date", "str"), ("kaggle", "share_same_day_booking", "Booked on the day of the appointment", "pct"),
        ("kaggle", "lead_days_median", "Median lead time (days)", "{:.0f}"), ("kaggle", "share_sms_received", "Received an SMS reminder", "pct"),
        ("kaggle", "n_train_rows", "Training part (appointments)", "int"), ("kaggle", "n_holdout_rows", "Locked holdout (appointments)", "int"),
        ("kaggle", "train_last_date", "Last training date", "str"), ("kaggle", "holdout_first_date", "First holdout date", "str"),
        ("kaggle", "cv_patients_in_both_sides", "Patients on both sides of any CV fold", "{:.0f}"),
        ("openml", "rows_raw", "Appointments in the raw file", "int"), ("openml", "rows_clean", "Appointments after cleaning", "int"),
        ("openml", "rows_removed", "Rows removed by cleaning", "int"), ("openml", "no_show_rate_clean", "No-show rate", "pct"),
        ("openml", "n_specialties", "Medical specialties", "{:.0f}"), ("openml", "n_train_rows", "Training part (appointments)", "int"),
        ("openml", "n_holdout_rows", "Locked holdout, the last month (appointments)", "int"),
        ("hangu", "rows_raw", "Consultations", "int"), ("hangu", "sessions", "Half-day sessions", "int"),
        ("hangu", "service_minutes_median", "Median consultation time (minutes)", "{:.1f}"),
        ("hangu", "service_minutes_mean", "Mean consultation time (minutes)", "{:.1f}"),
        ("hangu", "service_minutes_p95", "95th percentile consultation time (minutes)", "{:.1f}"),
    ]
    rows = [[DATASET[d], label, c("dataset_summary", {"dataset": d, "metric": m}, "value", FORMATS.get(f, f))] for d, m, label, f in plan]
    T.write("t01_dataset_summary", "E1", "The three datasets after cleaning, and how they were split", ["Dataset", "Measure", "Value"], rows,
            ["results/dataset_summary.csv"])

    # T02 cleaning log
    c, ci = T.quoters("t02_cleaning_log")
    rows = [[DATASET[k["dataset"]], c("cleaning_log", k, "step", "{:.0f}"), c("cleaning_log", k, "rule", "{}"),
             c("cleaning_log", k, "rows_before", "{:,.0f}"), c("cleaning_log", k, "rows_after", "{:,.0f}"),
             c("cleaning_log", k, "rows_changed", "{:,.0f}"), c("cleaning_log", k, "note", "{}")]
            for k in T.keys("cleaning_log", "dataset", "step")]
    T.write("t02_cleaning_log", "E1", "Cleaning log: how many rows each rule changed",
            ["Dataset", "Step", "Rule", "Rows before", "Rows after", "Rows changed", "What the rule does"], rows, ["results/cleaning_log.csv"])

    # T03 baselines
    c, ci = T.quoters("t03_baselines")
    rows = [[DATASET[k["dataset"]], MODEL[k["model"]], ci("e2_baselines", k, "auc_roc"), ci("e2_baselines", k, "avg_precision"),
             ci("e2_baselines", k, "brier"), c("e2_baselines", k, "accuracy")] for k in T.keys("e2_baselines", "dataset", "model")]
    T.write("t03_baselines", "E2", "Baselines on the locked holdouts (95% bootstrap intervals)",
            ["Dataset", "Model", "AUC-ROC", "Average precision", "Brier score", "Accuracy"], rows, ["results/e2_baselines.csv"],
            "Accuracy is shown only to make a point: always predicting show scores high on it and is useless.")

    # T04 / T08 model comparisons
    def model_table(name, experiment, title, file, keys, extra_note=""):
        c, ci = T.quoters(name)
        rows = [[MODEL[k["model"]], ci(file, k, "auc_roc"), ci(file, k, "avg_precision"), ci(file, k, "brier"),
                 ci(file, k, "sens_at_prec_30"), ci(file, k, "sens_at_prec_40"), ci(file, k, "sens_at_prec_50")] for k in keys]
        T.write(name, experiment, title, ["Model", "AUC-ROC", "Average precision", "Brier score", "Sensitivity at 30% precision",
                                          "Sensitivity at 40% precision", "Sensitivity at 50% precision"], rows, [f"results/{file}.csv"], extra_note)

    model_table("t04_kaggle_models", "E3", "Model comparison on the Kaggle holdout, all features (95% bootstrap intervals)", "e3_kaggle_models",
                T.keys("e3_kaggle_models", "model"),
                "The shuffled-label control should score no better than chance; the logistic regression uses class weights, so its Brier score is not comparable.")
    model_table("t08_openml_models", "E5", "The same model ladder on the OpenML holdout, the last month (95% bootstrap intervals)", "e5_openml_models",
                T.keys("e5_openml_models", "model"))

    # T05 ladder
    c, ci = T.quoters("t05_model_ladder")
    pretty = lambda step: " → ".join(MODEL[s.strip()] for s in step.split("->"))
    rows = []
    for k in T.keys("e3_ladder", "step"):
        rows.append([pretty(k["step"]), ci("e3_ladder", k, "delta_auc_roc", "{:+.3f}"), ci("e3_ladder", k, "delta_avg_precision", "{:+.3f}"),
                     ci("e3_ladder", k, "delta_sens_at_prec_40", "{:+.3f}"), yes_no(reg.raw("results/e3_ladder.csv", k, "beats_previous_step"))])
    T.write("t05_model_ladder", "E3", "Does each step up the ladder beat the one before? Paired differences on the holdout",
            ["Step", "Change in AUC-ROC", "Change in average precision", "Change in sensitivity at 40% precision", "Clearly better?"], rows,
            ["results/e3_ladder.csv"], "Clearly better means the intervals for both AUC-ROC and average precision lie above zero.")

    # T06 ablation
    c, ci = T.quoters("t06_ablation")
    rows = []
    for k in T.keys("e4_ablation", "variant"):
        delta = c("e4_ablation", k, "delta_auc_vs_all", "{:+.4f}")
        rows.append([VARIANT[k["variant"]], c("e4_ablation", k, "n_features", "{:.0f}"), c("e4_ablation", k, "cv_auc_mean", "{:.4f}"),
                     f"{delta} (SD {c('e4_ablation', k, 'delta_auc_fold_std', '{:.4f}')})", c("e4_ablation", k, "delta_ap_vs_all", "{:+.4f}")])
    T.write("t06_ablation", "E4", "Feature-group ablation: gradient boosting, patient-grouped cross-validation on the training part",
            ["Variant", "Features", "Cross-validated AUC-ROC", "Change in AUC-ROC vs all features (SD across folds)", "Change in average precision"],
            rows, ["results/e4_ablation.csv"], "The holdout is not used here.")

    # T07 SMS by lead time
    c, ci = T.quoters("t07_sms_by_lead_time")
    rows = []
    for k in T.keys("e4_sms_crosstab", "lead_time_band", "sms_received"):
        rows.append([k["lead_time_band"], "Yes" if k["sms_received"] == "1" else "No", c("e4_sms_crosstab", k, "n", "{:,.0f}"),
                     ci("e4_sms_crosstab", k, "rate", "{:.1%}", lo="ci_low", hi="ci_high"),
                     c("e4_sms_crosstab", k, "share_of_band_with_sms", "{:.1%}"),
                     c("e4_sms_crosstab", k, "rate_difference_sms_minus_none", "{pp}")])
    T.write("t07_sms_by_lead_time", "E4", "SMS reminder against no-show rate within each lead-time band (Kaggle training part)",
            ["Lead time (days)", "SMS received", "Appointments", "No-show rate (95% Wilson interval)", "Share of band with SMS",
             "Difference in no-show rate, SMS minus none (percentage points)"], rows, ["results/e4_sms_crosstab.csv"],
            "An association only: reminders were not sent at random, so this does not show that the SMS causes the difference.")

    # T09 transfer
    c, ci = T.quoters("t09_transfer")
    arrow = {"kaggle_to_openml": "Kaggle → OpenML", "openml_to_kaggle": "OpenML → Kaggle"}
    rows = [[arrow[k["direction"]], MODEL[k["model"]], ci("e6_transfer", k, "auc_within_target"), ci("e6_transfer", k, "auc_transfer"),
             ci("e6_transfer", k, "auc_drop", "{:+.3f}"), ci("e6_transfer", k, "ap_drop", "{:+.3f}")]
            for k in T.keys("e6_transfer", "direction", "model")]
    T.write("t09_transfer", "E6", "Transfer on the four common features (age, sex, lead time, weekday), both directions",
            ["Trained on → tested on", "Model", "AUC-ROC trained on the target's own training part", "AUC-ROC after transfer",
             "AUC-ROC drop", "Average-precision drop"], rows, ["results/e6_transfer.csv"], "A positive drop means the transferred model is worse.")

    # T10 imbalance
    c, ci = T.quoters("t10_imbalance")
    strategy = {"none": "None", "class_weights": "Class weights", "threshold_tuning": "Threshold tuning only", "random_under_sampling": "Random under-sampling",
                "smote": "SMOTE", "nearmiss": "NearMiss"}
    rows = [[MODEL[k["base_model"]], strategy[k["strategy"]], ci("e7_imbalance", k, "auc_roc"), ci("e7_imbalance", k, "avg_precision"),
             ci("e7_imbalance", k, "brier"), c("e7_imbalance", k, "mean_predicted")] for k in T.keys("e7_imbalance", "base_model", "strategy")]
    prevalence = c("e7_imbalance", {"base_model": "logistic_regression", "strategy": "none"}, "prevalence")
    T.write("t10_imbalance", "E7", "Imbalance handling on the Kaggle holdout",
            ["Base model", "Strategy", "AUC-ROC", "Average precision", "Brier score", "Mean predicted risk"], rows, ["results/e7_imbalance.csv"],
            f"The true no-show rate on the holdout is {prevalence}; a mean predicted risk far from it means the probabilities are distorted.")

    # T11 calibration
    c, ci = T.quoters("t11_calibration")
    method = {"uncalibrated": "None (uncalibrated)", "sigmoid": "Platt scaling (sigmoid)", "isotonic": "Isotonic regression"}
    rows = [["Training part, grouped CV", method[k["method"]], c("e8_calibration", k, "brier_cv", "{:.5f}"), c("e8_calibration", k, "ece_cv", "{:.4f}"),
             c("e8_calibration", k, "auc_cv", "{:.4f}")] for k in T.keys("e8_calibration", "stage", "method") if k["stage"] == "grouped_cv"]
    h = {"stage": "holdout", "method": "uncalibrated"}
    rows.append(["Locked holdout, scored once", method["uncalibrated"], ci("e8_calibration", h, "brier", "{:.4f}"), c("e8_calibration", h, "ece", "{:.4f}"),
                 ci("e8_calibration", h, "auc_roc", "{:.4f}")])
    T.write("t11_calibration", "E8", "Calibration: Brier score and expected calibration error (ECE), gradient boosting with all features",
            ["Data", "Calibration method", "Brier score", "Expected calibration error", "AUC-ROC"], rows, ["results/e8_calibration.csv"],
            "The calibrator is chosen by grouped-CV Brier score on the training part; a tie between methods means calibration adds nothing here.")

    # T12 deployable
    c, ci = T.quoters("t12_deployable")
    rows = []
    for k in T.keys("e9_deployable", "stage", "model"):
        if k["stage"] != "holdout":
            continue
        rows.append([MODEL[k["model"]], ci("e9_deployable", k, "auc_roc"), ci("e9_deployable", k, "avg_precision"), ci("e9_deployable", k, "brier"),
                     ci("e9_deployable", k, "sens_at_prec_40"), ci("e9_deployable", k, "delta_auc_roc_vs_research", "{:+.3f}")])
    T.write("t12_deployable", "E9", "The deployable model (age, sex, lead time, weekday, earlier appointments, earlier no-show rate) against the research model, holdout",
            ["Model", "AUC-ROC", "Average precision", "Brier score", "Sensitivity at 40% precision", "AUC-ROC difference from research model"], rows,
            ["results/e9_deployable.csv"], "The model in the app is not calibrated: see Table 11.")

    # T13 thresholds, T14 bands
    c, ci = T.quoters("t13_thresholds")
    rows = []
    for k in T.keys("e10_thresholds", "kind", "label"):
        if k["kind"] == "band":
            continue
        rule = {"fixed": f"Risk at least {k['label']}", "percentile": f"Top share above the {k['label']} percentile of risk",
                "chosen": f"Chosen {k['label'].title()} cut"}[k["kind"]]
        if k["kind"] == "chosen":
            rule += f" (target precision {c('e10_thresholds', k, 'target_precision', '{:.0%}')})"
        rows.append([rule, c("e10_thresholds", k, "threshold"), ci("e10_thresholds", k, "share_flagged", "{:.1%}"),
                     ci("e10_thresholds", k, "precision", "{:.1%}"), ci("e10_thresholds", k, "sensitivity", "{:.1%}")])
    T.write("t13_thresholds", "E10", "Flagging thresholds on the Kaggle holdout: how many bookings are flagged, and how many flagged are no-shows",
            ["Rule", "Risk threshold", "Share of bookings flagged", "Precision (share of flagged who miss)", "Sensitivity (share of no-shows flagged)"], rows,
            ["results/e10_thresholds.csv"], "Intervals are 95% bootstrap intervals over patients.")

    c, ci = T.quoters("t14_risk_bands")
    rows = []
    for k in T.keys("e10_thresholds", "kind", "label"):
        if k["kind"] != "band":
            continue
        upper = c("e10_thresholds", k, "upper_threshold") if k["label"] != "high" else "no upper limit"
        rows.append([k["label"].title(), c("e10_thresholds", k, "threshold"), upper, ci("e10_thresholds", k, "share_flagged", "{:.1%}"),
                     ci("e10_thresholds", k, "precision", "{:.1%}"), ci("e10_thresholds", k, "sensitivity", "{:.1%}")])
    T.write("t14_risk_bands", "E10", "The Low, Medium and High bands shown to staff",
            ["Band", "Risk from", "Risk below", "Share of bookings", "No-show rate in the band", "Share of all no-shows in the band"], rows,
            ["results/e10_thresholds.csv"], "These are the thresholds on the model card; the app reads them from there.")

    # simulation tables
    def sim_rows(c, ci, file, key_cols, lead_labels):
        rows = []
        for k in T.keys(file, *key_cols):
            rows.append(lead_labels(k) + [c(file, k, "double_slots_mean", "{:.2f}"), ci(file, k, "patients_served", "{:.2f}"),
                                         ci(file, k, "mean_wait_min", "{:.1f}"), ci(file, k, "overtime_min", "{:.1f}"),
                                         ci(file, k, "idle_min", "{:.1f}"), ci(file, k, "share_sessions_both_attend", "{:.1%}")])
        return rows

    sim_header = ["Double-booked slots per session", "Patients served", "Mean wait of attending patients (minutes)", "Doctor overtime (minutes)",
                  "Doctor idle time (minutes)", "Sessions where both patients of a double-booked slot attend"]
    c, ci = T.quoters("t15_policy_tradeoffs")
    rows = sim_rows(c, ci, "s2_policy_tradeoffs", ("policy",), lambda k: [k["policy"], c("s2_policy_tradeoffs", k, "threshold", "{:.3f}")])
    reps = c("s2_policy_tradeoffs", {"policy": "P0"}, "n_reps", "{:,.0f}")
    T.write("t15_policy_tradeoffs", "S2", "Booking policies: the trade-off table (no weighted score is computed)", ["Policy", "Risk threshold"] + sim_header, rows,
            ["results/s2_policy_tradeoffs.csv"], f"Means over {reps} replications with 95% intervals; one provider, one half-day session.")

    c, ci = T.quoters("t16_probability_error")
    shift = lambda k: "+0.00" if float(k["risk_shift"]) == 0 else f"{float(k['risk_shift']):+.2f}"
    rows = sim_rows(c, ci, "s3_probability_error", ("risk_shift", "policy"), lambda k: [shift(k), k["policy"]])
    T.write("t16_probability_error", "S3", "Sensitivity to error in the predicted risk (a constant added to every predicted probability)",
            ["Shift in predicted risk", "Policy"] + sim_header, rows, ["results/s3_probability_error.csv"],
            "Thresholds are fixed at their unshifted values, so a shift changes how many slots get doubled.")

    c, ci = T.quoters("t17_base_rate")
    rows = sim_rows(c, ci, "s4_base_rate", ("base_rate_scale", "policy"),
                    lambda k: [c("s4_base_rate", k, "base_no_show_rate", "{:.1%}"), k["policy"]])
    T.write("t17_base_rate", "S4", "Sensitivity to the clinic's base no-show rate", ["Base no-show rate", "Policy"] + sim_header, rows,
            ["results/s4_base_rate.csv"])

    c, ci = T.quoters("t18_service_variability")
    rows = sim_rows(c, ci, "s5_service_variability", ("variability_scale", "policy"),
                    lambda k: [c("s5_service_variability", k, "service_sd_min", "{:.1f}"), k["policy"]])
    T.write("t18_service_variability", "S5", "Sensitivity to the spread of consultation times (same mean)",
            ["Standard deviation of consultation time (minutes)", "Policy"] + sim_header, rows, ["results/s5_service_variability.csv"])

    # S6
    c, ci = T.quoters("t19_value_of_prediction")
    file = "s6_value_of_prediction"
    rows = []
    for k in T.keys(file, "scheme", "setting", "selector"):
        rows.append([("Matched to P1" if k["scheme"] == "matched_to_p1" else "Matched to P2"), k["setting"], SELECTOR[k["selector"]],
                     c(file, k, "double_slots_mean", "{:.2f}"), c(file, k, "mean_wait_min", "{:.1f}"),
                     ci(file, k, "mean_wait_min_vs_uniform", "{:+.2f}", lo="mean_wait_min_vs_uniform_lo", hi="mean_wait_min_vs_uniform_hi"),
                     ci(file, k, "mean_wait_min_vs_random", "{:+.2f}", lo="mean_wait_min_vs_random_lo", hi="mean_wait_min_vs_random_hi"),
                     ci(file, k, "overtime_min_vs_uniform", "{:+.2f}", lo="overtime_min_vs_uniform_lo", hi="overtime_min_vs_uniform_hi"),
                     ci(file, k, "overtime_min_vs_random", "{:+.2f}", lo="overtime_min_vs_random_lo", hi="overtime_min_vs_random_hi"),
                     c(file, k, "share_double_slots_both_attend", "{:.1%}")])
    T.write("t19_value_of_prediction", "S6", "Value of prediction at equal overbooking: the same number of double-booked slots, chosen four ways",
            ["Scheme", "Setting", "How the slots are chosen", "Double-booked slots per session", "Mean wait (minutes)", "Wait vs evenly spaced (minutes)",
             "Wait vs random (minutes)", "Overtime vs evenly spaced (minutes)", "Overtime vs random (minutes)", "Double-booked slots where both attend"], rows,
            [f"results/{file}.csv"], "Differences are paired over replications (95% intervals); below zero favours the row. Patients served are identical across "
                                      "rows with the same setting by construction, so they are not tabulated.")

    # service-time fit
    c, ci = T.quoters("t20_service_time_fit")
    rows = []
    for k in T.keys("sim_service_time_fit", "family"):
        raw = lambda col: reg.raw("results/sim_service_time_fit.csv", k, col)
        params = (f"{raw('param_1_name')} = {c('sim_service_time_fit', k, 'param_1', '{:.3f}')}, "
                  f"{raw('param_2_name')} = {c('sim_service_time_fit', k, 'param_2', '{:.3f}')}")
        rows.append([k["family"].title(), params, c("sim_service_time_fit", k, "log_likelihood", "{:,.1f}"), c("sim_service_time_fit", k, "aic", "{:,.1f}"),
                     c("sim_service_time_fit", k, "ks_statistic", "{:.4f}"), c("sim_service_time_fit", k, "mean_fitted", "{:.2f}"),
                     yes_no(reg.raw("results/sim_service_time_fit.csv", k, "chosen"))])
    k0 = {"family": "lognormal"}
    note = (f"Observed mean {c('sim_service_time_fit', k0, 'mean_data', '{:.2f}')} minutes over {c('sim_service_time_fit', k0, 'n', '{:,.0f}')} consultations; median "
            f"{c('sim_service_time_fit', k0, 'median_service_min', '{:.1f}')} minutes, so the default slot is {c('sim_service_time_fit', k0, 'slot_minutes_default', '{:.0f}')} minutes "
            f"and the session {c('sim_service_time_fit', k0, 'session_minutes_default', '{:.0f}')} minutes.")
    T.write("t20_service_time_fit", "S1", "Consultation-time distributions fitted to the Hangu data (minutes)",
            ["Distribution", "Parameters", "Log-likelihood", "AIC", "KS statistic", "Fitted mean (minutes)", "Chosen by AIC"], rows,
            ["results/sim_service_time_fit.csv"], note)

    # tests
    c, ci = T.quoters("t21_test_summary")
    label = {("total", "all"): "All tests", ("engine", "sqlite"): "Database tests on SQLite", ("engine", "mysql"): "Database tests on MySQL"}
    rows = [[label.get((k["group"], k["name"]), k["name"]), c("test_summary", k, "tests", "{:,.0f}"), c("test_summary", k, "passed", "{:,.0f}"),
             c("test_summary", k, "failed", "{:,.0f}"), c("test_summary", k, "skipped", "{:,.0f}")] for k in T.keys("test_summary", "group", "name")]
    T.write("t21_test_summary", "Chapter 7", "Test summary: tests run and passed, in total, per database engine and per test file",
            ["Scope", "Tests", "Passed", "Failed", "Skipped"], rows, ["results/test_summary.csv", "results/test_summary.txt"],
            "Written by scripts/run_test_summary.py from a run of the whole suite on both engines.")
    return T.index


# ---- figures ----------------------------------------------------------------------------------------------------
BG, INK, MUTED, GRID = "#FCFCFA", "#1A1A1A", "#555555", "#E4E4DF"
BLUE, ORANGE, GREEN, AMBER, GREY, PURPLE = "#2A78D6", "#EC6A35", "#1AA97C", "#E8A20C", "#8A8A85", "#7E57C2"
# the four chromatic colours pass the colour-vision-deficiency check; green and amber are light on this background,
# so every figure also uses marker shapes, direct labels and a table of the same numbers (see INDEX.md)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.size": 11, "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2.0, "font.family": "DejaVu Sans"})
    return plt


def _frame(plt, width, height, title, subtitle, footer, gap=0.45, xroom=0.65, **grid):
    """A figure with a left-aligned title, wrapped subtitle and footnote. Margins follow the amount of text (in inches),
    so nothing is cut off or overlaps: `gap` is room for panel titles or a legend above the axes, `xroom` for tick labels
    and the x-axis label below them."""
    fig = plt.figure(figsize=(width, height))
    sub = textwrap.fill(subtitle, int(width * 11.5))
    note = textwrap.fill(footer, int(width * 13.5))
    sub_lines, note_lines = sub.count("\n") + 1, note.count("\n") + 1
    size = min(15.0, 0.95 * width * 72 / (0.60 * len(title)))
    fig.text(0.012, 1 - 0.12 / height, title, fontsize=size, fontweight="bold", va="top", ha="left")
    fig.text(0.012, 1 - 0.62 / height, sub, fontsize=10.5, color=MUTED, va="top", ha="left", linespacing=1.4)
    fig.text(0.012, 0.12 / height, note, fontsize=8.5, color=GREY, va="bottom", ha="left", linespacing=1.4)
    top = 1 - (0.62 + 0.24 * sub_lines + gap) / height
    bottom = (0.12 + 0.17 * note_lines + 0.12 + xroom) / height
    fig.subplots_adjust(top=top, bottom=bottom, left=grid.pop("left", 0.09), right=grid.pop("right", 0.97), **grid)
    return fig


def _save(plt, fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _f(row, column) -> float:
    return float(row[column])


def _rows(reg, stem, **where):
    return [r for r in reg.table(f"results/{stem}.csv") if all(r[k] == v for k, v in where.items())]


def _pct_axis(ax, which="y"):
    from matplotlib.ticker import PercentFormatter

    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))


def _footer_repaired(src: Path, dst: Path, footer: str, cut_fraction: float, plt):
    """Reuse a result image, keep everything above the footnote strip, and write the footnote again in full."""
    import matplotlib.image as mpimg
    import numpy as np

    img = mpimg.imread(src)
    if img.shape[2] == 3:
        img = np.dstack([img, np.ones(img.shape[:2], dtype=img.dtype)])
    height, width = img.shape[:2]
    background = img[2, 2].copy()
    cut = int(height * cut_fraction)
    img = img.copy()
    img[cut:] = background
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.figimage(img, 0, 0, origin="upper", resize=False)
    left = 0.13 if width < 1500 else 0.065
    points = 11.0 if width < 1500 else 12.5
    chars = int((width * (1 - 2 * left)) / (points * 0.62 * dpi / 72))
    fig.text(left, (height - cut) * 0.5 / height, textwrap.fill(footer, chars), fontsize=points, color=GREY, va="center", ha="left", linespacing=1.3)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=dpi, facecolor=tuple(background))
    plt.close(fig)


# name, experiment, caption, source files, how it was made
FIGURE_META = [
    ("f01_e1_overall_noshow_rate", "E1", "Overall no-show rate in the two appointment datasets.", ["results/e1_eda_rates.csv", "results/figures/e1_overall_noshow_rate.png"], "copied unchanged from results/figures"),
    ("f02_e1_noshow_by_lead_time", "E1", "No-show rate by days between booking and appointment, both datasets.", ["results/e1_eda_rates.csv", "results/figures/e1_noshow_by_lead_time.png"], "copied unchanged from results/figures"),
    ("f03_e1_noshow_by_weekday", "E1", "No-show rate by appointment weekday, both datasets.", ["results/e1_eda_rates.csv", "results/figures/e1_noshow_by_weekday.png"], "copied unchanged from results/figures"),
    ("f04_e1_noshow_by_age_band", "E1", "No-show rate by age band, both datasets.", ["results/e1_eda_rates.csv", "results/figures/e1_noshow_by_age_band.png"], "copied unchanged from results/figures"),
    ("f05_e3_roc", "E3", "ROC curves of the three models on the Kaggle holdout.", ["results/e3_kaggle_models.csv", "results/figures/e3_roc.png"], "result image reused (curve points were not saved); footnote redrawn"),
    ("f06_e3_pr", "E3", "Precision-recall curves of the three models on the Kaggle holdout.", ["results/e3_kaggle_models.csv", "results/figures/e3_pr.png"], "result image reused (curve points were not saved); footnote redrawn"),
    ("f07_e4_ablation", "E4", "Change in cross-validated AUC-ROC when a feature group is removed, and when only that group is kept.", ["results/e4_ablation.csv"], "redrawn from the CSV"),
    ("f08_e6_transfer", "E6", "AUC-ROC within the target dataset and after transfer, both directions, common features only.", ["results/e6_transfer.csv"], "redrawn from the CSV"),
    ("f09_e8_reliability", "E8", "Reliability diagram on the Kaggle holdout, ten equal-count bins.", ["results/e8_reliability.csv", "results/e8_calibration.csv"], "redrawn from the CSV"),
    ("f10_e10_thresholds", "E10", "Precision, sensitivity and share of bookings flagged at each risk threshold, with the Medium and High cuts.", ["results/e10_thresholds.csv"], "redrawn from the CSV"),
    ("f11_s2_tradeoffs", "S2", "Patients served against mean wait and against doctor overtime for every booking policy.", ["results/s2_policy_tradeoffs.csv"], "redrawn from the CSV"),
    ("f12_s3_probability_error", "S3", "Double-booked slots, wait and overtime when the predicted risk is shifted by a constant.", ["results/s3_probability_error.csv"], "redrawn from the CSV"),
    ("f13_s4_base_rate", "S4", "Patients served, wait and overtime when the clinic's base no-show rate is scaled.", ["results/s4_base_rate.csv"], "redrawn from the CSV"),
    ("f14_s5_service_variability", "S5", "Wait and overtime when the spread of consultation times is scaled at the same mean.", ["results/s5_service_variability.csv"], "redrawn from the CSV"),
    ("f15_s6_value_of_prediction", "S6", "Collisions, waiting and overtime against the number of double-booked slots, by how the slots are chosen.", ["results/s6_value_of_prediction.csv", "results/figures/s6_value_of_prediction.png"], "copied unchanged from results/figures"),
    ("f16_s6_paired_differences", "S6", "Paired differences in waiting and overtime between risk-based, evenly spaced, random and oracle choices at equal overbooking.", ["results/s6_value_of_prediction.csv"], "redrawn from the CSV"),
    ("f17_service_time_fit", "S1", "Hangu consultation times with the lognormal and gamma fits.", ["results/sim_service_time_fit.csv", "results/figures/sim_service_time_fit.png"], "copied unchanged from results/figures"),
    ("f18_service_time_qq", "S1", "QQ plots of the Hangu consultation times against the fitted lognormal and gamma.", ["results/sim_service_time_fit.csv", "results/figures/sim_service_time_qq.png"], "result image reused (the quantiles were not saved); footnote redrawn"),
]


def build_figures(reg: Registry, out: Path) -> None:
    plt = _plt()
    figures = out / ASSETS / "figures"
    src = reg.root / "results" / "figures"
    known = {m[0] for m in FIGURE_META}

    def need(name):
        assert name in known, name
        return figures / f"{name}.png"

    summary = lambda dataset, metric: float(reg.raw("results/dataset_summary.csv", {"dataset": dataset, "metric": metric}, "value"))

    # E1: the experiment's own figures, unchanged
    for name, original, caption in [
        ("f01_e1_overall_noshow_rate", "e1_overall_noshow_rate", "Overall no-show rate in the two appointment datasets."),
        ("f02_e1_noshow_by_lead_time", "e1_noshow_by_lead_time", "No-show rate by days between booking and appointment, both datasets."),
        ("f03_e1_noshow_by_weekday", "e1_noshow_by_weekday", "No-show rate by appointment weekday, both datasets."),
        ("f04_e1_noshow_by_age_band", "e1_noshow_by_age_band", "No-show rate by age band, both datasets."),
    ]:
        target = need(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f"{original}.png", target)

    # E3 curves: images reused, footnote strip redrawn
    n_hold, n_dates = summary("kaggle", "n_holdout_dates"), summary("kaggle", "n_appointment_dates")
    foot = (f"Locked holdout (the latest {n_hold:.0f} of {n_dates:.0f} appointment dates, {summary('kaggle', 'n_holdout_rows'):,.0f} appointments), "
            "each model scored once. Numbers: results/e3_kaggle_models.csv.")
    for name, original, caption in [("f05_e3_roc", "e3_roc", "ROC curves of the three models on the Kaggle holdout."),
                                    ("f06_e3_pr", "e3_pr", "Precision-recall curves of the three models on the Kaggle holdout.")]:
        _footer_repaired(src / f"{original}.png", need(name), foot, 0.92, plt)

    # E4 ablation
    rows = _rows(reg, "e4_ablation")
    base = next(r for r in rows if r["variant"] == "all_features")
    folds = int(summary("kaggle", "cv_n_splits"))
    fig = _frame(plt, 11, 5.6, "What each group of features contributes",
                 f"Gradient boosting on the Kaggle training part, patient-grouped {folds}-fold cross-validation. Reference with every feature: "
                 f"AUC-ROC {_f(base, 'cv_auc_mean'):.3f}. Bars show the change in AUC-ROC; whiskers are the standard deviation of the fold-by-fold differences.",
                 "Numbers: results/e4_ablation.csv. SMS reminder is a research-only feature (the app cannot collect it). The holdout is not used here.",
                 gap=0.5, left=0.19, wspace=0.62)
    panels = [("Remove one group", [r for r in rows if r["variant"].startswith(("no_", "drop_"))], BLUE),
              ("Keep only one group", [r for r in rows if r["variant"].startswith("only_")], ORANGE)]
    for position, (title, group, color) in enumerate(panels):
        ax = fig.add_subplot(1, 2, position + 1)
        group = sorted(group, key=lambda r: _f(r, "delta_auc_vs_all"), reverse=True)
        labels = [GROUP[r["variant"]] for r in group]
        values = [_f(r, "delta_auc_vs_all") for r in group]
        ax.barh(range(len(group)), values, xerr=[_f(r, "delta_auc_fold_std") for r in group], color=color, height=0.55,
                error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
        for i, value in enumerate(values):
            ax.text(value - _f(group[i], "delta_auc_fold_std") - 0.004, i, f"{value:+.3f}", ha="right", va="center", fontsize=9.5, color=INK)
        ax.set_yticks(range(len(group)), labels)
        ax.invert_yaxis()
        ax.set_xlim(-0.30, 0.005)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_title(title, fontsize=11.5, loc="left", color=INK)
        ax.set_xlabel("Change in AUC-ROC vs all features")
        ax.grid(axis="y", visible=False)
    _save(plt, fig, need("f07_e4_ablation"))

    # E6 transfer
    rows = _rows(reg, "e6_transfer")
    fig = _frame(plt, 11, 5.4, "Models trained on one dataset, tested on the other",
                 "Only the four features both datasets share (age, sex, lead time, weekday). Blue bars: a model trained on the target dataset's own "
                 "training part. Orange bars: a model trained on the other dataset. Bars start at chance (0.5); whiskers are 95% bootstrap intervals.",
                 "Numbers: results/e6_transfer.csv. Both models are scored once on the target's locked holdout.", gap=0.45, wspace=0.08)
    for position, (direction, title) in enumerate([("kaggle_to_openml", "Trained on Kaggle, tested on OpenML"), ("openml_to_kaggle", "Trained on OpenML, tested on Kaggle")]):
        ax = fig.add_subplot(1, 2, position + 1)
        pair = [r for r in rows if r["direction"] == direction]
        for i, r in enumerate(pair):
            for j, (column, color, label) in enumerate([("auc_within_target", BLUE, "Target's own training part"), ("auc_transfer", ORANGE, "Trained on the other dataset")]):
                value = _f(r, column)
                ax.bar(i + (j - 0.5) * 0.36, value - 0.5, 0.34, bottom=0.5, color=color, label=label if i == 0 else None)
                ax.errorbar(i + (j - 0.5) * 0.36, value, yerr=[[value - _f(r, column + "_lo")], [_f(r, column + "_hi") - value]], color=INK, capsize=3, linewidth=1)
                ax.text(i + (j - 0.5) * 0.36, _f(r, column + "_hi") + 0.006, f"{value:.3f}", ha="center", fontsize=9.5)
        ax.axhline(0.5, color=GREY, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(pair)), [MODEL[r["model"]] for r in pair])
        ax.set_ylim(0.5, 0.80)
        ax.set_title(title, fontsize=11.5, loc="left", color=INK)
        ax.grid(axis="x", visible=False)
        if position == 0:
            ax.set_ylabel("AUC-ROC (axis starts at chance)")
            ax.legend(loc="upper left", fontsize=9.5)
        else:
            ax.set_yticklabels([])
    _save(plt, fig, need("f08_e6_transfer"))

    # E8 reliability
    bins = _rows(reg, "e8_reliability", version="uncalibrated")
    other = _rows(reg, "e8_reliability", version="calibrated")
    same = all(abs(_f(a, "mean_predicted") - _f(b, "mean_predicted")) < 1e-12 and abs(_f(a, "observed_rate") - _f(b, "observed_rate")) < 1e-12 for a, b in zip(bins, other))
    cv = {r["method"]: _f(r, "brier_cv") for r in _rows(reg, "e8_calibration", stage="grouped_cv")}
    chosen = min(cv, key=cv.get)
    holdout = _rows(reg, "e8_calibration", stage="holdout")[0]
    note = ("No calibrator is applied: Platt scaling and isotonic regression did not lower the grouped-CV Brier score, so the calibrated and uncalibrated curves are the same."
            if chosen == "uncalibrated" and same else f"Calibrator chosen by grouped-CV Brier score: {chosen}.")
    fig = _frame(plt, 7.5, 7.2, "Reliability of the gradient-boosting model on the Kaggle holdout",
                 f"Predicted risk against observed no-show rate in {len(bins)} equal-count bins; points on the diagonal are well calibrated. "
                 f"Expected calibration error {_f(holdout, 'ece'):.3f}. {note}",
                 "Whiskers: 95% Wilson interval. Numbers: results/e8_reliability.csv and results/e8_calibration.csv.", gap=0.15, left=0.13)
    ax = fig.add_subplot(1, 1, 1)
    top = max(max(_f(r, "ci_high") for r in bins), max(_f(r, "mean_predicted") for r in bins)) * 1.1
    ax.plot([0, top], [0, top], color=GREY, linewidth=1.2, linestyle="--", label="Perfect calibration")
    x, y = [_f(r, "mean_predicted") for r in bins], [_f(r, "observed_rate") for r in bins]
    ax.errorbar(x, y, yerr=[[b - _f(r, "ci_low") for b, r in zip(y, bins)], [_f(r, "ci_high") - b for b, r in zip(y, bins)]], color=BLUE,
                marker="o", markersize=6, capsize=3, label="Gradient boosting (observed rate per bin)")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    _pct_axis(ax)
    _pct_axis(ax, "x")
    ax.set_xlabel("Mean predicted probability of a no-show")
    ax.set_ylabel("Observed no-show rate")
    ax.legend(loc="upper left", fontsize=9.5)
    _save(plt, fig, need("f09_e8_reliability"))

    # E10 thresholds
    rows = sorted(_rows(reg, "e10_thresholds", kind="fixed"), key=lambda r: _f(r, "threshold"))
    cuts = {r["label"]: _f(r, "threshold") for r in _rows(reg, "e10_thresholds", kind="chosen")}
    prevalence = summary("kaggle", "holdout_no_show_rate")
    fig = _frame(plt, 9.5, 6.2, "What each flagging threshold catches",
                 "Kaggle holdout, deployable-family model. Raising the threshold flags fewer bookings, and a larger share of the flagged ones miss. "
                 "Bands show 95% bootstrap intervals; dotted lines mark the Medium and High cuts on the model card.",
                 "Numbers: results/e10_thresholds.csv. Above a threshold of about 0.45 fewer than 2% of bookings are flagged, so those points are noisy.",
                 gap=0.5, left=0.09, right=0.97)
    ax = fig.add_subplot(1, 1, 1)
    x = [_f(r, "threshold") for r in rows]
    for column, color, label, marker in [("precision", BLUE, "Precision", "o"), ("sensitivity", ORANGE, "Sensitivity", "s"), ("share_flagged", GREY, "Share of bookings flagged", "^")]:
        y = [_f(r, column) for r in rows]
        ax.plot(x, y, color=color, marker=marker, markersize=5, linestyle="--" if column == "share_flagged" else "-", label=label)
        if column != "share_flagged":
            ax.fill_between(x, [_f(r, column + "_lo") for r in rows], [_f(r, column + "_hi") for r in rows], color=color, alpha=0.15, linewidth=0)
    ax.axhline(prevalence, color=GREY, linewidth=1, linestyle=":")
    ax.text(0.455, prevalence + 0.015, "no-show rate: the precision of flagging at random", fontsize=8.5, color=MUTED)
    for label, threshold in cuts.items():
        ax.axvline(threshold, color=INK, linewidth=1, linestyle=":")
        ax.text(threshold + 0.004, 0.97, f"{label.title()} cut {threshold:.2f}", fontsize=9, va="top", color=INK)
    _pct_axis(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9.5)
    ax.set_xlim(0.15, 0.72)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Risk threshold (flag a booking when its predicted risk is at least this)")
    ax.set_ylabel("Share")
    _save(plt, fig, need("f10_e10_thresholds"))

    # S2 trade-offs
    rows = _rows(reg, "s2_policy_tradeoffs")
    fig = _frame(plt, 12, 5.8, "What each booking policy trades away",
                 "Each point is one policy, averaged over the simulated sessions. Serving more patients costs longer waits and more doctor overtime. "
                 "No single score is computed, because any weighting of the two would be the reader's choice.",
                 "Simulated half-day session for one doctor; consultation times from the Hangu data, no-show risk from the deployable model. Numbers: results/s2_policy_tradeoffs.csv.",
                 gap=0.2, wspace=0.18)
    style = {"P0": (GREY, "s"), "P1": (BLUE, "o"), "P2": (ORANGE, "^")}
    offsets = {"P1": (9, -13), "P2": (-9, 7)}
    for position, (metric, label) in enumerate([("mean_wait_min", "Mean wait of attending patients (minutes)"), ("overtime_min", "Doctor overtime past the session (minutes)")]):
        ax = fig.add_subplot(1, 2, position + 1)
        for family, (color, marker) in style.items():
            members = [r for r in rows if r["family"] == family]
            ax.scatter([_f(r, metric) for r in members], [_f(r, "patients_served") for r in members], color=color, marker=marker, s=60,
                       zorder=3, label={"P0": "P0 no overbooking", "P1": "P1 every k-th slot", "P2": "P2 risk threshold"}[family])
            for r in members:
                if r["policy"] in ("P0", "P2 0.5"):
                    continue
                name = r["parameter"] if family != "P0" else "P0"
                ha = "left" if family == "P1" else "right"
                ax.annotate(name, (_f(r, metric), _f(r, "patients_served")), xytext=offsets[family], textcoords="offset points", fontsize=9, ha=ha, color=INK)
        p0, p5 = next(r for r in rows if r["policy"] == "P0"), next(r for r in rows if r["policy"] == "P2 0.5")
        ax.annotate("P0 and P2 0.5", (_f(p0, metric), _f(p0, "patients_served")), xytext=(10, 8), textcoords="offset points", fontsize=9, color=INK)
        ax.set_xlabel(label)
        if position == 0:
            ax.set_ylabel("Patients served per session")
            ax.legend(loc="lower right", fontsize=9.5)
    _save(plt, fig, need("f11_s2_tradeoffs"))

    # S3, S4, S5 sensitivity
    def sensitivity(name, experiment, stem, xcolumn, xlabel, ticks, policies, title, subtitle, metrics):
        rows = _rows(reg, stem)
        fig = _frame(plt, 4.1 * len(metrics) + 0.6, 5.6, title, subtitle, f"Policies shown: {', '.join(p for p, *_ in policies)}. Every policy is in the table. Numbers: results/{stem}.csv.",
                     gap=0.5, xroom=1.0 if "\n" in "".join(ticks(rows)[1]) else 0.65, wspace=0.28)
        for position, (metric, label) in enumerate(metrics):
            ax = fig.add_subplot(1, len(metrics), position + 1)
            for policy, color, marker in policies:
                points = sorted(((_f(r, xcolumn), _f(r, metric)) for r in rows if r["policy"] == policy))
                if len(points) == 1:  # P0 only appears at the unshifted value: draw it across
                    ax.axhline(points[0][1], color=color, linewidth=1.4, linestyle="--", label=policy)
                else:
                    ax.plot(*zip(*points), color=color, marker=marker, markersize=6, label=policy)
            ax.set_xticks(*ticks(rows))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label)
            if position == 0:
                ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(policies))
        _save(plt, fig, need(name))

    chosen_policies = [("P0", GREY, "s"), ("P2 0.3", BLUE, "o"), ("P2 p70", ORANGE, "^"), ("P2 p80", GREEN, "D"), ("P2 p90", AMBER, "v")]
    xs = sorted({_f(r, "risk_shift") for r in _rows(reg, "s3_probability_error")})
    sensitivity("f12_s3_probability_error", "S3", "s3_probability_error", "risk_shift", "Constant added to every predicted risk",
                lambda rows: (xs, [f"{x:+.2f}" for x in xs]), chosen_policies, "If the predicted risk is off by a constant",
                "Thresholds stay at their unshifted values. Positive shifts overestimate risk, so more slots are double-booked; negative shifts underestimate it.",
                [("double_slots_mean", "Double-booked slots per session"), ("mean_wait_min", "Mean wait (minutes)"), ("overtime_min", "Doctor overtime (minutes)")])
    scale = {(_f(r, "base_rate_scale")): _f(r, "base_no_show_rate") for r in _rows(reg, "s4_base_rate")}
    sensitivity("f13_s4_base_rate", "S4", "s4_base_rate", "base_rate_scale", "Base no-show rate (scale in brackets)",
                lambda rows: (sorted(scale), [f"{scale[k]:.0%}\n(x{k:g})" for k in sorted(scale)]),
                [("P0", GREY, "s"), ("P1 k=3", BLUE, "o"), ("P2 0.3", ORANGE, "^"), ("P2 p80", GREEN, "D")], "If the clinic's no-show rate is different",
                "Every patient's no-show probability is scaled, then the policies are re-run. P1 and the percentile policies double the same number of slots at every rate.",
                [("patients_served", "Patients served per session"), ("mean_wait_min", "Mean wait (minutes)"), ("overtime_min", "Doctor overtime (minutes)")])
    sd = {(_f(r, "variability_scale")): _f(r, "service_sd_min") for r in _rows(reg, "s5_service_variability")}
    sensitivity("f14_s5_service_variability", "S5", "s5_service_variability", "variability_scale", "Standard deviation of consultation time (minutes)",
                lambda rows: (sorted(sd), [f"{sd[k]:.1f}\n(x{k:g})" for k in sorted(sd)]),
                [("P0", GREY, "s"), ("P1 k=3", BLUE, "o"), ("P2 0.3", ORANGE, "^"), ("P2 p80", GREEN, "D")], "If consultation times vary more or less",
                "The consultation-time distribution is rescaled around the same mean. The same patients attend in every setting, so patients served do not change; waits and overtime do.",
                [("mean_wait_min", "Mean wait (minutes)"), ("overtime_min", "Doctor overtime (minutes)")])

    # S6: the experiment's own figure, then the paired differences
    target = need("f15_s6_value_of_prediction")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "s6_value_of_prediction.png", target)

    rows = _rows(reg, "s6_value_of_prediction", scheme="matched_to_p2")
    settings = []
    for r in rows:
        if r["setting"] not in settings:
            settings.append(r["setting"])
    settings.sort(key=lambda s: -_f(next(r for r in rows if r["setting"] == s and r["selector"] == "p2_risk_threshold"), "double_slots_mean"))
    fig = _frame(plt, 12, 6.6, "Does choosing slots by risk help, at the same number of double-booked slots?",
                 "Paired differences over the simulated sessions with 95% intervals; below zero means the risk-based choice is better. Blue and orange: the risk "
                 "threshold against evenly spaced and against random slots. Amber: an oracle that knows who will not come, the most any model could gain.",
                 "Rows are P2 settings, ordered by how many slots they double-book (shown in brackets). Numbers: results/s6_value_of_prediction.csv.", gap=0.2, left=0.12, wspace=0.14)
    for position, (metric, label) in enumerate([("mean_wait_min", "Mean wait of attending patients (minutes)"), ("overtime_min", "Doctor overtime (minutes)")]):
        ax = fig.add_subplot(1, 2, position + 1)
        for i, setting in enumerate(settings):
            by = {r["selector"]: r for r in rows if r["setting"] == setting}
            for offset, selector, versus, color, marker, name in [(-0.22, "p2_risk_threshold", "uniform", BLUE, "o", "Risk threshold vs evenly spaced"),
                                                                   (0.0, "p2_risk_threshold", "random", ORANGE, "s", "Risk threshold vs random"),
                                                                   (0.22, "oracle", "uniform", AMBER, "D", "Oracle vs evenly spaced")]:
                r = by[selector]
                value = _f(r, f"{metric}_vs_{versus}")
                ax.errorbar(value, i + offset, xerr=[[value - _f(r, f"{metric}_vs_{versus}_lo")], [_f(r, f"{metric}_vs_{versus}_hi") - value]], color=color,
                            marker=marker, markersize=6, capsize=2.5, linewidth=1.3, label=name if i == 0 else None)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_yticks(range(len(settings)), [f"P2 {s} ({_f(next(r for r in rows if r['setting'] == s and r['selector'] == 'p2_risk_threshold'), 'double_slots_mean'):.1f})"
                                             for s in settings] if position == 0 else [""] * len(settings))
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(label + ": difference (below zero is better)")
        if position == 0:
            ax.legend(loc="lower left", fontsize=9)
    _save(plt, fig, need("f16_s6_paired_differences"))

    # service-time figures
    target = need("f17_service_time_fit")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "sim_service_time_fit.png", target)
    fit = _rows(reg, "sim_service_time_fit", family="lognormal")[0]
    foot = (f"Hangu Data.csv, consultation time in minutes (n = {_f(fit, 'n'):,.0f}). The shortest consultation in the file is {summary('hangu', 'service_seconds_min') / 60:.0f} minutes; "
            "shorter ones appear to have been removed by the dataset's authors, so the lowest quantiles sit above the line. Numbers: results/sim_service_time_fit.csv.")
    _footer_repaired(src / "sim_service_time_qq.png", need("f18_service_time_qq"), foot, 0.925, plt)


# ---- the results summary -----------------------------------------------------------------------------------------
class BareNumber(ValueError):
    """The summary template contains a number that does not come from a results file."""


SUMMARY_TEMPLATE = Path(__file__).with_name("results_summary.md.j2")
SUMMARY_PATH = "docs/RESULTS_SUMMARY.md"
# things that look like numbers but are names: experiment ids (E3, S6), policies (P2, p90, k = 2), files, code, chapters
NAMES = [r"`[^`]*`", r"\[[^\]]*\]\([^)]*\)", r"[\w./-]+\.(?:csv|png|md|py|json|txt|j2)\b", r"\b[A-Za-z]\d+\b", r"\bk\s*=\s*\d+",
         r"\b(?:Chapters?|Tables?|Figures?|Sections?)\s+\d+(?:\s*(?:,|and|to)\s*\d+)*", r"\bv\d+(?:\.\d+)*\b"]


def _formatter(f):
    return FORMATS.get(f, f) if isinstance(f, str) else "{:.%df}" % f


def render_summary(template_text: str, reg: Registry, used_in: str = SUMMARY_PATH) -> str:
    """Render the template. Numbers can only enter through q(), qci() and lit(); a bare number raises BareNumber."""
    def functions(scan: bool) -> dict:
        def q(stem, where, column, f=3):
            return "" if scan else reg.cell(f"results/{stem}.csv", where, column, _formatter(f), used_in)

        def qci(stem, where, column, f=3, lo=None, hi=None):
            return "" if scan else reg.ci(f"results/{stem}.csv", where, column, _formatter(f), used_in, lo, hi)

        def lit(text):  # a number that is a parameter of the design, not a result (for example a threshold label)
            return "" if scan else str(text)

        def v(stem, where, column):  # a value for deciding what a sentence may say; never printed
            return float(reg.raw(f"results/{stem}.csv", where, column))

        def expect(condition, message):
            if not scan and not condition:
                raise ValueError(f"results/ no longer supports this sentence of the summary: {message}")
            return ""

        def v6(scheme, setting, selector, column):  # the same, for the value-of-prediction table
            return v("s6_value_of_prediction", {"scheme": scheme, "setting": setting, "selector": selector}, column)

        return {"q": q, "qci": qci, "lit": lit, "v": v, "v6": v6, "expect": expect}

    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_text)
    scanned = template.render(**functions(True))
    for pattern in NAMES:
        scanned = re.sub(pattern, " ", scanned)
    bare = re.search(r".{0,40}\d.{0,40}", scanned)
    if bare:
        raise BareNumber(f"a number in the summary template does not come from results/: ...{bare.group(0).strip()}...")
    return template.render(**functions(False))


# ---- the index -------------------------------------------------------------------------------------------------------
CHAPTER = {"E1": "6 and 8", "E2": "8", "E3": "8", "E4": "8", "E5": "8", "E6": "8", "E7": "8", "E8": "8", "E9": "8", "E10": "8",
           "S1": "7 and 8", "S2": "8", "S3": "8", "S4": "8", "S5": "8", "S6": "8", "Chapter 7": "7"}


def write_index(docs: Path, tables: list, out: Path) -> None:
    lines = ["# Report assets: index", "",
             "Everything here is built from `results/` by `python -m scripts.make_report_assets`; nothing was typed by hand. "
             "Every number in the tables is listed in `docs/numbers_audit.csv`, and `python -m scripts.check_numbers` re-checks them against the results files.",
             "Table numbers below follow this folder; renumber them to fit the report. Figures in the PNG files carry their own titles and footnotes; "
             "green and amber are light on the page background, so each figure also uses marker shapes, direct labels, and has its numbers in a table.", "",
             "## Tables (CSV for editing, Markdown for pasting)", "", "| File | Plan item | Report chapter | What it shows | Source |", "|---|---|---|---|---|"]
    for name, experiment, title, sources in tables:
        lines.append(f"| `tables/{name}.csv`, `tables/{name}.md` | {experiment} | {CHAPTER.get(experiment, '8')} | {title} | {', '.join(f'`{s}`' for s in sources)} |")
    lines += ["", "## Figures (PNG)", "", "| File | Plan item | Report chapter | What it shows | Source | How it was made |", "|---|---|---|---|---|---|"]
    for name, experiment, caption, sources, how in FIGURE_META:
        lines.append(f"| `figures/{name}.png` | {experiment} | {CHAPTER.get(experiment, '8')} | {caption} | {', '.join(f'`{s}`' for s in sources)} | {how} |")
    lines += ["", "## Not produced here", "",
              "The architecture diagram, ER diagram, booking-flow diagram and the screenshots of each role's pages are drawn or taken by the team "
              "(PLAN section 8); `docs/DEMO_SCRIPT.md` lists the pages to capture.", ""]
    (docs / ASSETS / "INDEX.md").write_text("\n".join(lines))


# ---- everything --------------------------------------------------------------------------------------------------------
def build(root=ROOT, docs_dir=None, figures: bool = True) -> None:
    """Write docs/report_assets/, docs/RESULTS_SUMMARY.md and docs/numbers_audit.csv from results/ under `root`."""
    root = Path(root)
    docs = Path(docs_dir) if docs_dir else root / "docs"
    reg = Registry(root)
    tables = build_tables(reg, docs)
    template = SUMMARY_TEMPLATE.read_text()
    (docs / "RESULTS_SUMMARY.md").write_text(render_summary(template, reg))
    write_index(docs, tables, docs)
    if figures:
        build_figures(reg, docs)
    reg.write_audit(docs / "numbers_audit.csv")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-figures", action="store_true", help="tables, summary and audit only")
    args = parser.parse_args(argv)
    build(ROOT, ROOT / "docs", figures=not args.no_figures)
    print(f"wrote docs/{ASSETS}/ (tables, figures, INDEX.md), docs/RESULTS_SUMMARY.md and docs/numbers_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
