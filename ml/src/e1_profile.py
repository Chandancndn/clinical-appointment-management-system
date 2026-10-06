"""E1: dataset profile and cleaning.  python -m ml.src.e1_profile

Reads the three raw files, applies the logged cleaning rules, locks the time-based holdout and writes:
    results/cleaning_log.csv        one row per cleaning rule per dataset (rows before / after / changed)
    results/dataset_summary.csv     long table: dataset, metric, value, note
    results/e1_eda_rates.csv        no-show rate by lead time, weekday, age band (+ overall), with 95% intervals
    results/figures/e1_*.png        the figures drawn from e1_eda_rates.csv
    results/manifest.json           seed, raw-data hashes, date and git commit for each of the above

The profile of how no-show relates to lead time, weekday and age uses the TRAINING part only (the earlier 80% of
appointment dates for Kaggle, months 1-3 for OpenML). The holdout is described (size, dates, rate) but never
explored: it stays untouched until a final model is scored once (CLAUDE.md, Evaluation).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from . import data  # noqa: E402
from .manifest import record_result  # noqa: E402

SCRIPT = "ml/src/e1_profile.py"
Z95 = 1.959964
MIN_GROUP_N = 100  # a group smaller than this is not drawn as a bar (its rate is mostly noise); it stays in the CSV

LEAD_BANDS = ([-1, 0, 3, 7, 14, 30, np.inf], ["0 (same day)", "1-3", "4-7", "8-14", "15-30", "31+"])
AGE_BANDS = ([-1, 4, 17, 34, 49, 64, np.inf], ["0-4", "5-17", "18-34", "35-49", "50-64", "65+"])
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DATASET_LABELS = {"kaggle": "Kaggle (Vitória, Brazil, 2016)", "openml": "OpenML Medical-Appointment (2017)"}

# Colours: the validated reference palette, categorical slots 1 and 2 (blue, orange), light surface.
SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898881", "#e6e5e1"
SERIES = {"kaggle": "#2a78d6", "openml": "#eb6834"}


# ---- statistics -----------------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. (nan, nan) when n is 0."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def rate_rows(frame: pd.DataFrame, dataset: str, variable: str, groups: pd.Series, order: list[str]) -> list[dict]:
    """One row per non-empty group: n, no-shows, rate and Wilson interval."""
    rows = []
    for label in order:
        part = frame.loc[(groups == label).to_numpy(), data.TARGET]
        if len(part) == 0:
            continue
        low, high = wilson_interval(int(part.sum()), len(part))
        rows.append({"dataset": dataset, "variable": variable, "group": label, "n": len(part),
                     "no_show_n": int(part.sum()), "rate": part.mean(), "ci_low": low, "ci_high": high})
    return rows


def eda_rates(train: pd.DataFrame, dataset: str) -> list[dict]:
    lead = pd.cut(train["lead_days"], bins=LEAD_BANDS[0], labels=LEAD_BANDS[1]).astype(str)
    age = pd.cut(train["age"], bins=AGE_BANDS[0], labels=AGE_BANDS[1]).astype(str)
    weekday = train["appt_weekday"].map(dict(enumerate(WEEKDAYS)))
    return (rate_rows(train, dataset, "lead_time_band", lead, LEAD_BANDS[1])
            + rate_rows(train, dataset, "weekday", weekday, WEEKDAYS)
            + rate_rows(train, dataset, "age_band", age, AGE_BANDS[1]))


# ---- summary table --------------------------------------------------------------------------------
class Summary:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, dataset: str, metric: str, value, note: str = "") -> None:
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        self.rows.append({"dataset": dataset, "metric": metric, "value": value, "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["dataset", "metric", "value", "note"])


def _basic(summary: Summary, name: str, raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    summary.add(name, "rows_raw", len(raw), "rows in the raw file")
    summary.add(name, "rows_clean", len(clean), "rows after the cleaning rules in cleaning_log.csv")
    summary.add(name, "rows_removed", len(raw) - len(clean))
    summary.add(name, "no_show_n_clean", int(clean[data.TARGET].sum()))
    summary.add(name, "no_show_rate_clean", clean[data.TARGET].mean(), "share of cleaned rows with no_show = 1")


def _split_metrics(summary: Summary, name: str, split: data.HoldoutSplit, total: int) -> None:
    summary.add(name, "n_train_rows", len(split.train), "the locked training part")
    summary.add(name, "n_holdout_rows", len(split.holdout), "the locked holdout: scored once per final model")
    summary.add(name, "holdout_share_rows", len(split.holdout) / total)
    summary.add(name, "train_no_show_rate", split.train[data.TARGET].mean())
    summary.add(name, "holdout_no_show_rate", split.holdout[data.TARGET].mean(),
                "described only; the holdout is not used in the EDA rates")


def profile_kaggle(summary: Summary, raw: pd.DataFrame, clean: pd.DataFrame, seed: int):
    name = "kaggle"
    _basic(summary, name, raw, clean)
    per_patient = clean.groupby("patient_id").size()
    summary.add(name, "unique_patients_clean", clean["patient_id"].nunique())
    summary.add(name, "mean_appointments_per_patient", per_patient.mean())
    summary.add(name, "appointment_date_min", str(clean["appointment_date"].min().date()))
    summary.add(name, "appointment_date_max", str(clean["appointment_date"].max().date()))
    summary.add(name, "n_appointment_dates", clean["appointment_date"].nunique())
    summary.add(name, "share_same_day_booking", (clean["lead_days"] == 0).mean(), "lead_days == 0")
    summary.add(name, "lead_days_median", clean["lead_days"].median())
    summary.add(name, "lead_days_p95", clean["lead_days"].quantile(0.95))
    summary.add(name, "share_sms_received", clean["sms_received"].mean(), "SMS reminder: research feature only")
    summary.add(name, "share_female", (clean["sex"] == "F").mean())
    summary.add(name, "age_mean", clean["age"].mean())
    summary.add(name, "age_median", clean["age"].median())

    split = data.split_holdout_by_date(clean)
    _split_metrics(summary, name, split, len(clean))
    summary.add(name, "train_last_date", str(split.train["appointment_date"].max().date()))
    summary.add(name, "holdout_first_date", str(split.cut.date()))
    summary.add(name, "n_train_dates", split.train["appointment_date"].nunique())
    summary.add(name, "n_holdout_dates", split.holdout["appointment_date"].nunique(),
                f"round({data.HOLDOUT_FRACTION} * distinct dates), cut at a date boundary")

    train = split.train.reset_index(drop=True)
    folds = data.grouped_folds(train["patient_id"], n_splits=5, seed=seed)
    overlap = sum(len(set(train["patient_id"].iloc[t]) & set(train["patient_id"].iloc[v])) for t, v in folds)
    summary.add(name, "cv_n_splits", len(folds), f"GroupKFold on PatientId, seed {seed}, built on the training part")
    summary.add(name, "cv_valid_rows_min", min(len(v) for _, v in folds))
    summary.add(name, "cv_valid_rows_max", max(len(v) for _, v in folds))
    summary.add(name, "cv_patients_in_both_sides", overlap, "patients on both sides of any fold; must be 0")
    return split


def profile_openml(summary: Summary, raw: pd.DataFrame, clean: pd.DataFrame, seed: int):
    name = "openml"
    _basic(summary, name, raw, clean)
    summary.add(name, "appt_month_min", clean["appt_month"].min())
    summary.add(name, "appt_month_max", clean["appt_month"].max())
    summary.add(name, "lead_days_median", clean["lead_days"].median())
    summary.add(name, "lead_days_p95", clean["lead_days"].quantile(0.95))
    summary.add(name, "share_female", (clean["sex"] == "F").mean())
    summary.add(name, "age_mean", clean["age"].mean())
    summary.add(name, "age_median", clean["age"].median())
    summary.add(name, "n_specialties", clean["specialty"].nunique())
    for code, label in ((1, "call_centre"), (2, "personal"), (3, "web")):
        summary.add(name, f"share_channel_{label}", (clean["channel"] == code).mean(), f"canal = {code}")
    summary.add(name, "share_appt_type_procedures", (clean["appt_type"] == 2).mean(), "tipo = 2")

    split = data.split_holdout_last_month(clean)
    _split_metrics(summary, name, split, len(clean))
    summary.add(name, "holdout_month", split.cut, "the last month in the data")

    train = split.train.reset_index(drop=True)
    folds = data.stratified_folds(train[data.TARGET], n_splits=5, seed=seed)
    rates = [train[data.TARGET].iloc[v].mean() for _, v in folds]
    summary.add(name, "cv_n_splits", len(folds), f"stratified, seed {seed}, built on the training part")
    summary.add(name, "cv_valid_rows_min", min(len(v) for _, v in folds))
    summary.add(name, "cv_valid_rows_max", max(len(v) for _, v in folds))
    summary.add(name, "cv_valid_no_show_rate_min", min(rates))
    summary.add(name, "cv_valid_no_show_rate_max", max(rates))
    return split


def profile_hangu(summary: Summary, raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    name = "hangu"
    summary.add(name, "rows_raw", len(raw), "rows in Data.csv")
    summary.add(name, "rows_clean", len(clean))
    summary.add(name, "rows_removed", len(raw) - len(clean))
    summary.add(name, "sessions", clean["session"].nunique(), "half-day sessions")
    summary.add(name, "unique_patient_codes", clean["patient_code"].nunique())
    seconds = clean["service_seconds"]
    summary.add(name, "service_seconds_min", seconds.min(),
                "Data.csv holds no value below this: shorter consultations were removed by the dataset's authors")
    summary.add(name, "service_seconds_median", seconds.median())
    summary.add(name, "service_seconds_mean", seconds.mean())
    summary.add(name, "service_seconds_p95", seconds.quantile(0.95))
    summary.add(name, "service_seconds_max", seconds.max())
    minutes = clean["service_minutes"]
    summary.add(name, "service_minutes_median", minutes.median())
    summary.add(name, "service_minutes_mean", minutes.mean())
    summary.add(name, "service_minutes_p95", minutes.quantile(0.95))


# ---- figures --------------------------------------------------------------------------------------
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")  # solid hairline
    ax.set_axisbelow(True)


def _frame_figure(title: str, subtitle: str, note: str):
    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.70, bottom=0.19)
    fig.text(0.09, 0.955, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.09, 0.895, subtitle, fontsize=9, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)
    fig.text(0.09, 0.025, note, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom", linespacing=1.5)
    _style(ax)
    return fig, ax


def _legend(fig, datasets) -> None:
    """Legend in the header row, above the plot, so it can never sit on top of a bar."""
    handles = [Patch(facecolor=SERIES[d], label=DATASET_LABELS[d]) for d in datasets]
    fig.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.085, 0.775), ncol=len(handles),
               fontsize=8.5, labelcolor=INK_SECONDARY, handlelength=1.0, handleheight=1.0, columnspacing=1.8)


def grouped_rate_figure(rates: pd.DataFrame, variable: str, order: list[str], title: str, xlabel: str,
                        subtitle: str, path: Path) -> None:
    datasets = [d for d in ("kaggle", "openml") if d in set(rates["dataset"])]
    parts = {d: rates[(rates["dataset"] == d) & (rates["variable"] == variable)].set_index("group").reindex(order)
             for d in datasets}
    hidden = [f"{DATASET_LABELS[d].split(' ')[0]} {g} (n={int(parts[d].loc[g, 'n'])})"
              for d in datasets for g in order if parts[d].loc[g, "n"] < MIN_GROUP_N]  # NaN n (no rows) is not listed
    note = "Whiskers: 95% Wilson interval. Group sizes and exact rates: results/e1_eda_rates.csv"
    if hidden:
        note += f"\nNot drawn, fewer than {MIN_GROUP_N} appointments: " + ", ".join(hidden) + "."
    fig, ax = _frame_figure(title, subtitle, note)

    positions = np.arange(len(order))
    width = 0.30
    top = 0.0
    for i, dataset in enumerate(datasets):
        part = parts[dataset]
        drawn = (part["n"] >= MIN_GROUP_N).to_numpy()
        if not drawn.any():  # every group in this series is too small to draw
            continue
        x = positions + (i - (len(datasets) - 1) / 2) * (width + 0.04)
        ax.bar(x[drawn], part["rate"][drawn], width, color=SERIES[dataset], zorder=2)
        ax.errorbar(x[drawn], part["rate"][drawn],
                    yerr=[(part["rate"] - part["ci_low"])[drawn], (part["ci_high"] - part["rate"])[drawn]],
                    fmt="none", ecolor=INK_SECONDARY, elinewidth=0.9, capsize=2, zorder=3)
        top = max(top, float(part["ci_high"][drawn].max()))
        peak = part["rate"].where(drawn).idxmax()  # label only the highest drawn bar of each series
        ax.annotate(f"{part.loc[peak, 'rate']:.0%}", (x[order.index(peak)], part.loc[peak, "ci_high"]),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(positions, order)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(0, top * 1.15 if top else 1.0)
    _legend(fig, datasets)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def overall_figure(rates: pd.DataFrame, path: Path) -> None:
    overall = rates[rates["variable"] == "overall"].set_index("dataset")
    datasets = [d for d in ("kaggle", "openml") if d in overall.index]
    fig, ax = _frame_figure("Overall no-show rate", "All cleaned appointments in each dataset (positive class: no-show = 1)",
                            "Whiskers: 95% Wilson interval. Counts and exact rates: results/e1_eda_rates.csv")
    fig.subplots_adjust(top=0.82)  # no legend row here, so the plot can start higher
    x = np.arange(len(datasets))
    ax.set_xlim(-0.7, len(datasets) - 0.3)
    for i, dataset in enumerate(datasets):
        row = overall.loc[dataset]
        ax.bar(i, row["rate"], 0.22, color=SERIES[dataset], zorder=2)
        ax.errorbar(i, row["rate"], yerr=[[row["rate"] - row["ci_low"]], [row["ci_high"] - row["rate"]]], fmt="none",
                    ecolor=INK_SECONDARY, elinewidth=0.9, capsize=2, zorder=3)
        ax.annotate(f"{row['rate']:.1%}", (i, row["ci_high"]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{DATASET_LABELS[d]}\nn = {int(overall.loc[d, 'n']):,} appointments" for d in datasets])
    ax.set_ylim(0, float(overall["ci_high"].max()) * 1.18)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---- the experiment -------------------------------------------------------------------------------
FIGURES = {
    "e1_noshow_by_lead_time.png": ("lead_time_band", LEAD_BANDS[1], "No-show rate by lead time",
                                   "Days between booking and appointment"),
    "e1_noshow_by_weekday.png": ("weekday", WEEKDAYS, "No-show rate by appointment weekday", "Appointment weekday"),
    "e1_noshow_by_age_band.png": ("age_band", AGE_BANDS[1], "No-show rate by age band", "Age (years)"),
}


def run(kaggle_path=data.KAGGLE_PATH, openml_path=data.OPENML_PATH, hangu_path=data.HANGU_PATH,
        results_dir=data.RESULTS_DIR, seed: int = data.DEFAULT_SEED, repo_root=data.ROOT) -> dict:
    """Run E1 and write every output. Returns the tables so callers (and tests) can inspect them."""
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw = {"kaggle": data.load_kaggle(kaggle_path), "openml": data.load_openml(openml_path),
           "hangu": data.load_hangu(hangu_path)}
    log = data.CleaningLog()
    clean = {"kaggle": data.clean_kaggle(raw["kaggle"], log), "openml": data.clean_openml(raw["openml"], log),
             "hangu": data.clean_hangu(raw["hangu"], log)}

    summary = Summary()
    splits = {"kaggle": profile_kaggle(summary, raw["kaggle"], clean["kaggle"], seed),
              "openml": profile_openml(summary, raw["openml"], clean["openml"], seed)}
    profile_hangu(summary, raw["hangu"], clean["hangu"])

    rates = []
    for name, split in splits.items():
        rates += eda_rates(split.train, name)  # training part only
        everything = clean[name]
        low, high = wilson_interval(int(everything[data.TARGET].sum()), len(everything))
        rates.append({"dataset": name, "variable": "overall", "group": "all cleaned rows", "n": len(everything),
                      "no_show_n": int(everything[data.TARGET].sum()), "rate": everything[data.TARGET].mean(),
                      "ci_low": low, "ci_high": high})
    rates = pd.DataFrame(rates, columns=["dataset", "variable", "group", "n", "no_show_n", "rate", "ci_low", "ci_high"])

    outputs = [log.write(results_dir / "cleaning_log.csv")]
    for table, filename in ((summary.frame(), "dataset_summary.csv"), (rates, "e1_eda_rates.csv")):
        table.to_csv(results_dir / filename, index=False)
        outputs.append(results_dir / filename)

    subtitle = ("Training part only, the holdout is not used.\n"
                "Kaggle: earlier 80% of appointment dates.  OpenML: months before the last.")
    for filename, (variable, order, title, xlabel) in FIGURES.items():
        grouped_rate_figure(rates, variable, order, title, xlabel, subtitle, figures_dir / filename)
        outputs.append(figures_dir / filename)
    overall_figure(rates, figures_dir / "e1_overall_noshow_rate.png")
    outputs.append(figures_dir / "e1_overall_noshow_rate.png")

    inputs = [kaggle_path, openml_path, hangu_path]
    for output in outputs:
        record_result(output, script=SCRIPT, seed=seed, data_files=inputs, repo_root=repo_root,
                      manifest_path=results_dir / "manifest.json")
    return {"log": log.to_frame(), "summary": summary.frame(), "rates": rates, "outputs": outputs}


def main() -> None:
    result = run()
    summary = result["summary"]
    print("E1 done. Key numbers (all from results/):")
    for name in ("kaggle", "openml", "hangu"):
        part = summary[summary["dataset"] == name].set_index("metric")["value"]
        line = f"  {name:7s} rows raw {int(part['rows_raw']):>7,} -> clean {int(part['rows_clean']):>7,} (removed {int(part['rows_removed'])})"
        if "no_show_rate_clean" in part:
            line += f"   no-show rate {float(part['no_show_rate_clean']):.2%}"
        print(line)
    print("  outputs:", ", ".join(str(p.relative_to(data.ROOT)) for p in result["outputs"]))


if __name__ == "__main__":
    main()
