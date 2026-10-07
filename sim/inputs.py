"""Simulation inputs (PLAN section 5): the consultation-time distribution and the patient-risk pool.

Consultation time: lognormal and gamma are fitted by maximum likelihood to Hangu's service times in minutes and the
family with the lower AIC is used; QQ plots are drawn for both so the choice can be checked by eye. Variability can be
scaled at the SAME MEAN (S5): the standard deviation is multiplied by `variability`, the mean is kept.

Patient risk: the Kaggle holdout bookings, scored by the saved deployable model (never retrained here). The model
file's hash is checked against its model card before use.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from ml.src.plotting import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE  # noqa: E402

FAMILY_COLORS = {"lognormal": "#2a78d6", "gamma": "#eb6834"}  # categorical slots 1 and 2 (validated palette)
_V_MIN = 1e-12


# ---- defaults computed from data, never typed ----------------------------------------------------------
def default_slot_minutes(median_service_minutes: float) -> int:
    """Hangu's median consultation time rounded up to the next 5 minutes."""
    return int(math.ceil(median_service_minutes / 5.0) * 5)


def session_minutes_from_seed() -> int:
    """Length of the seeded app sessions (db/seed_synthetic.py: slots from SESSION[0] to SESSION[1])."""
    from db import seed_synthetic

    start, end = seed_synthetic.SESSION
    return (datetime.combine(date.min, end) - datetime.combine(date.min, start)).seconds // 60


# ---- the consultation-time model ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ServiceTimeModel:
    """family 'lognormal' (params mu, sigma of ln x) or 'gamma' (params shape, scale)."""

    family: str
    params: dict

    @property
    def mean(self) -> float:
        if self.family == "lognormal":
            return math.exp(self.params["mu"] + self.params["sigma"] ** 2 / 2)
        return self.params["shape"] * self.params["scale"]

    def ppf(self, v, variability: float = 1.0) -> np.ndarray:
        """Quantiles (minutes) for uniforms `v`. variability multiplies the coefficient of variation (so the standard
        deviation) while the mean stays the same."""
        v = np.clip(np.asarray(v, dtype=float), _V_MIN, 1 - _V_MIN)
        if self.family == "lognormal":
            cv_squared = math.expm1(self.params["sigma"] ** 2)  # CV^2 of a lognormal is e^(sigma^2) - 1
            sigma = math.sqrt(math.log1p(variability ** 2 * cv_squared))  # so the CV is scaled by exactly `variability`
            mu = math.log(self.mean) - sigma ** 2 / 2  # keeps the mean
            return np.exp(mu + sigma * stats.norm.ppf(v))
        shape = self.params["shape"] / variability ** 2  # the coefficient of variation scales with `variability`
        return stats.gamma.ppf(v, shape, scale=self.mean / shape)


@dataclass(frozen=True)
class FamilyFit:
    family: str
    params: dict
    log_likelihood: float
    aic: float
    ks_statistic: float
    mean: float

    def model(self) -> ServiceTimeModel:
        return ServiceTimeModel(self.family, self.params)


@dataclass
class FitResult:
    fits: list
    chosen: FamilyFit
    minutes: np.ndarray

    @property
    def model(self) -> ServiceTimeModel:
        return self.chosen.model()

    def table(self) -> pd.DataFrame:
        rows = []
        for fit in self.fits:
            (n1, v1), (n2, v2) = list(fit.params.items())
            rows.append({"family": fit.family, "param_1_name": n1, "param_1": v1, "param_2_name": n2, "param_2": v2,
                         "log_likelihood": fit.log_likelihood, "aic": fit.aic, "ks_statistic": fit.ks_statistic,
                         "mean_fitted": fit.mean, "mean_data": float(self.minutes.mean()), "n": len(self.minutes),
                         "chosen": fit is self.chosen})
        return pd.DataFrame(rows)


def fit_service_times(minutes) -> FitResult:
    """Maximum-likelihood lognormal and gamma (location fixed at 0, two parameters each); the lower AIC wins."""
    x = np.asarray(minutes, dtype=float)
    if (x <= 0).any():
        raise ValueError("service times must be positive")
    log_x = np.log(x)
    mu, sigma = float(log_x.mean()), float(log_x.std(ddof=0))  # the exact lognormal MLE
    ll_lognormal = float(np.sum(stats.norm.logpdf(log_x, mu, sigma) - log_x))
    shape, _, scale = stats.gamma.fit(x, floc=0)
    ll_gamma = float(stats.gamma.logpdf(x, shape, scale=scale).sum())
    fits = []
    for family, params, ll, cdf in (
        ("lognormal", {"mu": mu, "sigma": sigma}, ll_lognormal, lambda v: stats.norm.cdf(np.log(v), mu, sigma)),
        ("gamma", {"shape": float(shape), "scale": float(scale)}, ll_gamma, lambda v: stats.gamma.cdf(v, shape, scale=scale)),
    ):
        model = ServiceTimeModel(family, params)
        fits.append(FamilyFit(family, params, ll, 2 * 2 - 2 * ll, float(stats.kstest(x, cdf).statistic), model.mean))
    return FitResult(fits=fits, chosen=min(fits, key=lambda f: f.aic), minutes=x)


# ---- figures: QQ plots and the fitted densities ---------------------------------------------------------------
def _axes_style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def qq_figure(fit: FitResult, path) -> None:
    """Data quantiles against each fitted distribution's quantiles; the grey line is a perfect fit."""
    x = np.sort(fit.minutes)
    probabilities = (np.arange(len(x)) + 0.5) / len(x)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.9), dpi=200, facecolor=SURFACE, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.17, wspace=0.08)
    fig.text(0.08, 0.955, "QQ plots of Hangu consultation times", fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.08, 0.895, f"Each point is one quantile: points on the grey line mean the fitted distribution matches. "
             f"Chosen by AIC: {fit.chosen.family}.", fontsize=9, color=INK_SECONDARY, ha="left", va="top")
    limit = float(x.max()) * 1.04
    for ax, family_fit in zip(axes, fit.fits):
        model = family_fit.model()
        _axes_style(ax)
        ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1.0, zorder=1)
        ax.plot(model.ppf(probabilities), x, linestyle="none", marker="o", markersize=2.2, color=FAMILY_COLORS[family_fit.family],
                zorder=3)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        chosen = " (chosen)" if family_fit is fit.chosen else ""
        ax.set_title(f"{family_fit.family}{chosen}: AIC {family_fit.aic:,.1f}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Fitted quantile (minutes)", color=INK_SECONDARY, fontsize=9)
    axes[0].set_ylabel("Observed quantile (minutes)", color=INK_SECONDARY, fontsize=9)
    fig.text(0.08, 0.03, "Hangu Data.csv, service time in minutes. The data has no consultation shorter than 3 minutes "
             "(removed upstream), so the lowest quantiles are cut off.", fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def density_figure(fit: FitResult, path) -> None:
    """Histogram of the data with both fitted densities."""
    x = fit.minutes
    grid = np.linspace(max(x.min() * 0.8, 0.01), float(np.percentile(x, 99.5)), 400)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.16)
    fig.text(0.1, 0.955, "Hangu consultation times and the two fits", fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.1, 0.895, f"Median {np.median(x):.1f} min, mean {x.mean():.1f} min, n = {len(x):,}. Chosen by AIC: {fit.chosen.family}.",
             fontsize=9, color=INK_SECONDARY, ha="left", va="top")
    _axes_style(ax)
    ax.hist(x, bins=60, range=(0, float(grid.max())), density=True, color=GRID, edgecolor=SURFACE, zorder=2)
    for family_fit in fit.fits:
        p = family_fit.params
        density = (stats.lognorm.pdf(grid, p["sigma"], scale=math.exp(p["mu"])) if family_fit.family == "lognormal"
                   else stats.gamma.pdf(grid, p["shape"], scale=p["scale"]))
        ax.plot(grid, density, color=FAMILY_COLORS[family_fit.family], linewidth=1.8, zorder=3,
                label=f"{family_fit.family} (AIC {family_fit.aic:,.0f})")
    ax.set_xlabel("Consultation time (minutes)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Density", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5, labelcolor=INK_SECONDARY)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---- the patient-risk pool --------------------------------------------------------------------------------------
def load_risk_pool(artifacts_dir=None):
    """Predicted no-show risk of every Kaggle holdout booking, from the saved deployable model.

    Returns (risk array, input files for the manifest). The model is loaded, never retrained; its SHA-256 must match
    the model card, otherwise the artifact and its documentation have drifted apart and the run stops.
    """
    import joblib

    from ml.src import experiment

    artifacts = Path(artifacts_dir) if artifacts_dir else experiment.ARTIFACTS_DIR
    model_path, card_path = artifacts / "risk_model.joblib", artifacts / "model_card.json"
    card = json.loads(card_path.read_text())
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != card["artifact_sha256"]:
        raise RuntimeError("ml/artifacts/risk_model.joblib does not match model_card.json; re-run E9 and E10")
    prep = experiment.prepare_kaggle()
    _, X_holdout = experiment.deployable_data(prep)
    risk = joblib.load(model_path).predict_proba(X_holdout)[:, 1]
    return risk, [*prep.source_files, model_path]
