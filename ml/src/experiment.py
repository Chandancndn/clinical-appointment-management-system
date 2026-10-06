"""Shared plumbing for the research experiments E2 to E8: data preparation (locked holdout, folds) and results.

prepare_kaggle() / prepare_openml() load and clean the raw data, build the features and apply the split:
  * Kaggle: history features first (leak-free, see features.add_history), then the time-based holdout (latest 20%
    of appointment dates) and patient-grouped folds built on the TRAINING part only.
  * OpenML: the last month is the holdout, stratified folds on the rest (no patient id).
Tests pass `clean=` (and `source_files=`) to run the same code on small synthetic frames.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import data, features, train
from .data import DEFAULT_SEED
from .manifest import record_result

KAGGLE_INPUTS = [data.KAGGLE_PATH]
OPENML_INPUTS = [data.OPENML_PATH]


@dataclass
class Prepared:
    name: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    groups_train: Optional[np.ndarray]
    X_holdout: pd.DataFrame
    y_holdout: np.ndarray
    groups_holdout: Optional[np.ndarray]
    folds: list
    cut: object
    train_frame: pd.DataFrame
    holdout_frame: pd.DataFrame
    source_files: list

    @property
    def train_dates(self):
        return self.train_frame["appointment_date"] if "appointment_date" in self.train_frame else None

    @property
    def holdout_dates(self):
        return self.holdout_frame["appointment_date"] if "appointment_date" in self.holdout_frame else None


def prepare_kaggle(clean: Optional[pd.DataFrame] = None, seed: int = DEFAULT_SEED, source_files=None,
                   n_splits: int = 5) -> Prepared:
    if clean is None:
        clean = data.clean_kaggle(data.load_kaggle())
        source_files = KAGGLE_INPUTS if source_files is None else source_files
    frame = features.add_history(clean)  # outcomes known at booking time only; computed before any split
    split = data.split_holdout_by_date(frame)
    train_frame, holdout_frame = split.train.reset_index(drop=True), split.holdout.reset_index(drop=True)
    folds = data.grouped_folds(train_frame["patient_id"], n_splits=n_splits, seed=seed)
    return Prepared(
        name="kaggle", X_train=features.kaggle_features(train_frame), y_train=train_frame["no_show"].to_numpy(),
        groups_train=train_frame["patient_id"].to_numpy(), X_holdout=features.kaggle_features(holdout_frame),
        y_holdout=holdout_frame["no_show"].to_numpy(), groups_holdout=holdout_frame["patient_id"].to_numpy(),
        folds=folds, cut=split.cut, train_frame=train_frame, holdout_frame=holdout_frame,
        source_files=list(source_files or []))


def prepare_openml(clean: Optional[pd.DataFrame] = None, seed: int = DEFAULT_SEED, source_files=None,
                   n_splits: int = 5) -> Prepared:
    if clean is None:
        clean = data.clean_openml(data.load_openml())
        source_files = OPENML_INPUTS if source_files is None else source_files
    split = data.split_holdout_last_month(clean)
    train_frame, holdout_frame = split.train.reset_index(drop=True), split.holdout.reset_index(drop=True)
    folds = data.stratified_folds(train_frame["no_show"], n_splits=n_splits, seed=seed)
    return Prepared(
        name="openml", X_train=features.openml_features(train_frame), y_train=train_frame["no_show"].to_numpy(),
        groups_train=None, X_holdout=features.openml_features(holdout_frame),
        y_holdout=holdout_frame["no_show"].to_numpy(), groups_holdout=None, folds=folds, cut=split.cut,
        train_frame=train_frame, holdout_frame=holdout_frame, source_files=list(source_files or []))


# ---- results ----------------------------------------------------------------------------------------
class ResultWriter:
    """Writes a result file and records it in results/manifest.json (seed, raw-data hashes, date, commit)."""

    def __init__(self, results_dir, script: str, seed: int, inputs, repo_root=data.ROOT) -> None:
        self.results_dir, self.script, self.seed = Path(results_dir), script, seed
        self.inputs, self.repo_root = [Path(p) for p in inputs], repo_root
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _record(self, path: Path) -> Path:
        record_result(path, script=self.script, seed=self.seed, data_files=self.inputs, repo_root=self.repo_root,
                      manifest_path=self.results_dir / "manifest.json")
        return path

    def table(self, frame: pd.DataFrame, name: str) -> Path:
        path = self.results_dir / name
        frame.to_csv(path, index=False)
        return self._record(path)

    def json(self, obj: dict, name: str) -> Path:
        path = self.results_dir / name
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return self._record(path)

    def figure_path(self, name: str) -> Path:
        (self.results_dir / "figures").mkdir(exist_ok=True)
        return self.results_dir / "figures" / name

    def record_figure(self, path: Path) -> Path:
        return self._record(path)


def metric_columns() -> list[str]:
    from .evaluate import METRIC_NAMES

    return [c for m in METRIC_NAMES for c in (m, f"{m}_lo", f"{m}_hi")]


def ordered(frame: pd.DataFrame, leading: list[str]) -> pd.DataFrame:
    """Put `leading` columns first, then n/n_positive/n_boot, then the metric triples, then everything else."""
    head = [c for c in leading + ["n", "n_positive", "n_boot"] + metric_columns() if c in frame.columns]
    return frame[head + [c for c in frame.columns if c not in head]]


def load_selected(results_dir) -> dict:
    """The model E3 chose by cross-validation (E4, E7 and E8 build on it)."""
    path = Path(results_dir) / "e3_selected.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run E3 (python -m ml.src.e3_kaggle_models) first")
    return json.loads(path.read_text())


def tune_and_fit(model: str, prep: Prepared, columns, grids=None, seed: int = DEFAULT_SEED):
    """Tune on the training part (grouped or stratified CV), then fit the final model on all training rows."""
    grid = (grids or train.GRIDS).get(model, train.GRIDS[model])
    tuned = train.tune(model, prep.X_train, prep.y_train, prep.folds, columns, grid=grid, seed=seed)
    fitted = train.fit_final(model, prep.X_train, prep.y_train, columns, tuned.best_params, seed)
    return tuned, fitted


def holdout_probabilities(fitted, prep: Prepared, columns) -> np.ndarray:
    return fitted.predict_proba(prep.X_holdout[list(columns)])[:, 1]
