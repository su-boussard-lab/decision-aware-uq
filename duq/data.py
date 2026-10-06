"""
Dataset loading. Each loader returns train/validation/test `Splits`.

    dpr  preprocessed train/validation/test CSV files
    pou  cohort and covariate CSV files (restricted data; the repository ships
         only a synthetic example, which raises `DatasetUnavailable`)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig, ExperimentConfig
from .utils import describe_labels, log_elapsed

logger = logging.getLogger(__name__)

MIN_ROWS_FOR_ANALYSIS = 100


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset's real data is not present on disk."""


@dataclass
class Splits:
    """Model-ready train/validation/test matrices plus optional row metadata."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    name: str
    meta_test: pd.DataFrame | None = None

    def summary(self) -> pd.DataFrame:
        rows = [
            describe_labels(self.y_train, "train"),
            describe_labels(self.y_val, "validation"),
            describe_labels(self.y_test, "test"),
        ]
        logger.info("Feature count: %d", self.X_train.shape[1])
        return pd.DataFrame(rows).assign(n_features=self.X_train.shape[1])


class DatasetLoader(ABC):
    """Abstract loader; subclasses implement `load()` and return `Splits`."""

    key: str = ""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.data_cfg: DataConfig = cfg.data
        self.root = cfg.resolve(cfg.data.root)

    @abstractmethod
    def load(self) -> Splits:
        """Return model-ready splits for this dataset."""

    # -------------------------------------------------------------- helpers
    def _require(self, *paths: Path) -> None:
        missing = [str(p) for p in paths if not Path(p).exists()]
        if missing:
            raise DatasetUnavailable(
                f"{self.key}: required file(s) not found: {', '.join(missing)}"
            )

    def _subsample_train(self, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """Optionally cap training rows (stratified) to bound bootstrap runtime."""
        cap = self.data_cfg.max_train_rows
        if cap is None or len(y) <= cap:
            return X, y
        idx, _ = train_test_split(
            np.arange(len(y)),
            train_size=cap,
            random_state=self.data_cfg.split_seed,
            stratify=y,
        )
        logger.warning("Training set capped at %d rows (from %d)", cap, len(y))
        return X.iloc[idx].reset_index(drop=True), y[idx]


class SplitFileLoader(DatasetLoader):
    """Loader for train/validation/test CSV files; columns are aligned to the training split."""

    label_col: str = ""
    drop_cols: tuple[str, ...] = ()
    file_names: tuple[str, str, str] = ("train.csv", "validation.csv", "test.csv")

    def load(self) -> Splits:
        paths = [self.root / f for f in self.file_names]
        self._require(*paths)
        with log_elapsed(f"Loading {self.key} splits"):
            frames = [pd.read_csv(p) for p in paths]

        for name, df in zip(("train", "validation", "test"), frames):
            if self.label_col not in df.columns:
                raise DatasetUnavailable(
                    f"{self.key}: label column '{self.label_col}' missing from {name} split"
                )

        ys = [df[self.label_col].astype(int).to_numpy() for df in frames]
        Xs = [
            df.drop(columns=[self.label_col, *self.drop_cols], errors="ignore")
            for df in frames
        ]

        train_cols = Xs[0].columns
        Xs = [X.reindex(columns=train_cols) for X in Xs]
        for name, X in zip(("validation", "test"), Xs[1:]):
            n_missing = int(X.isna().any().sum())
            if n_missing:
                logger.warning(
                    "%s: %d feature(s) absent from %s split; filled with 0",
                    self.key,
                    n_missing,
                    name,
                )
        Xs = [X.fillna(0.0).astype(float).reset_index(drop=True) for X in Xs]

        X_train, y_train = self._subsample_train(Xs[0], ys[0])
        return Splits(X_train, y_train, Xs[1], ys[1], Xs[2], ys[2], name=self.key)


class DPRLoader(SplitFileLoader):
    """Diabetic patient 30-day readmission (released preprocessed splits)."""

    key = "dpr"
    label_col = "readmit_30d"


class POULoader(DatasetLoader):
    """
    Prolonged opioid use after surgery (STARR OMOP).

    Expects `cohort.csv` (rowId, outcome, evaluationType) and `dataset.csv`
    (rowId plus covariates) under `data.root`. The validation set is sampled from
    the predefined training partition.
    """

    key = "pou"

    def load(self) -> Splits:
        cohort_path, data_path = self._locate()
        with log_elapsed("Loading POU cohort"):
            cohort = pd.read_csv(cohort_path)
            data = pd.read_csv(data_path)

        if len(cohort) < MIN_ROWS_FOR_ANALYSIS:
            raise DatasetUnavailable(
                f"POU: only {len(cohort)} rows found at {cohort_path}. This is the "
                "synthetic demonstration subset, not the analysis cohort. Set "
                "data.root to the restricted POU data directory to run this arm."
            )

        merged = data.merge(cohort[["rowId", "outcome"]], on="rowId", how="inner")
        y = merged["outcome"].astype(int).to_numpy()
        X = merged.drop(columns=["rowId", "outcome"]).astype(float)

        if "evaluationType" in cohort.columns:
            is_train = (
                cohort.set_index("rowId")["evaluationType"]
                .reindex(merged["rowId"])
                .eq("Train")
                .to_numpy()
            )
        else:  # fall back to a stratified split when no predefined folds exist
            is_train = np.zeros(len(y), dtype=bool)
            tr, _ = train_test_split(
                np.arange(len(y)),
                train_size=1 - self.data_cfg.test_size,
                random_state=self.data_cfg.split_seed,
                stratify=y,
            )
            is_train[tr] = True

        tr_idx = np.flatnonzero(is_train)
        te_idx = np.flatnonzero(~is_train)
        val_frac = self.data_cfg.val_size / (1 - self.data_cfg.test_size)
        tr_idx, va_idx = train_test_split(
            tr_idx,
            test_size=val_frac,
            random_state=self.data_cfg.split_seed,
            stratify=y[tr_idx],
        )

        X_train, y_train = self._subsample_train(
            X.iloc[tr_idx].reset_index(drop=True), y[tr_idx]
        )
        return Splits(
            X_train,
            y_train,
            X.iloc[va_idx].reset_index(drop=True),
            y[va_idx],
            X.iloc[te_idx].reset_index(drop=True),
            y[te_idx],
            name=self.key,
            meta_test=cohort.set_index("rowId")
            .reindex(merged["rowId"].iloc[te_idx])
            .reset_index(),
        )

    def _locate(self) -> tuple[Path, Path]:
        for cohort_name, data_name in (
            ("cohort.csv", "dataset.csv"),
            ("cohort_subset.csv", "data_subset.csv"),
        ):
            cohort, data = self.root / cohort_name, self.root / data_name
            if cohort.exists() and data.exists():
                return cohort, data
        raise DatasetUnavailable(f"POU: no cohort/dataset CSV pair under {self.root}")


DATASET_REGISTRY: dict[str, type[DatasetLoader]] = {
    "dpr": DPRLoader,
    "pou": POULoader,
}


def load_dataset(cfg: ExperimentConfig) -> Splits:
    """Instantiate the registered loader for `cfg.data.dataset` and load the splits."""
    key = cfg.data.dataset.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"unknown dataset '{key}'; available: {sorted(DATASET_REGISTRY)}"
        )
    splits = DATASET_REGISTRY[key](cfg).load()
    splits.summary()
    return splits
