"""
Base classifier.

Models are returned unfitted so the bootstrap can `clone()` them. Standardisation
sits inside the pipeline, so it is refitted on every bootstrap resample.
"""

from __future__ import annotations

import logging
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


def build_lasso(cfg: ExperimentConfig) -> BaseEstimator:
    """L1-penalised logistic regression, optionally preceded by standardisation."""
    params = dict(cfg.model.params)
    params.setdefault("random_state", cfg.seed)
    steps = [("scaler", StandardScaler())] if cfg.model.standardize else []
    steps.append(("clf", LogisticRegression(**params)))
    return Pipeline(steps)


MODEL_REGISTRY: dict[str, Callable[[ExperimentConfig], BaseEstimator]] = {
    "lasso": build_lasso,
}


def build_model(cfg: ExperimentConfig, kind: str | None = None) -> BaseEstimator:
    """Instantiate an unfitted estimator from the model registry."""
    kind = (kind or cfg.model.kind).lower()
    if kind not in MODEL_REGISTRY:
        raise KeyError(f"unknown model kind '{kind}'; available: {sorted(MODEL_REGISTRY)}")
    logger.info("Model: %s", kind)
    return MODEL_REGISTRY[kind](cfg)
