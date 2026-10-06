"""
Uncertainty quantification arms: bootstrap ensemble and Bayesian logistic regression.

Both return a (members x samples) matrix of predicted probabilities per split,
from which every uncertainty metric is derived. Each bootstrap member has its own
pre-drawn seed, so serial and parallel fitting give identical results.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample
from threadpoolctl import threadpool_limits

from .config import ExperimentConfig
from .utils import log_elapsed

logger = logging.getLogger(__name__)


@dataclass
class MemberPredictions:
    """Per-member positive-class probabilities on one evaluation split."""

    probs: np.ndarray  # (n_members, n_samples)
    split: str
    method: str
    outcome_ratios: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.probs.ndim != 2 or self.probs.shape[0] < 2:
            raise ValueError("probs must be (n_members, n_samples) with n_members >= 2")


def bootstrap_ensemble(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    evaluation_sets: dict[str, pd.DataFrame],
    cfg: ExperimentConfig,
    method: str = "ensemble",
) -> dict[str, MemberPredictions]:
    """Fit `n_bootstrap` resampled copies of `model` and predict on every evaluation set."""
    ec = cfg.ensemble
    n_samples = ec.sample_size or len(y_train)
    rng_seeds = [int(s) for s in np.random.default_rng(cfg.seed).integers(0, 2**31 - 1, ec.n_bootstrap)]
    args = (model, X_train, y_train, evaluation_sets, n_samples, ec)
    splits = list(evaluation_sets)

    label = f"Bootstrap ensemble ({method}, B={ec.n_bootstrap}"
    with log_elapsed(f"{label}, {'parallel' if ec.parallel else 'serial'})"):
        runner = _fit_members_parallel if ec.parallel else _fit_members_serial
        members = runner(rng_seeds, *args)

    if ec.parallel and ec.verify_parallel:
        _assert_matches_serial(members, rng_seeds, *args)

    ratios = np.asarray([m[1] for m in members])
    logger.info("Bootstrap positive-class ratio: mean=%.4f  sd=%.4f", ratios.mean(), ratios.std())
    return {
        split: MemberPredictions(
            np.vstack([preds[split] for preds, _ in members]), split, method, ratios
        )
        for split in splits
    }


def _fit_member(
    seed: int,
    model: BaseEstimator,
    X_train,
    y_train,
    evaluation_sets: dict[str, pd.DataFrame],
    n_samples: int,
    ec,
) -> tuple[dict[str, np.ndarray], float]:
    """Fit one ensemble member on its bootstrap resample and score every evaluation split."""
    X_b, y_b = _draw(X_train, y_train, n_samples, seed, ec.stratify, ec.max_redraws)
    member = clone(model)
    member.fit(X_b, y_b)
    preds = {s: member.predict_proba(X)[:, 1] for s, X in evaluation_sets.items()}
    return preds, float(np.mean(y_b))


def _fit_members_serial(seeds: list[int], *args) -> list[tuple[dict[str, np.ndarray], float]]:
    """Fit members one at a time, logging progress every tenth of the ensemble."""
    out = []
    step = max(1, len(seeds) // 10)
    for b, seed in enumerate(seeds):
        out.append(_fit_member(seed, *args))
        if (b + 1) % step == 0:
            logger.info("  fitted %d/%d ensemble members", b + 1, len(seeds))
    return out


def _fit_members_parallel(seeds: list[int], *args) -> list[tuple[dict[str, np.ndarray], float]]:
    """Fit members in worker processes, one contiguous block of seeds per worker, preserving order."""
    from joblib import Parallel, delayed

    model, X_train, y_train, evaluation_sets, n_samples, ec = args
    n_jobs = ec.n_jobs if ec.n_jobs != -1 else (os.cpu_count() or 1)
    _warn_if_memory_heavy(X_train, n_jobs)

    chunks = [list(c) for c in np.array_split(np.asarray(seeds), min(n_jobs, len(seeds)))]

    logger.info(
        "Fitting %d members across %d worker processes (%d chunks)",
        len(seeds), n_jobs, len(chunks),
    )
    # null/none/0 leaves BLAS thread pools untouched.
    limit = ec.blas_threads_per_worker
    if isinstance(limit, str):
        limit = None if limit.lower() in {"null", "none", ""} else int(limit)
    with (threadpool_limits(limits=limit) if limit else nullcontext()):
        results = Parallel(n_jobs=ec.n_jobs, backend="loky", verbose=5, prefer="processes")(
            delayed(_fit_chunk)(chunk, model, X_train, y_train, evaluation_sets, n_samples, ec)
            for chunk in chunks
        )
    return [member for chunk_result in results for member in chunk_result]


def _fit_chunk(seeds: list[int], *args) -> list[tuple[dict[str, np.ndarray], float]]:
    """Fit a contiguous block of members inside one worker, in seed order."""
    return [_fit_member(seed, *args) for seed in seeds]


def _warn_if_memory_heavy(X_train, n_jobs: int, budget_gb: float = 8.0) -> None:
    """Warn when the training matrix will be copied into every worker."""
    nbytes = getattr(X_train, "memory_usage", lambda **_: None)(deep=True)
    nbytes = int(nbytes.sum()) if nbytes is not None else int(np.asarray(X_train).nbytes)
    total_gb = nbytes * n_jobs / 1e9
    if total_gb > budget_gb:
        logger.warning(
            "Training matrix is %.2f GB and is copied per worker: ~%.1f GB across %d jobs. "
            "Reduce ensemble.n_jobs if this run is memory-bound.",
            nbytes / 1e9, total_gb, n_jobs,
        )


def _assert_matches_serial(members, seeds: list[int], *args) -> None:
    """Refit serially and require exact equality; raises rather than warns."""
    logger.info("verify_parallel: refitting serially to check for exact agreement")
    reference = _fit_members_serial(seeds, *args)
    for b, ((par, r_par), (ser, r_ser)) in enumerate(zip(members, reference)):
        for split in par:
            if not np.array_equal(par[split], ser[split]):
                delta = float(np.max(np.abs(par[split] - ser[split])))
                raise AssertionError(
                    f"parallel and serial ensembles differ at member {b}, split '{split}' "
                    f"(max |difference| = {delta:.3e})"
                )
        if r_par != r_ser:
            raise AssertionError(f"outcome ratio differs at member {b}: {r_par} vs {r_ser}")
    logger.info("verify_parallel: %d members match the serial fit exactly", len(members))


def _draw(X, y, n_samples: int, seed: int, stratify: bool, max_redraws: int):
    """Draw one bootstrap resample, redrawing if it is single-class."""
    for attempt in range(max_redraws):
        X_b, y_b = resample(
            X,
            y,
            n_samples=n_samples,
            replace=True,
            random_state=seed + attempt,
            stratify=y if stratify else None,
        )
        if len(np.unique(y_b)) > 1:
            if attempt:
                logger.warning("Redrew bootstrap sample %d time(s): single-class draw", attempt)
            return X_b, y_b
    raise RuntimeError(
        "bootstrap resample was single-class after "
        f"{max_redraws} redraws; set ensemble.stratify=true"
    )


def bnn_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    evaluation_sets: dict[str, pd.DataFrame],
    cfg: ExperimentConfig,
) -> dict[str, MemberPredictions]:
    """Fit the Bayesian logistic regression and draw `num_samples` posterior predictive samples."""
    from .bnn_vi import fit_bnn, predict_bnn

    X = X_train.to_numpy(dtype=np.float32) if hasattr(X_train, "to_numpy") else np.asarray(
        X_train, dtype=np.float32
    )
    net = fit_bnn(X, np.asarray(y_train), cfg.bnn, seed=cfg.seed)

    out: dict[str, MemberPredictions] = {}
    for split, X_eval in evaluation_sets.items():
        Xe = X_eval.to_numpy(dtype=np.float32) if hasattr(X_eval, "to_numpy") else np.asarray(
            X_eval, dtype=np.float32
        )
        probs = predict_bnn(net, Xe, cfg.bnn, seed=cfg.seed)
        out[split] = MemberPredictions(probs, split, "bnn")
    return out
