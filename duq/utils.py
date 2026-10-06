"""
Logging, timing, seeding, and numerically safe probability transforms.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Clipping bound applied before any log/logit transform.
EPS = 1e-12


def configure_logging(level: int = logging.INFO, log_file: str | os.PathLike | None = None) -> None:
    """Configure root logging for entry-point scripts (never called by library modules)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


@contextmanager
def log_elapsed(label: str, level: int = logging.INFO):
    """Context manager that logs wall-clock elapsed time for a named block."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.log(level, "%s - elapsed: %.2fs", label, time.perf_counter() - t0)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is needed only for the Bayesian arm
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_array(x: Any, dtype=float) -> np.ndarray:
    """Convert a pandas/NumPy/list object to a flat NumPy array of the given dtype."""
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asarray(x, dtype=dtype).ravel()


def clip_proba(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip probabilities into (eps, 1 - eps) so log/logit transforms stay finite."""
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Log-odds transform with clipping."""
    p = clip_proba(p, eps)
    return np.log(p) - np.log1p(-p)



def binary_entropy(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Shannon entropy (nats) of Bernoulli(p), evaluated elementwise."""
    p = clip_proba(p, eps)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def save_table(df: pd.DataFrame, path: str | os.PathLike, index: bool = False) -> Path:
    """Write a DataFrame to CSV, creating parent directories, and log the destination."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    logger.info("Table saved -> %s", path)
    return path


def save_json(obj: Any, path: str | os.PathLike) -> Path:
    """Serialise a JSON-compatible object, coercing NumPy scalars to Python types."""
    path = Path(path)
    ensure_dir(path.parent)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not JSON serialisable: {type(o)}")

    path.write_text(json.dumps(obj, indent=2, default=default))
    logger.info("JSON saved -> %s", path)
    return path


def describe_labels(y: Iterable, name: str) -> dict[str, float]:
    """Return and log the size and positive-class prevalence of a label vector."""
    y = as_array(y)
    y = y[~np.isnan(y)]
    n, n_pos = len(y), int((y == 1).sum())
    summary = {
        "split": name,
        "n": n,
        "n_positive": n_pos,
        "n_negative": n - n_pos,
        "prevalence": n_pos / n if n else float("nan"),
    }
    logger.info(
        "%s: n=%d  positives=%d  prevalence=%.4f", name, n, n_pos, summary["prevalence"]
    )
    return summary
