"""
Geometry adjustment: SD_perp = SD_logit - s(logit p-bar), where s is a natural
cubic spline fitted on the validation split and applied to test.

`standardize=True` additionally divides the residual by a fitted conditional scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import OrthogonalizationConfig
from .utils import EPS, clip_proba, logit

logger = logging.getLogger(__name__)


class NaturalCubicSpline:
    """Natural cubic regression spline (ESL S5.2.1); linear beyond the boundary knots."""

    def __init__(self, n_knots: int = 5):
        if n_knots < 3:
            raise ValueError("a natural cubic spline needs at least 3 knots")
        self.n_knots = n_knots
        self.knots_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def _basis(self, x: np.ndarray) -> np.ndarray:
        k = self.knots_
        K = len(k)

        def d(j: int) -> np.ndarray:
            return (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[-1], 0) ** 3) / (k[-1] - k[j])

        cols = [np.ones_like(x), x]
        d_last = d(K - 2)
        cols += [d(j) - d_last for j in range(K - 2)]
        return np.column_stack(cols)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NaturalCubicSpline":
        """Place knots at equally spaced quantiles of x and solve by least squares."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        quantiles = np.linspace(0, 100, self.n_knots)
        knots = np.unique(np.percentile(x, quantiles))
        if len(knots) < 3:  # near-degenerate predictor (e.g. constant p-bar)
            knots = np.linspace(x.min(), x.max() + EPS, 3)
            logger.warning("Spline predictor is near-degenerate; using 3 evenly spaced knots")
        self.knots_ = knots
        self.coef_, *_ = np.linalg.lstsq(self._basis(x), y, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("spline is not fitted")
        return self._basis(np.asarray(x, float)) @ self.coef_


@dataclass
class OrthogonalizedSD:
    """Fitted trend E[SD_logit | p-bar] (`location`) and, optionally, its conditional scale."""

    cfg: OrthogonalizationConfig
    location: object = None
    scale: object | None = None
    fit_n: int = 0
    fit_r2: float = float("nan")
    _extras: dict = field(default_factory=dict)

    def _x(self, mean_proba: np.ndarray) -> np.ndarray:
        """Spline predictor: logit(p-bar) by default, a monotone reparameterisation of p-bar."""
        p = clip_proba(mean_proba)
        return logit(p) if self.cfg.predictor == "logit" else p

    def _smoother(self):
        return NaturalCubicSpline(self.cfg.n_knots)

    def fit(self, mean_proba: np.ndarray, sd_logit: np.ndarray) -> "OrthogonalizedSD":
        """Fit the trend (and optional scale) of SD_logit on p-bar."""
        x, y = self._x(mean_proba), np.asarray(sd_logit, float)
        self.location = self._smoother().fit(x, y)
        resid = y - self.location.predict(x)

        var_y = float(np.var(y))
        self.fit_r2 = 1.0 - float(np.var(resid)) / var_y if var_y > 0 else float("nan")
        self.fit_n = len(y)

        if self.cfg.standardize:
            log_abs = np.log(np.abs(resid) + EPS)
            self.scale = self._smoother().fit(x, log_abs)

        logger.info(
            "Orthogonalisation fitted on n=%d (spline, %d knots): R^2 of p-bar on SD_logit = %.3f",
            self.fit_n,
            self.cfg.n_knots,
            self.fit_r2,
        )
        return self

    def transform(self, mean_proba: np.ndarray, sd_logit: np.ndarray) -> np.ndarray:
        """Return SD_perp for new data using the fitted trend."""
        if self.location is None:
            raise RuntimeError("OrthogonalizedSD is not fitted")
        x = self._x(mean_proba)
        resid = np.asarray(sd_logit, float) - self.location.predict(x)
        if self.scale is not None:
            resid = resid / (np.exp(self.scale.predict(x)) + EPS)
        return resid

    def expected(self, mean_proba: np.ndarray) -> np.ndarray:
        """Fitted E[SD_logit | p-bar]; the curve overlaid on the diagnostic scatter."""
        return self.location.predict(self._x(mean_proba))
