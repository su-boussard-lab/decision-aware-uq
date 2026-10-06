"""
Uncertainty metrics, each percentile-ranked against the validation split and
oriented so that larger means more uncertain.

    sd_logit       SD of member logits                 logit-scale dispersion
    sd_perp        SD_logit - s(logit p-bar)           geometry-adjusted dispersion
    sd_prob        SD of member probabilities          probability-scale dispersion
    null_mi        H(p-bar) - E[H(p_b)]                mutual information
    null_entropy   H(p-bar)                            predictive entropy
    null_dm        -|p-bar - tau|                      decision margin
    null_dm_logit  -|logit p-bar - logit tau|          decision margin (logit scale)

`is_null` marks the information-free baselines (mutual information is grouped
with them; see Methods).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .config import UncertaintyConfig
from .ensemble import MemberPredictions
from .orthogonalize import OrthogonalizedSD
from .utils import as_array, binary_entropy, clip_proba, logit

logger = logging.getLogger(__name__)


def percentile_rank(x: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Empirical CDF of `x` under `reference` (mid-ranks for ties)."""
    x = np.asarray(x, float)
    ref = np.sort(np.asarray(x if reference is None else reference, float))
    n = len(ref)
    if n == 0:
        raise ValueError("reference distribution is empty")
    left = np.searchsorted(ref, x, side="left")
    right = np.searchsorted(ref, x, side="right")
    return (left + right) / (2.0 * n)


@dataclass
class PredictionBundle:
    """Per-sample summaries of a (members x samples) probability matrix on one split."""

    y_true: np.ndarray
    member_probs: np.ndarray
    split: str
    method: str
    mean_proba: np.ndarray = field(init=False)
    mean_logit: np.ndarray = field(init=False)
    sd_prob: np.ndarray = field(init=False)
    sd_logit: np.ndarray = field(init=False)
    entropy: np.ndarray = field(init=False)
    aleatoric_entropy: np.ndarray = field(init=False)
    mutual_information: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        probs = clip_proba(self.member_probs)
        member_logits = logit(probs)
        self.mean_proba = probs.mean(axis=0)
        self.mean_logit = member_logits.mean(axis=0)
        # ddof=1: the ensemble is a sample of models, not the population of models
        self.sd_prob = probs.std(axis=0, ddof=1)
        self.sd_logit = member_logits.std(axis=0, ddof=1)

        self.entropy = binary_entropy(self.mean_proba)
        self.aleatoric_entropy = binary_entropy(probs).mean(axis=0)
        # Non-negative by Jensen; the clip absorbs floating-point error.
        self.mutual_information = np.maximum(self.entropy - self.aleatoric_entropy, 0.0)

        n_members = probs.shape[0]
        if n_members < 30:
            logger.warning(
                "Mutual information is upward-biased with few members (B=%d): sampling "
                "noise inflates apparent disagreement. Treat null_mi as unreliable below "
                "about 30 members.",
                n_members,
            )
        if len(self.y_true) != probs.shape[1]:
            raise ValueError("y_true length does not match the number of scored samples")

    @classmethod
    def from_members(cls, members: MemberPredictions, y_true) -> "PredictionBundle":
        return cls(as_array(y_true, int), members.probs, members.split, members.method)

    @property
    def n(self) -> int:
        return len(self.y_true)

    @property
    def prevalence(self) -> float:
        return float(np.mean(self.y_true))


class UncertaintyMetric(ABC):
    """Base metric: `raw` returns an oriented score, `score` returns its percentile rank."""

    name: str = ""
    label: str = ""
    requires_tau: bool = False
    is_null: bool = False

    @abstractmethod
    def raw(self, bundle: PredictionBundle, tau: float) -> np.ndarray:
        """Raw metric values, oriented so that larger means more uncertain."""

    def score(
        self, bundle: PredictionBundle, reference: PredictionBundle | None, tau: float
    ) -> np.ndarray:
        ref = None if reference is None else self.raw(reference, tau)
        return percentile_rank(self.raw(bundle, tau), ref)


class SdLogitMetric(UncertaintyMetric):
    """SD of member predictions on the logit scale."""

    name, label = "sd_logit", r"SD$_{logit}$"

    def raw(self, bundle, tau):
        return bundle.sd_logit


class SdProbMetric(UncertaintyMetric):
    """SD of member predictions on the probability scale."""

    name, label = "sd_prob", r"SD$_{prob}$ (uncorrected)"

    def raw(self, bundle, tau):
        return bundle.sd_prob


class SdPerpMetric(UncertaintyMetric):
    """SD_logit minus the validation-fitted spline in logit p-bar."""

    name, label = "sd_perp", r"SD$_\perp$"

    def __init__(self, orthogonalizer: OrthogonalizedSD):
        self.orthogonalizer = orthogonalizer

    def raw(self, bundle, tau):
        return self.orthogonalizer.transform(bundle.mean_proba, bundle.sd_logit)


class EntropyNull(UncertaintyMetric):
    """Predictive entropy H(p-bar)."""

    name, label, is_null = "null_entropy", r"Null: $H(\bar{p})$", True

    def raw(self, bundle, tau):
        return bundle.entropy


class DecisionMarginNull(UncertaintyMetric):
    """Decision margin -|p-bar - tau|."""

    name, label = "null_dm", r"Null: DM $=-|\bar{p}-\tau|$"
    requires_tau, is_null = True, True

    def raw(self, bundle, tau):
        return -np.abs(bundle.mean_proba - tau)


class DecisionMarginLogitNull(UncertaintyMetric):
    """Decision margin on the logit scale, -|logit p-bar - logit tau|."""

    name, label = "null_dm_logit", r"DM$_{logit}$"
    requires_tau, is_null = True, True

    def raw(self, bundle, tau):
        return -np.abs(bundle.mean_logit - float(logit(np.array([tau]))[0]))


class MutualInformationNull(UncertaintyMetric):
    """
    Mutual information H(p-bar) - E[H(p_b)]. Grouped with the baselines because
    MI ~= 0.5 p-bar(1 - p-bar) Var(logit), so it scales with p-bar(1 - p-bar).
    """

    name, label, is_null = "null_mi", r"MI (uncorrected)", True

    def raw(self, bundle, tau):
        return bundle.mutual_information


#: Constructible without extra state; `sd_perp` needs a fitted orthogonaliser.
METRIC_REGISTRY: dict[str, type[UncertaintyMetric]] = {
    m.name: m
    for m in (
        SdLogitMetric,
        SdProbMetric,
        EntropyNull,
        DecisionMarginNull,
        DecisionMarginLogitNull,
        MutualInformationNull,
    )
}


class MetricSet:
    """The metrics of one experiment, sharing one fitted orthogonaliser."""

    def __init__(
        self,
        cfg: UncertaintyConfig,
        orthogonalizer: OrthogonalizedSD,
        names: list[str] | None = None,
    ):
        self.cfg = cfg
        self.orthogonalizer = orthogonalizer
        names = names or list(cfg.metrics)
        self.metrics: dict[str, UncertaintyMetric] = {}
        for name in dict.fromkeys(names):
            if name == SdPerpMetric.name:
                self.metrics[name] = SdPerpMetric(orthogonalizer)
            elif name in METRIC_REGISTRY:
                self.metrics[name] = METRIC_REGISTRY[name]()
            else:
                raise KeyError(
                    f"unknown uncertainty metric '{name}'; available: "
                    f"{sorted([*METRIC_REGISTRY, SdPerpMetric.name])}"
                )
        logger.info("Uncertainty metrics: %s", ", ".join(self.metrics))

    @property
    def names(self) -> list[str]:
        return list(self.metrics)

    def labels(self) -> dict[str, str]:
        return {name: m.label for name, m in self.metrics.items()}

    def scores(
        self, bundle: PredictionBundle, reference: PredictionBundle | None, tau: float
    ) -> dict[str, np.ndarray]:
        """Percentile-ranked scores for every metric, keyed by metric name."""
        return {name: m.score(bundle, reference, tau) for name, m in self.metrics.items()}

    def raws(self, bundle: PredictionBundle, tau: float) -> dict[str, np.ndarray]:
        """Unranked metric values, used for the diagnostic scatter plots."""
        return {name: m.raw(bundle, tau) for name, m in self.metrics.items()}


def fit_orthogonalizer(
    cfg: UncertaintyConfig,
    reference: PredictionBundle,
    target: PredictionBundle,
) -> OrthogonalizedSD:
    """Fit the SD_logit trend on validation ('val') or in-sample on the target split ('self')."""
    ocfg = cfg.orthogonalization
    source = reference if ocfg.fit_split == "val" else target
    if ocfg.fit_split == "self":
        logger.warning(
            "orthogonalization.fit_split='self': SD_perp is residualised in-sample "
            "(sensitivity analysis only)."
        )
    return OrthogonalizedSD(ocfg).fit(source.mean_proba, source.sd_logit)


def correlation_with_pbar(values: np.ndarray, mean_proba: np.ndarray) -> dict[str, float]:
    """Spearman and Pearson correlation of a metric with p-bar."""
    from scipy.stats import pearsonr, spearmanr

    values, mean_proba = np.asarray(values, float), np.asarray(mean_proba, float)
    if np.std(values) == 0 or np.std(mean_proba) == 0:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    rho, p = spearmanr(values, mean_proba)
    return {
        "pearson_r": float(pearsonr(values, mean_proba)[0]),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
    }
