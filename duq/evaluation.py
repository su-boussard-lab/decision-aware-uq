"""
Effect sizes, threshold sweep, net benefit and deferral analyses.

Effect sizes are rank-biserial correlations, r = 2 P(X > Y) - 1.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from .utils import as_array, log_elapsed, logit

logger = logging.getLogger(__name__)

CONFUSION_GROUPS = ("TP", "FP", "FN", "TN")


def best_f1_threshold(y_true, y_score, eps: float = 1e-12) -> tuple[float, float]:
    """Threshold maximising F1 (called on the validation split)."""
    y_true, y_score = as_array(y_true, int), as_array(y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / (precision + recall + eps)
    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best])


def confusion_masks(y_true, mean_proba, tau: float) -> dict[str, np.ndarray]:
    """Boolean masks for TP/FP/FN/TN at operating threshold `tau`."""
    y = as_array(y_true, int)
    pred = (as_array(mean_proba) >= tau).astype(int)
    return {
        "TP": (pred == 1) & (y == 1),
        "FP": (pred == 1) & (y == 0),
        "FN": (pred == 0) & (y == 1),
        "TN": (pred == 0) & (y == 0),
    }


def rank_biserial(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
    n_boot: int = 0,
    ci_level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Rank-biserial r = 2U / (n_x n_y) - 1 with Mann-Whitney P and optional bootstrap CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    out = {"n_x": len(x), "n_y": len(y), "r": np.nan, "U": np.nan, "p_value": np.nan}
    if len(x) == 0 or len(y) == 0:
        return out

    res = mannwhitneyu(x, y, alternative=alternative)
    out.update(
        U=float(res.statistic),
        p_value=float(res.pvalue),
        r=float(2.0 * res.statistic / (len(x) * len(y)) - 1.0),
    )

    if n_boot:
        rng = np.random.default_rng(seed)
        draws = np.empty(n_boot)
        for i in range(n_boot):
            xb = rng.choice(x, len(x), replace=True)
            yb = rng.choice(y, len(y), replace=True)
            u = mannwhitneyu(xb, yb, alternative="two-sided").statistic
            draws[i] = 2.0 * u / (len(xb) * len(yb)) - 1.0
        alpha = (1 - ci_level) / 2
        out["r_lo"], out["r_hi"] = np.quantile(draws, [alpha, 1 - alpha])
    return out


def group_comparison(
    scores: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    pairs: tuple[tuple[str, str], ...] = (("FP", "FN"), ("FP", "TP"), ("TP", "TN")),
    n_boot: int = 0,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Effect size and test for each metric and each confusion-group contrast."""
    rows = []
    for metric, values in scores.items():
        for a, b in pairs:
            stats = rank_biserial(
                values[masks[a]], values[masks[b]], "two-sided", n_boot, ci_level, seed
            )
            rows.append({"metric": metric, "contrast": f"{a} vs {b}", **stats})
    return pd.DataFrame(rows)


def group_summary(scores: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Median and IQR of every metric within each confusion group."""
    rows = []
    for metric, values in scores.items():
        for group in CONFUSION_GROUPS:
            v = values[masks[group]]
            rows.append(
                {
                    "metric": metric,
                    "group": group,
                    "n": len(v),
                    "median": np.median(v) if len(v) else np.nan,
                    "q25": np.quantile(v, 0.25) if len(v) else np.nan,
                    "q75": np.quantile(v, 0.75) if len(v) else np.nan,
                    "mean": np.mean(v) if len(v) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def threshold_sweep(
    bundle,
    metric_set,
    reference,
    taus: np.ndarray,
    contrast: tuple[str, str] = ("FP", "FN"),
) -> pd.DataFrame:
    """FP-vs-FN effect size at each tau; confusion groups and tau-dependent metrics are recomputed."""
    a, b = contrast
    rows = []
    for tau in taus:
        masks = confusion_masks(bundle.y_true, bundle.mean_proba, tau)
        if masks[a].sum() < 2 or masks[b].sum() < 2:
            continue
        for name, metric in metric_set.metrics.items():
            values = metric.score(bundle, reference, float(tau))
            stats = rank_biserial(values[masks[a]], values[masks[b]])
            rows.append(
                {
                    "tau": float(tau),
                    "metric": name,
                    "contrast": f"{a} vs {b}",
                    "n_a": int(masks[a].sum()),
                    "n_b": int(masks[b].sum()),
                    **{k: stats[k] for k in ("r", "U", "p_value")},
                }
            )
    df = pd.DataFrame(rows)
    logger.info("Threshold sweep: %d tau values x %d metrics", df["tau"].nunique(), len(metric_set.names))
    return df


def net_benefit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: float,
    n_total: int | None = None,
) -> float:
    """NB = TP/N - (FP/N) * tau / (1 - tau), with N fixed to `n_total` when given."""
    y_true, y_pred = as_array(y_true, int), as_array(y_pred, int)
    n = n_total or len(y_true)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    return tp / n - (fp / n) * (tau / (1.0 - tau))


def deferral_curve(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    tau: float,
    coverages,
    include_random: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Standardised net benefit vs coverage for each deferral rule.

    At coverage c the (1 - c) most uncertain samples are deferred and not treated;
    the full test cohort is always the denominator.
    """
    y = as_array(y_true, int)
    p = as_array(mean_proba)
    n_total, prevalence = len(y), float(np.mean(y))
    rules = dict(scores)
    if include_random:
        rules["random"] = np.random.default_rng(seed).random(n_total)

    rows = []
    for name, values in rules.items():
        order = np.argsort(np.asarray(values, float), kind="stable")  # low uncertainty first
        for cov in coverages:
            k = int(np.floor(cov * n_total))
            if k == 0:
                continue
            retained = np.zeros(n_total, dtype=bool)
            retained[order[:k]] = True
            deferred = ~retained

            pred = np.zeros(n_total, dtype=int)
            pred[retained] = (p[retained] >= tau).astype(int)

            masks = confusion_masks(y[retained], p[retained], tau)
            nb = net_benefit(y, pred, tau, n_total)
            rows.append(
                {
                    "rule": name,
                    "coverage": k / n_total,
                    "n_retained": k,
                    "net_benefit": nb,
                    "standardized_net_benefit": nb / prevalence if prevalence else np.nan,
                    "TP_retained": int(masks["TP"].sum()),
                    "FP_retained": int(masks["FP"].sum()),
                    "FN_retained": int(masks["FN"].sum()),
                    "deferred_positives": int(y[deferred].sum()),
                    "deferred_fraction": float(deferred.mean()),
                }
            )

    df = pd.DataFrame(rows)
    nb_all = net_benefit(y, np.ones(n_total, dtype=int), tau, n_total)
    df.attrs["reference"] = {
        "tau": tau,
        "prevalence": prevalence,
        "treat_all_snb": nb_all / prevalence if prevalence else np.nan,
        "treat_none_snb": 0.0,
        "deferred_policy": "treat_none",
    }
    return df


#: Plain-text metric names for tabular output.
PLAIN_METRIC_NAMES = {
    "sd_logit": "SD_logit",
    "sd_perp": "SD_perp",
    "sd_prob": "SD_prob",
    "null_mi": "Mutual information",
    "null_entropy": "Predictive entropy",
    "null_dm": "Decision margin",
    "null_dm_logit": "Decision margin (logit)",
}


def _deferral_stats(
    y: np.ndarray,
    pred: np.ndarray,
    order: np.ndarray,
    k_snb: int,
    k_sens: int,
    weight: float,
    n_total: int,
) -> tuple[float, float]:
    """Net benefit at `k_snb` retained cases and retained sensitivity at `k_sens` (treat-none, fixed N)."""
    tp_cum = np.cumsum((pred & (y == 1))[order])
    fp_cum = np.cumsum((pred & (y == 0))[order])
    pos_cum = np.cumsum((y == 1)[order])

    nb = tp_cum[k_snb - 1] / n_total - (fp_cum[k_snb - 1] / n_total) * weight
    positives = pos_cum[k_sens - 1]
    sens = tp_cum[k_sens - 1] / positives if positives else np.nan
    return float(nb), float(sens)


def _decision_impact_replicate(
    y: np.ndarray,
    p: np.ndarray,
    scores: dict[str, np.ndarray],
    tau: float,
    snb_coverage: float,
    sens_coverage: float,
) -> dict[str, tuple[float, float, float]]:
    """One evaluation of every rule plus the no-deferral reference, on one sample."""
    n = len(y)
    prevalence = float(np.mean(y))
    weight = tau / (1.0 - tau)
    pred = p >= tau

    k_snb, k_sens = max(1, int(np.floor(snb_coverage * n))), max(1, int(np.floor(sens_coverage * n)))
    identity = np.arange(n)
    nb_full, _ = _deferral_stats(y, pred, identity, n, n, weight, n)
    snb_full = nb_full / prevalence if prevalence else np.nan
    positives = int((y == 1).sum())
    sens_full = float((pred & (y == 1)).sum() / positives) if positives else np.nan

    # Expected sNB of random deferral: E[sNB_random] = coverage * sNB_full.
    snb_random = (k_snb / n) * snb_full

    out = {"__full__": (np.nan, np.nan, sens_full)}
    for name, values in scores.items():
        order = np.argsort(np.asarray(values, float), kind="stable")  # least uncertain first
        nb, sens = _deferral_stats(y, pred, order, k_snb, k_sens, weight, n)
        snb = nb / prevalence if prevalence else np.nan
        out[name] = (snb - snb_full, snb - snb_random, sens)
    return out


def decision_impact(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    tau: float,
    dataset: str,
    method: str = "ensemble",
    snb_coverage: float = 0.95,
    sens_coverage: float = 0.50,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Decision impact of deferring the most uncertain cases (Table 3).

      d_snb_vs_full    change in standardised net benefit at `snb_coverage`
                       relative to no deferral
      d_snb_vs_random  the same relative to expected random deferral
      retained_sens    sensitivity among retained cases at `sens_coverage`

    Deferred cases are not treated and N is the full cohort. Intervals are
    percentile bootstrap over resampled rows (or `groups`), with cut-points and
    the random comparator recomputed in each resample.
    """
    y, p = as_array(y_true, int), as_array(mean_proba)
    n = len(y)
    rng = np.random.default_rng(seed)
    point = _decision_impact_replicate(y, p, scores, tau, snb_coverage, sens_coverage)

    if groups is not None:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        index_of = {g: np.flatnonzero(groups == g) for g in unique}
        logger.info("Decision impact: clustered bootstrap over %d groups", len(unique))
    draws: dict[str, list[tuple[float, float, float]]] = {k: [] for k in point}

    with log_elapsed(f"Decision impact ({method}, {n_boot} bootstrap resamples)"):
        for _ in range(n_boot):
            if groups is None:
                idx = rng.integers(0, n, n)
            else:
                picked = rng.choice(unique, len(unique), replace=True)
                idx = np.concatenate([index_of[g] for g in picked])
            y_b = y[idx]
            if y_b.min() == y_b.max():
                continue
            rep = _decision_impact_replicate(
                y_b, p[idx], {k: v[idx] for k, v in scores.items()},
                tau, snb_coverage, sens_coverage,
            )
            for key, values in rep.items():
                draws[key].append(values)

    alpha = (1 - ci_level) / 2
    columns = ("d_snb_vs_full", "d_snb_vs_random", "retained_sens")

    def _row(key: str, rule: str) -> dict:
        arr = np.asarray(draws[key], float) if draws[key] else np.empty((0, 3))
        row = {"method": method, "dataset": dataset, "rule": rule}
        for j, col in enumerate(columns):
            estimate = point[key][j]
            row[col] = estimate
            if np.isnan(estimate) or not len(arr):
                row[f"{col}_lo"] = row[f"{col}_hi"] = np.nan
            else:
                lo, hi = np.nanquantile(arr[:, j], [alpha, 1 - alpha])
                row[f"{col}_lo"], row[f"{col}_hi"] = float(lo), float(hi)
        return row

    rows = [_row("__full__", "no deferral")]
    rows += [_row(name, PLAIN_METRIC_NAMES.get(name, name)) for name in scores]
    df = pd.DataFrame(rows)
    df.attrs["reference"] = {
        "tau": tau,
        "snb_coverage": snb_coverage,
        "sens_coverage": sens_coverage,
        "n_test": n,
        "n_boot_used": len(draws["__full__"]),
    }
    return df


def residual_coverage(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    coverages,
) -> pd.DataFrame:
    """Mean |y - p-bar| among the least uncertain samples vs coverage, per true class."""
    y, p = as_array(y_true, int), as_array(mean_proba)
    residual = np.abs(y - p)
    rows = []
    for name, values in scores.items():
        order = np.argsort(np.asarray(values, float), kind="stable")
        for cls in (0, 1):
            cls_order = order[y[order] == cls]
            n = len(cls_order)
            if n == 0:
                continue
            for cov in coverages:
                k = int(np.floor(cov * n))
                if k == 0:
                    continue
                rows.append(
                    {
                        "metric": name,
                        "y_true": cls,
                        "coverage": k / n,
                        "n": k,
                        "mean_abs_residual": float(residual[cls_order[:k]].mean()),
                    }
                )
    return pd.DataFrame(rows)


def risk_coverage(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    tau: float,
    coverages,
) -> pd.DataFrame:
    """Selective risk (1 - AUC, 1 - accuracy, FNR) and retained prevalence vs coverage."""
    y, p = as_array(y_true, int), as_array(mean_proba)
    rows = []
    for name, values in scores.items():
        order = np.argsort(np.asarray(values, float), kind="stable")
        for cov in coverages:
            k = int(np.floor(cov * len(y)))
            if k == 0:
                continue
            idx = order[:k]
            yk, pk = y[idx], p[idx]
            pred = (pk >= tau).astype(int)
            auc = 1 - roc_auc_score(yk, pk) if len(np.unique(yk)) > 1 else np.nan
            fn = int(np.sum((pred == 0) & (yk == 1)))
            n_pos = int(yk.sum())
            rows.append(
                {
                    "metric": name,
                    "coverage": k / len(y),
                    "n": k,
                    "risk_auc": auc,
                    "risk_accuracy": 1 - accuracy_score(yk, pred),
                    "f1": f1_score(yk, pred, zero_division=0),
                    "fnr": fn / n_pos if n_pos else np.nan,
                    "retained_prevalence": float(yk.mean()),
                }
            )
    return pd.DataFrame(rows)


def calibration_table(y_true, mean_proba, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Binned reliability table; quantile bins avoid empty bins under class imbalance."""
    y, p = as_array(y_true, int), as_array(mean_proba)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_rate": float(y[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(y_true, mean_proba) -> dict[str, float]:
    """Prevalence, p-bar location, Brier score and Brier skill score for one split."""
    y, p = as_array(y_true, int), as_array(mean_proba)
    prevalence = float(y.mean())
    climatology = prevalence * (1.0 - prevalence)
    brier = float(np.mean((p - y) ** 2))
    return {
        "n": len(y),
        "prevalence": prevalence,
        "mean_pbar": float(p.mean()),
        "median_pbar": float(np.median(p)),
        "brier_score": brier,
        "climatology": climatology,
        "brier_skill_score": 1.0 - brier / climatology if climatology > 0 else np.nan,
    }


def performance_summary(y_true, mean_proba, tau: float) -> dict[str, float]:
    """Headline discrimination and operating-point metrics at threshold `tau`."""
    y, p = as_array(y_true, int), as_array(mean_proba)
    pred = (p >= tau).astype(int)
    masks = confusion_masks(y, p, tau)
    tp, fp, fn, tn = (int(masks[g].sum()) for g in ("TP", "FP", "FN", "TN"))
    return {
        "n": len(y),
        "prevalence": float(y.mean()),
        "tau": float(tau),
        "tau_logit": float(logit(np.array([tau]))[0]),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
    }
