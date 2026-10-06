"""
Reliability analyses for one UQ method.

    error_separation, threshold_transfer
        pooled and within-predicted-label separation of errors; transfer of a
        validation-derived uncertainty threshold to test
    residual_uncertainty_curve, residual_partial_association
        |y - p-bar| vs uncertainty against the calibration-implied 2 p-bar(1 - p-bar),
        and the within-p-bar-decile association
    group_overlap_diagnostic, uncertainty_group_brier, risk_sorting_index,
    uncertainty_group_calibration, subgroup_net_benefit
        comparability, reliability and net benefit of the least- and
        most-uncertain groups
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .evaluation import calibration_table, confusion_masks, rank_biserial
from .utils import as_array, clip_proba, log_elapsed

logger = logging.getLogger(__name__)

CORRECTNESS_LABELS = ("correct", "incorrect")


def correctness_masks(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Collapse the four confusion groups into correct (TP+TN) and incorrect (FP+FN)."""
    return {
        "correct": masks["TP"] | masks["TN"],
        "incorrect": masks["FP"] | masks["FN"],
        "pred_positive": masks["TP"] | masks["FP"],
        "pred_negative": masks["TN"] | masks["FN"],
    }


def error_separation(
    scores: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    n_boot: int = 0,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Rank-biserial effects per metric: `pooled` (incorrect vs correct),
    `within_pred_pos` (FP vs TP), `within_pred_neg` (FN vs TN) and
    `stratum_weighted` (size-weighted mean of the two within-stratum effects).
    """
    cm = correctness_masks(masks)
    contrasts = {
        "pooled": (cm["incorrect"], cm["correct"]),
        "within_pred_pos": (masks["FP"], masks["TP"]),
        "within_pred_neg": (masks["FN"], masks["TN"]),
    }

    rows = []
    for metric, values in scores.items():
        stats_by_name = {}
        for name, (m_err, m_ok) in contrasts.items():
            stats = rank_biserial(
                values[m_err], values[m_ok], "two-sided", n_boot, ci_level, seed
            )
            stats_by_name[name] = stats
            rows.append(
                {
                    "metric": metric,
                    "contrast": name,
                    "n_incorrect": stats["n_x"],
                    "n_correct": stats["n_y"],
                    "r": stats["r"],
                    "auc_error": (stats["r"] + 1) / 2 if np.isfinite(stats["r"]) else np.nan,
                    "p_value": stats["p_value"],
                    "r_lo": stats.get("r_lo", np.nan),
                    "r_hi": stats.get("r_hi", np.nan),
                }
            )

        pos, neg = stats_by_name["within_pred_pos"], stats_by_name["within_pred_neg"]
        weights = np.array([pos["n_x"] + pos["n_y"], neg["n_x"] + neg["n_y"]], dtype=float)
        effects = np.array([pos["r"], neg["r"]], dtype=float)
        ok = np.isfinite(effects) & (weights > 0)
        weighted = float(np.average(effects[ok], weights=weights[ok])) if ok.any() else np.nan
        rows.append(
            {
                "metric": metric,
                "contrast": "stratum_weighted",
                "n_incorrect": pos["n_x"] + neg["n_x"],
                "n_correct": pos["n_y"] + neg["n_y"],
                "r": weighted,
                "auc_error": (weighted + 1) / 2 if np.isfinite(weighted) else np.nan,
                "p_value": np.nan,
                "r_lo": np.nan,
                "r_hi": np.nan,
            }
        )

    df = pd.DataFrame(rows)
    wide = df.pivot_table(index="metric", columns="contrast", values="r")
    for metric, row in wide.iterrows():
        logger.info(
            "  %-14s r(pooled)=%+.3f  r(within +)=%+.3f  r(within -)=%+.3f",
            metric,
            row.get("pooled", np.nan),
            row.get("within_pred_pos", np.nan),
            row.get("within_pred_neg", np.nan),
        )
    return df


def uncertainty_threshold_bounds(
    val_scores: dict[str, np.ndarray],
    val_masks: dict[str, np.ndarray],
    q_correct: float = 0.75,
    q_incorrect: float = 0.25,
    include_minmax: bool = False,
) -> pd.DataFrame:
    """
    Validation-derived bounds per metric: `t_correct` (q_correct quantile of correct
    predictions) and `t_incorrect` (q_incorrect quantile of incorrect ones).
    `band_inverted` is True when no uncertainty value separates the two groups.
    """
    cm = correctness_masks(val_masks)
    rows = []
    for metric, values in val_scores.items():
        ok, err = values[cm["correct"]], values[cm["incorrect"]]
        if len(ok) == 0 or len(err) == 0:
            continue
        t_ok = float(np.quantile(ok, q_correct))
        t_err = float(np.quantile(err, q_incorrect))
        row = {
            "metric": metric,
            "q_correct": q_correct,
            "q_incorrect": q_incorrect,
            "t_correct": t_ok,
            "t_incorrect": t_err,
            "band_width": t_ok - t_err,
            "band_inverted": bool(t_err > t_ok),
        }
        if include_minmax:
            row.update(t_correct_max=float(ok.max()), t_incorrect_min=float(err.min()))
        rows.append(row)
    return pd.DataFrame(rows)


def _flag_stats(values: np.ndarray, correct: np.ndarray, incorrect: np.ndarray, t: float) -> dict:
    """Operating characteristics of the rule 'flag as uncertain if score > t'."""
    flagged = values > t
    n_err, n_ok = int(incorrect.sum()), int(correct.sum())
    n_flag = int(flagged.sum())
    return {
        "flag_rate": float(flagged.mean()),
        "sensitivity": float(flagged[incorrect].mean()) if n_err else np.nan,
        "specificity": float((~flagged)[correct].mean()) if n_ok else np.nan,
        "ppv": float(incorrect[flagged].mean()) if n_flag else np.nan,
    }


def threshold_transfer(
    val_scores: dict[str, np.ndarray],
    val_masks: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    bounds: pd.DataFrame,
    scale: str = "rank",
) -> pd.DataFrame:
    """Flag rate, sensitivity, specificity and PPV of the validation threshold on validation and test."""
    val_cm, test_cm = correctness_masks(val_masks), correctness_masks(test_masks)
    rows = []
    for row in bounds.itertuples():
        metric, t = row.metric, row.t_correct
        if metric not in test_scores:
            continue
        v = _flag_stats(val_scores[metric], val_cm["correct"], val_cm["incorrect"], t)
        s = _flag_stats(test_scores[metric], test_cm["correct"], test_cm["incorrect"], t)
        rows.append(
            {
                "metric": metric,
                "scale": scale,
                "threshold": t,
                "intended_specificity": row.q_correct,
                **{f"val_{k}": val for k, val in v.items()},
                **{f"test_{k}": val for k, val in s.items()},
                **{f"gap_{k}": s[k] - v[k] for k in v},
                "band_inverted": row.band_inverted,
            }
        )
    return pd.DataFrame(rows)


def correctness_long_frame(
    scores_by_split: dict[str, dict[str, np.ndarray]],
    masks_by_split: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Tidy frame of (split, metric, correctness, score) for the correctness figure."""
    records = []
    for split, scores in scores_by_split.items():
        cm = correctness_masks(masks_by_split[split])
        for metric, values in scores.items():
            for label in CORRECTNESS_LABELS:
                for v in values[cm[label]]:
                    records.append(
                        {"split": split, "metric": metric, "correctness": label, "value": v}
                    )
    return pd.DataFrame(records)


def calibration_implied_residual(mean_proba: np.ndarray) -> np.ndarray:
    """E|y - p-bar| under perfect calibration: 2 * p-bar * (1 - p-bar)."""
    p = clip_proba(mean_proba)
    return 2.0 * p * (1.0 - p)


def residual_uncertainty_curve(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    n_bins: int = 20,
) -> pd.DataFrame:
    """Binned mean |y - p-bar| against uncertainty, with the calibration-implied reference."""
    y, p = as_array(y_true, int), as_array(mean_proba)
    residual = np.abs(y - p)
    implied = calibration_implied_residual(p)

    rows = []
    for metric, values in scores.items():
        values = np.asarray(values, float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            logger.warning("%s: too few distinct values to bin; skipping curve", metric)
            continue
        idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            m = idx == b
            if not m.any():
                continue
            rows.append(
                {
                    "metric": metric,
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_uncertainty": float(values[m].mean()),
                    "mean_abs_residual": float(residual[m].mean()),
                    "implied_residual": float(implied[m].mean()),
                    "mean_pbar": float(p[m].mean()),
                }
            )
    return pd.DataFrame(rows)


def residual_uncertainty_loess(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    frac: float = 0.5,
    n_out: int = 100,
) -> pd.DataFrame:
    """LOESS fit of |y - p-bar| and of the calibration-implied residual on each metric."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    y, p = as_array(y_true, int), as_array(mean_proba)
    residual, implied = np.abs(y - p), calibration_implied_residual(p)

    rows = []
    for metric, values in scores.items():
        values = np.asarray(values, float)
        grid = np.linspace(values.min(), values.max(), n_out)
        obs = lowess(residual, values, frac=frac, return_sorted=True)
        imp = lowess(implied, values, frac=frac, return_sorted=True)
        rows.append(
            pd.DataFrame(
                {
                    "metric": metric,
                    "uncertainty": grid,
                    "loess_residual": np.interp(grid, obs[:, 0], obs[:, 1]),
                    "loess_implied": np.interp(grid, imp[:, 0], imp[:, 1]),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def residual_partial_association(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    n_strata: int = 10,
    min_stratum: int = 20,
) -> pd.DataFrame:
    """
    Spearman association of uncertainty with |y - p-bar|, overall and pooled
    within p-bar strata (Fisher z, weights n - 3).
    """
    y, p = as_array(y_true, int), as_array(mean_proba)
    residual = np.abs(y - p)

    feasible = max(2, len(p) // min_stratum)
    if feasible < n_strata:
        logger.warning(
            "n=%d supports only %d p-bar strata at min_stratum=%d; reducing from %d",
            len(p),
            feasible,
            min_stratum,
            n_strata,
        )
        n_strata = feasible

    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_strata + 1)))
    strata = np.clip(np.digitize(p, edges[1:-1]), 0, max(len(edges) - 2, 0))

    rows = []
    for metric, values in scores.items():
        values = np.asarray(values, float)
        rho_all = spearmanr(values, residual).statistic if np.std(values) > 0 else np.nan

        zs, ws = [], []
        for s in np.unique(strata):
            m = strata == s
            if m.sum() < min_stratum or np.std(values[m]) == 0 or np.std(residual[m]) == 0:
                continue
            rho = spearmanr(values[m], residual[m]).statistic
            if not np.isfinite(rho) or abs(rho) >= 1:
                continue
            zs.append(np.arctanh(rho))
            ws.append(m.sum() - 3)

        if not zs:
            logger.warning(
                "%s: no p-bar stratum met the minimum size; within-stratum association "
                "unavailable, report rho_overall only",
                metric,
            )
        pooled = float(np.tanh(np.average(zs, weights=ws))) if zs else np.nan
        rows.append(
            {
                "metric": metric,
                "rho_overall": float(rho_all) if np.isfinite(rho_all) else np.nan,
                "rho_within_pooled": pooled,
                "n_strata_used": len(zs),
                "attenuation": (float(rho_all) - pooled) if np.isfinite(rho_all) and zs else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    for row in df.itertuples():
        logger.info(
            "  %-14s rho(unc, residual): overall=%+.3f  within-p-bar=%+.3f",
            row.metric,
            row.rho_overall,
            row.rho_within_pooled,
        )
    return df


def uncertainty_group_masks(
    values: np.ndarray, low_pct: float = 20.0, high_pct: float = 20.0
) -> dict[str, np.ndarray]:
    """Least- and most-uncertain quantile groups of a single metric."""
    values = np.asarray(values, float)
    lo = np.percentile(values, low_pct)
    hi = np.percentile(values, 100.0 - high_pct)
    return {"low_uncertainty": values <= lo, "high_uncertainty": values >= hi}


def group_overlap_diagnostic(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    low_pct: float,
    high_pct: float,
) -> pd.DataFrame:
    """Median p-bar, p-bar overlap and prevalence of the low/high uncertainty groups."""
    y, p = as_array(y_true, int), as_array(mean_proba)
    rows = []
    for metric, values in scores.items():
        g = uncertainty_group_masks(values, low_pct, high_pct)
        p_lo, p_hi = p[g["low_uncertainty"]], p[g["high_uncertainty"]]
        if len(p_lo) == 0 or len(p_hi) == 0:
            continue
        lo_range = np.quantile(p_lo, [0.05, 0.95])
        hi_range = np.quantile(p_hi, [0.05, 0.95])
        inside_hi = float(np.mean((p_hi >= lo_range[0]) & (p_hi <= lo_range[1])))
        inside_lo = float(np.mean((p_lo >= hi_range[0]) & (p_lo <= hi_range[1])))
        rows.append(
            {
                "metric": metric,
                "n_low": int(g["low_uncertainty"].sum()),
                "n_high": int(g["high_uncertainty"].sum()),
                "low_pbar_median": float(np.median(p_lo)),
                "high_pbar_median": float(np.median(p_hi)),
                "low_pbar_q05": lo_range[0],
                "low_pbar_q95": lo_range[1],
                "high_pbar_q05": hi_range[0],
                "high_pbar_q95": hi_range[1],
                "overlap_fraction": (inside_hi + inside_lo) / 2,
                "low_prevalence": float(y[g["low_uncertainty"]].mean()),
                "high_prevalence": float(y[g["high_uncertainty"]].mean()),
            }
        )
    df = pd.DataFrame(rows)
    for row in df.itertuples():
        logger.info(
            "  %-14s p-bar median low/high = %.4f / %.4f  overlap=%.2f",
            row.metric,
            row.low_pbar_median,
            row.high_pbar_median,
            row.overlap_fraction,
        )
    return df


def brier_stats(y_true, mean_proba, global_climatology: float) -> dict[str, float]:
    """
    Brier score for a subset with two skill scores:

        BSS_within  = 1 - BS / [pi_g (1 - pi_g)]      group base rate
        BSS_global  = 1 - BS / [pi_all (1 - pi_all)]  cohort base rate
    """
    y, p = as_array(y_true, int), as_array(mean_proba)
    n = len(y)
    if n == 0:
        return dict(n=0, prevalence=np.nan, brier_score=np.nan,
                    group_climatology=np.nan, bss_within=np.nan, bss_global=np.nan)

    bs = float(np.mean((p - y) ** 2))
    prev = float(y.mean())
    group_clim = prev * (1.0 - prev)
    if group_clim <= 0:
        # Single-class group: BSS_within is undefined.
        logger.warning(
            "BSS_within undefined: group of n=%d has prevalence %.4f (single class).",
            n, prev,
        )
    return dict(
        n=n,
        prevalence=prev,
        brier_score=bs,
        group_climatology=group_clim,
        bss_within=float(1.0 - bs / group_clim) if group_clim > 0 else np.nan,
        bss_global=float(1.0 - bs / global_climatology) if global_climatology > 0 else np.nan,
    )


def _bootstrap_group_brier(
    y, p, scores, low_pct, high_pct, global_clim, n_boot, ci_level, seed
):
    """Bootstrap percentile CIs for per-group statistics and low-minus-high gaps (cut-points recomputed)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ("brier_score", "bss_within", "bss_global", "prevalence")
    per_group = {(m, g): {k: [] for k in keys} for m in scores for g in
                 ("low_uncertainty", "high_uncertainty")}
    gaps = {m: {"bss_within": [], "bss_global": [], "prevalence": []} for m in scores}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        clim_b = yb.mean() * (1 - yb.mean())
        for m, v in scores.items():
            vb = np.asarray(v)[idx]
            masks = {
                "low_uncertainty": vb <= np.percentile(vb, low_pct),
                "high_uncertainty": vb >= np.percentile(vb, 100.0 - high_pct),
            }
            stats = {g: brier_stats(yb[mk], pb[mk], clim_b) for g, mk in masks.items()}
            for g, s in stats.items():
                for k in keys:
                    per_group[(m, g)][k].append(s[k])
            for k in ("bss_within", "bss_global", "prevalence"):
                gaps[m][k].append(stats["low_uncertainty"][k] - stats["high_uncertainty"][k])

    alpha = (1 - ci_level) / 2
    q = lambda a: (float(np.nanquantile(a, alpha)), float(np.nanquantile(a, 1 - alpha)))
    return ({k: {kk: q(vv) for kk, vv in v.items()} for k, v in per_group.items()},
            {k: {kk: q(vv) for kk, vv in v.items()} for k, v in gaps.items()})


def uncertainty_group_brier(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    low_pct: float = 20.0,
    high_pct: float = 20.0,
    n_boot: int = 0,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-group Brier score with both skill-score references and optional CIs."""
    y, p = as_array(y_true, int), as_array(mean_proba)
    pi_all = float(y.mean()) if len(y) else np.nan
    global_clim = pi_all * (1.0 - pi_all) if np.isfinite(pi_all) else np.nan
    if not global_clim > 0:
        logger.warning("Evaluation split is single-class; skill scores are undefined.")

    overall = brier_stats(y, p, global_clim)
    rows = [{"metric": "all", "group": "all", **overall, "global_climatology": global_clim}]
    for metric, values in scores.items():
        for group, mask in uncertainty_group_masks(values, low_pct, high_pct).items():
            rows.append({
                "metric": metric, "group": group,
                **brier_stats(y[mask], p[mask], global_clim),
                "global_climatology": global_clim,
            })
    df = pd.DataFrame(rows)

    if n_boot:
        with log_elapsed(f"Brier bootstrap ({n_boot} replicates)"):
            per_group, _ = _bootstrap_group_brier(
                y, p, scores, low_pct, high_pct, global_clim, n_boot, ci_level, seed
            )
        for stat in ("brier_score", "bss_within", "bss_global", "prevalence"):
            df[f"{stat}_lo"] = [
                per_group.get((r.metric, r.group), {}).get(stat, (np.nan, np.nan))[0]
                for r in df.itertuples()
            ]
            df[f"{stat}_hi"] = [
                per_group.get((r.metric, r.group), {}).get(stat, (np.nan, np.nan))[1]
                for r in df.itertuples()
            ]

    logger.info(
        "  cohort BS=%.5f  prevalence=%.4f  (global climatology=%.5f)",
        overall["brier_score"], pi_all, global_clim,
    )
    for row in df[df.metric != "all"].itertuples():
        logger.info(
            "  %-14s %-17s BS=%.5f  BSS_within=%+.3f  prevalence=%.4f",
            row.metric, row.group, row.brier_score, row.bss_within, row.prevalence,
        )
    return df


def risk_sorting_index(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    low_pct: float = 20.0,
    high_pct: float = 20.0,
    n_boot: int = 0,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Low-minus-high gaps: `risk_sorting` (BSS_global gap) and
    `reliability_separation` (BSS_within gap).
    """
    y, p = as_array(y_true, int), as_array(mean_proba)
    pi_all = float(y.mean())
    global_clim = pi_all * (1 - pi_all)

    rows = []
    for metric, values in scores.items():
        g = uncertainty_group_masks(values, low_pct, high_pct)
        lo = brier_stats(y[g["low_uncertainty"]], p[g["low_uncertainty"]], global_clim)
        hi = brier_stats(y[g["high_uncertainty"]], p[g["high_uncertainty"]], global_clim)
        rows.append({
            "metric": metric,
            "prevalence_gap": lo["prevalence"] - hi["prevalence"],
            "risk_sorting": lo["bss_global"] - hi["bss_global"],
            "reliability_separation": lo["bss_within"] - hi["bss_within"],
        })
    df = pd.DataFrame(rows)

    if n_boot:
        _, gaps = _bootstrap_group_brier(
            y, p, scores, low_pct, high_pct, global_clim, n_boot, ci_level, seed
        )
        for col, key in (("risk_sorting", "bss_global"),
                         ("reliability_separation", "bss_within"),
                         ("prevalence_gap", "prevalence")):
            df[f"{col}_lo"] = [gaps[m][key][0] for m in df.metric]
            df[f"{col}_hi"] = [gaps[m][key][1] for m in df.metric]

    df = df.sort_values("reliability_separation", ascending=False)
    for r in df.itertuples():
        ci = (f"  [{r.reliability_separation_lo:+.3f}, {r.reliability_separation_hi:+.3f}]"
              if n_boot else "")
        logger.info(
            "  %-14s risk-sorting=%+.3f   reliability separation=%+.3f%s",
            r.metric, r.risk_sorting, r.reliability_separation, ci,
        )
    return df



def uncertainty_group_calibration(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    low_pct: float = 20.0,
    high_pct: float = 20.0,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability table for the least- and most-uncertain group of every metric."""
    y, p = as_array(y_true, int), as_array(mean_proba)
    frames = []
    for metric, values in scores.items():
        for group, mask in uncertainty_group_masks(values, low_pct, high_pct).items():
            if mask.sum() < n_bins:
                logger.warning("%s/%s: too few samples to bin calibration", metric, group)
                continue
            tab = calibration_table(y[mask], p[mask], n_bins=n_bins)
            tab.insert(0, "group", group)
            tab.insert(0, "metric", metric)
            tab["group_n"] = int(mask.sum())
            tab["group_prevalence"] = float(y[mask].mean())
            frames.append(tab)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def subgroup_net_benefit(
    y_true: np.ndarray,
    mean_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: np.ndarray,
    low_pct: float = 20.0,
    high_pct: float = 20.0,
    fixed_denominator: bool = True,
) -> pd.DataFrame:
    """
    Decision curves for the least- and most-uncertain groups: `snb_conditional`
    (within the group) and `snb_fixed` (full-cohort denominator, other cases untreated).
    """
    y, p = as_array(y_true, int), as_array(mean_proba)
    n_total, prevalence = len(y), float(np.mean(y))

    rows = []
    for metric, values in scores.items():
        for group, mask in uncertainty_group_masks(values, low_pct, high_pct).items():
            n_g = int(mask.sum())
            if n_g == 0:
                continue
            y_g, p_g = y[mask], p[mask]
            prev_g = float(y_g.mean())
            for t in thresholds:
                w = t / (1.0 - t)
                pred = p_g >= t
                tp = int(np.sum(pred & (y_g == 1)))
                fp = int(np.sum(pred & (y_g == 0)))

                nb_cond = tp / n_g - (fp / n_g) * w
                nb_all_cond = prev_g - (1 - prev_g) * w
                row = {
                    "metric": metric,
                    "group": group,
                    "threshold": float(t),
                    "n_group": n_g,
                    "group_prevalence": prev_g,
                    "net_benefit": nb_cond,
                    "snb_conditional": nb_cond / prev_g if prev_g else np.nan,
                    "treat_all_conditional": nb_all_cond / prev_g if prev_g else np.nan,
                    "treat_none": 0.0,
                }
                if fixed_denominator:
                    nb_fixed = tp / n_total - (fp / n_total) * w
                    row["snb_fixed"] = nb_fixed / prevalence if prevalence else np.nan
                    row["treat_all_fixed"] = (
                        (prevalence - (1 - prevalence) * w) / prevalence if prevalence else np.nan
                    )
                rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["reference"] = {"prevalence": prevalence, "n_total": n_total}
    return df


def run_reliability_suite(
    val_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    val_bundle,
    test_bundle,
    tau: float,
    cfg,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Compute every reliability table for one UQ method; returns named DataFrames."""
    val_masks = confusion_masks(val_bundle.y_true, val_bundle.mean_proba, tau)
    test_masks = confusion_masks(test_bundle.y_true, test_bundle.mean_proba, tau)
    out: dict[str, pd.DataFrame] = {}

    if cfg.error_separation.enabled:
        ecfg = cfg.error_separation
        logger.info("Correct-vs-incorrect separation")
        out["error_separation"] = error_separation(
            test_scores, test_masks, cfg.n_bootstrap_ci, cfg.ci_level, seed
        )
        bounds = uncertainty_threshold_bounds(
            val_scores, val_masks, ecfg.q_correct, ecfg.q_incorrect, ecfg.include_minmax
        )
        out["uncertainty_bounds"] = bounds
        out["threshold_transfer"] = threshold_transfer(
            val_scores, val_masks, test_scores, test_masks, bounds
        )
        out["correctness_distribution"] = correctness_long_frame(
            {"validation": val_scores, "test": test_scores},
            {"validation": val_masks, "test": test_masks},
        )

    if cfg.residual_uncertainty.enabled:
        rcfg = cfg.residual_uncertainty
        logger.info("Residual vs uncertainty")
        out["residual_uncertainty_binned"] = residual_uncertainty_curve(
            test_bundle.y_true, test_bundle.mean_proba, test_scores, rcfg.n_bins
        )
        out["residual_uncertainty_loess"] = residual_uncertainty_loess(
            test_bundle.y_true, test_bundle.mean_proba, test_scores, rcfg.loess_frac
        )
        out["residual_partial_association"] = residual_partial_association(
            test_bundle.y_true, test_bundle.mean_proba, test_scores, rcfg.n_pbar_strata
        )

    if cfg.uncertainty_groups.enabled:
        gcfg = cfg.uncertainty_groups
        logger.info("Uncertainty-stratified calibration and net benefit")
        out["group_overlap"] = group_overlap_diagnostic(
            test_bundle.y_true, test_bundle.mean_proba, test_scores, gcfg.low_pct, gcfg.high_pct
        )
        out["group_brier"] = uncertainty_group_brier(
            test_bundle.y_true, test_bundle.mean_proba, test_scores,
            gcfg.low_pct, gcfg.high_pct,
            gcfg.brier_n_boot, cfg.ci_level, seed,
        )
        out["risk_sorting_index"] = risk_sorting_index(
            test_bundle.y_true, test_bundle.mean_proba, test_scores,
            gcfg.low_pct, gcfg.high_pct,
            gcfg.brier_n_boot, cfg.ci_level, seed,
        )
        out["group_calibration"] = uncertainty_group_calibration(
            test_bundle.y_true,
            test_bundle.mean_proba,
            test_scores,
            gcfg.low_pct,
            gcfg.high_pct,
            gcfg.n_calibration_bins,
        )
        thresholds = np.linspace(gcfg.threshold_min, gcfg.threshold_max, gcfg.threshold_n)
        out["group_net_benefit"] = subgroup_net_benefit(
            test_bundle.y_true,
            test_bundle.mean_proba,
            test_scores,
            thresholds,
            gcfg.low_pct,
            gcfg.high_pct,
            gcfg.fixed_denominator,
        )
    return out
