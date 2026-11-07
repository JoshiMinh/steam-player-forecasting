"""Statistical significance tools for forecast comparison.

Provides a unit-level bootstrap for mean (and paired mean-difference) scores and
the Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def _as_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[~np.isnan(arr)] if np.isnan(arr).any() else arr


def block_bootstrap_ci(
    values: Any,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean, resampling whole units.

    Each element of ``values`` is one independent unit (here a game-fold score),
    so resampling units keeps the dependence between the 12 horizon steps inside a
    forecast window intact.

    Returns:
        (mean, lower, upper)
    """
    arr = _as_1d(values)
    if arr.size == 0:
        raise ValueError("values must contain at least one non-NaN element")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lower, upper = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(arr.mean()), float(lower), float(upper)


def paired_bootstrap_diff(
    scores_a: Any,
    scores_b: Any,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired unit bootstrap for the mean difference ``a - b``.

    ``scores_a`` and ``scores_b`` must be aligned unit by unit. A negative mean
    difference means model A has the lower (better) error.

    Returns:
        dict with ``mean_diff``, ``ci_low``, ``ci_high`` and a two-sided
        bootstrap ``p_value`` (twice the smaller tail mass around zero).
    """
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have the same length")
    keep = ~(np.isnan(a) | np.isnan(b))
    diff = (a - b)[keep]
    if diff.size == 0:
        raise ValueError("no paired units available")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot_means = diff[idx].mean(axis=1)
    lower, upper = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    below = float((boot_means <= 0).mean())
    above = float((boot_means >= 0).mean())
    p_value = min(1.0, 2.0 * min(below, above))
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_value": p_value,
    }


def diebold_mariano(
    loss_a: Any,
    loss_b: Any,
    h: int = 12,
    harvey_correction: bool = True,
) -> dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Args:
        loss_a: Per-step loss of model A (e.g. scaled absolute error).
        loss_b: Per-step loss of model B, aligned with ``loss_a``.
        h: Forecast horizon; Bartlett-weighted autocovariances up to lag
            ``h - 1`` enter the long-run variance (Newey-West).
        harvey_correction: Apply the Harvey-Leybourne-Newbold small-sample
            adjustment and use Student-t(T - 1) critical values.

    Returns:
        dict with ``statistic``, ``p_value`` (two-sided), ``mean_diff`` (A - B,
        negative means A is better) and ``n``. ``statistic`` and ``p_value`` are
        NaN when the loss differential is degenerate.
    """
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same length")
    d = a - b
    n = d.size
    if n < 2:
        raise ValueError("Need at least 2 observations")
    mean_diff = float(d.mean())

    centered = d - d.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_var = gamma0
    for k in range(1, min(h, n)):
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / n)
        # Bartlett weights keep the long-run variance non-negative, which the
        # unweighted sum of the first h - 1 autocovariances does not guarantee.
        long_run_var += 2.0 * (1.0 - k / h) * gamma_k

    if long_run_var <= 1e-15:
        # Identical (or constant-difference) losses: no evidence either way.
        p = 1.0 if abs(mean_diff) < 1e-12 else float("nan")
        return {"statistic": float("nan"), "p_value": p, "mean_diff": mean_diff, "n": n}

    stat = mean_diff / np.sqrt(long_run_var / n)
    if harvey_correction:
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        if factor <= 0:
            return {"statistic": float("nan"), "p_value": float("nan"), "mean_diff": mean_diff, "n": n}
        stat = stat * np.sqrt(factor)
        p_value = 2.0 * float(stats.t.sf(abs(stat), df=n - 1))
    else:
        p_value = 2.0 * float(stats.norm.sf(abs(stat)))
    return {"statistic": float(stat), "p_value": p_value, "mean_diff": mean_diff, "n": n}
