"""NaN-safe reductions and bootstrap confidence intervals.
"""
import warnings

import numpy as np


def safe_nanmean(a, axis=0):
    """np.nanmean without RuntimeWarning for all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def safe_nanpercentile(a, q, axis=0):
    """np.nanpercentile without RuntimeWarning for all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(a, q, axis=axis)


# ---------------------------------------------------------------------------
# Bootstrap settings  (single authoritative location)
# ---------------------------------------------------------------------------
def get_bootstrap_settings(cfg):
    bcfg = cfg.get("bootstrap", {})
    return {
        "n_boot":             int(bcfg.get("n_boot", 1000)),
        "ci":                 float(bcfg.get("ci", 95)),
        "seed":               int(bcfg.get("seed", 0)),
        "plot_significance":  bool(bcfg.get("plot_significance", True)),
    }


# ---------------------------------------------------------------------------
# Bootstrap statistics
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(samples, n_boot=1000, ci=95, seed=0):
    samples = np.asarray(samples, dtype=np.float64)
    ntime   = samples.shape[0]
    mean    = safe_nanmean(samples, axis=0)

    if ntime < 2:
        nan = np.full_like(mean, np.nan)
        return mean, nan, nan

    rng        = np.random.default_rng(seed)
    boot_means = [
        safe_nanmean(samples[rng.integers(0, ntime, size=ntime), ...], axis=0)
        for _ in range(int(n_boot))
    ]
    boot_means = np.asarray(boot_means)

    alpha    = (100.0 - ci) / 2.0
    ci_low   = safe_nanpercentile(boot_means, alpha,         axis=0)
    ci_high  = safe_nanpercentile(boot_means, 100.0 - alpha, axis=0)
    return mean, ci_low, ci_high


def bootstrap_reduction_ci(ref_samples, exp_samples, n_boot=1000, ci=95, seed=0):
    """Bootstrap CI for percent reduction = 100 * (ref - exp) / ref.
    Positive values indicate improvement."""
    ref_samples = np.asarray(ref_samples, dtype=np.float64)
    exp_samples = np.asarray(exp_samples, dtype=np.float64)

    ntime       = min(ref_samples.shape[0], exp_samples.shape[0])
    ref_samples = ref_samples[:ntime, ...]
    exp_samples = exp_samples[:ntime, ...]

    def _reduction(ref, exp):
        red   = np.full_like(ref, np.nan, dtype=np.float64)
        valid = np.isfinite(ref) & np.isfinite(exp) & (np.abs(ref) > 1.0e-12)
        red[valid] = 100.0 * (ref[valid] - exp[valid]) / ref[valid]
        return red

    reduction = _reduction(safe_nanmean(ref_samples, axis=0),
                           safe_nanmean(exp_samples, axis=0))

    if ntime < 2:
        nan = np.full_like(reduction, np.nan)
        return reduction, nan, nan

    rng            = np.random.default_rng(seed)
    boot_reductions = []
    for _ in range(int(n_boot)):
        ind = rng.integers(0, ntime, size=ntime)
        boot_reductions.append(
            _reduction(safe_nanmean(ref_samples[ind, ...], axis=0),
                       safe_nanmean(exp_samples[ind, ...], axis=0))
        )
    boot_reductions = np.asarray(boot_reductions)

    alpha   = (100.0 - ci) / 2.0
    ci_low  = safe_nanpercentile(boot_reductions, alpha,         axis=0)
    ci_high = safe_nanpercentile(boot_reductions, 100.0 - alpha, axis=0)
    return reduction, ci_low, ci_high
