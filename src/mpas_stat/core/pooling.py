"""Aggregating per-cycle statistics over many cycles ("time aggregation").

Each statistics file holds one cycle's mean / RMS / STD / spread plus the
sample weight behind them ("weight" in model-space files, "Count" in
obs-space files). Averaging the per-cycle statistics directly ("mean"
aggregation, the pre-2026-11 behaviour) gives an average of RMS values, not
the RMS over the period, and lets a cycle with 5 observations count as much
as one with 5,000. "pooled" aggregation instead pools the underlying sums:

    mean   = sum(w * mean_c) / sum(w)
    rms    = sqrt(sum(w * rms_c^2) / sum(w))
    std    = sqrt(sum(w * (std_c^2 + mean_c^2)) / sum(w) - mean^2)
    spread = sqrt(sum(w * spread_c^2) / sum(w))          (and total spread)

which equals the statistic of all samples of all cycles taken together.
Other statistics (e.g. Count) keep the plain mean over cycles.

A *pool* is a dict ``{"kind", "w", "m1", "m2"}`` of per-cycle arrays
(cycle axis 0) that every collector attaches to its entries as
``entry["pool"]``; bootstrap confidence intervals resample cycles and
re-pool, and percent reductions are computed from pooled values.
"""
import warnings

import numpy as np

from .bootstrap import bootstrap_mean_ci, bootstrap_reduction_ci, safe_nanmean

# Name aliases: model-space files use lower-case names, obs-space files
# display names.
_ALIASES = {
    "mean": ("mean", "Bias"),
    "rms": ("rms", "RMS"),
    "std": ("std", "STD"),
    "weight": ("weight", "Count"),
}


def stat_kind(stat_name):
    """How a statistic is pooled over cycles: ``"mean"``, ``"rms"``, ``"std"`` or ``"plain"``.

    Mean-like statistics (mean, bias) pool linearly, RMS-like ones (rms,
    spread, Desroziers errors, normalized-departure RMS) as the root of the
    pooled mean square, STD from the pooled first and second moments, and
    anything else (e.g. Count) as a plain mean.
    """
    s = str(stat_name).lower()
    if s in ("mean", "bias"):
        return "mean"
    if s in ("rms", "rmse"):
        return "rms"
    if s == "std":
        return "std"
    if s in ("spread", "totalspread", "totspread"):
        return "rms"          # pooled as sqrt(mean of squares)
    if s in ("obserrordesroziers", "bkgerrordesroziers", "obserrorassumed"):
        return "rms"          # sqrt of a pooled mean product / mean square
    if s in ("normdeprms", "normdeptotalrms"):
        return "rms"
    if s in ("normdepmean", "normdeptotalmean"):
        return "mean"
    return "plain"


def resolve_time_aggregation(cfg):
    """'pooled' (default) or 'mean' from a graphics cfg: top-level
    time_aggregation (model-space graphic.yaml) or statistics.time_aggregation
    (obs-space config); statistics.method: legacy implies 'mean'."""
    # Model-space graphic.yaml uses "statistics" for the LIST of statistic
    # names to plot; only the obs-space config has a "statistics" mapping.
    stats_cfg = cfg.get("statistics")
    if not isinstance(stats_cfg, dict):
        stats_cfg = {}
    if stats_cfg.get("method") == "legacy":
        return "mean"
    mode = cfg.get("time_aggregation", stats_cfg.get("time_aggregation", "pooled"))
    if mode not in ("pooled", "mean"):
        raise ValueError(f"time_aggregation must be 'pooled' or 'mean', got {mode!r}")
    return mode


def _lookup(variables, name):
    """First present alias of `name` in a netCDF variables mapping, else None."""
    for alias in _ALIASES.get(name, (name,)):
        if alias in variables:
            return np.asarray(variables[alias][...], dtype=np.float64)
    return None


def read_components(group, stat_name):
    """(value, w, m1, m2) arrays for one statistic of one cycle, from the
    netCDF group holding it. Raises KeyError if the statistic is missing.
    Missing weights default to 1 wherever the value is finite."""
    if stat_name not in group.variables:
        raise KeyError(stat_name)
    value = np.asarray(group.variables[stat_name][...], dtype=np.float64)
    kind = stat_kind(stat_name)

    w = _lookup(group.variables, "weight")
    if w is None or w.shape != value.shape:
        w = np.ones_like(value)
    w = np.where(np.isfinite(value) & np.isfinite(w), w, 0.0)

    nan = np.full_like(value, np.nan)
    if kind == "mean":
        return value, w, value, nan
    if kind == "rms":
        return value, w, nan, value * value
    if kind == "std":
        mean = _lookup(group.variables, "mean")
        rms = _lookup(group.variables, "rms")
        if mean is None or mean.shape != value.shape:
            return value, w, value, nan          # cannot pool: fall back to plain
        ms = rms * rms if (rms is not None and rms.shape == value.shape) else value * value + mean * mean
        return value, w, mean, ms
    return value, w, value, nan


def missing_components(shape=()):
    """Pool components ``(value, w, m1, m2)`` for a missing sample: NaN value and moments, zero weight."""
    nan = np.full(shape, np.nan)
    return nan, np.zeros(shape), nan, nan


class PoolBuilder:
    """Collects per-cycle components alongside a collector's samples."""

    def __init__(self, stat_name):
        self.kind = stat_kind(stat_name)
        self.w, self.m1, self.m2 = [], [], []

    def add(self, comps):
        _, w, m1, m2 = comps
        self.w.append(w)
        self.m1.append(m1)
        self.m2.append(m2)

    def build(self, transpose=None):
        def stack(x):
            arr = np.asarray(x, dtype=np.float64)
            return arr.transpose(transpose) if transpose is not None else arr
        return {"kind": self.kind, "w": stack(self.w), "m1": stack(self.m1), "m2": stack(self.m2)}


def pooled_value(pool, index=None):
    """Pooled statistic over cycles (axis 0), optionally for a resampled
    cycle `index`. NaN where no cycle has weight."""
    w, m1, m2 = pool["w"], pool["m1"], pool["m2"]
    if index is not None:
        w, m1, m2 = w[index], m1[index], m2[index]
    kind = pool["kind"]
    if kind == "std" and not np.any(np.isfinite(m2)):
        kind = "plain"
    if kind == "plain":
        return safe_nanmean(m1, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        need = m2 if kind == "rms" else m1
        ok = (w > 0) & np.isfinite(need)
        if kind == "std":
            ok &= np.isfinite(m2)
        wsum = np.sum(np.where(ok, w, 0.0), axis=0)
        has = wsum > 0
        out = np.full(wsum.shape, np.nan)
        if kind == "mean":
            s1 = np.sum(np.where(ok, w * m1, 0.0), axis=0)
            out[has] = (s1 / np.where(has, wsum, 1.0))[has]
        elif kind == "rms":
            s2 = np.sum(np.where(ok, w * m2, 0.0), axis=0)
            out[has] = np.sqrt(s2 / np.where(has, wsum, 1.0))[has]
        else:  # std
            s1 = np.sum(np.where(ok, w * m1, 0.0), axis=0)
            s2 = np.sum(np.where(ok, w * m2, 0.0), axis=0)
            mean = s1 / np.where(has, wsum, 1.0)
            var = np.maximum(s2 / np.where(has, wsum, 1.0) - mean * mean, 0.0)
            out[has] = np.sqrt(var)[has]
    return out


def _percentile_ci(boot, ci):
    alpha = (100.0 - ci) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(boot, alpha, axis=0), np.nanpercentile(boot, 100.0 - alpha, axis=0)


def bootstrap_pool_ci(pool, n_boot=1000, ci=95, seed=0):
    """Pooled value and bootstrap CI, resampling whole cycles."""
    value = pooled_value(pool)
    ntime = pool["w"].shape[0]
    if ntime < 2:
        nan = np.full_like(value, np.nan)
        return value, nan, nan
    rng = np.random.default_rng(seed)
    boot = np.asarray([pooled_value(pool, rng.integers(0, ntime, size=ntime))
                       for _ in range(int(n_boot))])
    lo, hi = _percentile_ci(boot, ci)
    return value, lo, hi


def _reduction(ref, exp):
    red = np.full(np.shape(ref), np.nan, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(exp) & (np.abs(ref) > 1.0e-12)
    red[valid] = 100.0 * (ref[valid] - exp[valid]) / ref[valid]
    return red


def bootstrap_pool_reduction_ci(ref_pool, exp_pool, n_boot=1000, ci=95, seed=0):
    """Percent reduction 100*(ref-exp)/ref of pooled values, with a paired
    bootstrap CI over cycles (same resampled cycles for both)."""
    ntime = min(ref_pool["w"].shape[0], exp_pool["w"].shape[0])
    ref_pool = {k: (v[:ntime] if k != "kind" else v) for k, v in ref_pool.items()}
    exp_pool = {k: (v[:ntime] if k != "kind" else v) for k, v in exp_pool.items()}
    reduction = _reduction(pooled_value(ref_pool), pooled_value(exp_pool))
    if ntime < 2:
        nan = np.full_like(reduction, np.nan)
        return reduction, nan, nan
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_boot)):
        ind = rng.integers(0, ntime, size=ntime)
        boot.append(_reduction(pooled_value(ref_pool, ind), pooled_value(exp_pool, ind)))
    lo, hi = _percentile_ci(np.asarray(boot), ci)
    return reduction, lo, hi


# ---------------------------------------------------------------------------
# Entry-level helpers used by collectors and plot functions. An entry uses
# pooled aggregation when it carries a "pool"; otherwise it falls back to the
# plain-mean behaviour on entry["samples"] (time_aggregation: mean, or data
# collected without pooling information).
# ---------------------------------------------------------------------------
def aggregate(samples, pool, boot):
    """(value, ci_low, ci_high) for a collector: pooled if `pool`, else mean."""
    if pool is not None:
        return bootstrap_pool_ci(pool, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])
    return bootstrap_mean_ci(samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])


def entry_reduction_ci(ref_entry, exp_entry, boot, key="samples"):
    """Percent reduction of exp vs ref entry, with bootstrap CI."""
    if ref_entry.get("pool") is not None and exp_entry.get("pool") is not None:
        return bootstrap_pool_reduction_ci(ref_entry["pool"], exp_entry["pool"],
                                           n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])
    return bootstrap_reduction_ci(ref_entry[key], exp_entry[key],
                                  n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])


def entry_value(entry, key="samples"):
    """Aggregated value of an entry over cycles (pooled or plain mean)."""
    if entry.get("pool") is not None:
        return pooled_value(entry["pool"])
    return safe_nanmean(entry[key], axis=0)


def pool_from_rows(stat_name, rows):
    """Pool from nested rows of component tuples (as read_components /
    missing_components return them), e.g. rows[cycle][lead] -> arrays of
    shape (n_cycles, n_leads, ...)."""
    def take(i):
        return np.asarray([[np.asarray(c[i], dtype=np.float64) for c in row] for row in rows],
                          dtype=np.float64)
    return {"kind": stat_kind(stat_name), "w": take(1), "m1": take(2), "m2": take(3)}


def entry_difference_ci(ref_entry, exp_entry, boot, key="samples"):
    """Percent DIFFERENCE of exp vs ref, 100 * (exp - ref) / ref, with its
    bootstrap CI: negative = lower than the reference (an improvement for
    RMS / STD). This is what the comparison figures plot ("diff. (%)");
    it is the negative of entry_reduction_ci()."""
    red, lo, hi = entry_reduction_ci(ref_entry, exp_entry, boot, key=key)
    return -red, -hi, -lo
