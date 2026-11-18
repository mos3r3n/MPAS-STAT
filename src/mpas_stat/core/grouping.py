"""Grouped (binned) statistics shared by the model-space and obs-space stages."""
import numpy as np


def grouped_stats_1d(values: np.ndarray, group_ids: np.ndarray, n_groups: int):
    """
    Compute mean, rms, std for 1D values grouped by integer group_ids.
    NaNs are ignored.
    """
    valid = np.isfinite(values)
    if not np.any(valid):
        shape = (n_groups,)
        return (
            np.full(shape, np.nan, dtype=np.float64),
            np.full(shape, np.nan, dtype=np.float64),
            np.full(shape, np.nan, dtype=np.float64),
        )

    gids = group_ids[valid]
    vals = values[valid]

    count = np.bincount(gids, minlength=n_groups).astype(np.float64)
    s1 = np.bincount(gids, weights=vals, minlength=n_groups).astype(np.float64)
    s2 = np.bincount(gids, weights=vals * vals, minlength=n_groups).astype(np.float64)

    mean = np.full(n_groups, np.nan, dtype=np.float64)
    rms = np.full(n_groups, np.nan, dtype=np.float64)
    std = np.full(n_groups, np.nan, dtype=np.float64)

    nonzero = count > 0
    mean[nonzero] = s1[nonzero] / count[nonzero]
    rms[nonzero] = np.sqrt(s2[nonzero] / count[nonzero])

    var = np.full(n_groups, np.nan, dtype=np.float64)
    var[nonzero] = s2[nonzero] / count[nonzero] - mean[nonzero] ** 2
    var[nonzero] = np.maximum(var[nonzero], 0.0)
    std[nonzero] = np.sqrt(var[nonzero])

    return mean, rms, std


def grouped_weighted_stats(values, weights, group_ids=None, n_groups=1, spread=None):
    """Weighted, pooled statistics of ``values`` per group.

    Every element of ``values`` is one sample; ``weights`` and ``group_ids``
    broadcast against it (e.g. per-cell area weights against a (nCells,
    nLevels) field). Samples with a non-finite value or a non-positive /
    non-finite weight are ignored. For each group returns float64 arrays:

        mean   = sum(w x) / sum(w)
        rms    = sqrt(sum(w x^2) / sum(w))
        std    = sqrt(sum(w (x - mean)^2) / sum(w))     (two-pass)
        weight = sum(w)                                (0 for empty groups)
        spread = sqrt(sum(w s^2) / sum(w))             (only if spread given)

    The spread is aggregated as the square root of the mean ensemble
    variance -- the quantity that should match the RMSE of the ensemble
    mean for a reliable ensemble -- not as the mean standard deviation.
    Empty groups are NaN (weight 0).
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape)
    if group_ids is None:
        group_ids = np.zeros(values.shape, dtype=np.int64)
    else:
        group_ids = np.broadcast_to(np.asarray(group_ids, dtype=np.int64), values.shape)

    v = values.ravel()
    w = weights.ravel()
    g = group_ids.ravel()

    def _moments(x, ok):
        wsum = np.bincount(g[ok], weights=w[ok], minlength=n_groups)
        s1 = np.bincount(g[ok], weights=w[ok] * x[ok], minlength=n_groups)
        nonzero = wsum > 0
        mean = np.full(n_groups, np.nan)
        mean[nonzero] = s1[nonzero] / wsum[nonzero]
        return wsum, nonzero, mean

    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    wsum, nonzero, mean = _moments(v, ok)

    s2 = np.bincount(g[ok], weights=w[ok] * v[ok] * v[ok], minlength=n_groups)
    dev = v[ok] - mean[g[ok]]
    sdev = np.bincount(g[ok], weights=w[ok] * dev * dev, minlength=n_groups)

    rms = np.full(n_groups, np.nan)
    std = np.full(n_groups, np.nan)
    rms[nonzero] = np.sqrt(s2[nonzero] / wsum[nonzero])
    std[nonzero] = np.sqrt(sdev[nonzero] / wsum[nonzero])

    out = {"mean": mean, "rms": rms, "std": std, "weight": wsum}

    if spread is not None:
        s = np.asarray(spread, dtype=np.float64).ravel()
        ok_s = np.isfinite(s) & np.isfinite(w) & (w > 0)
        wsum_s = np.bincount(g[ok_s], weights=w[ok_s], minlength=n_groups)
        ss = np.bincount(g[ok_s], weights=w[ok_s] * s[ok_s] * s[ok_s], minlength=n_groups)
        sprd = np.full(n_groups, np.nan)
        nz = wsum_s > 0
        sprd[nz] = np.sqrt(ss[nz] / wsum_s[nz])
        out["spread"] = sprd

    return out
