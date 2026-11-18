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
