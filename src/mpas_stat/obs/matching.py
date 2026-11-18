"""Common-sample selection for obs-space verification.

When OMB and OMA, or several experiments, are compared, each would otherwise
be computed on the observations that passed ITS OWN QC. Differences in the
statistic then mix a better fit with a different set of observations (e.g. an
outer loop rejecting the worst-fitting ones). restrict_to_common_sample() keeps,
for every vector being compared, only the observations that are valid (QC
passed and finite innovation) in ALL of them.

Observations are matched by value -- latitude, longitude, vertical coordinate
(pressure or height) and time -- not by position in the file, because
different experiments (and the separate ``_ana.h5`` file of an ensemble DA
run) need not store observations in the same order. Exact repeats of the same
key within one file are matched in order of occurrence.
"""
from typing import List, Optional

import numpy as np

# Rounding applied before comparing coordinates, so float32/float64 copies of
# the same observation compare equal.
_DECIMALS = {"lat": 4, "lon": 4, "pres": 2, "hgt": 4}


def _column(vec, name, n):
    arr = vec.get(name)
    if arr is None:
        return np.zeros(n, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if name in _DECIMALS:
        arr = np.round(arr, _DECIMALS[name])
    return np.nan_to_num(arr, nan=-9.99e30)


def observation_keys(vec) -> np.ndarray:
    """One hashable (void) key per observation of an obs vector dict."""
    n = int(np.asarray(vec["lat"]).size)
    cols = [_column(vec, name, n) for name in ("lat", "lon", "pres", "hgt", "time")]
    base = np.ascontiguousarray(np.column_stack(cols))
    void = base.view(np.dtype((np.void, base.dtype.itemsize * base.shape[1]))).ravel()

    # Occurrence index among identical keys (0 for the first, 1 for the
    # second, ...), so exact duplicates still match one-to-one.
    _, inverse = np.unique(void, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    sorted_inv = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inv)) + 1]
    run_start = np.repeat(starts, np.diff(np.r_[starts, sorted_inv.size]))
    occurrence = np.empty(n, dtype=np.float64)
    occurrence[order] = np.arange(n) - run_start

    full = np.ascontiguousarray(np.column_stack(cols + [occurrence]))
    return full.view(np.dtype((np.void, full.dtype.itemsize * full.shape[1]))).ravel()


def valid_mask(vec) -> np.ndarray:
    """QC passed (qc == 0, or no QC) and finite innovation."""
    innov = np.asarray(vec["innov"], dtype=np.float64)
    ok = np.isfinite(innov)
    if vec.get("qc") is not None:
        ok &= np.asarray(vec["qc"]) == 0
    return ok


# Warn when the common sample keeps less than this fraction of some input's
# valid observations (usually: the compared runs do not share observations).
WARN_FRACTION = 0.5


def restrict_to_common_sample(vectors: List[Optional[dict]],
                              label: Optional[str] = None) -> List[Optional[dict]]:
    """Return copies of `vectors` whose "qc" keeps only the common sample.

    Entries that are None, or have no innovation, are passed through
    unchanged and do not constrain the others. With fewer than two usable
    vectors nothing is changed. In the returned vectors, qc is 0 for
    observations in the common sample and NaN otherwise.
    """
    usable = [i for i, v in enumerate(vectors) if v is not None and v.get("innov") is not None]
    if len(usable) < 2:
        return list(vectors)

    keys, masks = {}, {}
    common = None
    for i in usable:
        keys[i] = observation_keys(vectors[i])
        masks[i] = valid_mask(vectors[i])
        valid_keys = np.unique(keys[i][masks[i]])
        common = valid_keys if common is None else np.intersect1d(common, valid_keys, assume_unique=True)

    out = list(vectors)
    for i in usable:
        keep = masks[i] & np.isin(keys[i], common)
        n_valid, n_keep = int(masks[i].sum()), int(keep.sum())
        if n_valid and n_keep < WARN_FRACTION * n_valid:
            print(f"WARNING: common sample{' for ' + label if label else ''} keeps only "
                  f"{n_keep} of {n_valid} valid observations of one compared run; the runs "
                  f"may not share observations (statistics.common_sample: false verifies "
                  f"each on its own sample)")
        new = dict(vectors[i])
        new["qc"] = np.where(keep, 0.0, np.nan)
        out[i] = new
    return out
