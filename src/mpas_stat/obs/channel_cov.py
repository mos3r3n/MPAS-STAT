"""Inter-channel error covariances of radiance sensors (Desroziers et al. 2005).

For a radiance sensor and a "da" experiment with OMA, per cycle and domain:

    R_ij ~ mean[(O-A)_i (O-B)_j]          observation-error covariance
    B_ij ~ mean[(A-B)_i (O-B)_j]          background-error covariance (obs space)
    D_ij =  mean[(O-B)_i (O-B)_j]          innovation covariance (= R + B ideally)

over the locations where channels i and j both pass QC in OMB and OMA.
Means and pair counts are written per cycle, so cycles pool exactly
(count-weighted). R from this estimate is not symmetric in general; it is
symmetrized when the correlation matrix is formed (see correlation_from_cov).
"""
import h5py
import numpy as np

from .binning import _apply_domain_mask
from .matching import observation_keys

__all__ = ["CHANNEL_COV_KEYS", "correlation_channels", "departure_matrix", "pair_rows",
           "channel_covariance", "correlation_from_cov"]

# (netCDF name, results prefix)
CHANNEL_COV_KEYS = [
    ("ObsErrorCovDesroziers", "chcov_r"),
    ("BkgErrorCovDesroziers", "chcov_b"),
    ("OMBCov", "chcov_d"),
    ("PairCount", "chcov_n"),
]


def correlation_channels(settings, diag_obs, configured, bak_file=None):
    """Channels to correlate: statistics.correlation_channels[diag_obs]
    (a list, or "all" = every channel in the file), else the sensor's
    configured channels."""
    choice = (settings.get("correlation_channels") or {}).get(diag_obs)
    if choice is None:
        return [int(c) for c in configured]
    if isinstance(choice, str) and choice.lower() == "all":
        if bak_file is None:
            return [int(c) for c in configured]
        with h5py.File(bak_file, "r") as f:
            for grp in ("ombg", "ObsValue", "hofx0", "hofx"):
                if grp in f and "brightnessTemperature" in f[grp]:
                    return list(range(1, f[grp]["brightnessTemperature"].shape[1] + 1))
        return [int(c) for c in configured]
    return [int(c) for c in choice]


def departure_matrix(vectors):
    """(n_obs, n_channel) departures with NaN where a channel failed QC or
    is missing, from per-channel vectors of ONE file (rows aligned)."""
    cols = []
    for vec in vectors:
        d = np.asarray(vec["innov"], dtype=np.float64).copy()
        if vec.get("qc") is not None:
            d[np.asarray(vec["qc"]) != 0] = np.nan
        cols.append(d)
    return np.column_stack(cols)


def pair_rows(vec_b, vec_a):
    """Row indices (ib, ia) of the same locations in the OMB and OMA files
    (identity when the files are the same / identically ordered)."""
    kb, ka = observation_keys(vec_b), observation_keys(vec_a)
    if kb.size == ka.size and np.array_equal(kb, ka):
        idx = np.arange(kb.size)
        return idx, idx
    _, ib, ia = np.intersect1d(kb, ka, assume_unique=True, return_indices=True)
    return ib, ia


def channel_covariance(d_ob, d_oa):
    """Mean cross-products and pair counts for one set of paired rows:
    {"chcov_r", "chcov_b", "chcov_d": (nch, nch) means, "chcov_n": counts}."""
    ok = np.isfinite(d_ob) & np.isfinite(d_oa)
    xb = np.where(ok, d_ob, 0.0)
    xa = np.where(ok, d_oa, 0.0)
    m = ok.astype(np.float64)
    n = m.T @ m
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xa.T @ xb) / n
        b = ((xb - xa).T @ xb) / n
        d = (xb.T @ xb) / n
    nan = n <= 0
    for a in (r, b, d):
        a[nan] = np.nan
    return {"chcov_r": r, "chcov_b": b, "chcov_d": d, "chcov_n": n}


def store_channel_covariance(results, idx, omb_vecs, oma_vecs, domains):
    """Fill results[<prefix>_omb][iexp, idom, it] for every domain."""
    ib, ia = pair_rows(omb_vecs[0], oma_vecs[0])
    if ib.size == 0:
        return
    d_ob = departure_matrix(omb_vecs)[ib]
    d_oa = departure_matrix(oma_vecs)[ia]
    lat = np.asarray(omb_vecs[0]["lat"])[ib]
    lon = np.asarray(omb_vecs[0]["lon"])[ib]
    iexp, it = idx
    for idom, (name, coords) in enumerate(domains.items()):
        mask = _apply_domain_mask(np.zeros(ib.size), lat, lon, name, coords)
        keep = np.ones(ib.size, bool) if mask is None else (np.asarray(mask) == 0)
        cov = channel_covariance(d_ob[keep], d_oa[keep])
        for _, prefix in CHANNEL_COV_KEYS:
            results[f"{prefix}_omb"][iexp, idom, it] = cov[prefix]


def correlation_from_cov(cov):
    """(sigma, correlation, min eigenvalue) of a covariance estimate after
    symmetrizing it; channels with a non-positive variance get NaN."""
    c = 0.5 * (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T)
    var = np.diag(c).copy()
    sigma = np.where(var > 0, np.sqrt(np.where(var > 0, var, 0.0)), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = c / np.outer(sigma, sigma)
    good = np.isfinite(sigma)
    min_eig = np.nan
    if good.sum() >= 1 and np.all(np.isfinite(c[np.ix_(good, good)])):
        min_eig = float(np.linalg.eigvalsh(c[np.ix_(good, good)]).min())
    return sigma, corr, min_eig
