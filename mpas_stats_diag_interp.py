import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from numba import njit, prange, set_num_threads, get_num_threads
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def configure_numba_threads(n_threads: Optional[int] = None) -> int:
    """Set and return the number of numba threads used by interpolation kernels."""
    if not HAS_NUMBA:
        return 1
    if n_threads is not None and n_threads > 0:
        set_num_threads(int(n_threads))
    return get_num_threads()


def _prepare_profile_coord_arrays(
    vert_coord: np.ndarray,
    log_space: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Preprocess vertical coordinates profile by profile:
      - remove non-finite values
      - optionally require > 0 for log space
      - sort
      - remove duplicate coordinates (keep first)

    Returns
    -------
    coord_list : list of 1D ndarrays
        Cleaned coordinate values in working space (linear or log).
    idx_list : list of 1D ndarrays
        Original vertical indices corresponding to coord_list.
    """
    npoints = vert_coord.shape[0]
    coord_list: List[np.ndarray] = []
    idx_list: List[np.ndarray] = []

    for i in range(npoints):
        coord = np.asarray(vert_coord[i, :], dtype=np.float64)

        valid = np.isfinite(coord)
        if log_space:
            valid &= coord > 0.0

        if np.count_nonzero(valid) < 2:
            coord_list.append(np.empty(0, dtype=np.float64))
            idx_list.append(np.empty(0, dtype=np.int32))
            continue

        coord_valid = coord[valid]
        orig_idx = np.where(valid)[0].astype(np.int32)

        sort_idx = np.argsort(coord_valid)
        coord_sorted = coord_valid[sort_idx]
        idx_sorted = orig_idx[sort_idx]

        unique_coord, unique_pos = np.unique(coord_sorted, return_index=True)
        idx_unique = idx_sorted[unique_pos]

        if unique_coord.size < 2:
            coord_list.append(np.empty(0, dtype=np.float64))
            idx_list.append(np.empty(0, dtype=np.int32))
            continue

        coord_work = np.log(unique_coord) if log_space else unique_coord

        coord_list.append(np.asarray(coord_work, dtype=np.float64))
        idx_list.append(np.asarray(idx_unique, dtype=np.int32))

    return coord_list, idx_list


def _pack_profile_coord_arrays(
    coord_list: List[np.ndarray],
    idx_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack variable-length profile arrays into fixed-size 2D arrays for numba.

    Returns
    -------
    coord_pack : ndarray, shape (npoints, maxlen)
    idx_pack   : ndarray, shape (npoints, maxlen)
    lengths    : ndarray, shape (npoints,)
    """
    npoints = len(coord_list)
    maxlen = max((len(c) for c in coord_list), default=0)

    coord_pack = np.full((npoints, maxlen), np.nan, dtype=np.float64)
    idx_pack = np.full((npoints, maxlen), -1, dtype=np.int32)
    lengths = np.zeros(npoints, dtype=np.int32)

    for i, (coord, idx) in enumerate(zip(coord_list, idx_list)):
        n = len(coord)
        lengths[i] = n
        if n > 0:
            coord_pack[i, :n] = coord
            idx_pack[i, :n] = idx

    return coord_pack, idx_pack, lengths


if HAS_NUMBA:
    @njit
    def _searchsorted_left(arr, n, value):
        """
        searchsorted(arr[:n], value, side='left') for 1D sorted array.
        """
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        return lo


    @njit(parallel=True)
    def _build_interp_coefficients_by_level_numba(
        coord_pack,
        idx_pack,
        lengths,
        target_vals,
    ):
        """
        Build coefficients in parallel over flattened (target level, point) space.

        This is usually faster than parallelizing over target levels only, because
        the number of target height/pressure levels is small while npoints is large.
        """
        npoints = coord_pack.shape[0]
        ntarget = target_vals.size
        ntotal = ntarget * npoints

        k0 = np.full((ntarget, npoints), -1, dtype=np.int32)
        k1 = np.full((ntarget, npoints), -1, dtype=np.int32)
        w = np.full((ntarget, npoints), np.nan, dtype=np.float64)
        valid_target = np.zeros((ntarget, npoints), dtype=np.bool_)

        for ij in prange(ntotal):
            k = ij // npoints
            i = ij - k * npoints
            n = lengths[i]
            if n < 2:
                continue

            tval = target_vals[k]
            ind = _searchsorted_left(coord_pack[i], n, tval)

            if ind == 0 or ind == n:
                continue

            lo = ind - 1
            hi = ind

            x0 = coord_pack[i, lo]
            x1 = coord_pack[i, hi]
            if x1 == x0:
                continue

            wt = (tval - x0) / (x1 - x0)

            k0[k, i] = idx_pack[i, lo]
            k1[k, i] = idx_pack[i, hi]
            w[k, i] = wt
            valid_target[k, i] = True

        return k0, k1, w, valid_target
else:
    def _build_interp_coefficients_by_level_numba(*args, **kwargs):
        raise ImportError("numba is not installed")


def build_interp_coefficients_by_level(
    vert_coord: np.ndarray,
    target_levels: np.ndarray,
    log_space: bool = False,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build interpolation coefficients organized by target level.

    This version preprocesses each profile once, then can parallelize
    coefficient building across target levels.

    Parameters
    ----------
    vert_coord : ndarray, shape (npoints, nz)
        Vertical coordinate for each profile.
    target_levels : array-like, shape (ntarget,)
        Target interpolation levels.
    log_space : bool
        Whether interpolation is in log(vertical coordinate).
    use_numba : bool
        Whether to use numba for coefficient building if available.

    Returns
    -------
    coeff : dict
        k0           : shape (ntarget, npoints)
        k1           : shape (ntarget, npoints)
        w            : shape (ntarget, npoints)
        valid_target : shape (ntarget, npoints)
    """
    target_levels = np.asarray(target_levels, dtype=np.float64)

    if log_space and np.any(target_levels <= 0.0):
        raise ValueError("All target_levels must be > 0 when log_space=True")

    target_vals = np.log(target_levels) if log_space else target_levels

    coord_list, idx_list = _prepare_profile_coord_arrays(
        vert_coord,
        log_space=log_space,
    )
    coord_pack, idx_pack, lengths = _pack_profile_coord_arrays(coord_list, idx_list)

    if use_numba and HAS_NUMBA:
        k0, k1, w, valid_target = _build_interp_coefficients_by_level_numba(
            coord_pack,
            idx_pack,
            lengths,
            target_vals,
        )
    else:
        npoints = coord_pack.shape[0]
        ntarget = target_vals.size

        k0 = np.full((ntarget, npoints), -1, dtype=np.int32)
        k1 = np.full((ntarget, npoints), -1, dtype=np.int32)
        w = np.full((ntarget, npoints), np.nan, dtype=np.float64)
        valid_target = np.zeros((ntarget, npoints), dtype=np.bool_)

        for k, tval in enumerate(target_vals):
            for i in range(npoints):
                n = lengths[i]
                if n < 2:
                    continue

                coord_i = coord_pack[i, :n]
                ind = np.searchsorted(coord_i, tval, side="left")

                if ind == 0 or ind == n:
                    continue

                lo = ind - 1
                hi = ind

                x0 = coord_i[lo]
                x1 = coord_i[hi]
                if x1 == x0:
                    continue

                wt = (tval - x0) / (x1 - x0)

                k0[k, i] = idx_pack[i, lo]
                k1[k, i] = idx_pack[i, hi]
                w[k, i] = wt
                valid_target[k, i] = True

    return {
        "k0": k0,
        "k1": k1,
        "w": w,
        "valid_target": valid_target,
        "target_levels": target_levels,
        "log_space": log_space,
    }


def apply_interp_coefficients_by_level_numpy(
    field: np.ndarray,
    coeff: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Apply precomputed coefficients with vectorized numpy indexing.

    This fallback is much faster than a pure Python nested loop when numba is
    unavailable.
    """
    field = np.asarray(field, dtype=np.float64)

    k0 = coeff["k0"]
    k1 = coeff["k1"]
    w = coeff["w"]
    valid_target = coeff["valid_target"]

    ntarget, npoints = k0.shape
    out_t = np.full((ntarget, npoints), np.nan, dtype=np.float64)

    point_idx = np.broadcast_to(np.arange(npoints, dtype=np.int64), (ntarget, npoints))
    valid = valid_target & (k0 >= 0) & (k1 >= 0)
    if np.any(valid):
        v0 = field[point_idx[valid], k0[valid]]
        v1 = field[point_idx[valid], k1[valid]]
        good = np.isfinite(v0) & np.isfinite(v1)
        vals = np.full(v0.shape, np.nan, dtype=np.float64)
        vals[good] = (1.0 - w[valid][good]) * v0[good] + w[valid][good] * v1[good]
        out_t[valid] = vals

    return out_t.T

if HAS_NUMBA:
    @njit(parallel=True)
    def apply_interp_coefficients_by_level_numba(field, k0, k1, w, valid_target):
        """
        Apply coefficients in parallel over flattened (target level, point) space.
        """
        ntarget, npoints = k0.shape
        ntotal = ntarget * npoints
        out = np.full((npoints, ntarget), np.nan, dtype=np.float64)

        for ij in prange(ntotal):
            k = ij // npoints
            i = ij - k * npoints
            if valid_target[k, i]:
                kk0 = k0[k, i]
                kk1 = k1[k, i]
                if kk0 >= 0 and kk1 >= 0:
                    v0 = field[i, kk0]
                    v1 = field[i, kk1]
                    if np.isfinite(v0) and np.isfinite(v1):
                        out[i, k] = (1.0 - w[k, i]) * v0 + w[k, i] * v1

        return out
else:
    def apply_interp_coefficients_by_level_numba(field, k0, k1, w, valid_target):
        raise ImportError("numba is not installed")


def apply_interp_by_level(
    field: np.ndarray,
    coeff: Dict[str, np.ndarray],
    use_numba: bool = True,
) -> np.ndarray:
    """
    Apply interpolation coefficients using level-parallel numba if available.
    """
    field = np.asarray(field, dtype=np.float64)

    if use_numba and HAS_NUMBA:
        return apply_interp_coefficients_by_level_numba(
            field,
            coeff["k0"],
            coeff["k1"],
            coeff["w"],
            coeff["valid_target"],
        )

    return apply_interp_coefficients_by_level_numpy(field, coeff)


def interpolate_to_levels(
    field: np.ndarray,
    vert_coord: np.ndarray,
    target_levels: np.ndarray,
    log_space: bool = False,
    use_numba: bool = True,
) -> np.ndarray:
    """
    Backward-compatible wrapper.
    """
    coeff = build_interp_coefficients_by_level(
        vert_coord,
        target_levels,
        log_space=log_space,
        use_numba=use_numba,
    )
    return apply_interp_by_level(field, coeff, use_numba=use_numba)


def build_shared_interp_coeffs(
    support: Dict[str, np.ndarray],
    config,
    use_numba: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build shared interpolation coefficient caches once.

    Expected keys in support:
      - init_z
      - init_p
      - fcst_p_mean
    """
    hgt_levels = np.asarray(config.hgt_levels, dtype=np.float64)
    pres_levels = np.asarray(config.pres_levels, dtype=np.float64)

    coeffs = {
        "hgt_from_init_z": build_interp_coefficients_by_level(
            support["init_z"],
            hgt_levels,
            log_space=False,
            use_numba=use_numba,
        ),
        "pres_from_init_p": build_interp_coefficients_by_level(
            support["init_p"],
            pres_levels,
            log_space=True,
            use_numba=use_numba,
        ),
        "pres_from_fcst_p_mean": build_interp_coefficients_by_level(
            support["fcst_p_mean"],
            pres_levels,
            log_space=True,
            use_numba=use_numba,
        ),
    }
    return coeffs
