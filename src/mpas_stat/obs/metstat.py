"""MET-format ``.stat`` output for observation-space statistics.

Each statistics file ``obsstat_<obs>.<date>.nc`` gets a companion
``obsstat_<obs>.<date>.stat`` holding the same pairs as MET line types, so
the MET/METplus tools can be used on them directly: ``stat_analysis`` (for
example ``-job aggregate_stat -line_type SL1L2 -out_line_type CNT`` to pool
cycles), METviewer and METplotpy.

Line types written (column layout of MET 12, see the MET User's Guide,
Point-Stat and Ensemble-Stat output):

``SL1L2``
    Scalar partial sums TOTAL FBAR OBAR FOBAR FFBAR OOBAR MAE, with the
    forecast F = H(x) and the observation O = ObsValue. One line per
    variable and domain (``FCST_LEV ALL``) and per vertical bin.
``ECNT``
    Ensembles: TOTAL, N_ENS, ME, RMSE, SPREAD, SPREAD_PLUS_OERR, CRPS_EMP,
    CRPS_EMP_FAIR, MAE (of the ensemble mean); the other columns are NA.
``RHIST``
    Ensembles, O−B only: the rank histogram (observations perturbed with
    their error before ranking, unlike MET's default).

How the header columns are filled:

* ``MODEL`` = experiment, ``DESC`` = ``OMB``, ``OMA`` or ``OMF``,
  ``OBTYPE`` = sensor.
* ``FCST_LEAD``: the cycle interval for O−B (the background is a forecast
  from the previous cycle), 0 for O−A, the forecast lead for O−F.
* ``FCST_VALID_BEG`` = ``FCST_VALID_END`` = the valid time.
* ``FCST_LEV``: ``P<hPa>`` for pressure bins, ``Z<m>`` for height bins,
  ``ALL`` for the whole column. Radiance latitude bins are written as
  ``VX_MASK`` ``<domain>_LAT<deg>``.
* ``INTERP_MTHD`` ``NA``, ``INTERP_PNTS`` 1; thresholds and ``ALPHA`` NA.

Note MET's sign convention: ME = mean(F − O), the opposite of the mean
departure O − F shown in the observation-space figures.
"""
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np

from ..graphics.labels import plain_units

MET_VERSION = "V12.0.0"

HEADER = ["VERSION", "MODEL", "DESC", "FCST_LEAD", "FCST_VALID_BEG", "FCST_VALID_END",
          "OBS_LEAD", "OBS_VALID_BEG", "OBS_VALID_END", "FCST_VAR", "FCST_UNITS",
          "FCST_LEV", "OBS_VAR", "OBS_UNITS", "OBS_LEV", "OBTYPE", "VX_MASK",
          "INTERP_MTHD", "INTERP_PNTS", "FCST_THRESH", "OBS_THRESH", "COV_THRESH",
          "ALPHA", "LINE_TYPE"]

SL1L2_COLUMNS = ["TOTAL", "FBAR", "OBAR", "FOBAR", "FFBAR", "OOBAR", "MAE"]

ECNT_COLUMNS = ["TOTAL", "N_ENS", "CRPS", "CRPSS", "IGN", "ME", "RMSE", "SPREAD",
                "ME_OERR", "RMSE_OERR", "SPREAD_OERR", "SPREAD_PLUS_OERR", "CRPSCL",
                "CRPS_EMP", "CRPSCL_EMP", "CRPSS_EMP", "CRPS_EMP_FAIR", "SPREAD_MD",
                "MAE", "MAE_OERR", "BIAS_RATIO", "N_GE_OBS", "ME_GE_OBS", "N_LT_OBS",
                "ME_LT_OBS", "IGN_CONV_OERR", "IGN_CORR_OERR"]


def _num(x) -> str:
    """A number in MET text form: NA when missing, integers without a point."""
    if x is None:
        return "NA"
    x = float(x)
    if not np.isfinite(x):
        return "NA"
    if x == int(x) and abs(x) < 1.0e15:
        return str(int(x))
    return f"{x:.12g}"


def _lead(hours: float) -> str:
    """MET lead time HHMMSS (hours may exceed 99)."""
    total = int(round(float(hours) * 3600))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}{m:02d}{sec:02d}"


def _time(t) -> str:
    return t.strftime("%Y%m%d_%H%M%S")


def _token(s) -> str:
    s = "NA" if s is None or str(s) == "" else str(s)
    return s.replace(" ", "_")


def _units(u) -> str:
    """Plain-text units (m/s, K, g/kg) as one token."""
    return _token(plain_units(u) if u else None)


def _var_name(var, bin_method: int) -> str:
    return f"brightnessTemperature_{var}" if bin_method == 3 else str(var)


def _level_name(level: float, bin_method: int) -> str:
    if bin_method == 1:
        return f"P{level:g}"
    if bin_method == 2:
        return f"Z{level * 1000.0:g}"
    return "ALL"


def sl1l2_values(get) -> List[str]:
    """SL1L2 columns from ``get(prefix)`` (prefixes as in MET_SL1L2_KEYS)."""
    return [_num(get(p)) for p in ("sl_total", "sl_fbar", "sl_obar", "sl_fobar",
                                   "sl_ffbar", "sl_oobar", "sl_mae")]


def ecnt_values(get, n_ens: int) -> List[str]:
    """ECNT columns: CRPS and spread terms from ``get``; ME, RMSE and MAE of
    the ensemble mean from the SL1L2 sums of the same vector."""
    fbar, obar = get("sl_fbar"), get("sl_obar")
    ffbar, fobar, oobar = get("sl_ffbar"), get("sl_fobar"), get("sl_oobar")
    me = rmse = None
    if None not in (fbar, obar) and np.isfinite(fbar) and np.isfinite(obar):
        me = fbar - obar
        mse = ffbar - 2.0 * fobar + oobar
        rmse = np.sqrt(mse) if np.isfinite(mse) and mse >= 0 else None
    vals = dict.fromkeys(ECNT_COLUMNS, None)
    vals.update({
        "TOTAL": get("ec_total"), "N_ENS": n_ens, "ME": me, "RMSE": rmse,
        "SPREAD": get("ec_spread"), "SPREAD_PLUS_OERR": get("ec_spread_oerr"),
        "CRPS_EMP": get("ec_crps"), "CRPS_EMP_FAIR": get("ec_crps_fair"),
        "MAE": get("sl_mae"),
    })
    return [_num(vals[c]) for c in ECNT_COLUMNS]


def stat_lines(results: Dict, diag_obs: str, bin_method: int, exp_name: str,
               iexp: int, time_idx: tuple, diagtypes, valid_time, leads: Dict[str, float],
               n_member: int) -> List[str]:
    """Every MET line for one experiment and time index (see module doc)."""
    variables = list(results["variables"])
    units = list(results.get("units") or [None] * len(variables))
    domains = list(results["domain_names"])
    levels = np.asarray(results["levels"], dtype=np.float64)
    vt = _time(valid_time)

    def header(desc, lead, var, unit, lev, mask, line_type):
        return [MET_VERSION, _token(exp_name), desc, _lead(lead), vt, vt, "000000", vt, vt,
                _token(var), _units(unit), lev, _token(var), _units(unit), lev,
                _token(diag_obs), _token(mask), "NA", "1", "NA", "NA", "NA", "NA", line_type]

    lines = []
    for diag in diagtypes:
        desc, lead = diag.upper(), leads.get(diag, 0.0)
        if f"sl_total_{diag}" not in results:
            continue
        has_ecnt = n_member > 1 and f"ec_total_{diag}" in results
        for iv, var in enumerate(variables):
            vname, unit = _var_name(var, bin_method), units[iv] if iv < len(units) else None
            for idom, dom in enumerate(domains):
                base = (iexp, idom) + time_idx

                def get0(prefix):
                    return results[f"{prefix}_{diag}"][base + (iv,)]

                if get0("sl_total") > 0:
                    lines.append(header(desc, lead, vname, unit, "ALL", dom, "SL1L2")
                                 + sl1l2_values(get0))
                    if has_ecnt and get0("ec_total") > 0:
                        lines.append(header(desc, lead, vname, unit, "ALL", dom, "ECNT")
                                     + ecnt_values(get0, n_member))
                    if diag == "omb" and "rank_omb" in results and n_member > 1:
                        counts = np.asarray(results["rank_omb"][base + (iv,)])[:n_member + 1]
                        if counts.sum() > 0:
                            lines.append(header(desc, lead, vname, unit, "ALL", dom, "RHIST")
                                         + [_num(counts.sum()), _num(counts.size)]
                                         + [_num(c) for c in counts])

                if bin_method == 3 and dom != "global":
                    continue                      # radiance latitude bins: global only
                for il, lev in enumerate(levels):
                    def getb(prefix, il=il):
                        return results[f"{prefix}_{diag}_bin"][base + (il, iv)]
                    if not getb("sl_total") > 0:
                        continue
                    if bin_method == 3:
                        lev_name, mask = "ALL", f"{dom}_LAT{lev:g}"
                    else:
                        lev_name, mask = _level_name(lev, bin_method), dom
                    lines.append(header(desc, lead, vname, unit, lev_name, mask, "SL1L2")
                                 + sl1l2_values(getb))
                    if has_ecnt and getb("ec_total") > 0:
                        lines.append(header(desc, lead, vname, unit, lev_name, mask, "ECNT")
                                     + ecnt_values(getb, n_member))
    return [" ".join(cols) for cols in lines]


def write_stat_file(path: str, lines: List[str]) -> Optional[str]:
    """Write a .stat file (header row + lines); nothing when there are no lines."""
    if not lines:
        return None
    with open(path, "w") as f:
        f.write(" ".join(HEADER) + "\n")
        f.write("\n".join(lines) + "\n")
    return path


def cycling_leads(config: Dict) -> Dict[str, float]:
    """FCST_LEAD per diagtype for cycling files: the cycle interval for
    the background, 0 for the analysis."""
    interval = float((config.get("time_settings") or {}).get("interval", 6))
    return {"omb": interval, "oma": 0.0}


def forecast_valid_time(init_time, lead_hours) -> object:
    """Valid time of a forecast: init time plus the lead in hours."""
    return init_time + timedelta(hours=float(lead_hours))
