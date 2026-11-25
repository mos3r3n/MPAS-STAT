#!/usr/bin/env python3
"""MET Python-embedding script for the radar GridStat run.

GridStat runs it twice per cycle, once for the forecast and once for the
observations, through the commands that
:func:`mpas_stat.radar.common._build_gridstat_conf` puts in
``FCST_VAR1_NAME`` / ``OBS_VAR1_NAME``::

    python3 metplus_embed.py /scratch/.../data.npz FCST
    python3 metplus_embed.py /scratch/.../data.npz OBS

It loads the cache written by
:func:`mpas_stat.radar.common._write_metplus_npz_cache` and sets two
module-level names, which MET reads from the script's namespace:

``met_data``
    2-D ``float64`` array, the gridded field.
``attrs``
    Metadata dict: ``valid``, ``init``, ``lead``, ``accum``, ``name``,
    ``long_name``, ``level``, ``units`` and ``grid`` (a regular lat-lon
    grid description).

It does no gridding or date arithmetic of its own; everything comes from
the cache.

.. note::
   Grid cells with no observation are NaN, on the assumption that MET's
   Python embedding treats NaN as missing data. Check this first if a
   whole domain scores as missing, or if empty cells are being scored.
"""
import sys

import numpy as np

if __name__ != "__main__":
    # MET's Python embedding execs this file as a script (it is not
    # imported as a module), but guard anyway so a stray `import
    # mpas_stat.radar.metplus_embed` elsewhere (e.g. from a test) doesn't try to
    # parse sys.argv.
    met_data = np.full((1, 1), np.nan)
    attrs = {}
else:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: src/mpas_stat/radar/metplus_embed.py <npz_cache_path> <FCST|OBS>\n")
        sys.exit(1)

    _cache_path, _role = sys.argv[1], sys.argv[2].upper()
    if _role not in ("FCST", "OBS"):
        sys.stderr.write(f"src/mpas_stat/radar/metplus_embed.py: ROLE must be FCST or OBS, got {sys.argv[2]!r}\n")
        sys.exit(1)

    with np.load(_cache_path, allow_pickle=True) as _z:
        met_data = np.asarray(_z["fcst_grid" if _role == "FCST" else "obs_grid"], dtype=np.float64)
        grid = dict(
            type="LatLon",
            name=str(_z["grid_name"].item()),
            lat_ll=float(_z["lat_ll"]),
            lon_ll=float(_z["lon_ll"]),
            delta_lat=float(_z["delta_lat"]),
            delta_lon=float(_z["delta_lon"]),
            Nlat=int(_z["nlat"]),
            Nlon=int(_z["nlon"]),
        )
        attrs = dict(
            valid=str(_z["valid"].item()),
            init=str(_z["init"].item()),
            lead=str(_z["lead"].item()),
            accum=str(_z["accum"].item()),
            name=str(_z["name"].item()),
            long_name=f"{str(_z['long_name'].item())} ({_role})",
            level=str(_z["level"].item()),
            units=str(_z["units"].item()),
            grid=grid,
        )
