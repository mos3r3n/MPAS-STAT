#!/usr/bin/env python3
"""
MET Python-embedding script for the radar composite-reflectivity /
METplus grid_stat integration (see radar_verify_common.py's
process_one_cycle() and its module docstring for the full design).

grid_stat invokes this script TWICE per cycle -- once to supply the
forecast field, once for the observation field -- via a command line
built from the METplus conf's FCST_VAR1_NAME / OBS_VAR1_NAME
(radar_verify_common._build_gridstat_conf() sets those to the literal
command below), e.g.:

    python3 radar_metplus_embed.py /scratch/.../data.npz FCST
    python3 radar_metplus_embed.py /scratch/.../data.npz OBS

It must set two names at MODULE scope before it exits -- MET's Python
embedding support reads them directly out of this script's namespace, not
from a return value or stdout:
  met_data : 2-D numpy.ndarray (float64) -- the gridded field.
  attrs    : dict -- metadata MET needs to treat met_data as a proper
             gridded field: valid, init, lead, accum, name, long_name,
             level, units, grid (grid itself a dict describing a regular
             LatLon grid -- see below).

Deliberately minimal: this script does NOT grid anything itself, does NOT
know any real file paths (HofX obsout files, diagnosis files, etc.), and
does NOT do any date/time arithmetic of its own -- every value below was
already computed, once, by radar_verify_common._write_metplus_npz_cache()
in the orchestrating process (which already has the real datetimes and
grid spacing in hand from gridding the cycle). This script's only job is
to load that one cache file back and present it the way MET expects. That
is a deliberate design choice -- see process_one_cycle()'s module
docstring -- to avoid depending on METplus's own filename-into-script-
argument templating conventions, which were not confirmed with certainty
against documentation during this integration: the only things this
script depends on from its own command line are the two plain positional
arguments below (a literal, fully-resolved cache path, and a role flag),
nothing METplus-templated.

ASSUMPTION flagged for on-system validation -- please check this first if
results look wrong (e.g. the whole domain scoring as missing, or missing
cells being scored instead of skipped): no MET/METplus installation was
available in the sandbox this integration was developed in, so this
script could not be run inside an actual grid_stat invocation. It is
written on the documented assumption that MET's Python embedding treats a
non-finite (NaN) met_data value as missing/bad data -- matching how
calc_composite_reflectivity() already marks "no observation fell in this
grid cell" -- but that specific behavior could not be exercised against a
real MET build here. A single-cycle dry run on your own system (real
run_metplus.py, real MET) is the one step of this whole integration most
worth doing before trusting results.
"""
import sys

import numpy as np

if __name__ != "__main__":
    # MET's Python embedding execs this file as a script (it is not
    # imported as a module), but guard anyway so a stray `import
    # radar_metplus_embed` elsewhere (e.g. from a test) doesn't try to
    # parse sys.argv.
    met_data = np.full((1, 1), np.nan)
    attrs = {}
else:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: radar_metplus_embed.py <npz_cache_path> <FCST|OBS>\n")
        sys.exit(1)

    _cache_path, _role = sys.argv[1], sys.argv[2].upper()
    if _role not in ("FCST", "OBS"):
        sys.stderr.write(f"radar_metplus_embed.py: ROLE must be FCST or OBS, got {sys.argv[2]!r}\n")
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
