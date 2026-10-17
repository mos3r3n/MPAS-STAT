#!/bin/ksh
#####################################
# DIRECTIVES:
#PBS -N mpas_vefy_obs
#PBS -o mpas_vefy_obs.out
#PBS -e mpas_vefy_obs.err
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -k eod
#PBS -P your_project_id
#PBS -m ae
#PBS -l select=1:ncpus=8:mpiprocs=1:mem=64GB

set -e
set -u
set -o pipefail

#####################################
# Observation-space verification -- OMB/OMA (background/analysis)
#
# Verifies IODA obs-space innovations from CYCLING DATA ASSIMILATION runs:
#   <da_dir>/<exp_name>/CyclingDA/<cycle>/dbOut/obsout_da_<obs>.h5    (OMB/OMA)
#   <hofx_dir>/<exp_name>/<cycle>/dbOut/obsout_hofx_<obs>.h5         (OMB only,
#                                                                      exp_type=hofx)
#
# Unlike run_verify_model.ksh, this script does NOT loop per cycle: both
# src/mpas_stat/obs/diag.py and src/mpas_stat/obs/plot.py already sweep the
# whole [start_time, end_time] cycle range (and every configured sensor)
# internally from a single config file, so this job runs the two stages
# exactly once each. See run_verify_obs_fcst.ksh for the companion OMF
# (obs-minus-forecast, extended-forecast skill) verification job.
#####################################

#####################################
# Experiment(s) to verify/compare -- space-separated, same length lists.
#   exp_types entries: "da" (OMB+OMA, CyclingDA) | "hofx" (OMB only)
#   exp_outerloop: which analysis outer-loop's OMA to read (exp_type=da,
#                  exp_nmember==1 only -- see exp_nmember note below)
#   exp_nmember:   >1 enables ensemble spread. Also changes WHERE OMA is
#                  read from (exp_type=da only): with exp_nmember==1
#                  (deterministic DA), OMB and OMA are separate groups in
#                  the SAME obsout_da_<obs>.h5 file; with exp_nmember>1
#                  (ensemble DA, e.g. LETKF/GETKF), OMA instead lives in a
#                  SEPARATE obsout_da_<obs>_ana.h5 file alongside it --
#                  handled automatically by build_filename() in
#                  src/mpas_stat/obs/diag.py.
#####################################

export exp_names="exp_control exp_test"
export exp_labels="Control Test"
export exp_types="da da"
export exp_outerloop="2 2"
export exp_nmember="1 1"

#####################################
# Cycle range
#####################################

export start_time="2024-09-13 06:00:00"
export end_time="2024-09-21 12:00:00"
export cycle_interval=6

#####################################
# Sensors (see src/mpas_stat/obs/sensor_config.py's OBSERVATION_CONFIGS for valid keys)
#####################################

export sensors="sondes profiler aircraft satwind satwnd sfc gnssrorefncep amsua_n18 amsua_n19 amsua_metop-b amsua_metop-c mhs_n19 mhs_metop-b mhs_metop-c"

#####################################
# Domains to compute/plot (apply_domains) -- keys into the "domains" map
# written into the config below; edit that map directly to add a new one.
#####################################

export apply_domains="global NXTro"

#####################################
# Plot / statistics settings
#####################################

export plot_oma=true
export figure_format=png
export max_time_ticks=12
export generate_lat_lon_2d=true
export latlon_lat_bin_width=10.0
export latlon_lon_bin_width=10.0
# Which `domains` entry (defined below) bounds the 2-D lat-lon map grid.
# Leave as "global" for a global application; for a REGIONAL MPAS-JEDI
# application, point this at a domain covering the mesh's own extent (add
# one to the "domains" map below if needed) so the 2-D map -- and its
# resolution, set by latlon_lat/lon_bin_width above -- covers only the area
# that actually has observations instead of the whole globe.
export latlon_domain=global

#####################################
# 2-D REAL-experiment comparison figures (lat-lon map, lev-time, lev-lat):
# experiments.names[0] (first exp_names entry above) plots as the absolute
# reference; every other configured experiment plots as a bootstrap %
# reduction vs. it, for whichever statistics are listed here -- mirroring
# the model-space package's own relative_reduction_statistics, but scoped
# to just these 2-D products (0D time series / 1D profiles keep their
# existing OMB-solid/OMA-dashed absolute-valued convention regardless).
# OMB and OMA are drawn as two separate sets of figures under omb/ and
# oma/ subfolders since they can't share this real-experiment axis with
# the reference/reduction convention at the same time.
#####################################

export comparison_relative_reduction_statistics="RMS STD"
export generate_time_level_2d=true   # cycle x level Hovmoller, reuses the existing 1D profile groups
export generate_lat_vert_2d=true     # latitude x level cross-section (new 2-D statistic; skipped for radiances)

#####################################
# Directories
#####################################

export main_dir=/path/to/regional_mpasjedi
export da_dir=${main_dir}
export hofx_dir=${main_dir}
export graphic_dir=/path/to/verify_obs
export diag_dir=""      # blank -> defaults to <graphic_dir>/diag_files
export package_dir=/path/to/MPAS-STAT
export PYTHONPATH="${package_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

#####################################
# Retry settings
#####################################

export max_retries=3
export retry_delay=30

ulimit -v unlimited

#####################################
# Working directory
#####################################

mkdir -p "${graphic_dir}"
cd "${graphic_dir}"

#####################################
# Helpers: space-separated env var -> quoted YAML flow list
#####################################

yaml_list() {
    # $1 = space-separated tokens; numeric tokens are left unquoted.
    out="["
    first=1
    for tok in $1; do
        if [[ ${first} -eq 0 ]]; then out="${out}, "; fi
        first=0
        case "${tok}" in
            ''|*[!0-9.\-]*) out="${out}\"${tok}\"" ;;   # non-numeric -> quote
            *)               out="${out}${tok}" ;;       # numeric -> bare
        esac
    done
    out="${out}]"
    echo "${out}"
}

#####################################
# Build the obs-space YAML config (both stages read the same file)
#####################################

config_file="${graphic_dir}/obs_verify.yaml"

{
    echo "paths:"
    echo "  da_dir: ${da_dir}"
    echo "  hofx_dir: ${hofx_dir}"
    echo "  diag_dir: ${diag_dir:-null}"
    echo "  graphic_dir: ${graphic_dir}"
    echo
    echo "experiments:"
    echo "  names: $(yaml_list "${exp_names}")"
    echo "  labels: $(yaml_list "${exp_labels}")"
    echo "  types: $(yaml_list "${exp_types}")"
    echo "  outerloop: $(yaml_list "${exp_outerloop}")"
    echo "  n_member: $(yaml_list "${exp_nmember}")"
    echo
    echo "sensors: $(yaml_list "${sensors}")"
    echo
    echo "domains:"
    echo "  global: [null, null, null, null]"
    echo "  NXTro: [30.0, 90.0, -180.0, 180.0]"
    echo "  SXTro: [-90.0, -30.0, -180.0, 180.0]"
    echo "  NTro: [5.0, 30.0, -180.0, 180.0]"
    echo "  STro: [-30.0, -5.0, -180.0, 180.0]"
    echo "  ITCZ: [-5.0, 5.0, -180.0, 180.0]"
    echo "  Tro: [-23.5, 23.5, -180.0, 180.0]"
    echo
    echo "plot_settings:"
    echo "  plot_oma: ${plot_oma}"
    echo "  apply_domains: $(yaml_list "${apply_domains}")"
    echo "  figure_format: ${figure_format}"
    echo "  max_time_ticks: ${max_time_ticks}"
    echo "  generate_lat_lon_2d: ${generate_lat_lon_2d}"
    echo "  latlon_domain: ${latlon_domain}"
    echo "  latlon_lat_bin_width: ${latlon_lat_bin_width}"
    echo "  latlon_lon_bin_width: ${latlon_lon_bin_width}"
    echo "  comparison_relative_reduction_statistics: $(yaml_list "${comparison_relative_reduction_statistics}")"
    echo "  generate_time_level_2d: ${generate_time_level_2d}"
    echo "  generate_lat_vert_2d: ${generate_lat_vert_2d}"
    echo
    echo "time_settings:"
    echo "  start_time: \"${start_time}\""
    echo "  end_time: \"${end_time}\""
    echo "  interval: ${cycle_interval}"
} > "${config_file}"

echo "Wrote config: ${config_file}"

#####################################
# Run one stage with retry-on-failure (mirrors run_verify_model.ksh)
#####################################

run_with_retry() {
    description=$1
    shift

    attempt=1
    status=1

    while [[ ${attempt} -le ${max_retries} ]]; do
        echo "Attempt ${attempt}/${max_retries}: ${description}"

        if "$@" > "${description}.log" 2>&1; then
            status=0
        else
            status=$?
        fi

        if [[ ${status} -eq 0 ]]; then
            echo "SUCCESS: ${description}"
            break
        fi

        echo "WARNING: failed attempt ${attempt} for ${description}"

        if [[ ${attempt} -lt ${max_retries} ]]; then
            echo "Retrying in ${retry_delay}s..."
            sleep ${retry_delay}
        fi

        ((attempt++))
    done

    if [[ ${status} -ne 0 ]]; then
        echo "ERROR: ${description} failed after ${max_retries} attempts"
        echo "Check: ${graphic_dir}/${description}.log"
        exit ${status}
    fi
}

#####################################
# Run
#####################################

run_with_retry verify_obs_stats  python -m mpas_stat.cli.obs_stats "${config_file}"
run_with_retry verify_obs_plots  python -m mpas_stat.cli.obs_graphics "${config_file}"

echo
echo "=================================================="
echo "Observation-space (OMB/OMA) verification completed successfully"
echo "=================================================="
