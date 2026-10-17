#!/bin/ksh
#####################################
# DIRECTIVES:
#PBS -N mpas_vefy_obs_fcst
#PBS -o mpas_vefy_obs_fcst.out
#PBS -e mpas_vefy_obs_fcst.err
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
# Observation-space verification -- OMF (obs-minus-forecast, extended-
# forecast skill)
#
# Verifies IODA obs-space innovations from EXTENDED-FORECAST HofX runs:
#   <hofx_dir>/<exp_name>/HofX/<init_cycle>/f<HHH>/dbOut/obsout_hofx_<obs>.h5
# for every configured forecast lead hour, producing the classic
# "statistic vs. forecast lead time" skill plots via plot_forecast_lead_0d.
#
# Companion to run_verify_obs.ksh (OMB/OMA, cycling DA) -- the two verify
# different things (short-range background/analysis fit vs. extended-
# forecast skill growth) and are normally run as separate jobs, though a
# single config/job could mix "da"/"hofx" and "hofx_forecast" experiments
# if desired (src/mpas_stat/obs/diag.py handles both in one pass). Like
# run_verify_obs.ksh, this script does NOT loop per cycle or per lead hour
# in the shell -- both python stages sweep the whole cycle range x lead-hour
# grid internally from a single config file.
#####################################

#####################################
# Experiment(s) to verify/compare -- space-separated, same length lists.
# exp_types must be "hofx_forecast" for every entry processed by this job.
#####################################

export exp_names="exp_control"
export exp_labels="Control"
export exp_types="hofx_forecast"
export exp_outerloop="0"
export exp_nmember="1"

#####################################
# Initialization-cycle range (one HofX forecast is launched per cycle)
#####################################

export start_time="2024-09-13 06:00:00"
export end_time="2024-09-21 12:00:00"
export cycle_interval=6

#####################################
# Extended forecast settings (same schema as the model-space graphic.yaml's
# "forecast:" block / run_verify_model_fcst.ksh's forecast_length/interval)
#####################################

export forecast_length=24
export forecast_interval=6
# export forecast_start=6   # optional; defaults to forecast_interval

#####################################
# Sensors (see src/mpas_stat/obs/sensor_config.py's OBSERVATION_CONFIGS for valid keys)
#####################################

export sensors="sondes aircraft satwind satwnd sfc amsua_n18 amsua_n19"

#####################################
# Domains to compute/plot (apply_domains)
#####################################

export apply_domains="global NXTro"

#####################################
# Plot / statistics settings
#####################################

export figure_format=png
export generate_lat_lon_2d=true
export latlon_lat_bin_width=10.0
export latlon_lon_bin_width=10.0
# Which `domains` entry (defined below) bounds the 2-D lat-lon map grid --
# see the matching note in run_verify_obs.ksh. Also applies to OMF's 2-D
# maps (process_forecast_experiment uses the same resolve_latlon_bounds()).
export latlon_domain=global

#####################################
# Directories
#####################################

export main_dir=/path/to/regional_mpasjedi
export da_dir=${main_dir}
export hofx_dir=${main_dir}
export graphic_dir=/path/to/verify_obs_fcst
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

config_file="${graphic_dir}/obs_verify_fcst.yaml"

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
    echo "  apply_domains: $(yaml_list "${apply_domains}")"
    echo "  figure_format: ${figure_format}"
    echo "  generate_lat_lon_2d: ${generate_lat_lon_2d}"
    echo "  latlon_domain: ${latlon_domain}"
    echo "  latlon_lat_bin_width: ${latlon_lat_bin_width}"
    echo "  latlon_lon_bin_width: ${latlon_lon_bin_width}"
    echo
    echo "time_settings:"
    echo "  start_time: \"${start_time}\""
    echo "  end_time: \"${end_time}\""
    echo "  interval: ${cycle_interval}"
    echo
    echo "forecast:"
    echo "  length: ${forecast_length}"
    echo "  interval: ${forecast_interval}"
    if [[ -n "${forecast_start:-}" ]]; then
        echo "  start: ${forecast_start}"
    fi
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

run_with_retry verify_obs_fcst_stats  python -m mpas_stat.cli.obs_stats "${config_file}"
run_with_retry verify_obs_fcst_plots  python -m mpas_stat.cli.obs_graphics "${config_file}"

echo
echo "=================================================="
echo "Observation-space (OMF extended-forecast) verification completed successfully"
echo "=================================================="
