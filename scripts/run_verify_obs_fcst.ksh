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
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=64GB

set -e
set -u
set -o pipefail

#####################################
# Observation-space verification -- OMF (extended-forecast skill)
#
# Same layout as run_verify_model_fcst.ksh: one job per experiment, one
# statistics run per (initialization cycle, forecast hour), written into
# the experiment's own folder:
#
#   input : ${main_dir}/${exp_name}/HofX/<init>/f<HHH>/dbOut/obsout_hofx_<obs>.h5
#   output: ${main_dir}/${exp_name}/VerifyOBS/<init>/f<HHH>/obsstat_<obs>.<valid>.nc
#
# Figures are drawn separately, for all experiments together:
#   python -m mpas_stat.cli.obs_graphics -c configs/obs_graphic.yaml
# with exp types "hofx_forecast" and a "forecast:" block in that config.
#####################################

#####################################
# Experiment
#####################################

export exp_name=exp_control

export ini_date=2024091306
export end_date=2024092112

# Interval between forecast initializations
export verify_interval=6

# --------------------------------------------------
# Extended forecast settings
# --------------------------------------------------
export forecast_length=24
export forecast_interval=6

# Ensemble size of the HofX forecast (>1: ensemble spread)
export nMember=1

# --------------------------------------------------
# Experiments to compare against
#
# OMF statistics are computed on ONE common observation sample: only
# observations valid in this experiment AND in every experiment listed
# here (same init cycle and forecast hour) are used. Their files are only
# read, never written. Leave empty to verify on this experiment's own sample.
# --------------------------------------------------
export compare_exps="exp_test"

#####################################
# Sensors
# (keys of OBSERVATION_CONFIGS in src/mpas_stat/obs/sensor_config.py)
#####################################

export sensors="sondes aircraft satwind satwnd sfc amsua_n18 amsua_n19"

#####################################
# Python environment
#
# Conda activation hooks often read variables that are not set yet (e.g.
# GEOMETRIC_DATA_DIR in geometric_features-activate.sh), which aborts the
# job under `set -u`. Relax it while activating.
#####################################

set +u
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate mpas_mesh
set -u

#####################################
# Directories
#####################################

export main_dir=/path/to/regional_mpasjedi

export package_dir=/path/to/MPAS-STAT
export PYTHONPATH="${package_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

export working_dir=${main_dir}/${exp_name}/VerifyOBS

#####################################
# Retry settings
#####################################

export max_retries=3
export retry_delay=30

ulimit -v unlimited

#####################################
# Helper: space-separated words -> YAML flow list
#####################################

yaml_list() {
    out=""
    for tok in $1; do
        out="${out:+${out}, }${tok}"
    done
    echo "[${out}]"
}

repeat_list() {
    # $1 = value, $2 = count -> "value value ..."
    out=""
    i=0
    while [[ ${i} -lt $2 ]]; do
        out="${out:+${out} }$1"
        i=$((i + 1))
    done
    echo "${out}"
}

all_exps="${exp_name} ${compare_exps}"
n_exps=$(echo ${all_exps} | wc -w)

#####################################
# Main cycling loop
#####################################

idate=${ini_date}

while [[ ${idate} -le ${end_date} ]]; do

    echo
    echo "=================================================="
    echo "Cycle         : ${idate}"
    echo "Experiment    : ${exp_name}"
    echo "Compared with : ${compare_exps:-none}"
    echo "=================================================="

    #################################
    # Cycle date
    #################################

    YY=$(echo ${idate} | cut -c1-4)
    MM=$(echo ${idate} | cut -c5-6)
    DD=$(echo ${idate} | cut -c7-8)
    HH=$(echo ${idate} | cut -c9-10)

    cycle_time="${YY}-${MM}-${DD} ${HH}:00:00"

    #################################
    # Forecast-time loop
    #################################

    forecast_hour=${forecast_interval}

    while [[ ${forecast_hour} -le ${forecast_length} ]]; do

        valid_date=$(date \
            -d "${YY}-${MM}-${DD} ${HH} + ${forecast_hour} hours" \
            "+%Y%m%d%H")

        #################################
        # Working directory
        #################################

        verify_dir="${working_dir}/${idate}/f$(printf '%03d' ${forecast_hour})"

        mkdir -p "${verify_dir}"
        cd "${verify_dir}"

        #################################
        # Information
        #################################

        echo
        echo "--------------------------------------------------"
        echo "Cycle         : ${idate}"
        echo "Valid time    : ${valid_date}"
        echo "Forecast hour : ${forecast_hour}"
        echo "--------------------------------------------------"

        #################################
        # Configuration
        #################################

        cp "${package_dir}/configs/obs_input.yaml" input.yaml

        sed -i \
            -e "s|expNames|$(yaml_list "${all_exps}")|g" \
            -e "s|expTypes|$(yaml_list "$(repeat_list hofx_forecast ${n_exps})")|g" \
            -e "s|expOuterloop|$(yaml_list "$(repeat_list 0 ${n_exps})")|g" \
            -e "s|expNmember|$(yaml_list "$(repeat_list ${nMember} ${n_exps})")|g" \
            -e "s|writeExp|$(yaml_list "${exp_name}")|g" \
            -e "s|sensorList|$(yaml_list "${sensors}")|g" \
            -e "s|mainDir|${main_dir}|g" \
            -e "s|workDir|${verify_dir}|g" \
            -e "s|cycleTime|${cycle_time}|g" \
            -e "s|cycleInterval|${verify_interval}|g" \
            -e "s|forecastStart|${forecast_hour}|g" \
            -e "s|forecastLength|${forecast_hour}|g" \
            -e "s|forecastInterval|${forecast_interval}|g" \
            input.yaml

        #################################
        # Run
        #################################

        attempt=1
        status=1

        while [[ ${attempt} -le ${max_retries} ]]; do

            echo "Attempt ${attempt}/${max_retries}"

            if python -m mpas_stat.cli.obs_stats -c input.yaml \
                > verify_obs.log 2>&1; then
                status=0
            else
                status=$?
            fi

            if [[ ${status} -eq 0 ]]; then
                echo "SUCCESS"
                break
            fi

            echo "WARNING: failed attempt ${attempt}"

            if [[ ${attempt} -lt ${max_retries} ]]; then
                echo "Retrying in ${retry_delay}s..."
                sleep ${retry_delay}
            fi

            ((attempt++))

        done

        #################################
        # Check
        #################################

        if [[ ${status} -ne 0 ]]; then

            echo "ERROR: verification failed"
            echo "Cycle         : ${idate}"
            echo "Valid time    : ${valid_date}"
            echo "Forecast hour : ${forecast_hour}"
            echo "Directory     : ${verify_dir}"
            echo "Check         : ${verify_dir}/verify_obs.log"

            exit ${status}

        fi

        #################################
        # Next forecast time
        #################################

        forecast_hour=$((forecast_hour + forecast_interval))

    done

    #################################
    # Next cycle
    #################################

    idate=$(date \
        -d "${YY}-${MM}-${DD} ${HH} + ${verify_interval} hours" \
        "+%Y%m%d%H")

done

echo
echo "=================================================="
echo "Verification completed successfully"
echo "=================================================="
