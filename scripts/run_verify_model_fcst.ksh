#!/bin/ksh
#####################################
# DIRECTIVES:
#PBS -N mpas_vefy_model
#PBS -o mpas_vefy_model.out
#PBS -e mpas_vefy_model.err
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -k eod
#PBS -P your_project_id
#PBS -m ae
#PBS -l select=1:ncpus=4:mpiprocs=1:mem=64GB

set -e
set -u
set -o pipefail

#####################################
# Experiment
#####################################

export exp_name=exp_control

export ini_date=2024091706
export end_date=2024092700

# Cycling interval
export verify_interval=6

# --------------------------------------------------
# Verification mode
#
#   background : verify background fields
#   forecast   : verify extended forecasts
# --------------------------------------------------
export verify_mode=forecast

# --------------------------------------------------
# Forecast type
#
#   deterministic
#   ensemble
# --------------------------------------------------
export forecast_type=deterministic

# --------------------------------------------------
# Cycling interval settings
# --------------------------------------------------
export cycling_interval=1

# --------------------------------------------------
# Extended forecast settings
# --------------------------------------------------
export forecast_length=24
export forecast_interval=6

#####################################
# Directories
#####################################

export main_dir=/path/to/regional_mpasjedi

export init_dir=/path/to/ExternalAnalyses

export package_dir=/path/to/MPAS-STAT
export PYTHONPATH="${package_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

export working_dir=${main_dir}/${exp_name}/VerifyModel

#####################################
# MPAS / ensemble settings
#####################################

export mesh_prefix='x1.522172'

export nMember=1
export nProcesses=4

export ens_prefix="mem"
export ens_digits="03"

#####################################
# Retry settings
#####################################

export max_retries=3
export retry_delay=30

ulimit -v unlimited

#####################################
# Check settings
#####################################

if [[ "${verify_mode}" != "background" && \
      "${verify_mode}" != "forecast" ]]; then

    echo "ERROR: invalid verify_mode=${verify_mode}"
    exit 1
fi

if [[ "${forecast_type}" != "deterministic" && \
      "${forecast_type}" != "ensemble" ]]; then

    echo "ERROR: invalid forecast_type=${forecast_type}"
    exit 1
fi

#####################################
# Main cycling loop
#####################################

idate=${ini_date}

while [[ ${idate} -le ${end_date} ]]; do

    echo
    echo "=================================================="
    echo "Cycle         : ${idate}"
    echo "Verification  : ${verify_mode}"
    echo "Forecast type : ${forecast_type}"
    echo "=================================================="

    #################################
    # Cycle date
    #################################

    YY=$(echo ${idate} | cut -c1-4)
    MM=$(echo ${idate} | cut -c5-6)
    DD=$(echo ${idate} | cut -c7-8)
    HH=$(echo ${idate} | cut -c9-10)

    cycle_date_file="${YY}-${MM}-${DD}_${HH}.00.00"

    #################################
    # Forecast directory
    #################################
    
    if [[ "${verify_mode}" == "background" ]]; then
	jdate=$(date -d "${YY}-${MM}-${DD} ${HH} - ${cycling_interval} hours" "+%Y%m%d%H")
        forecast_dir="${main_dir}/${exp_name}/CyclingFC/${jdate}"
    else
	forecast_dir="${main_dir}/${exp_name}/ExtendedFC/${idate}"
    fi

    #################################
    # Background / initial condition
    #################################

    initFile="${init_dir}/${idate}/${mesh_prefix}.init.${cycle_date_file}.nc"

    #################################
    # Set forecast hours
    #################################

    if [[ "${verify_mode}" == "background" ]]; then
        forecast_start=0
        forecast_end=0
        forecast_step=1
    else
        forecast_start=${forecast_interval}
        forecast_end=${forecast_length}
        forecast_step=${forecast_interval}

    fi

    #################################
    # Forecast-time loop
    #################################

    forecast_hour=${forecast_start}

    while [[ ${forecast_hour} -le ${forecast_end} ]]; do

        #################################
        # Valid time
        #################################

        valid_date=$(date \
            -d "${YY}-${MM}-${DD} ${HH} + ${forecast_hour} hours" \
            "+%Y%m%d%H")

        VYY=$(echo ${valid_date} | cut -c1-4)
        VMM=$(echo ${valid_date} | cut -c5-6)
        VDD=$(echo ${valid_date} | cut -c7-8)
        VHH=$(echo ${valid_date} | cut -c9-10)

        valid_date_file="${VYY}-${VMM}-${VDD}_${VHH}.00.00"

        #################################
        # Working directory
        #################################

        if [[ "${verify_mode}" == "background" ]]; then
            verify_dir="${working_dir}/${idate}/background"
        else
            verify_dir="${working_dir}/${idate}/f$(printf '%03d' ${forecast_hour})"
        fi

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
        echo "Verification  : ${verify_mode}"
        echo "Forecast type : ${forecast_type}"
        echo "--------------------------------------------------"

        #################################
        # Configuration
        #################################

        cp "${package_dir}/configs/model_input.yaml" input.yaml

        #################################
        # Forecast file
        #################################

        if [[ "${forecast_type}" == "ensemble" ]]; then

            forecastFile="${forecast_dir}/${ens_prefix}{:${ens_digits}d}/mpasout.${valid_date_file}.nc"

        else

            forecastFile="${forecast_dir}/mpasout.${valid_date_file}.nc"

        fi

        #################################
        # Output
        #################################

        outputFile="${verify_dir}/mpasstat.${valid_date_file}.nc"

        #################################
        # YAML configuration
        #################################

        sed -i \
            -e "s|forecastFile|${forecastFile}|g" \
            -e "s|initFile|${initFile}|g" \
            -e "s|outputFile|${outputFile}|g" \
            -e "s|nMember|${nMember}|g" \
            -e "s|nProcesses|${nProcesses}|g" \
            input.yaml

        #################################
        # Run
        #################################

        attempt=1
        status=1

        while [[ ${attempt} -le ${max_retries} ]]; do

            echo "Attempt ${attempt}/${max_retries}"

            if python -m mpas_stat.cli.model_stats -c input.yaml \
                > verify_model.log 2>&1; then
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

            exit ${status}

        fi

        #################################
        # Next forecast time
        #################################

        forecast_hour=$((forecast_hour + forecast_step))

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
