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
#PBS -l select=1:ncpus=16:mpiprocs=1:mem=64GB

set echo

export exp_name=exp_control
export ini_date=2024091700
export end_date=2024092700
export cycle_period=6
export fcst_range=1
export main_dir=/path/to/regional_mpasjedi
export init_dir=/path/to/ExternalAnalyses
export package_dir=/path/to/MPAS-STAT
export PYTHONPATH="${package_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

export mesh_prefix='x1.522172'
export nMember=32
export nProcesses=16

# ensemble format
export ens_prefix="mem"
export ens_digits="03"

# safe mode
export max_retries=3
export retry_delay=30

ulimit -v unlimited

# Working dir
export working_dir=${main_dir}/${exp_name}/VerifyModel

idate=${ini_date}
while [[ ${idate} -le ${end_date} ]]; do

   echo "running job for: $idate"

   export YY=$(echo $idate | cut -c1-4)
   export MM=$(echo $idate | cut -c5-6)
   export DD=$(echo $idate | cut -c7-8)
   export HH=$(echo $idate | cut -c9-10)
   export DATE_FILE=${YY}-${MM}-${DD}_${HH}.00.00

   mkdir -p ${working_dir}/${idate}

   export jdate=$(date -d "${YY}-${MM}-${DD} ${HH} - ${fcst_range} hours" "+%Y%m%d%H")

   cd ${working_dir}/${idate}

   cp "${package_dir}/configs/model_input.yaml" input.yaml
   
   export forecastFile="${main_dir}/${exp_name}/CyclingFC/${jdate}/${ens_prefix}{:${ens_digits}d}/mpasout.${DATE_FILE}.nc"
   export initFile=${init_dir}/${idate}/${mesh_prefix}.init.${DATE_FILE}.nc
   export outputFile=${working_dir}/${idate}/mpasstat.${DATE_FILE}.nc

   sed -i "s|forecastFile|${forecastFile}|g" input.yaml
   sed -i "s|initFile|${initFile}|g" input.yaml
   sed -i "s|outputFile|${outputFile}|g" input.yaml
   sed -i "s|nMember|${nMember}|g" input.yaml
   sed -i "s|nProcesses|${nProcesses}|g" input.yaml

   ## The following will run the scripts multiple times
   attempt=1
   status=1

   while [[ ${attempt} -le ${max_retries} ]]; do
      echo "Attempt ${attempt}/${max_retries} for ${idate}"

      python -m mpas_stat.cli.model_stats -c input.yaml > verify_model.log 2>&1
      status=$?

      if [[ ${status} -eq 0 ]]; then
         echo "SUCCESS at ${idate} (attempt ${attempt})"
         break
      fi

      echo "WARNING: failed attempt ${attempt} for ${idate}"

      if [[ ${attempt} -lt ${max_retries} ]]; then
         echo "Retrying in ${retry_delay}s..."
         sleep ${retry_delay}
      fi

      ((attempt++))
   done

   # final check
   if [[ ${status} -ne 0 ]]; then
      echo "ERROR: failed after ${max_retries} attempts at ${idate}"
      echo "Check: ${working_dir}/${idate}/verify_model.log"
      exit ${status}
   fi

   # advance to next cycle
   export idate=$(date -d "${YY}-${MM}-${DD} ${HH} + ${cycle_period} hours" "+%Y%m%d%H")

done
