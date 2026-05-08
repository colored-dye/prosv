#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "cfg" "" "2b_l10; 9b_l20; q25_32b_l32" "string" "required"
define_arg "obj" "" "lang; simpo; none" "string" "required"
define_arg "loc" "" "all; prosv; prompt" "string" "required"

parse_args "$@"

export LOGURU_LEVEL='WARNING'

SCRIPT=suppressed_evaluate_sv.py

# Load hyperparameters from file
config_file="${cfg}_${obj}_${loc}.sh"
config_file="../axbench/configs/${config_file}"
if [ ! -e ${config_file} ]; then
  echo -e "${config_file} does not exist!"
  exit 1
fi
source ${config_file}

if [[ ${obj} == lang ]]; then
  CONCEPT_DATA_DIR=../../data/concept500/prod_${cfg}_v1/generate/
elif [[ ${obj} == simpo ]]; then
  CONCEPT_DATA_DIR=../../data/concept500_contrast/prod_${cfg}_v1/generate/
else
  CONCEPT_DATA_DIR=../../data/concept500/prod_${cfg}_v1/generate/
fi

OUTPUT_BASE_DIR=/mnt/data/byt/reft_data/adv_robustness/${cfg}/outputs_${SV}/${POSITIONS}/${obj}

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || Method: ${SV} || Positions: ${POSITIONS} || Obj: ${obj}"
echo -e "####################################################################\n"

START=$(/usr/bin/date +%s)

python ${SCRIPT} \
  --concept_data_dir ${CONCEPT_DATA_DIR} \
  --output_dir ${OUTPUT_BASE_DIR} \
  --batch_size 10 \
  --positions ${POSITIONS} \
  --max_concepts 15


END=$(/usr/bin/date +%s)
total_seconds=$(( $END - $START ))
d=$((total_seconds / 86400))
h=$(((total_seconds % 86400) / 3600))
m=$(((total_seconds % 3600) / 60))
s=$((total_seconds % 60))

echo -e "\n################################################"
echo -e "Finished at $(date) || Elapsed: ${d}d, ${h}h, ${m}m, ${s}s"
echo -e "################################################\n"
