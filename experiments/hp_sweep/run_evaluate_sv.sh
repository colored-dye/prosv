#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "cfg" "" "" "string" "required"
define_arg "sv" "" "" "string" "required"
define_arg "pos" "" "" "string" "required"
define_arg "obj" "" "" "string" "required"

parse_args "$@"

export LOGURU_LEVEL='WARNING'

SCRIPT=evaluate_sv.py

ADAPTER=$sv
POSITIONS=$pos

BATCH_SIZE_ARR=(3 6 12)
EPOCHS_ARR=(6 12 18 24 30)
VEC_INIT_ARR=(1 8 16)
VEC_LR_ARR=(4e-2 8e-2)
FACTOR_LR_ARR=(1e-2 1e-1 1 10 20 40)
FACTOR_INIT_ARR=(1 2 4 8 16)

CONCEPT_DATA_DIR=../../data/hp_tuning/
OUTPUT_BASE_DIR=/mnt/data/byt/reft_data/hp_sweep/${cfg}/outputs_${ADAPTER}/${POSITIONS}/${obj}

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || Method: ${ADAPTER} || Positions: ${POSITIONS} || Obj: ${obj}"
echo -e "####################################################################\n"

START=$(/usr/bin/date +%s)

python ${SCRIPT} \
  --concept_data_dir ${CONCEPT_DATA_DIR} \
  --output_dir ${OUTPUT_BASE_DIR} \
  --batch_size 10 \
  --positions ${POSITIONS} \
  --batch_size_list ${BATCH_SIZE_ARR[@]} \
  --epochs_list ${EPOCHS_ARR[@]} \
  --vector_learning_rate_list ${VEC_LR_ARR[@]} \
  --vector_init_scale_list ${VEC_INIT_ARR[@]} \
  --factor_learning_rate_list ${FACTOR_LR_ARR[@]} \
  --factor_init_scale_list ${FACTOR_INIT_ARR[@]}


END=$(/usr/bin/date +%s)
total_seconds=$(( $END - $START ))
d=$((total_seconds / 86400))
h=$(((total_seconds % 86400) / 3600))
m=$(((total_seconds % 3600) / 60))
s=$((total_seconds % 60))

echo -e "\n################################################"
echo -e "Finished at $(date) || Elapsed: ${d}d, ${h}h, ${m}m, ${s}s"
echo -e "################################################\n"
