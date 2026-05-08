#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "devices" "" "" "string" "required"
define_arg "cfg" "" "" "string" "required"
define_arg "sv" "" "" "string" "required"
define_arg "pos" "" "" "string" "required"

parse_args "$@"

export LOGURU_LEVEL='WARNING'

SCRIPT=sweep_sv.py

ADAPTER=$sv
POSITIONS=$pos

export CUDA_VISIBLE_DEVICES=$devices

BATCH_SIZE=12

# MODEL_BASE_DIR=/home/Dataset/Models
MODEL_BASE_DIR=/mnt/data/byt/Models
if [[ ${cfg} == 2b_l10 ]]; then
  MODEL_PATH=google/gemma-2-2b-it
  LAYERS=(10)
  VEC_INIT=6.92
elif [[ ${cfg} == 9b_l20 ]]; then
  MODEL_PATH=google/gemma-2-9b-it
  LAYERS=(20)
  VEC_INIT=7.737
elif [[ ${cfg} == l3_8b_l15 ]]; then
  MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
  LAYERS=(15)
  VEC_INIT=8
elif [[ ${cfg} == l3_3b_l13 ]]; then
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
  LAYERS=(13)
  VEC_INIT=8
elif [[ ${cfg} == q3_4b_l18 ]]; then
  MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507-FP8
  LAYERS=(18)
elif [[ ${cfg} == o2_1b_l8 ]]; then
  MODEL_PATH=allenai/OLMo-2-0425-1B-Instruct
  LAYERS=(8)
  VEC_INIT=8
elif [[ ${cfg} == q25_32b_l32 ]]; then
  MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
  LAYERS=(32)
  VEC_INIT=8.459
  BATCH_SIZE=6
else
  echo "Unknown configuration: ${cfg}"
  exit 1
fi
MODEL_PATH=${MODEL_BASE_DIR}/${MODEL_PATH}

LR=4e-2

SEED_ARR=(42 43 44)
# SEED_ARR=(42)
FACTOR_LR_ARR=(1e-2 1e-1 1 10 20 40)
FACTOR_INIT_ARR=(1 2 4 8 16)

CONCEPT_DATA_DIR=../../data/concept10/prod_${cfg}_v1/generate/

if (( $(echo "${VEC_INIT} == 1.0" | bc -l) )); then
  save_key=default
else
  save_key=larger_vec
fi
OUTPUT_BASE_DIR=/mnt/data/byt/reft_data/concept_sweep_grid_${save_key}/${cfg}/outputs_${ADAPTER}/${POSITIONS}

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || Method: ${ADAPTER} || Positions: ${POSITIONS}"
echo -e "####################################################################\n"

START=$(/usr/bin/date +%s)

# for FACTOR_LR in ${FACTOR_LR_ARR[@]}; do
#   for FACTOR_INIT in ${FACTOR_INIT_ARR[@]}; do
#     for SEED in ${SEED_ARR[@]}; do

#       echo -e "\n===================================================================="
#       echo -e "Seed: ${SEED} || Factor lr: ${FACTOR_LR} || Factor init: ${FACTOR_INIT}"
#       echo -e "====================================================================\n"

#       OUTPUT_DIR=${OUTPUT_BASE_DIR}/seed=${SEED}_scale=${FACTOR_INIT}_lr=${FACTOR_LR}
#       if [[ -e ${OUTPUT_DIR}/0/state_dict.pt ]]; then
#         echo "Skipping"
#         continue
#       fi

python ${SCRIPT} \
  --seed_list ${SEED_ARR[@]} \
  --model_path ${MODEL_PATH} \
  --concept_data_dir ${CONCEPT_DATA_DIR} \
  --output_dir ${OUTPUT_BASE_DIR} \
  --layers ${LAYERS[@]} \
  --learning_rate ${LR} \
  --factor_learning_rate_list ${FACTOR_LR_ARR[@]} \
  --factor_init_scale_list ${FACTOR_INIT_ARR[@]} \
  --vector_init_scale ${VEC_INIT} \
  --epochs 6 \
  --low_rank_dim 1 \
  --batch_size ${BATCH_SIZE} \
  --adapter_type ${ADAPTER} \
  --positions ${POSITIONS} \
  --optimizer adam

#       if [[ $? -ne 0 ]]; then
#         echo -e "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#         echo -e "Error! Early exit"
#         echo -e "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
#         exit 1
#       fi

#     done
#   done
# done

END=$(/usr/bin/date +%s)
total_seconds=$(( $END - $START ))
d=$((total_seconds / 86400))
h=$(((total_seconds % 86400) / 3600))
m=$(((total_seconds % 3600) / 60))
s=$((total_seconds % 60))

echo -e "\n#########################################################"
echo -e "Finished at $(date) || Elapsed: ${d}d, ${h}h, ${m}m, ${s}s"
echo -e "#########################################################\n"
