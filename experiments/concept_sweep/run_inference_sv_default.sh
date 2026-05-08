#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "devices" "" "" "string" "required"
define_arg "cfg" "" "" "string" "required"
define_arg "sv" "" "" "string" "required"
define_arg "pos" "" "" "string" "required"

parse_args "$@"

export LOGURU_LEVEL='WARNING'

SCRIPT=inference_sv.py

ADAPTER=$sv
POSITIONS=$pos

# SEED="$3"

export CUDA_VISIBLE_DEVICES=$devices

START=$(/usr/bin/date +%s)

# MODEL_BASE_DIR=/home/Dataset/Models
MODEL_BASE_DIR=/mnt/data/byt/Models
if [[ ${cfg} == 2b_l10 ]]; then
  MODEL_PATH=google/gemma-2-2b-it
elif [[ ${cfg} == 9b_l20 ]]; then
  MODEL_PATH=google/gemma-2-9b-it
elif [[ ${cfg} == l3_8b_l16 ]]; then
  MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
elif [[ ${cfg} == l3_3b_l13 ]]; then
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
elif [[ ${cfg} == q3_4b_l18 ]]; then
  MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507-FP8
elif [[ ${cfg} == o2_1b_l8 ]]; then
  MODEL_PATH=allenai/OLMo-2-0425-1B-Instruct
elif [[ ${cfg} == q25_32b_l32 ]]; then
  MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
else
  echo "Unknown configuration: ${cfg}"
  exit 1
fi
MODEL_PATH=${MODEL_BASE_DIR}/${MODEL_PATH}

CONCEPT_DATA_DIR=../../data/concept10/prod_${cfg}_v1/generate/
ALPACA_EVAL_PATH=../../data/alpaca_eval.json
# OUTPUT_DIR=${HOME}/share/reft_data/concept_sweep/${cfg}/outputs_${ADAPTER}/
OUTPUT_BASE_DIR=/mnt/data/byt/reft_data/concept_sweep_grid_default/${cfg}/outputs_${ADAPTER}/${POSITIONS}

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || Method: ${ADAPTER} || Positions: ${POSITIONS}"
echo -e "####################################################################\n"

SEED_ARR=(42 43 44)
# SEED_ARR=(42)
FACTOR_LR_ARR=(1e-2 1e-1 1 10 20 40)
FACTOR_INIT_ARR=(1 2 4 8 16)

python ${SCRIPT} \
  --seed_list ${SEED_ARR[@]} \
  --factor_learning_rate_list ${FACTOR_LR_ARR[@]} \
  --factor_init_scale_list ${FACTOR_INIT_ARR[@]} \
  --model_path ${MODEL_PATH} \
  --concept_data_dir ${CONCEPT_DATA_DIR} \
  --output_dir ${OUTPUT_BASE_DIR} \
  --alpaca_eval_path ${ALPACA_EVAL_PATH} \
  --batch_size 10 \
  --max_new_tokens 128 \
  --positions ${POSITIONS}

END=$(/usr/bin/date +%s)
total_seconds=$(( $END - $START ))
d=$((total_seconds / 86400))
h=$(((total_seconds % 86400) / 3600))
m=$(((total_seconds % 3600) / 60))
s=$((total_seconds % 60))

echo -e "\n#########################################################"
echo -e "Finished at $(date) || Elapsed: ${d}d, ${h}h, ${m}m, ${s}s"
echo -e "#########################################################\n"
