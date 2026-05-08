#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "devices" "" "" "string" "required"
define_arg "cfg" "" "" "string" "required"
define_arg "obj" "" "" "string" "required"
define_arg "loc" "" "" "string" "required"
define_arg "benchmark" "" "" "string" "required"

parse_args "$@"

export LOGURU_LEVEL='WARNING'

SCRIPT=benchmark_sv.py

export CUDA_VISIBLE_DEVICES=$devices

# Load hyperparameters from file
config_file="${cfg}_${obj}_${loc}.sh"
config_file="../axbench/configs/${config_file}"
if [ ! -e ${config_file} ]; then
  echo -e "${config_file} does not exist!"
  exit 1
fi
source ${config_file}

# MODEL_BASE_DIR=/home/Dataset/Models
# MODEL_BASE_DIR=${HOME}/Models
MODEL_BASE_DIR=/mnt/data/byt/Models
if [[ ${cfg} == 2b_l10 ]]; then
  MODEL_PATH=google/gemma-2-2b-it
elif [[ ${cfg} == 9b_l20 ]]; then
  MODEL_PATH=google/gemma-2-9b-it
elif [[ ${cfg} == q25_32b_l32 ]]; then
  # MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
  MODEL_PATH=unsloth/Qwen2.5-32B-Instruct-bnb-4bit
elif [[ ${cfg} == l3_8b_l16 ]]; then
  MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
elif [[ ${cfg} == q3_4b_l18 ]]; then
  MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507-FP8
elif [[ ${cfg} == o2_1b_l8 ]]; then
  MODEL_PATH=allenai/OLMo-2-0425-1B-Instruct
else
  echo "Unknown configuration: ${cfg}"
  exit 1
fi
MODEL_PATH=${MODEL_BASE_DIR}/${MODEL_PATH}

BENCHMARK_NAME=${benchmark}
if [[ ${BENCHMARK_NAME} == tinymmlu ]]; then
  BENCHMARK_PATH=tinyBenchmarks/tinyMMLU
  NUM_SHOTS=0 # few-shots are fixed by the benchmark
  MAX_NEW_TOKENS=128
elif [[ ${BENCHMARK_NAME} == tinygsm8k ]]; then
  BENCHMARK_PATH=tinyBenchmarks/tinyGSM8k
  NUM_SHOTS=0 # few-shots are fixed by the benchmark
  MAX_NEW_TOKENS=512
elif [[ ${BENCHMARK_NAME} == mmlu ]]; then
  BENCHMARK_PATH=cais/mmlu
  NUM_SHOTS=5
  MAX_NEW_TOKENS=5
elif [[ ${BENCHMARK_NAME} == mbpp ]]; then
  BENCHMARK_PATH=google-research-datasets/mbpp
  NUM_SHOTS=3
  MAX_NEW_TOKENS=512
else
  echo -e "\n!!!!!!!!!!!!!!!!"
  echo -e "Benchmark unknown or not specified"
  echo -e "!!!!!!!!!!!!!!!!\n"
fi
BENCHMARK_ROOT_DIR=/mnt/data/byt/Dataset
BENCHMARK_PATH=${BENCHMARK_ROOT_DIR}/${BENCHMARK_PATH}

CONCEPT_DATA_DIR=../../data/concept500/prod_${cfg}_v1/generate/

CKPT_DIR=/mnt/data/byt/reft_data/axbench/${cfg}/outputs_${SV}/${POSITIONS}/${obj}
SAVE_DIR=/mnt/data/byt/reft_data/benchmark/${cfg}/outputs_${SV}/${POSITIONS}/${obj}

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || Method: ${SV} || Positions: ${POSITIONS}"
echo -e "####################################################################\n"


START=$(/usr/bin/date +%s)

python ${SCRIPT} \
  --model_path ${MODEL_PATH} \
  --ckpt_dir ${CKPT_DIR} \
  --save_dir ${SAVE_DIR} \
  --concept_data_dir ${CONCEPT_DATA_DIR} \
  --batch_size 4 \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --positions ${POSITIONS} \
  --num_shots ${NUM_SHOTS} \
  --benchmark_name ${BENCHMARK_NAME} \
  --benchmark_hf_path ${BENCHMARK_PATH} \
  --max_concepts 10 \
  # --prompt_steering
  # --no_interventions

END=$(/usr/bin/date +%s)
total_seconds=$(( $END - $START ))
d=$((total_seconds / 86400))
h=$(((total_seconds % 86400) / 3600))
m=$(((total_seconds % 3600) / 60))
s=$((total_seconds % 60))

echo -e "\n#############################################################"
echo -e "Finished at $(date) || Elapsed: ${d}d, ${h}h, ${m}m, ${s}s"
echo -e "#############################################################\n"
