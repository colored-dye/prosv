#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "devices" "" "" "string" "required"
define_arg "cfg" "" "2b_l10; 9b_l20; q25_32b_l32" "string" "required"
define_arg "obj" "" "lang; simpo; none" "string" "required"
define_arg "loc" "" "all; prosv; prompt" "string" "required"

parse_args "$@"

export LOGURU_LEVEL='WARNING'

SCRIPT=suppressed_inference_sv.py

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
MODEL_BASE_DIR=/mnt/data/byt/Models
if [[ ${cfg} == 2b_l* ]]; then
  MODEL_PATH=google/gemma-2-2b-it
elif [[ ${cfg} == 9b_l* ]]; then
  MODEL_PATH=google/gemma-2-9b-it
elif [[ ${cfg} == q25_32b_l* ]]; then
  # MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
  MODEL_PATH=unsloth/Qwen2.5-32B-Instruct-bnb-4bit
else
  echo "Unknown configuration: ${cfg}"
  exit 1
fi
MODEL_PATH=${MODEL_BASE_DIR}/${MODEL_PATH}

CONCEPT_DATA_DIR=../../data/concept500/prod_${cfg}_v1/generate/

CKPT_BASE_DIR=/mnt/data/byt/reft_data/axbench/${cfg}/outputs_${SV}/${POSITIONS}/${obj}
SAVE_BASE_DIR=/mnt/data/byt/reft_data/adv_robustness/${cfg}/outputs_${SV}/${POSITIONS}/${obj}

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || SV: ${SV} || Positions: ${POSITIONS}"
echo -e "####################################################################\n"

START=$(/usr/bin/date +%s)

python ${SCRIPT} \
  --model_path ${MODEL_PATH} \
  --concept_data_dir ${CONCEPT_DATA_DIR} \
  --ckpt_dir ${CKPT_BASE_DIR} \
  --save_dir ${SAVE_BASE_DIR} \
  --batch_size 10 \
  --max_new_tokens 128 \
  --positions ${POSITIONS} \
  --max_concepts 15

END=$(/usr/bin/date +%s)
total_seconds=$(( $END - $START ))
d=$((total_seconds / 86400))
h=$(((total_seconds % 86400) / 3600))
m=$(((total_seconds % 3600) / 60))
s=$((total_seconds % 60))

msg="Finished at $(date) || Elapsed: ${d}d, ${h}h, ${m}m, ${s}s"
n=${#msg}
banner=$(printf "%0.s#" $(seq 1 $n))
echo -e "\n${banner}"
echo $msg
echo -e "${banner}\n"
