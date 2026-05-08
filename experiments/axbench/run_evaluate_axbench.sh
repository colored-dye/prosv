#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "cfg" "" "2b_l10; 9b_l20; 27b_l21; g3_12b_l22; q25_32b_l32" "string" "required"
define_arg "obj" "" "lang; simpo; none" "string" "required"
define_arg "loc" "" "all; prosv; prompt" "string" "required"

parse_args "$@"

export LOGURU_LEVEL='WARNING'

SCRIPT=evaluate_axbench.py

# Load hyperparameters from file
config_file="${cfg}_${obj}_${loc}.sh"
if [ ! -e configs/${config_file} ]; then
  echo -e "configs/${config_file} does not exist!"
  exit 1
fi
source configs/${config_file}

# MODEL_BASE_DIR=/home/Dataset/Models
MODEL_BASE_DIR=/mnt/data/byt/Models
if [[ ${cfg} == 2b_l* ]]; then
  MODEL_PATH=google/gemma-2-2b-it
elif [[ ${cfg} == 9b_l* ]]; then
  MODEL_PATH=google/gemma-2-9b-it
elif [[ ${cfg} == q25_32b_l* ]]; then
  MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
elif [[ ${cfg} == 27b_l* ]]; then
  MODEL_PATH=unsloth/gemma-2-27b-it-bnb-4bit
elif [[ ${cfg} == g3_12b_l* ]]; then
  MODEL_PATH=google/gemma-3-12b-it
else
  echo "Unknown configuration: ${cfg}"
  exit 1
fi
MODEL_PATH=${MODEL_BASE_DIR}/${MODEL_PATH}

CONCEPT_DATA_DIR=../../data/concept500/prod_${cfg}_v1/generate/
if [[ ${obj} == simpo ]]; then
  CONCEPT_DATA_DIR=../../data/concept500_contrast/prod_${cfg}_v1/generate/
fi

OUTPUT_BASE_DIR=/mnt/data/byt/reft_data/axbench/${cfg}/outputs_${SV}/${POSITIONS}/${obj}

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || Method: ${SV} || Positions: ${POSITIONS} || Obj: ${obj}"
echo -e "####################################################################\n"

START=$(/usr/bin/date +%s)

python ${SCRIPT} \
  --model_path ${MODEL_PATH} \
  --concept_data_dir ${CONCEPT_DATA_DIR} \
  --output_dir ${OUTPUT_BASE_DIR} \
  --batch_size 10 \
  --positions ${POSITIONS} \
  --max_new_tokens 128 \
  --max_concepts 100


END=$(/usr/bin/date +%s)
total_seconds=$(( $END - $START ))
d=$((total_seconds / 86400))
h=$(((total_seconds % 86400) / 3600))
m=$(((total_seconds % 3600) / 60))
s=$((total_seconds % 60))

echo -e "\n################################################"
echo -e "Finished at $(date) || Elapsed: ${d}d, ${h}h, ${m}m, ${s}s"
echo -e "################################################\n"
