#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "devices" "" "" "string" "required"
define_arg "cfg" "" "2b_l10; 9b_l20; 27b_l21; g3_12b_l22; q25_32b_l32" "string" "required"
define_arg "obj" "" "lang; simpo" "string" "required"
define_arg "loc" "" "all; prosv" "string" "required"

parse_args "$@"

export LOGURU_LEVEL='WARNING'
export CUDA_VISIBLE_DEVICES=${devices}

SCRIPT=train_axbench.py

GRAD_ACCUM_STEPS=1

# Load hyperparameters from file
config_file="${cfg}_${obj}_${loc}.sh"
config_file=${config_file}
if [ ! -e configs/${config_file} ]; then
  echo -e "configs/${config_file} does not exist!"
  exit 1
fi
source configs/${config_file}

# MODEL_BASE_DIR=${HOME}/Models
MODEL_BASE_DIR=/mnt/data/byt/Models
if [[ ${cfg} == 2b_l10 ]]; then
  MODEL_PATH=google/gemma-2-2b-it
  LAYERS=(10)
elif [[ ${cfg} == 2b_l20 ]]; then
  MODEL_PATH=google/gemma-2-2b-it
  LAYERS=(20)
elif [[ ${cfg} == 9b_l20 ]]; then
  MODEL_PATH=google/gemma-2-9b-it
  LAYERS=(20)
elif [[ ${cfg} == 9b_l31 ]]; then
  MODEL_PATH=google/gemma-2-9b-it
  LAYERS=(31)
elif [[ ${cfg} == q25_32b_l32 ]]; then
  MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
  LAYERS=(32)
elif [[ ${cfg} == 27b_l21 ]]; then
  MODEL_PATH=unsloth/gemma-2-27b-it-bnb-4bit
  LAYERS=(21)
elif [[ ${cfg} == g3_12b_l22 ]]; then
  MODEL_PATH=google/gemma-3-12b-it
  LAYERS=(22)
else
  echo "Unknown configuration: ${cfg}"
  exit 1
fi
MODEL_PATH=${MODEL_BASE_DIR}/${MODEL_PATH}

CONCEPT_DATA_DIR=../../data/concept500/prod_${cfg}_v1/generate/
if [[ ${obj} == simpo ]]; then
  CONCEPT_DATA_DIR=../../data/concept500_contrast/prod_${cfg}_v1/generate/
fi

OUTPUT_ROOT_DIR=/mnt/data/byt/reft_data
OUTPUT_BASE_DIR=${OUTPUT_ROOT_DIR}/axbench/${cfg}/outputs_${SV}/${POSITIONS}/${obj}

mkdir -p ${OUTPUT_BASE_DIR}
cp configs/${config_file} ${OUTPUT_BASE_DIR}/${config_file}

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || Method: ${SV} || Positions: ${POSITIONS} || Obj: ${obj} || Finit: ${FACTOR_INIT} || Flr: ${FACTOR_LR} || Vinit: ${VEC_INIT} || Vlr: ${VEC_LR}"
echo -e "####################################################################\n"

START=$(/usr/bin/date +%s)

python ${SCRIPT} \
  --model_path ${MODEL_PATH} \
  --concept_data_dir ${CONCEPT_DATA_DIR} \
  --output_dir ${OUTPUT_BASE_DIR} \
  --layers ${LAYERS[@]} \
  --adapter_type ${SV} \
  --positions ${POSITIONS} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --vector_learning_rate ${VEC_LR} \
  --vector_init_scale ${VEC_INIT} \
  --factor_learning_rate ${FACTOR_LR} \
  --factor_init_scale ${FACTOR_INIT} \
  --objective ${obj} \
  --max_concepts 100 \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS}
  # --max_output_length 100

END=$(/usr/bin/date +%s)
total_seconds=$(( $END - $START ))
d=$((total_seconds / 86400))
h=$(((total_seconds % 86400) / 3600))
m=$(((total_seconds % 3600) / 60))
s=$((total_seconds % 60))

echo -e "\n#########################################################"
echo -e "Finished at $(date) || Elapsed: ${d}d, ${h}h, ${m}m, ${s}s"
echo -e "#########################################################\n"
