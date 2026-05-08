#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "devices" "" "" "string" "required"
define_arg "cfg" "" "" "string" "required"
define_arg "sv" "" "" "string" "required"
define_arg "pos" "" "" "string" "required"
define_arg "obj" "" "" "string" "required"

parse_args "$@"

if [[ ${obj} == bisimpo && ${sv} != pref ]]; then
  echo -e "\n!!!!!!!!!!!!!!!!!!!!"
  echo -e "bisimpo objective and pref SV are binded!"
  echo -e "!!!!!!!!!!!!!!!!!!!!\n"
  exit 1
fi

export LOGURU_LEVEL='WARNING'

SCRIPT=sweep_sv.py

ADAPTER=$sv
POSITIONS=$pos

export CUDA_VISIBLE_DEVICES=$devices

# EPOCHS_ARR=(6 12 18)
# VEC_LR_ARR=(4e-2 8e-2)
EPOCHS_ARR=(12)
VEC_LR_ARR=(4e-2)

# MODEL_BASE_DIR=/home/Dataset/Models
MODEL_BASE_DIR=/mnt/data/byt/Models

GRAD_ACCUM_STEPS=1

if [[ ${obj} == simpo ]]; then
  BATCH_SIZE_ARR=(6)
elif [[ ${obj} == lang ]]; then
  BATCH_SIZE_ARR=(12)
elif [[ ${obj} == bisimpo ]]; then
  BATCH_SIZE_ARR=(3)
  EPOCHS_ARR=(12 18)
  VEC_LR_ARR=(8e-2)
fi

if [[ ${pos} == all ]]; then
  FACTOR_LR_ARR=(1e-2 1e-1 1)
  FACTOR_INIT_ARR=(1 2)
  VEC_INIT_ARR=(1 8)
elif [[ ${pos} == f2+l2 ]]; then
  FACTOR_LR_ARR=(1e-1 1 10 20)
  FACTOR_INIT_ARR=(1 2 4 8)
  VEC_INIT_ARR=(8)
fi

if [[ ${cfg} == 2b_l10 ]]; then
  MODEL_PATH=google/gemma-2-2b-it
  LAYERS=(10)
elif [[ ${cfg} == 2b_l20 ]]; then
  MODEL_PATH=google/gemma-2-2b-it
  LAYERS=(20)

  if [[ ${pos} == f2+l2 ]]; then
    EPOCHS_ARR=(12 18)
  elif [[ ${pos} == all_prompt ]]; then
    FACTOR_LR_ARR=(1 10 20)
    FACTOR_INIT_ARR=(1 2)
    VEC_INIT_ARR=(8)
  fi
elif [[ ${cfg} == 9b_l20 ]]; then
  MODEL_PATH=google/gemma-2-9b-it
  LAYERS=(20)
  BATCH_SIZE_ARR=(3 6)
  # VEC_LR_ARR=(4e-2 8e-2)
  VEC_INIT_ARR=(1 8)
  VEC_LR_ARR=(8e-2)
  # EPOCHS_ARR=(18 24)
  EPOCHS_ARR=(6 12 18)

  if [[ ${obj} == bisimpo ]]; then
    BATCH_SIZE_ARR=(3)
    EPOCHS_ARR=(18)
    GRAD_ACCUM_STEPS=2
    VEC_LR_ARR=(8e-2)
    VEC_INIT_ARR=(1)
    FACTOR_INIT_ARR=(1 2)
    FACTOR_LR_ARR=(1e-2 1e-1)
    GRAD_ACCUM_STEPS=2
  fi
elif [[ ${cfg} == 9b_l31 ]]; then
  MODEL_PATH=google/gemma-2-9b-it
  LAYERS=(31)
elif [[ ${cfg} == q25_32b_l32 ]]; then
  MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
  LAYERS=(32)
  # BATCH_SIZE_ARR=(3 6)
  # EPOCHS_ARR=(6 12 18)
  # VEC_LR_ARR=(4e-2 8e-2)
  if [[ ${obj} == bisimpo ]]; then
    BATCH_SIZE_ARR=(3)
    EPOCHS_ARR=(18)
    GRAD_ACCUM_STEPS=2
    VEC_LR_ARR=(8e-2)
    VEC_INIT_ARR=(1)
    FACTOR_INIT_ARR=(1 2)
    FACTOR_LR_ARR=(1e-2 1e-1)
    GRAD_ACCUM_STEPS=2
  fi
elif [[ ${cfg} == g3_12b_l22 ]]; then
  MODEL_PATH=google/gemma-3-12b-it
  LAYERS=(22)
  BATCH_SIZE_ARR=(6 12)
  VEC_INIT_ARR=(8)
  VEC_LR_ARR=(8e-2)
  EPOCHS_ARR=(12 18 24)

  if [[ ${obj} == simpo ]]; then
    BATCH_SIZE_ARR=(3 6)
  fi
elif [[ ${cfg} == 27b_l21 ]]; then
  MODEL_PATH=unsloth/gemma-2-27b-it-bnb-4bit
  LAYERS=(21)
  BATCH_SIZE_ARR=(12)
  VEC_INIT_ARR=(8 1)
  VEC_LR_ARR=(4e-2 8e-2)
  EPOCHS_ARR=(12 18)

  if [[ ${obj} == simpo ]]; then
    BATCH_SIZE_ARR=(6)
  fi
else
  echo "Unknown configuration: ${cfg}"
  exit 1
fi
MODEL_PATH=${MODEL_BASE_DIR}/${MODEL_PATH}

CONCEPT_DATA_DIR=../../data/hp_tuning/
OUTPUT_BASE_DIR=/mnt/data/byt/reft_data/hp_sweep/${cfg}/outputs_${ADAPTER}/${POSITIONS}/${obj}

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || Method: ${ADAPTER} || Positions: ${POSITIONS} || Obj: ${obj}"
echo -e "####################################################################\n"

START=$(/usr/bin/date +%s)

python ${SCRIPT} \
  --model_path ${MODEL_PATH} \
  --concept_data_dir ${CONCEPT_DATA_DIR} \
  --output_dir ${OUTPUT_BASE_DIR} \
  --layers ${LAYERS[@]} \
  --low_rank_dim 1 \
  --adapter_type ${ADAPTER} \
  --positions ${POSITIONS} \
  --optimizer adam \
  --batch_size_list ${BATCH_SIZE_ARR[@]} \
  --epochs_list ${EPOCHS_ARR[@]} \
  --vector_learning_rate_list ${VEC_LR_ARR[@]} \
  --vector_init_scale_list ${VEC_INIT_ARR[@]} \
  --factor_learning_rate_list ${FACTOR_LR_ARR[@]} \
  --factor_init_scale_list ${FACTOR_INIT_ARR[@]} \
  --objective $obj \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  # --prefix_length 4

END=$(/usr/bin/date +%s)
total_seconds=$(( $END - $START ))
d=$((total_seconds / 86400))
h=$(((total_seconds % 86400) / 3600))
m=$(((total_seconds % 3600) / 60))
s=$((total_seconds % 60))

echo -e "\n#########################################################"
echo -e "Finished at $(date) || Elapsed: ${d}d, ${h}h, ${m}m, ${s}s"
echo -e "#########################################################\n"
