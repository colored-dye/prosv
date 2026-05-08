#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
export LOGURU_LEVEL='WARNING'

SCRIPT=train_rep.py

ADAPTER="$1"
CONCEPT_ID=299

cfg='2b_l10'
# cfg='9b_l20'
cfg=l3_8b_l16
cfg=o2_1b_l8

echo -e "\n####################################################################"
echo -e "CFG: ${cfg} || Representation adapter: ${ADAPTER} || Concept id: ${CONCEPT_ID}"
echo -e "####################################################################\n"

if [[ ${cfg} == 2b_l10 ]]; then
    MODEL_PATH=/home/Dataset/Models/google/gemma-2-2b-it
    LAYERS=(10)
elif [[ ${cfg} == 9b_l20 ]]; then
    MODEL_PATH=/home/Dataset/Models/google/gemma-2-9b-it
    LAYERS=(20)
elif [[ ${cfg} == l3_8b_l16 ]]; then
    MODEL_PATH=$HOME/Models/meta-llama/Llama-3.1-8B-Instruct
    LAYERS=(16)
elif [[ ${cfg} == o2_1b_l8 ]]; then
    MODEL_PATH=$HOME/Models/allenai/OLMo-2-0425-1B-Instruct
    LAYERS=(8)
fi

OUTPUT_DIR=${HOME}/share/reft_data/concept/${cfg}/outputs_${ADAPTER}/
LR=5e-3
FACTOR_LR=10.0
FACTOR_INIT=4.0


python ${SCRIPT} \
    --seed 42 \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --layers ${LAYERS[@]} \
    --learning_rate ${LR} \
    --factor_learning_rate ${FACTOR_LR} \
    --epochs 30 \
    --concept_id ${CONCEPT_ID} \
    --low_rank_dim 1 \
    --batch_size 12 \
    --adapter_type ${ADAPTER} \
    --factor_init_scale ${FACTOR_INIT} \
    --vector_init_scale 1.0 \
    --positions f4 \
    --optimizer adam

echo -e "\n################################################"
echo -e "Finished at $(date)"
echo -e "################################################\n"
