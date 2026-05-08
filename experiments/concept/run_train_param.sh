#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

SCRIPT=train_param.py

ADAPTER=$1
CONCEPT_ID=44

echo -e "\n####################################################"
echo -e "Parameter adapter type: ${ADAPTER}"
echo -e "####################################################\n"

python ${SCRIPT} \
    --model_path /home/Dataset/Models/google/gemma-2-2b-it \
    --output_dir outputs_param_${ADAPTER}/ \
    --layers 6 8 10 12 \
    --learning_rate 1e-2 \
    --epochs 50 \
    --concept_id ${CONCEPT_ID} \
    --low_rank_dim 1 \
    --batch_size 8 \
    --adapter_type ${ADAPTER}

echo -e "\n################################################"
echo -e "Finished at $(date)"
echo -e "################################################\n"
