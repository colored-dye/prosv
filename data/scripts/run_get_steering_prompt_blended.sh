#!/usr/bin/env bash

cfg=9b_l20

ALPACA_EVAL_PATH=../alpaca_eval.json
DATA_DIR=../concept500/prod_${cfg}_v1/generate/

python get_steering_prompt_blended.py \
    --concept_data_dir ${DATA_DIR} \
    --batch_size 10 \
    --alpaca_eval_path ${ALPACA_EVAL_PATH}
