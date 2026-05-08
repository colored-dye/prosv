#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "subset" "" "concept500" "string" "required"
define_arg "cfg" "" "2b_l10; 9b_l20" "string" "required"

parse_args "$@"


DATA_DIR="../${subset}/prod_${cfg}_v1/generate/"
ALPACA_EVAL_PATH=../alpaca_eval.json

python get_suppression_prompt_blended.py \
    --concept_data_dir ${DATA_DIR} \
    --alpaca_eval_path ${ALPACA_EVAL_PATH} \
    --batch_size 10 \
    --max_concepts 20
