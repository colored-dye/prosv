#!/usr/bin/env bash

source $HOME/argparse.sh

define_arg "subset" "" "concept500" "string" "required"
define_arg "cfg" "" "2b_l10; 9b_l20" "string" "required"

parse_args "$@"


DATA_DIR="../${subset}/prod_${cfg}_v1/generate/"

python get_steering_prompt_prepend.py \
    --concept_data_dir ${DATA_DIR} \
    --batch_size 10
