#!/usr/bin/env bash

sv="$1"
pos="$2"

# OUTPUT_DIR=${HOME}/share/reft_data/concept_sweep/9b_l20/outputs_${sv}/${pos}/${i}
OUTPUT_DIR=${HOME}/reft_data/concept_sweep/9b_l20/outputs_${sv}/${pos}

for ((i=0; i<10; i++)); do
    python merge_parquet.py \
        --output_dir $OUTPUT_DIR/$i \
        --file_names steered_generations.parquet eval.parquet
done
