#!/usr/bin/env bash

source $HOME/argparse.sh

# define_arg "option" "" "" "string" "required"
# define_arg "pos" "" "" "string" "required"
define_arg "cfg" "" "" "string" "required"

parse_args "$@"

export LOGURU_LEVEL='WARNING'

SCRIPT=plot_sv.py

SEED_ARR=(42 43 44)
FACTOR_LR_ARR=(1e-2 1e-1 1 10 20 40)
FACTOR_INIT_ARR=(1 2 4 8 16)

OPTION_ARR=(default larger_vec)
POS_ARR=(all all_prompt f4+l4 f8 l8 f2+l2 f4 l4 f1+l1 f2 l2)
SV_ARR=(AddInv ClampInv)
SCORE_TYPE_ARR=(concept overall)

OUTPUT_ROOT_DIR=/mnt/data/byt/reft_data/

for option in ${OPTION_ARR[@]}; do
  for pos in ${POS_ARR[@]}; do
    echo -e "\n####################################################################"
    echo -e "Option: ${option} || CFG: ${cfg} || Positions: ${pos}"
    echo -e "####################################################################\n"

    python ${SCRIPT} \
      --seed_list ${SEED_ARR[@]} \
      --factor_lr_list ${FACTOR_LR_ARR[@]} \
      --factor_init_size_list ${FACTOR_INIT_ARR[@]} \
      --output_base_dir ${OUTPUT_ROOT_DIR} \
      --cfg ${cfg} \
      --option ${option} \
      --pos ${pos} \
      --svs ${SV_ARR[@]} \
      --score_types ${SCORE_TYPE_ARR[@]}

    if [[ $? -ne 0 ]]; then
      echo -e "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo -e "Error! Early exit"
      echo -e "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
      exit 1
    fi

done
done
