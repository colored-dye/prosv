#!/usr/bin/env bash

mkdir -p logs

./run_benchmark_sv.sh --devices 6 --cfg q25_32b_l32 --obj simpo --loc all --benchmark tinygsm8k 2>&1 | tee logs/q25_32b_l32_simpo_all_tinygsm8k.log

./run_benchmark_sv.sh --devices 6 --cfg q25_32b_l32 --obj lang --loc all --benchmark tinygsm8k 2>&1 | tee logs/q25_32b_l32_lang_all_tinygsm8k.log

./run_benchmark_sv.sh --devices 6 --cfg q25_32b_l32 --obj lang --loc prosv --benchmark tinygsm8k 2>&1 | tee logs/q25_32b_l32_lang_prosv_tinygsm8k.log

./run_benchmark_sv.sh --devices 6 --cfg q25_32b_l32 --obj simpo --loc prosv --benchmark tinygsm8k 2>&1 | tee logs/q25_32b_l32_simpo_prosv_tinygsm8k.log


./run_benchmark_sv.sh --devices 6 --cfg q25_32b_l32 --obj simpo --loc all --benchmark tinymmlu 2>&1 | tee logs/q25_32b_l32_simpo_all_tinymmlu.log

./run_benchmark_sv.sh --devices 6 --cfg q25_32b_l32 --obj lang --loc all --benchmark tinymmlu 2>&1 | tee logs/q25_32b_l32_lang_all_tinymmlu.log

./run_benchmark_sv.sh --devices 6 --cfg q25_32b_l32 --obj lang --loc prosv --benchmark tinymmlu 2>&1 | tee logs/q25_32b_l32_lang_prosv_tinymmlu.log

./run_benchmark_sv.sh --devices 6 --cfg q25_32b_l32 --obj simpo --loc prosv --benchmark tinymmlu 2>&1 | tee logs/q25_32b_l32_simpo_prosv_tinymmlu.log
