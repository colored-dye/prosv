# Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions

<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.05983)  [![Github](https://img.shields.io/badge/PrOSV_Code-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/colored-dye/prosv)  [![Hugging Face Data](https://img.shields.io/badge/Data-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/colored-dye/concept500-contrastive)  [![Hugging Face Model](https://img.shields.io/badge/SV_Checkpoints-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/colored-dye/axbench-steering-vector)

</div>

PrOSV is a repository for training and evaluating steering vectors for language models, with an emphasis on prompt-only interventions that preserve KV-cache efficiency.

The repo supports two intervention modes:

- PrOSV: prompt-only steering.
- FSSV: full-sequence steering.

It also includes AxBench-based evaluation and benchmark scripts for measuring steering quality and downstream task performance.

## Repository Layout

- `src/reft`: core intervention, training, and evaluation code.
- `data`: concept datasets, prompt-generation assets, and helper scripts.
- `experiments`: runnable workflows for AxBench, benchmarks, concept sweeps, and related experiments.

## Quick Setup

Requirements:

- Python 3.10+
- `uv`
- `argparse.sh` for the shell entrypoints

Install the environment:

```sh
uv sync
```

Install `argparse.sh` where the shell scripts expect it:

```sh
git clone https://github.com/colored-dye/argparse-sh "$HOME/argparse-sh"
ln -s "$HOME/argparse-sh/argparse.sh" "$HOME/argparse.sh"
```

If you want to generate steering prompts with the scripts under `data/scripts`, set an OpenAI-compatible API key first:

```sh
export OPENAI_API_KEY=...
```

## Quick Use

### 0. Configure bash script

Customizable variables:

- `MODEL_BASE_DIR`
- `OUTPUT_ROOT_DIR`

The actual output directory is `${OUTPUT_ROOT_DIR}/axbench/${cfg}/outputs_${SV}/${POSITIONS}/${obj}`, which we write as `outputs` in shorthand below.

### 1. Train a steering vector with AxBench

From `experiments/axbench`:

```sh
./run_train_axbench.sh --devices 0 --cfg 2b_l10 --obj lang --loc prosv
```

This trains a prompt-only steering vector for one of the provided model-layer configurations.

Expected output: checkpoints are written under `outputs/<concept_id>/`.

### 2. Run AxBench inference and evaluation

Prompt-only baseline:

```sh
./run_inference_axbench.sh --devices 0 --cfg q25_32b_l32 --obj none --loc prompt
./run_evaluate_axbench.sh --cfg q25_32b_l32 --obj none --loc prompt
```

Expected output: inference writes `outputs/<concept_id>/steered_generations.parquet`, and evaluation writes `outputs/<concept_id>/eval.parquet`.

Steering-vector evaluation:

```sh
./run_inference_axbench.sh --devices 0 --cfg 2b_l10 --obj lang --loc prosv
./run_evaluate_axbench.sh --cfg 2b_l10 --obj lang --loc prosv
```

Expected output: the same per-concept files under `outputs/<concept_id>/`, now using the trained steering vector.

### 3. Run benchmark evaluation

From `experiments/benchmark`:

```sh
./run_benchmark_sv.sh --devices 0 --cfg 2b_l10 --obj lang --loc prosv --benchmark tinygsm8k
```

This evaluates an intervened model on standard benchmarks such as MMLU and GSM8K-style tasks.

Expected output: per-concept benchmark results are saved to `outputs/<concept_id>/steered_benchmark_<benchmark>.parquet`.

## Data and Configs

- `data/concept10`: verification split.
- `data/concept500`: AxBench evaluation data.
- `data/concept500_contrast`: augmented contrastive evaluation data.
- `experiments/axbench/configs`: model and training hyperparameter presets.

The shell scripts expect model checkpoints to be available locally or through the referenced Hugging Face model names in each experiment script.

## Where To Look Next

- `data/README.md` for dataset details.
- `experiments/axbench/README.md` for AxBench-specific usage.
- `experiments/benchmark/README.md` for benchmark notes and results.
- `experiments/concept_sweep/README.md` for sweep workflows.

## Acknowledgements

Many thanks to these projects:

- [baukit](https://github.com/davidbau/baukit)
- [pyvene](https://github.com/stanfordnlp/pyvene)
- [pyreft](https://github.com/stanfordnlp/pyreft)
- [axbench](https://github.com/stanfordnlp/axbench)

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{bao2026towards,
  title = {Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions},
  author = {Bao, Yuntai and Li, Qinfeng and Yu, Xinyan and Zhang, Xuhong and Su, Ge and Zhang, Wenqi and Yan, Liu and Weng, Haiqin and Yin, Jianwei},
  booktitle = {Forty-third International Conference on Machine Learning},
  year = {2026},
  url = {https://openreview.net/forum?id=AaT3liS5PE},
}
```
