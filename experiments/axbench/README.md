
AxBench evaluation on Concept500 dataset.

Hyperparameter configurations are in `configs/`.

## Instructions

### Prompt steering

Prompt steering does not involve training.

```bash
./run_inference_axbench.sh --devices 0 --cfg q25_32b_l32 --obj none --loc prompt
```

```bash
./run_evaluate_axbench.sh --cfg q25_32b_l32 --obj none --loc prompt
```

### SV steering

```bash
./run_train_axbench.sh --devices 0 --cfg 2b_l10 --obj lang --loc prosv
```

```bash
./run_inference_axbench.sh --devices 0 --cfg 2b_l10 --obj lang --loc prosv
```

```bash
./run_evaluate_axbench.sh --cfg q25_32b_l32 --obj none --loc prompt
```

---

Our SimPO SVs with cfg 9b_l20 often underperform original RePS;
tuning hyperparameters to maximize scores!


