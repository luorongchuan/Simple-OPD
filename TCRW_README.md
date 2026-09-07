# TCRW-OPD experimental branch

This branch adds an initial **Teacher-Compatible Representation Warm-up (TCRW)** implementation on top of Simple-OPD.

The downstream OPD code is intentionally unchanged. TCRW modifies only the pre-OPD LoRA warm-up by optimizing

\[
L_{warm}=\alpha L_{SFT}+\beta L_{rep}.
\]

`L_rep` aligns frozen-Teacher and LoRA-Student hidden states on the **same Teacher-generated CoT under teacher forcing**. The first version supports cosine distance or normalized MSE and defaults to the final hidden layer only.

## Files added

- `verl/verl/trainer/tcrw_sft_trainer.py`: representation-aware FSDP/LoRA warm-up trainer.
- `verl/examples/warm_up/run_tcrw_warmup.sh`: two-GPU launcher with workspace_134 defaults.
- `verl/examples/warm_up/inspect_tcrw_data.py`: scans local parquet files and identifies files containing `prompt` + `response` (and optionally `is_correct`).

## Local defaults

The launcher is preconfigured for:

```text
repo:     /home/luorongchuan/workspace_134/Simple-OPD
student:  /home/luorongchuan/workspace_134/models/DeepSeek-R1-Distill-Qwen-1.5B
teacher:  /home/luorongchuan/workspace_134/models/JustRL-DeepSeek-1.5B
datasets: /home/luorongchuan/workspace_134/datasets
GPUs:     4,5
```

All of these are environment-variable overrides rather than hard requirements.

## 1. Pull this branch

```bash
cd /home/luorongchuan/workspace_134/Simple-OPD
git fetch origin
git checkout tcrw-representation-warmup
git pull origin tcrw-representation-warmup
```

## 2. Find the Teacher-CoT parquet

Do **not** assume that raw `DAPO-Math-17k` is already a warm-up dataset. TCRW needs a parquet containing at least `prompt` and `response`, where `response` is the Teacher-generated CoT. For correct/wrong ablations it should also contain `is_correct`.

```bash
cd /home/luorongchuan/workspace_134/Simple-OPD/verl/examples/warm_up
python inspect_tcrw_data.py --root /home/luorongchuan/workspace_134/datasets
```

If no compatible file exists, first use the original Simple-OPD rollout/data-generation pipeline to create Teacher CoT responses.

## 3. First smoke run

```bash
cd /home/luorongchuan/workspace_134/Simple-OPD/verl/examples/warm_up

CUDA_VISIBLE_DEVICES=4,5 \
SFT_PARQUET=/absolute/path/to/teacher_cot.parquet \
SFT_SAMPLE_MODE=all \
CUTOFF_LEN=1024 \
MAX_STEPS=10 \
SAVE_FREQ=10 \
LORA_RANK=16 \
REP_BETA=0.3 \
REP_LAYERS='[-1]' \
bash run_tcrw_warmup.sh
```

Expected console metrics include:

```text
train/loss
train/sft_loss
train/rep_loss
```

A successful run writes `FINAL_HF_PATH.txt` under the run directory.

## 4. Mechanism run

After the 10-step smoke run succeeds:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
SFT_PARQUET=/absolute/path/to/teacher_cot.parquet \
CUTOFF_LEN=1024 \
MAX_STEPS=40 \
SAVE_FREQ=40 \
LORA_RANK=16 \
REP_BETA=0.3 \
REP_LAYERS='[-1]' \
bash run_tcrw_warmup.sh
```

Then increase to 100/150/175 steps and compare against Simple-OPD under matched data, LoRA rank, learning rate, and OPD budget.

## 5. Key ablations

```bash
# Simple-OPD baseline: use the repository's original warm-up launcher.

# Hybrid TCRW
REP_ALPHA=1.0 REP_BETA=0.3 REP_LOSS=cosine REP_LAYERS='[-1]'

# Representation-heavy
REP_ALPHA=1.0 REP_BETA=1.0 REP_LOSS=cosine REP_LAYERS='[-1]'

# Multiple selected layers (only after the last-layer run is stable)
REP_LAYERS='[-1,-5,-9]'

# Correct/wrong Teacher rollout ablation when is_correct exists
SFT_SAMPLE_MODE=correct
SFT_SAMPLE_MODE=wrong
```

## Important implementation constraints in v1

- `ulysses_sequence_parallel_size=1`.
- `use_remove_padding=False`.
- Student and Teacher hidden sizes must match for direct alignment.
- The frozen Teacher is replicated once per training rank. This is reasonable for the 1.5B/1.5B setup on 2 x A100 80GB, but should be redesigned for much larger Teachers.
- Start with 1024-token CoT and final-layer alignment before increasing sequence length or the number of aligned layers.

These constraints are deliberate so the first experiment tests the research hypothesis without adding unnecessary engineering complexity.
