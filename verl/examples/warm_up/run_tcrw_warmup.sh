#!/bin/bash
# TCRW-OPD representation-aware warm-up launcher.
#
# Default paths match luorongchuan's workspace_134 layout. Every default can
# be overridden from the environment. This script only changes the warm-up;
# the merged HF checkpoint can be passed to the repository's existing OPD
# launcher afterwards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${VERL_ROOT}/.." && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true

# ---------------- local machine defaults ----------------
: "${WORKSPACE_ROOT:=/home/luorongchuan/workspace_134}"
: "${MODEL_PATH:=${WORKSPACE_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${TEACHER_MODEL_PATH:=${WORKSPACE_ROOT}/models/JustRL-DeepSeek-1.5B}"
: "${OUTPUT_ROOT:=${WORKSPACE_ROOT}/outputs/tcrw_opd}"

# GPU 4 and 5 are the default free cards reported for workspace_134.
: "${CUDA_VISIBLE_DEVICES:=4,5}"
export CUDA_VISIBLE_DEVICES
: "${NPROC:=2}"

# SFT_PARQUET must contain at least prompt + response. For correct/wrong
# filtering it must also contain is_correct. Run inspect_tcrw_data.py first
# if you are unsure which local parquet is the teacher-CoT warm-up file.
if [ -z "${SFT_PARQUET:-}" ]; then
    echo "[TCRW][ERROR] SFT_PARQUET is not set."
    echo "Run:"
    echo "  python ${SCRIPT_DIR}/inspect_tcrw_data.py --root ${WORKSPACE_ROOT}/datasets"
    echo "Then launch, for example:"
    echo "  SFT_PARQUET=/abs/path/to/teacher_cot.parquet bash ${BASH_SOURCE[0]}"
    exit 2
fi
if [ ! -f "${SFT_PARQUET}" ]; then
    echo "[TCRW][ERROR] parquet not found: ${SFT_PARQUET}"
    exit 2
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[TCRW][ERROR] student model not found: ${MODEL_PATH}"
    exit 2
fi
if [ ! -d "${TEACHER_MODEL_PATH}" ]; then
    echo "[TCRW][ERROR] teacher model not found: ${TEACHER_MODEL_PATH}"
    exit 2
fi

# ---------------- experiment knobs ----------------
: "${SFT_SAMPLE_MODE:=all}"          # all | correct | wrong
: "${CUTOFF_LEN:=1024}"              # start small for mechanism validation
: "${LR:=5e-6}"
: "${MICRO_BSZ:=1}"
: "${GRAD_ACC:=4}"
: "${MAX_STEPS:=40}"                 # smoke/mechanism run; later use 100/150/175
: "${SAVE_FREQ:=40}"
: "${LORA_RANK:=16}"
: "${LORA_ALPHA:=${LORA_RANK}}"
: "${LORA_TARGET_MODULES:=all-linear}"
: "${REP_ALPHA:=1.0}"
: "${REP_BETA:=0.3}"
: "${REP_LOSS:=cosine}"              # cosine | normalized_mse
: "${REP_LAYERS:=[-1]}"              # Hydra list, e.g. '[-1,-5,-9]'
: "${REP_RESPONSE_ONLY:=true}"
: "${TEACHER_DTYPE:=bf16}"
: "${SKIP_MERGE:=0}"
: "${SKIP_EVAL:=1}"                  # first validate training before expensive eval

case "${SFT_SAMPLE_MODE}" in
    all)
        FILTER_ARGS=(--only_correct 0)
        ;;
    correct)
        FILTER_ARGS=(--only_correct 1)
        ;;
    wrong)
        FILTER_ARGS=(--only_correct 0 --only_wrong 1)
        ;;
    *)
        echo "[TCRW][ERROR] SFT_SAMPLE_MODE must be all, correct, or wrong"
        exit 2
        ;;
esac

GLOBAL_BSZ=$(( MICRO_BSZ * NPROC * GRAD_ACC ))
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="tcrw-r${LORA_RANK}-b${REP_BETA}-${SFT_SAMPLE_MODE}-${TS}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
mkdir -p "${RUN_DIR}"
FILTERED_PARQUET="${RUN_DIR}/warmup_${SFT_SAMPLE_MODE}.parquet"

python "${SCRIPT_DIR}/filter_correct_sft.py" \
    --input_parquet "${SFT_PARQUET}" \
    --output_parquet "${FILTERED_PARQUET}" \
    "${FILTER_ARGS[@]}"

echo "============================================================"
echo "TCRW warm-up"
echo "  GPUs             : ${CUDA_VISIBLE_DEVICES}"
echo "  student          : ${MODEL_PATH}"
echo "  teacher          : ${TEACHER_MODEL_PATH}"
echo "  data             : ${FILTERED_PARQUET}"
echo "  output           : ${RUN_DIR}"
echo "  max length       : ${CUTOFF_LEN}"
echo "  global batch     : ${GLOBAL_BSZ}"
echo "  steps            : ${MAX_STEPS}"
echo "  LoRA rank        : ${LORA_RANK}"
echo "  rep layers       : ${REP_LAYERS}"
echo "  alpha/beta       : ${REP_ALPHA}/${REP_BETA}"
echo "  rep loss         : ${REP_LOSS}"
echo "============================================================"

# TCRW v1 deliberately disables remove-padding and Ulysses SP. This keeps
# Teacher and Student hidden states in the same [B,T,H] layout.
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    -m verl.trainer.tcrw_sft_trainer \
    data.train_files="${FILTERED_PARQUET}" \
    data.val_files="${FILTERED_PARQUET}" \
    data.val_max_samples=4 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length="${CUTOFF_LEN}" \
    data.truncation=right \
    data.train_batch_size="${GLOBAL_BSZ}" \
    data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
    model.partial_pretrain="${MODEL_PATH}" \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    model.fsdp_config.model_dtype=bf16 \
    model.lora_rank="${LORA_RANK}" \
    model.lora_alpha="${LORA_ALPHA}" \
    model.target_modules="${LORA_TARGET_MODULES}" \
    optim.lr="${LR}" \
    optim.lr_warmup_steps_ratio=0.0 \
    optim.lr_scheduler=cosine \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=False \
    trainer.default_local_dir="${RUN_DIR}/ckpt" \
    trainer.project_name="${WANDB_PROJECT:-tcrw_warmup}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.logger='[console]' \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${MAX_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=-1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NPROC}" \
    trainer.resume_mode=disable \
    trainer.device=cuda \
    +representation_warmup.teacher_model_path="${TEACHER_MODEL_PATH}" \
    +representation_warmup.teacher_dtype="${TEACHER_DTYPE}" \
    +representation_warmup.alpha="${REP_ALPHA}" \
    +representation_warmup.beta="${REP_BETA}" \
    +representation_warmup.loss_type="${REP_LOSS}" \
    +representation_warmup.layers="${REP_LAYERS}" \
    +representation_warmup.response_only="${REP_RESPONSE_ONLY}" \
    2>&1 | tee "${RUN_DIR}/train.log"

if [ "${SKIP_MERGE}" = "1" ]; then
    echo "[TCRW] SKIP_MERGE=1; training finished at ${RUN_DIR}/ckpt"
    exit 0
fi

HF_ROOT="${RUN_DIR}/hf"
mkdir -p "${HF_ROOT}"
CKPT_LIST=$(ls -1d "${RUN_DIR}"/ckpt/global_step_* 2>/dev/null | sort -V || true)
if [ -z "${CKPT_LIST}" ]; then
    echo "[TCRW][ERROR] no global_step_* checkpoint found"
    exit 3
fi

LAST_HF_DIR=""
for CKPT_DIR in ${CKPT_LIST}; do
    STEP_NAME="$(basename "${CKPT_DIR}")"
    HF_DIR="${HF_ROOT}/${STEP_NAME}"
    mkdir -p "${HF_DIR}"
    echo "[TCRW] merging ${STEP_NAME} -> ${HF_DIR}"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${CKPT_DIR}" \
        --target_dir "${HF_DIR}" \
        --trust-remote-code \
        2>&1 | tee -a "${RUN_DIR}/merge.log"

    # Fold LoRA adapter into the top-level HF weights, matching Simple-OPD.
    if [ -d "${HF_DIR}/lora_adapter" ] && [ ! -f "${HF_DIR}/.lora_merged" ]; then
        python "${SCRIPT_DIR}/merge_lora_into_base.py" \
            --hf-dir "${HF_DIR}" \
            --lora-alpha "${LORA_ALPHA}" \
            --base-model-name "${MODEL_PATH}" \
            --dtype bf16 \
            2>&1 | tee -a "${RUN_DIR}/lora_merge.log"
    fi
    LAST_HF_DIR="${HF_DIR}"
done

echo "[TCRW] warm-up complete"
echo "[TCRW] final merged checkpoint: ${LAST_HF_DIR}"
echo "[TCRW] train log: ${RUN_DIR}/train.log"

echo "${LAST_HF_DIR}" > "${RUN_DIR}/FINAL_HF_PATH.txt"

if [ "${SKIP_EVAL}" != "1" ]; then
    GPUS="${CUDA_VISIBLE_DEVICES}" \
    MODEL_PATH="${LAST_HF_DIR}" \
        bash "${REPO_ROOT}/math_eval/start_math_eval.sh"
fi
