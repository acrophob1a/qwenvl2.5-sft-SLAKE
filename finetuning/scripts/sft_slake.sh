#!/bin/bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache

cd /root/autodl-tmp/qwenvl-sft

NGPU=1
run_name="slake-vqa-sft-v1"

mkdir -p work_dirs/$run_name

torchrun --nnodes=1 --nproc_per_node=$NGPU finetuning/train.py \
    --config finetuning/configs/sft_slake.py \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --output_dir work_dirs/$run_name \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --save_only_model True \
    --bf16 \
    --learning_rate 2e-5 \
    --mm_projector_lr 2e-5 \
    --optim adamw_torch \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name $run_name \
    --report_to none \
    |& tee work_dirs/$run_name/output.log
