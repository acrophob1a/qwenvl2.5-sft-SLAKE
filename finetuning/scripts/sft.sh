export HF_ENDPOINT=https://hf-mirror.com

NGPU=1
run_name="sft-1"

if [ ! -d work_dirs/$run_name ]; then
    mkdir -p work_dirs/$run_name
fi

torchrun --nnodes=1 --nproc_per_node=$NGPU finetuning/train.py \
    --config finetuning/configs/sft.py \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --output_dir work_dirs/$run_name \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 3 \
    --bf16 \
    --learning_rate 1e-5 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name $run_name \
    --report_to none \
    |& tee -a work_dirs/$run_name/output.log
