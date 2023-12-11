#!/bin/bash

torchrun --nproc_per_node=4 --master_port=9292 cog_stanford_alpaca/train.py \
    --model_name_or_path EleutherAI/polyglot-ko-12.8b \
    --data_path /workspace/KoAlpaca/ko_alpaca_data.json \
    --fp16 True \
    --output_dir /workspace/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./cog_stanford_alpaca/configs/default_offload_opt_param.json" \
    --gradient_checkpointing 1
