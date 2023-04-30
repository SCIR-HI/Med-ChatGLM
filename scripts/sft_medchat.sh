wandb online
exp_tag="chatglm_tuning"

python run_clm.py \
    --model_name_or_path MODEL_PATH \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.txt \
    --max_seq_length 256 \
    --output_dir ./output/ \
    --do_train \
    --logging_steps 30 \
    --log_file ./log/$exp_tag \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --group_by_length False \
    --num_train_epochs 3 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --logging_dir ./log \
    --logging_steps 10 \
    --save_strategy epoch \
    --seed 2023 \
    --remove_unused_columns False \
    --torch_dtype auto \
    --adam_epsilon 1e-3 \
    --report_to wandb \
    --run_name $exp_tag
