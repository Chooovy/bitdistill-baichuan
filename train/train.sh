export MODEL_PATH='/aifs4su/gov/models/Qwen2-1.5B-Instruct'
export DATA_PATH=
export SAVE_PATH='/aifs4su/gov/models/Qwen2-1.5B-Instruct-baichuan-distill'
export LOGGING_PATH='/home/lilujun/workspace/BitDistiller/train/logs/Qwen2-1.5B-Instruct/int3-g128/'
export EPOCHS=4


export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_max_length 1024 \
    --output_dir $SAVE_PATH \
    --logging_dir $LOGGING_PATH \
    --num_train_epochs $EPOCHS \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 4 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 4 \
    --save_total_limit 15 \
    --learning_rate 8e-6 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed config/zero.json \
    --bits 3 \
    --quant_type int3-asym \
    --q_group_size 128 \
    --train_kd True \
    --kd_loss_type 'cakld' \
    --max_train_samples 999999 \
    --clip /home/lilujun/workspace/BitDistiller/quantization/clip_cache/Qwen2-1.5B-Instruct/int3-g128.pt
