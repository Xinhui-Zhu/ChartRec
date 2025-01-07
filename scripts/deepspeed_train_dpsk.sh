export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

MODEL_PATH='deepseek-ai/deepseek-coder-7b-base-v1.5'

# DATA_PATH='./data/v2/trainset_size_exp/TableBench_instructions_trainset_3904.jsonl'
DATA_PATH='./data/v2/train/TableInstruct_instructions.jsonl'
SAVE_PATH='./ckpt/dpsk-coder-v2/'

CUDA_VISIBLE_DEVICES="0,1,2,3" deepspeed train/deepspeed_train_llama3.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --use_cot false \
    --save_steps 10000 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --cache_dir /proj/arise/arise/xz3276/model \
    --deepspeed config/ds_config.json




