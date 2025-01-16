export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

MODEL_PATH='deepseek-ai/deepseek-coder-7b-base-v1.5'

# DATA_PATH='./data/v2/train/TableInstruct_instructions.jsonl'
DATA_PATH='./data/v3/train'
SAVE_PATH='./ckpt/dpsk-coder-v2/'

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=4 --use_env train/train_llama3.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --use_cot false \
    --save_steps 50 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --gradient_checkpointing True \
    --cache_dir /proj/arise/arise/xz3276/model \
    --report_to "wandb" \
    --logging_steps 1 \
    --resume_from_checkpoint ckpt/dpsk-coder-v2/checkpoint-900

# DATA_PATH='./data/v2/trainset_size_exp/TableBench_instructions_trainset_7838.jsonl'
# SAVE_PATH='./ckpt/dpsk-coder-v2-part-8k/'

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_llama3.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_path $DATA_PATH \
#     --bf16 True \
#     --output_dir $SAVE_PATH \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --use_cot false \
#     --save_steps 10000 \
#     --save_total_limit 40 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --gradient_checkpointing True \
#     --tf32 True



# DATA_PATH='./data/v2/trainset_size_exp/TableBench_instructions_trainset_11769.jsonl'
# SAVE_PATH='./ckpt/dpsk-coder-v2-part-12k/'

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_llama3.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_path $DATA_PATH \
#     --bf16 True \
#     --output_dir $SAVE_PATH \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --use_cot false \
#     --save_steps 10000 \
#     --save_total_limit 40 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --gradient_checkpointing True \
#     --tf32 True



# DATA_PATH='./data/v2/trainset_size_exp/TableBench_instructions_trainset_15703.jsonl'
# SAVE_PATH='./ckpt/dpsk-coder-v2-part-16k/'

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_llama3.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_path $DATA_PATH \
#     --bf16 True \
#     --output_dir $SAVE_PATH \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --use_cot false \
#     --save_steps 10000 \
#     --save_total_limit 40 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --gradient_checkpointing True \
#     --tf32 True