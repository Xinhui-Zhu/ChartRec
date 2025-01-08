export PYTHONPATH=/proj/arise/arise/xz3276/data/ChartRec

DATA_PATH='./data/v2/test'

# export TRANSFORMERS_CACHE=/proj/arise/arise/xz3276/data
export HF_HOME=/proj/arise/arise/xz3276/data
export TORCHDYNAMO_CACHE_DIR=/proj/arise/arise/xz3276/data
export TRITON_CACHE=/proj/arise/arise/xz3276/data

# MODEL_DIR='ckpt/llama-3-1-8b-v2'
# MODEL_DIR='Multilingual-Multimodal-NLP/TableLLM-Llama3.1-8B'
MODEL_DIR='ckpt/dpsk-coder-v2/checkpoint-300'

python inference/infer.py \
    --data_path $DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'tablebench'  \
    --temperature 0 \
    --sample_n 1 \
    --outdir eval/experiment_results/20250107/outputs_size/2

python eval/batch_parse_response_script.py --exp_version '20250107'
python eval/batch_eval_response_script.py --exp_version '20250107'



