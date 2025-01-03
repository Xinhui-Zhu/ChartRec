export PYTHONPATH=/proj/arise/arise/xz3276/data/TableBench

DATA_PATH='./data/v2/test'

# export TRANSFORMERS_CACHE=/proj/arise/arise/xz3276/data
export HF_HOME=/proj/arise/arise/xz3276/data
export TORCHDYNAMO_CACHE_DIR=/proj/arise/arise/xz3276/data
export TRITON_CACHE=/proj/arise/arise/xz3276/data


# MODEL_DIR='ckpt/llama-3-1-8b-v2'
MODEL_DIR='Multilingual-Multimodal-NLP/TableLLM-Llama3.1-8B'

python inference/infer.py \
    --data_path $DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'tablebench'  \
    --temperature 1 \
    --sample_n 1 \
    --outdir eval/experiment_results/20250102/outputs_size/2

python eval/batch_parse_response_script.py --exp_version '20250102'
python eval/batch_eval_response_script.py --exp_version '20250102'




python eval/batch_parse_response_script.py --exp_version '20250102-tem=1'
python eval/batch_eval_response_script.py --exp_version '20250102-tem=1'

