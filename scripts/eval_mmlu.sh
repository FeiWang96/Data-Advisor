CUDA_ID=0
MODEL_PATH="path/to/base/llm"
LORA_PATH="path/to/save/checkpoints"
OUTPUT_NAME="custom_name"

CUDA_VISIBLE_DEVICES=$CUDA_ID python -m eval.mmlu_eval \
    --model_path ${MODEL_PATH} \
    --prediction_file predictions/${OUTPUT_NAME}_mmlu.jsonl \
    --lora_path ${LORA_PATH}

