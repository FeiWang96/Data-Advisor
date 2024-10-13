CUDA_ID=0
MODEL_PATH="path/to/base/llm"
LORA_PATH="path/to/save/checkpoints"
OUTPUT_NAME="custom_name"

CUDA_VISIBLE_DEVICES=$CUDA_ID python -m eval.beavertails_eval \
    --model_path ${MODEL_PATH} \
    --prediction_file predictions/${OUTPUT_NAME}_beavertails.jsonl \
    --lora_path ${LORA_PATH}

CUDA_VISIBLE_DEVICES=$CUDA_ID python -m evaluator.llamaguard \
    --prediction_file predictions/${OUTPUT_NAME}_beavertails.jsonl \
    --output_file outputs/${OUTPUT_NAME}_beavertails.jsonl \
    --score_file final_scores/beavertails_${OUTPUT_NAME}.csv 
