#!/bin/bash

# ==============================================================================
# 0. Run safety checks and GPU settings: ./shell/train.sh 0
# ==============================================================================
if [ -z "$1" ]; then
  echo "Error: Please enter the number of the GPU!"
  echo "Usage: $0 <GPU_ID>"
  echo "Example: $0 0"
  exit 1
fi

GPU_ID=$1
echo "Using GPU: ${GPU_ID}"


# ==============================================================================
# 1. Environment variables and path configuration
# ==============================================================================
# Dataset: ClothingShoesAndJewelry, MoviesAndTV, TripAdvisor
DATASET_NAME="ClothingShoesAndJewelry"
DATA_DIR="./data/"
LOG_DIR="./logs/"
OUTPUT_DIR="./output/"


# ==============================================================================
# 2. Hyperparameter settings
# ==============================================================================
EPOCHS=3
LR=1e-3
ACC_STEPS=1
DELTA=0.2
WORD_LEN=20
ID_HIDDEN=1024
BATCH_SIZE=40
R=4
EXPERT_NUM=4
TOP_K=2
BAL_REG=0.1
LORA_MODULES=2
CKPT_DIR="./checkpoints/"
MODEL_NAME="../autodl-fs/Qwen2.5-7B/"
LOG_NAME="train_${DATASET_NAME}_qwen.log"


# ==============================================================================
# 3. run main.py
# ==============================================================================
echo "===================================================================="
echo "Starting training process..."
echo "Model:   ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Log:     ${LOG_DIR}${DATASET_NAME}/${LOG_NAME}"
echo "===================================================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --data_dir "${DATA_DIR}" \
    --ckpt_dir "${CKPT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --log_name "${LOG_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --accumulation_steps ${ACC_STEPS} \
    --delta ${DELTA} \
    --word ${WORD_LEN} \
    --id_hidden ${ID_HIDDEN} \
    --lora_modules ${LORA_MODULES} \
    --r ${R} \
    --expert_num ${EXPERT_NUM} \
    --top_k ${TOP_K} \
    --bal_reg ${BAL_REG} \
    --use_uiadapter \

echo "Process finished! Please check the logs in ${LOG_DIR}${DATASET_NAME}/${LOG_NAME}"