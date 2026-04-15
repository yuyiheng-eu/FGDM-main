#!/bin/bash
# SLT 回译模型测试脚本
# 用法：./slt_test.sh [GPU_ID] [CHECKPOINT] [OUTPUT_DIR]

GPU_ID=${1:-0}
CKPT=${2:-SLT-main/sign_skels_model/best.ckpt}
OUTPUT_DIR=${3:-slt_results}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "GPU: ${GPU_ID}"
echo "Config: SLT-main/configs/sign.yaml"
echo "Checkpoint: ${CKPT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo ""

cd ${SCRIPT_DIR}/SLT-main
PYTHONPATH=${SCRIPT_DIR}/SLT-main python -m signjoey test configs/sign.yaml \
    --ckpt ${CKPT} \
    --output_path ${OUTPUT_DIR} \
    --gpu_id ${GPU_ID} 
