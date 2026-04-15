#!/bin/bash
# SLT 回译模型训练脚本
# 用法：./slp_train.sh [GPU_ID] [CONFIG]

GPU_ID=${1:-0}
CONFIG=${2:-./SLT-main/configs/sign.yaml}

cd SLT-main
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m signjoey train ${CONFIG}
