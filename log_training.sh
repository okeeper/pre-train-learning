#!/bin/bash

# 创建日志目录
mkdir -p logs

# 获取当前时间作为日志文件名
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# 运行训练脚本，并将所有输出同时发送到终端和日志文件
bash run_deepspeed_balanced.sh 2>&1 | tee -a $LOG_FILE

echo "训练日志已保存到: $LOG_FILE" 