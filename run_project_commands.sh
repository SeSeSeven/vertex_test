#!/bin/bash
# 停止脚本在遇到错误时
set -e

# 依次执行命令
python data.py
python model.py
python main.py train --lr 1e-4 --bucket_name "my_lmu_mlops_data_bucket"
python main.py evaluate trained_model.pt

# 保持容器运行（如果需要）
# tail -f /dev/null