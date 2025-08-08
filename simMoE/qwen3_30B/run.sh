#!/bin/bash
# 运行Qwen3-30B在4个H800 GPU上的SimAI仿真

# 设置环境变量
export AS_HIGH_PRECISION=1
export AS_SEND_LAT=3
export AS_NVLS_ENABLE=0

# 确保脚本有执行权限
chmod +x scripts/run_qwen3_30b_simulation.py

# 运行仿真
python3 scripts/run_qwen3_30b_simulation.py