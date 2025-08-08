#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用AICB生成Qwen3-30B模型的工作负载文件
包含所有指定的模型参数和并行配置
"""

import os
import json
import numpy as np
from h800_gpu_config import get_h800_params
from parallel_inference_config import get_parallel_config

def generate_qwen3_30b_workload():
    """
    使用AICB生成Qwen3-30B模型的工作负载文件
    
    返回:
        str: 生成的工作负载文件路径
    """
    # 获取H800 GPU参数和并行配置
    gpu_params = get_h800_params()
    parallel_config = get_parallel_config()
    
    # Qwen3-30B模型参数 - 根据提供的表格
    model_params = {
        "name": "Qwen3-30B",
        "hidden_size": 2048,                    # 隐藏层大小
        "num_attention_heads": 32,              # 注意力头数量
        "num_kv_heads": 4,                      # KV头数量
        "head_dim": 128,                        # 头维度
        "num_hidden_layers": 48,                # 隐藏层层数
        "intermediate_size": 6144,              # 中间层大小(累积)
        "moe_intermediate_size": 768,           # 每个专家的中间层大小
        "num_experts": 128,                     # 专家数量
        "active_experts_per_token": 8,          # 每个token激活的专家数量
        "activation_function": "silu",          # 激活函数
        "rms_norm_eps": 1e-6,                   # RMS归一化epsilon
        "vocab_size": 151936,                   # 词汇表大小
        "data_type": "bfloat16"                 # 数据类型
    }
    
    # 推理批次大小 - 根据经验值设置
    # 对于30B级别的MoE模型，在4个H800 GPU上，批次大小通常在8-32之间
    batch_size = 16
    
    # 序列长度 - 常见值
    seq_length = 2048
    
    # 创建AICB工作负载配置
    workload_config = {
        "model": model_params,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "gpu_params": gpu_params,
        "parallel_config": parallel_config,
        "inference_only": True,  # 仅推理模式
        "precision": "bfloat16"  # 使用bfloat16精度
    }
    
    # 确保输出目录存在
    os.makedirs("workload", exist_ok=True)
    
    # 将配置写入JSON文件
    config_file = "workload/qwen3_30b_workload_config.json"
    with open(config_file, "w") as f:
        json.dump(workload_config, f, indent=4)
    
    print(f"已生成工作负载配置文件：{config_file}")
    
    # 使用AICB生成工作负载文件
    workload_file = "workload/qwen3_30b_workload.json"
    
    # 注意：这里我们假设AICB工具已经安装并可用
    # 获取SimAI文件夹的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 从当前路径向上找到SimAI目录
    simai_dir = current_dir
    while simai_dir != '/' and not simai_dir.endswith('SimAI'):
        simai_dir = os.path.dirname(simai_dir)
    
    if not simai_dir.endswith('SimAI'):
        # 如果没找到SimAI目录，使用相对路径推导
        simai_dir = os.path.join(current_dir, "..", "..", "..")
        simai_dir = os.path.abspath(simai_dir)
    simulator_path = os.path.join(simai_dir, "aicb")

    aicb_command = f"./bin/aicb_workload_generator --config {config_file} --output {workload_file}"
    print(f"执行AICB命令：{aicb_command}")
    
    os.system(aicb_command)
    
    
    print(f"已生成Qwen3-30B模型的工作负载文件：{workload_file}")
    return workload_file

if __name__ == "__main__":
    generate_qwen3_30b_workload()