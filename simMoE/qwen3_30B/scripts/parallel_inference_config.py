#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行推理配置文件
定义attention dp4和ffn ep4的并行推理行为
"""

def get_parallel_config():
    """
    配置attention dp4和ffn ep4并行推理行为
    
    dp4: 数据并行度为4，将输入数据分割到4个GPU上
    ep4: 专家并行度为4，将MoE专家分布到4个GPU上
    
    返回:
        dict: 并行配置参数
    """
    config = {
        # 数据并行配置
        "data_parallel": {
            "degree": 4,                # 数据并行度为4
            "input_split_axis": 0,      # 在批次维度上分割输入
            "output_split_axis": 0,     # 在批次维度上分割输出
            "communication_pattern": "all_reduce"  # 使用all_reduce进行梯度聚合
        },
        
        # 专家并行配置
        "expert_parallel": {
            "degree": 4,                # 专家并行度为4
            "num_experts": 128,         # 总专家数量
            "experts_per_gpu": 32,      # 每个GPU上的专家数量 (128/4=32)
            "active_experts_per_token": 8,  # 每个token激活的专家数量
            "communication_pattern": "all_to_all"  # 使用all_to_all进行专家路由
        },
        
        # 注意力机制配置 - 使用数据并行
        "attention": {
            "parallel_mode": "data_parallel",  # 注意力机制使用数据并行
            "heads": 32,                       # 注意力头数量
            "kv_heads": 4,                     # KV头数量
            "head_dim": 128                    # 头维度
        },
        
        # 前馈网络配置 - 使用专家并行
        "ffn": {
            "parallel_mode": "expert_parallel",  # 前馈网络使用专家并行
            "moe_enabled": True,                # 启用MoE
            "hidden_size": 2048,                # 隐藏层大小
            "intermediate_size": 6144,          # 中间层大小(累积)
            "expert_intermediate_size": 768     # 每个专家的中间层大小
        }
    }
    return config

if __name__ == "__main__":
    # 打印并行配置，用于测试
    import json
    config = get_parallel_config()
    print(json.dumps(config, indent=4))