#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H800 GPU参数配置文件
定义NVIDIA H800 GPU的关键参数，用于SimAI仿真
"""

def get_h800_params():
    """
    返回NVIDIA H800 GPU的关键参数
    """
    params = {
        # 核心参数
        "num_sm": 132,                  # 流多处理器数量
        "max_threads_per_sm": 2048,     # 每个SM的最大线程数
        "max_blocks_per_sm": 32,        # 每个SM的最大块数
        "max_shared_memory_per_sm": 164 * 1024,  # 每个SM的共享内存 (字节)
        "max_registers_per_sm": 65536,  # 每个SM的寄存器数量
        
        # 内存参数
        "memory_clock_rate": 1215 * 1000 * 1000,  # 内存时钟频率 (Hz)
        "memory_bus_width": 5120,       # 内存总线宽度 (bit)
        "l2_cache_size": 51 * 1024 * 1024,  # L2缓存大小 (字节)
        "global_memory_size": 80 * 1024 * 1024 * 1024,  # 全局内存大小 (字节)
        
        # 计算参数
        "clock_rate": 1755 * 1000 * 1000,  # 核心时钟频率 (Hz)
        "tensor_core_performance": 989,  # Tensor Core性能 (TFLOPS, FP16)
        
        # 其他参数
        "compute_capability_major": 9,  # 计算能力主版本
        "compute_capability_minor": 0,  # 计算能力次版本
    }
    return params

if __name__ == "__main__":
    # 打印H800 GPU参数，用于测试
    import json
    params = get_h800_params()
    print(json.dumps(params, indent=4))