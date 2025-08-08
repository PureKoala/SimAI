# Qwen3-30B在4个H800 GPU上的SimAI仿真实现指南

本指南提供了使用SimAI工具在4个H800 GPU上模拟Qwen3-30B模型的完整实现方案。我们将详细介绍环境配置、GPU参数设置、网络拓扑配置、并行推理行为实现以及使用AICB生成工作负载的过程。

## 1. SimAI环境设置与依赖安装

### 1.1 克隆SimAI代码库

首先，我们需要克隆SimAI代码库并初始化子模块：

```bash
# 克隆SimAI代码库
git clone https://github.com/alibaba/SimAI.git
cd SimAI

# 初始化并更新子模块
git submodule init
git submodule update
```

### 1.2 安装依赖

SimAI依赖于以下软件包：

```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-pip libboost-all-dev

# 安装Python依赖
pip3 install numpy pandas matplotlib networkx pyyaml
```

### 1.3 编译SimAI

使用提供的构建脚本编译SimAI：

```bash
# 编译SimAI
./build.sh
```

### 1.4 设置环境变量

为了获得高精度的仿真结果，我们需要设置以下环境变量：

```bash
# 设置高精度仿真环境变量
export AS_HIGH_PRECISION=1  # 启用高精度仿真模式
export AS_SEND_LAT=3        # 设置发送延迟参数
export AS_NVLS_ENABLE=0     # 禁用NVLink，因为我们只使用IBGDA网络
```

## 2. 配置H800 GPU参数

创建一个Python脚本来定义H800 GPU的参数配置：

```python
# h800_gpu_config.py

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
```

## 3. 设置4 GPU全连接拓扑与IBGDA 400Gbps网络

创建一个Python脚本来生成4个H800 GPU的全连接拓扑，使用IBGDA 400Gbps网络，不使用NVLink：

```python
# gen_4_H800_ibgda_topo.py

import json

def generate_4_h800_ibgda_topology():
    """
    生成4个H800 GPU的全连接拓扑，使用IBGDA 400Gbps网络，不使用NVLink
    """
    # IBGDA 400Gbps网络参数
    ibgda_bandwidth = 400 * 1024 * 1024 * 1024 / 8  # 400Gbps转换为字节/秒
    ibgda_latency = 1.5e-6  # 1.5微秒延迟
    
    # 创建4个节点（每个节点一个GPU）
    nodes = []
    for i in range(4):
        node = {
            "id": i,
            "type": "gpu",
            "name": f"H800_GPU_{i}",
            "nics": [
                {
                    "id": 0,
                    "type": "ibgda",
                    "name": f"nic_{i}",
                    "bandwidth": ibgda_bandwidth,
                    "latency": ibgda_latency
                }
            ]
        }
        nodes.append(node)
    
    # 创建全连接拓扑的链接
    links = []
    link_id = 0
    for i in range(4):
        for j in range(i+1, 4):
            link = {
                "id": link_id,
                "name": f"link_{i}_{j}",
                "src_node": i,
                "src_nic": 0,
                "dst_node": j,
                "dst_nic": 0,
                "bandwidth": ibgda_bandwidth,
                "latency": ibgda_latency
            }
            links.append(link)
            link_id += 1
    
    # 创建拓扑配置
    topology = {
        "name": "4_H800_IBGDA_400Gbps_FullyConnected",
        "nodes": nodes,
        "links": links
    }
    
    # 将拓扑配置写入JSON文件
    with open("4_h800_ibgda_topo.json", "w") as f:
        json.dump(topology, f, indent=4)
    
    print("已生成4个H800 GPU的全连接拓扑配置文件：4_h800_ibgda_topo.json")
    return "4_h800_ibgda_topo.json"

if __name__ == "__main__":
    generate_4_h800_ibgda_topology()
```

## 4. 实现attention dp4和ffn ep4并行推理行为

创建一个Python脚本来定义attention dp4（数据并行）和ffn ep4（专家并行）的并行推理行为：

```python
# parallel_inference_config.py

def get_parallel_config():
    """
    配置attention dp4和ffn ep4并行推理行为
    dp4: 数据并行度为4，将输入数据分割到4个GPU上
    ep4: 专家并行度为4，将MoE专家分布到4个GPU上
    """
    config = {
        # 数据并行配置
        "data_parallel": {
            "degree": 4,               # 数据并行度为4
            "input_split_axis": 0,      # 在批次维度上分割输入
            "output_split_axis": 0,     # 在批次维度上分割输出
            "communication_pattern": "all_reduce"  # 使用all_reduce进行梯度聚合
        },
        
        # 专家并行配置
        "expert_parallel": {
            "degree": 4,               # 专家并行度为4
            "num_experts": 128,         # 总专家数量
            "experts_per_gpu": 32,      # 每个GPU上的专家数量 (128/4=32)
            "active_experts_per_token": 8,  # 每个token激活的专家数量
            "communication_pattern": "all_to_all"  # 使用all_to_all进行专家路由
        },
        
        # 注意力机制配置
        "attention": {
            "parallel_mode": "data_parallel",  # 注意力机制使用数据并行
            "heads": 32,                       # 注意力头数量
            "kv_heads": 4,                     # KV头数量
            "head_dim": 128                    # 头维度
        },
        
        # 前馈网络配置
        "ffn": {
            "parallel_mode": "expert_parallel",  # 前馈网络使用专家并行
            "moe_enabled": True,                # 启用MoE
            "hidden_size": 2048,                # 隐藏层大小
            "intermediate_size": 6144,          # 中间层大小(累积)
            "expert_intermediate_size": 768     # 每个专家的中间层大小
        }
    }
    return config
```

## 5. 使用AICB生成Qwen3-30B的工作负载文件

创建一个Python脚本，使用AICB生成Qwen3-30B模型的工作负载文件，包含所有指定的模型参数：

```python
# generate_qwen3_30b_workload.py

import os
import json
import numpy as np
from h800_gpu_config import get_h800_params
from parallel_inference_config import get_parallel_config

def generate_qwen3_30b_workload():
    """
    使用AICB生成Qwen3-30B模型的工作负载文件
    """
    # 获取H800 GPU参数和并行配置
    gpu_params = get_h800_params()
    parallel_config = get_parallel_config()
    
    # Qwen3-30B模型参数
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
    
    # 将配置写入JSON文件
    with open("qwen3_30b_workload_config.json", "w") as f:
        json.dump(workload_config, f, indent=4)
    
    # 使用AICB生成工作负载文件
    os.system("./bin/aicb_workload_generator --config qwen3_30b_workload_config.json --output qwen3_30b_workload.json")
    
    print("已生成Qwen3-30B模型的工作负载文件：qwen3_30b_workload.json")
    return "qwen3_30b_workload.json"

if __name__ == "__main__":
    generate_qwen3_30b_workload()
```

## 6. 确定适合推理的批次大小

在上面的工作负载生成脚本中，我们已经设置了批次大小为16。这是基于以下考虑：

1. **GPU内存限制**：H800 GPU有80GB内存，对于30B级别的MoE模型，每个GPU需要存储约7.5B参数（30B/4）
2. **MoE特性**：MoE模型只激活部分专家，因此内存效率更高
3. **并行策略**：使用数据并行和专家并行组合可以更有效地利用资源
4. **经验值**：对于类似规模的模型，在4个80GB GPU上，批次大小通常在8-32之间
5. **延迟与吞吐量平衡**：批次大小16在保证合理延迟的同时提供较好的吞吐量

可以根据实际仿真结果调整批次大小，以找到最佳平衡点。

## 7. 运行仿真的完整Python代码

创建一个主脚本，集成所有步骤并运行仿真：

```python
# run_qwen3_30b_simulation.py

import os
import time
import json
import matplotlib.pyplot as plt
from gen_4_H800_ibgda_topo import generate_4_h800_ibgda_topology
from generate_qwen3_30b_workload import generate_qwen3_30b_workload

def run_simulation():
    """
    运行Qwen3-30B在4个H800 GPU上的仿真
    """
    print("开始Qwen3-30B在4个H800 GPU上的SimAI仿真...")
    
    # 步骤1：设置环境变量
    os.environ["AS_HIGH_PRECISION"] = "1"
    os.environ["AS_SEND_LAT"] = "3"
    os.environ["AS_NVLS_ENABLE"] = "0"
    
    # 步骤2：生成网络拓扑
    print("\n步骤1：生成4个H800 GPU的全连接拓扑...")
    topo_file = generate_4_h800_ibgda_topology()
    print(f"网络拓扑文件已生成：{topo_file}")
    
    # 步骤3：生成工作负载
    print("\n步骤2：生成Qwen3-30B模型的工作负载...")
    workload_file = generate_qwen3_30b_workload()
    print(f"工作负载文件已生成：{workload_file}")
    
    # 步骤4：运行仿真
    print("\n步骤3：运行仿真...")
    simulation_command = f"./bin/SimAI_simulator --workload {workload_file} --network {topo_file} --output simulation_results"
    print(f"执行命令：{simulation_command}")
    
    start_time = time.time()
    os.system(simulation_command)
    end_time = time.time()
    
    print(f"\n仿真完成，耗时：{end_time - start_time:.2f}秒")
    
    # 步骤5：分析结果
    print("\n步骤4：分析仿真结果...")
    analyze_results()

def analyze_results():
    """
    分析仿真结果
    """
    try:
        # 读取仿真结果
        with open("simulation_results/summary.json", "r") as f:
            results = json.load(f)
        
        # 提取关键指标
        total_time = results.get("total_time_ms", 0) / 1000  # 转换为秒
        compute_time = results.get("compute_time_ms", 0) / 1000
        communication_time = results.get("communication_time_ms", 0) / 1000
        overlap_time = results.get("overlap_time_ms", 0) / 1000
        
        # 打印关键指标
        print("\n仿真结果摘要:")
        print(f"总执行时间: {total_time:.4f}秒")
        print(f"计算时间: {compute_time:.4f}秒 ({compute_time/total_time*100:.2f}%)")
        print(f"通信时间: {communication_time:.4f}秒 ({communication_time/total_time*100:.2f}%)")
        print(f"重叠时间: {overlap_time:.4f}秒 ({overlap_time/total_time*100:.2f}%)")
        
        # 计算有效带宽利用率
        if "network_stats" in results:
            total_data_transferred = results["network_stats"].get("total_data_transferred_bytes", 0) / (1024**3)  # 转换为GB
            effective_bandwidth = total_data_transferred / total_time  # GB/s
            print(f"总传输数据量: {total_data_transferred:.2f} GB")
            print(f"有效带宽: {effective_bandwidth:.2f} GB/s")
        
        # 可视化结果
        create_visualization(compute_time, communication_time, overlap_time)
        
    except Exception as e:
        print(f"分析结果时出错: {e}")

def create_visualization(compute_time, communication_time, overlap_time):
    """
    创建可视化图表
    """
    try:
        # 创建时间分布饼图
        plt.figure(figsize=(10, 6))
        
        # 饼图数据
        labels = ['计算时间', '通信时间', '重叠时间']
        sizes = [compute_time, communication_time, overlap_time]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.1, 0, 0)  # 突出计算时间
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Qwen3-30B推理时间分布')
        
        # 保存图表
        plt.savefig("simulation_results/time_distribution.png")
        print("\n已生成时间分布可视化图表: simulation_results/time_distribution.png")
        
    except Exception as e:
        print(f"创建可视化时出错: {e}")

if __name__ == "__main__":
    run_simulation()
```

## 8. 完整的配置文件和脚本

为了方便使用，我们将所有配置文件和脚本整合到一个目录中：

```bash
mkdir -p qwen3_30b_simulation
cd qwen3_30b_simulation

# 复制所有脚本
cp ../h800_gpu_config.py .
cp ../gen_4_H800_ibgda_topo.py .
cp ../parallel_inference_config.py .
cp ../generate_qwen3_30b_workload.py .
cp ../run_qwen3_30b_simulation.py .

# 创建README文件
cat > README.md << 'EOF'
# Qwen3-30B在4个H800 GPU上的SimAI仿真

本目录包含在4个H800 GPU上使用SimAI模拟Qwen3-30B模型的所有必要脚本和配置文件。

## 文件说明

- `h800_gpu_config.py`: H800 GPU参数配置
- `gen_4_H800_ibgda_topo.py`: 生成4 GPU全连接拓扑与IBGDA 400Gbps网络
- `parallel_inference_config.py`: 实现attention dp4和ffn ep4并行推理行为
- `generate_qwen3_30b_workload.py`: 使用AICB生成Qwen3-30B的工作负载文件
- `run_qwen3_30b_simulation.py`: 运行仿真的主脚本

## 使用方法

1. 确保已安装SimAI及其依赖
2. 运行主脚本：`python run_qwen3_30b_simulation.py`
3. 查看仿真结果：`simulation_results/`目录

## 仿真配置

- 模型：Qwen3-30B (隐藏层大小2048，注意力头32，KV头4，128专家，每token激活8专家)
- GPU：4x H800 (全连接拓扑)
- 网络：IBGDA 400Gbps (无NVLink)
- 并行策略：attention dp4，ffn ep4
- 批次大小：16
EOF

# 创建运行脚本
cat > run.sh << 'EOF'
#!/bin/bash

# 设置环境变量
export AS_HIGH_PRECISION=1
export AS_SEND_LAT=3
export AS_NVLS_ENABLE=0

# 运行仿真
python run_qwen3_30b_simulation.py
EOF

chmod +x run.sh

echo "所有文件已准备就绪，可以通过运行 ./run.sh 开始仿真"
```

## 9. 总结

本指南提供了在SimAI上模拟Qwen3-30B模型在4个H800 GPU上运行的完整实现方案。我们详细介绍了：

1. SimAI环境设置与依赖安装
2. H800 GPU参数配置
3. 4 GPU全连接拓扑与IBGDA 400Gbps网络设置
4. Attention dp4和ffn ep4并行推理行为实现
5. 使用AICB生成Qwen3-30B工作负载文件
6. 确定适合推理的批次大小
7. 运行仿真的完整Python代码

通过这些步骤，您可以在不需要实际硬件的情况下，精确模拟Qwen3-30B模型在特定硬件配置下的性能表现，为实际部署提供参考依据。

这种高精度的仿真可以帮助您：

- 评估不同并行策略的效果
- 优化批次大小以平衡延迟和吞吐量
- 分析通信和计算的重叠情况
- 识别潜在的性能瓶颈
- 在实际部署前进行系统调优

SimAI的高精度仿真能力（与真实世界结果的一致性达到98.1%）使其成为大型语言模型部署前评估的强大工具。
