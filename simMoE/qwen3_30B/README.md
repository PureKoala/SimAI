# Qwen3-30B在4个H800 GPU上的SimAI仿真

本项目提供了使用SimAI工具在4个H800 GPU上模拟Qwen3-30B模型的完整实现方案。通过高精度仿真，可以在不需要实际硬件的情况下评估模型性能和优化部署策略。

## 项目概述

本项目实现了以下关键功能：

1. 配置4个H800 GPU参数和IBGDA 400Gbps网络（不使用NVLink）
2. 实现attention dp4（数据并行）和ffn ep4（专家并行）的并行推理行为
3. 使用AICB生成Qwen3-30B模型的工作负载文件
4. 运行高精度仿真并分析结果

## 文件结构

```
.
├── README.md                           # 项目说明文档
├── Qwen3-30B在4个H800_GPU上的SimAI仿真实现指南.md  # 详细实现指南
├── h800_gpu_config.py                  # H800 GPU参数配置
├── gen_4_H800_ibgda_topo.py            # 4 GPU全连接拓扑生成
├── parallel_inference_config.py         # 并行推理配置
├── generate_qwen3_30b_workload.py      # Qwen3-30B工作负载生成
├── run_qwen3_30b_simulation.py         # 主仿真脚本
├── run.sh                              # 运行脚本
└── simulation_results/                 # 仿真结果目录（运行后生成）
    ├── summary.json                    # 结果摘要
    └── figures/                        # 可视化图表
```

## 环境要求

- SimAI及其依赖已安装（请参考SimAI官方文档）
- Python 3.6+
- NumPy, Matplotlib, JSON
- 足够的磁盘空间用于存储仿真结果

## 安装步骤

1. 克隆SimAI代码库并安装依赖：

```bash
git clone https://github.com/alibaba/SimAI.git
cd SimAI
git submodule init
git submodule update
./build.sh
```

2. 复制本项目文件到SimAI目录：

```bash
cp -r /path/to/this/project/* /path/to/SimAI/
```

## 使用方法

1. 进入项目目录：

```bash
cd /path/to/SimAI
```

2. 运行仿真：

```bash
chmod +x run.sh
./run.sh
```

3. 查看结果：

仿真完成后，结果将保存在`simulation_results`目录中，包括：
- `summary.json`：仿真结果摘要
- `figures/`：可视化图表

## 模型参数

Qwen3-30B模型参数：

| 参数 | 值 |
|------|-----|
| 隐藏层大小 | 2048 |
| 注意力头数 | 32 |
| KV头数 | 4 |
| 头维度 | 128 |
| 激活函数 | silu |
| 隐藏层层数 | 48 |
| 中间层大小 (累积) | 6144 |
| MoE中间层大小 (per Expert) | 768 |
| 专家数量 | 128 |
| 每个token激活的专家数 | 8 |
| RMS归一化epsilon | 1e-06 |
| 数据类型 | bfloat16 |
| 词汇表大小 | 151936 |

## 并行策略

- **注意力机制**：数据并行（dp4），将输入数据分割到4个GPU上
- **前馈网络**：专家并行（ep4），将MoE专家分布到4个GPU上

## 批次大小选择

对于Qwen3-30B模型在4个H800 GPU上的推理，我们选择批次大小为16，这是基于以下考虑：

1. **GPU内存限制**：H800 GPU有80GB内存，对于30B级别的MoE模型，每个GPU需要存储约7.5B参数
2. **MoE特性**：MoE模型只激活部分专家，因此内存效率更高
3. **并行策略**：使用数据并行和专家并行组合可以更有效地利用资源
4. **经验值**：对于类似规模的模型，在4个80GB GPU上，批次大小通常在8-32之间
5. **延迟与吞吐量平衡**：批次大小16在保证合理延迟的同时提供较好的吞吐量

## 注意事项

1. 本项目假设SimAI已正确安装并配置
2. 实际仿真时间取决于主机性能
3. 工作负载生成需要AICB工具，请确保其可用
4. 仿真结果的准确性取决于参数配置的精确性

## 参考资料

- SimAI官方文档
- AICB使用指南
- Qwen3-30B模型规格
- H800 GPU技术规格

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
