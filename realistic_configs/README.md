# SimAI 现实世界 GPU 通信延迟仿真配置

本目录包含基于真实硬件规格的 SimAI 配置文件，用于进行准确的端到端 GPU 通信延迟仿真。

## 🎯 目标

替换 `demo_latency` 文件夹中的自定义参数，使用基于以下真实硬件的配置：
- **NVIDIA H100 SXM GPU** (80GB HBM3, 3.35TB/s 带宽)
- **NVLink 4.0** (18 links × 50GB/s = 900GB/s)
- **InfiniBand NDR** (400Gbps 网络)
- **Spectrum-X 网络拓扑** (NVIDIA 数据中心标准)

## 📁 文件结构

```
realistic_configs/
├── README.md                          # 本文件
├── 配置对比说明.md                     # 详细的配置对比分析
├── realistic_gpu_config.ini            # 主配置文件
├── realistic_busbw.yaml               # 通信带宽配置
├── realistic_system_config.json       # 系统级配置
├── realistic_network_config.json      # 网络详细配置
├── generate_realistic_workload.py     # 工作负载生成器
├── run_realistic_simulation.sh        # 一键运行脚本
└── results/                           # 仿真结果目录
```

## 🚀 快速开始

### 1. 一键运行完整仿真

```bash
cd /home/bytedance/SimAI/realistic_configs
./run_realistic_simulation.sh
```

脚本将自动完成：
- ✅ 检查 SimAI 构建状态
- ✅ 生成 H100 SuperPOD 网络拓扑
- ✅ 生成 GPT-3 175B 真实训练工作负载
- ✅ 运行分析仿真和/或详细网络仿真
- ✅ 分析结果并生成性能报告

### 2. 选择仿真模式

脚本提供三种仿真模式：
1. **快速分析仿真** (Analytical, ~1分钟): 使用数学模型快速评估
2. **详细网络仿真** (NS3, ~30分钟): 完整的网络行为仿真
3. **两种模式都运行**: 获得完整的对比分析

## 🔧 自定义配置

### 修改硬件规格

编辑 `realistic_gpu_config.ini`:

```ini
[hardware_specs]
gpu_type = H100                        # 可选: A100, H100, H800
gpu_memory_bandwidth_gbps = 3350       # H100 HBM3 带宽
nvlink_bandwidth_per_link_gbps = 50    # NVLink 4.0
network_bandwidth_gbps = 400           # InfiniBand NDR

[realistic_latency_model]
memory_access_ns = 150                 # HBM3 延迟
nvlink_latency_ns = 25                 # NVLink 延迟
network_latency_ns = 1000              # 网络基础延迟
```

### 调整通信带宽

编辑 `realistic_busbw.yaml`:

```yaml
TP:
  allreduce: 280     # 节点内 NVLink AllReduce
  allgather: 320     # 节点内 AllGather
  
DP:
  allreduce: 45      # 跨节点 AllReduce
  allgather: 50      # 跨节点 AllGather
```

### 生成不同模型的工作负载

```bash
# GPT-3 175B
python3 generate_realistic_workload.py --model gpt3-175b --tp 8 --pp 8 --dp 32

# LLaMA 65B  
python3 generate_realistic_workload.py --model llama-65b --tp 4 --pp 4 --dp 16

# PaLM 540B
python3 generate_realistic_workload.py --model palm-540b --tp 8 --pp 16 --dp 64
```

## 📊 与 Demo 配置的主要差异

| 配置类别 | Demo 配置 | 现实配置 | 改进 |
|----------|-----------|----------|------|
| **GPU** | 自定义/未指定 | H100 SXM 真实规格 | ✅ 准确硬件建模 |
| **内存** | 300ns 延迟 | 150ns HBM3 | ✅ 实际内存性能 |
| **NVLink** | 2880Gbps | 900GB/s (18×50GB/s) | ✅ NVLink 4.0 真实带宽 |
| **网络** | 100Gbps | 400Gbps NDR | ✅ 现代网络标准 |
| **拓扑** | 抽象 Fat-Tree | Spectrum-X SuperPOD | ✅ 实际数据中心设计 |
| **工作负载** | 简单测试 | GPT-3 175B 训练 | ✅ 真实 LLM 通信模式 |
| **通信带宽** | 任意设置 | 基于实测数据 | ✅ NCCL 性能基准 |

## 📈 结果分析

### 仿真完成后查看结果

1. **性能报告**: `results/performance_report.md`
   - 配置摘要和硬件规格
   - 通信带宽对比表
   - 延迟分析和优化建议

2. **详细数据**: `results/simulation_results.json`
   - 量化的延迟数据
   - 通信时间分解
   - 网络利用率统计

3. **仿真日志**: 
   - `results/analytical_simulation.log`
   - `results/ns3_simulation.log`

### 典型结果示例

```
现实世界 H100 集群延迟分析:
========================================
GPU 计算延迟:      100ns   (kernel launch)
内存访问延迟:      150ns   (HBM3)
NVLink 延迟:       25ns    (intra-node)
网络传播延迟:      1000ns  (InfiniBand)
协议开销:          200ns   (NCCL + RDMA)
----------------------------------------
端到端延迟:        ~50μs   (大消息 AllReduce)
小消息延迟:        ~20μs   (延迟主导)
通信带宽:          280GB/s (TP), 45GB/s (DP)
```

## 🎯 应用场景

### 1. 硬件选型决策
- 对比不同 GPU 型号的通信性能
- 评估网络升级的性能收益
- 分析内存带宽对整体性能的影响

### 2. 软件优化指导
- 优化 NCCL 参数设置
- 调整并行策略 (TP/PP/DP)
- 计算通信重叠优化

### 3. 系统设计验证
- 验证网络拓扑设计
- 分析规模扩展的性能趋势
- 识别系统瓶颈

## 📚 参考配置来源

### 硬件规格参考
- **NVIDIA H100 SXM**: [官方规格](https://www.nvidia.com/en-us/data-center/h100/)
- **NVLink 4.0**: NVIDIA NVLink 技术文档
- **InfiniBand NDR**: [Mellanox 产品规格](https://www.mellanox.com/)

### 性能数据参考
- **NCCL 性能指南**: NVIDIA Deep Learning Performance Guide
- **MLPerf 基准**: [MLPerf Training 结果](https://mlcommons.org/en/training-normal-21/)
- **DGX SuperPOD**: NVIDIA DGX SuperPOD 架构文档

### 通信算法参考
- **Megatron-LM**: [论文和实现](https://github.com/NVIDIA/Megatron-LM)
- **DeepSpeed**: [通信优化技术](https://github.com/microsoft/DeepSpeed)
- **FairScale**: [并行策略实现](https://github.com/facebookresearch/fairscale)

## ⚠️ 注意事项

### 1. 系统要求
- 确保 SimAI 已正确编译 (Analytical + NS3)
- 推荐使用 Ubuntu 20.04 + GCC 9.4.0
- 需要足够的内存和存储空间用于大规模仿真

### 2. 仿真时间
- **Analytical 仿真**: 1-5 分钟
- **NS3 详细仿真**: 30-60 分钟 (取决于工作负载复杂度)
- **大规模配置**: 可能需要数小时

### 3. 结果解释
- 仿真结果是基于模型的估算，与实际测量可能有差异
- 建议与实际 GPU 集群测试结果进行对比验证
- 关注相对性能趋势而非绝对数值

## 🔍 故障排除

### 常见问题

1. **编译错误**
   ```bash
   cd /home/bytedance/SimAI
   ./scripts/build.sh -c analytical
   ./scripts/build.sh -c ns3
   ```

2. **拓扑生成失败**
   ```bash
   cd astra-sim-alibabacloud/inputs/topo
   python3 gen_Topo_Template.py -topo Spectrum-X -g 2048 -gps 8 -gt H100
   ```

3. **工作负载错误**
   ```bash
   python3 generate_realistic_workload.py --model gpt3-175b --output-dir ./test
   ```

### 获取帮助

如有问题，请：
1. 查看 `results/` 目录下的日志文件
2. 参考 `配置对比说明.md` 了解配置细节
3. 联系 SimAI 团队获取技术支持

## 🎉 开始使用

现在您可以使用这些基于真实硬件规格的配置来进行准确的 GPU 通信延迟仿真：

```bash
cd /home/bytedance/SimAI/realistic_configs
./run_realistic_simulation.sh
```

享受真实世界级别的 GPU 延迟仿真体验！
