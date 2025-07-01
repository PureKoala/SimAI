# SimAI GPU网络延迟仿真测试工具

基于SimAI框架的GPU网络通信延迟仿真测试工具，提供纳秒级精度的延迟分析。

## 🚀 功能特性

- **纳秒级精度**: 基于SimAI/Astra-sim的高精度仿真
- **多维度分析**: GPU计算、内存访问、PCIe传输、网络传播等延迟组件分解
- **多种通信模式**: All-Reduce、点对点、广播等集合通信
- **网络拓扑支持**: Fat-Tree、Dragonfly等数据中心网络拓扑
- **可视化分析**: 自动生成延迟分析图表
- **配置灵活**: 支持自定义测试参数和网络配置

## 📁 文件结构

```
demo_latency/
├── simai_latency_simulator.py    # 主仿真程序
├── run_test.py                   # 快速运行脚本
├── Makefile                      # 自动化构建文件
├── README.md                     # 说明文档
├── configs/                      # 配置文件目录
│   ├── simai_latency_config.ini  # 主配置文件
│   └── system_config.json        # 系统配置文件
├── workloads/                    # 自动生成的工作负载文件
├── results/                      # 测试结果目录
│   └── plots/                    # 生成的图表
```

## 🛠️ 安装和环境配置

### 1. 安装Python依赖

```bash
# 方法1: 使用Makefile
make install-deps

# 方法2: 手动安装
pip install pandas numpy matplotlib seaborn configparser
```

### 2. 检查环境

```bash
make check-env
```

## 🏃‍♂️ 快速开始

### 1. 创建配置文件

```bash
make create-config
```

### 2. 运行快速测试

```bash
# 使用Makefile (推荐)
make quick-test

# 或者使用Python脚本
python run_test.py quick
```

### 3. 查看结果

```bash
# 显示最近的结果文件
make show-results

# 为最近的结果生成图表
make plot-last
```

## 📊 测试模式

### 快速测试 (3分钟)
```bash
make quick-test
```
- 通信类型: All-Reduce, P2P
- 数据大小: 1KB, 4KB, 16KB, 64KB
- 网络拓扑: Fat-Tree
- 迭代次数: 3

### 完整测试 (10分钟)
```bash
make full-test
```
- 通信类型: All-Reduce, P2P, Broadcast
- 数据大小: 1KB - 1MB (6个数据点)
- 网络拓扑: Fat-Tree, Dragonfly
- 迭代次数: 5

### 自定义测试
```bash
make custom-test COMM_TYPES="all_reduce p2p" DATA_SIZES="1024 4096" ITERATIONS=5
```

### 专项测试

```bash
# 网络拓扑对比测试
make topology-test

# 数据大小扫描测试  
make size-sweep-test

# 详细测试 (包含所有选项)
make detailed-test
```

## ⚙️ 配置说明

### 主配置文件 (`configs/simai_latency_config.ini`)

```ini
[simulation]
network_backend = ns3          # 网络后端: ns3/analytical/physical
simulator_binary = auto        # 仿真器路径
verbose_logging = true         # 详细日志

[latency_model]
# 基础延迟参数 (纳秒)
gpu_compute_ns = 100          # GPU计算延迟
memory_access_ns = 300        # 内存访问延迟
pcie_gen4_per_gb_ns = 8000   # PCIe Gen4每GB传输延迟
network_base_ns = 1000       # 基础网络延迟
switch_processing_ns = 500   # 交换机处理延迟
protocol_overhead_ns = 200   # 协议开销

[test_parameters]
comm_types = ["all_reduce", "p2p", "broadcast"]
data_sizes_kb = [1, 4, 16, 64, 256, 1024, 4096]
topologies = ["fat_tree", "dragonfly"]
num_iterations = 5
```

## 📈 延迟组件分析

工具会分解以下延迟组件：

1. **GPU计算延迟** (100ns基础): GPU内核启动和基本计算延迟
2. **内存访问延迟** (300ns基础): GPU内存访问延迟，与数据大小相关
3. **PCIe传输延迟** (8μs/GB): PCIe Gen4传输延迟
4. **网络传播延迟** (1μs基础): 网络物理传播延迟
5. **交换机处理延迟** (500ns): 网络交换机处理延迟
6. **协议开销** (200ns): 通信协议栈开销
7. **序列化延迟**: 数据序列化和反序列化延迟

## 📊 结果分析

### 1. 输出文件

- **JSON结果**: `results/simai_latency_results_TIMESTAMP.json`
- **CSV数据**: `results/simai_latency_results_TIMESTAMP.csv`

### 2. 自动生成图表

- **延迟vs数据大小**: 显示不同通信类型的延迟随数据大小变化
- **延迟组件分解**: 饼图显示各组件延迟占比
- **拓扑对比**: 不同网络拓扑的延迟对比
- **组件堆叠图**: 延迟组件随数据大小的变化

### 3. 分析报告示例

```
SimAI GPU网络延迟仿真完成
============================================================
总测试数: 120
平均总延迟: 45.67 μs
最小延迟: 12.34 μs
最大延迟: 234.56 μs

延迟组件分解:
  Gpu Compute: 100.0 ns (0.2%)
  Memory Access: 1500.0 ns (3.3%)
  Pcie Transfer: 12000.0 ns (26.3%)
  Network Propagation: 1000.0 ns (2.2%)
  Switch Processing: 500.0 ns (1.1%)
  Protocol Overhead: 200.0 ns (0.4%)
  Serialization: 30400.0 ns (66.5%)
```

## 🔧 高级用法

### 1. 直接使用Python API

```python
from simai_latency_simulator import SimAILatencySimulator

# 创建仿真器
simulator = SimAILatencySimulator("configs/simai_latency_config.ini")

# 运行单次测试
workload_file = simulator.generate_workload_file("all_reduce", 65536)
network_config = simulator.generate_network_config("fat_tree")
result = simulator.run_simulation(workload_file, network_config)

print(f"总延迟: {result['total_latency_ns']} ns")
```

### 2. 批量测试

```python
# 运行延迟扫描
results_df = simulator.run_latency_sweep(
    comm_types=["all_reduce", "p2p"],
    data_sizes=[1024, 4096, 16384],
    topologies=["fat_tree"],
    num_iterations=5
)

# 分析结果
analysis = simulator.analyze_results(results_df)

# 生成图表
simulator.generate_plots(results_df)
```

### 3. 自定义延迟模型

编辑配置文件中的 `[latency_model]` 部分来调整延迟参数：

```ini
[latency_model]
# 针对特定硬件的参数调整
gpu_compute_ns = 150          # 较慢的GPU
memory_access_ns = 250        # 更快的内存
pcie_gen4_per_gb_ns = 6000   # PCIe Gen5
network_base_ns = 500        # 低延迟网络
```

## 🎯 使用场景

### 1. 网络架构设计
- 比较不同网络拓扑的性能
- 分析带宽和延迟的权衡
- 优化数据中心网络设计

### 2. 通信优化
- 选择最优的集合通信算法
- 分析不同数据大小的性能特征
- 优化通信模式

### 3. 性能建模
- 预测大规模集群的通信性能
- 分析延迟瓶颈
- 指导硬件采购决策

## 🐛 故障排除

### 1. 仿真器未找到
```
警告: 未找到SimAI仿真器二进制文件，将使用数学模型模拟
```
这是正常现象，工具会自动使用内置的数学模型进行延迟计算。

### 2. Python依赖缺失
```bash
make install-deps
```

### 3. 结果文件未生成
检查 `results/` 目录权限，确保程序有写入权限。

### 4. 图表无法显示
确保安装了 `matplotlib` 和 `seaborn`：
```bash
pip install matplotlib seaborn
```

## 📚 参考资料

### SimAI框架相关
- **Astra-sim**: 分布式训练仿真器
- **ns-3**: 网络仿真后端
- **AICB**: AI通信基准测试工具

### 网络延迟建模
- **Fat-Tree拓扑**: 数据中心常用的三层网络架构
- **Dragonfly拓扑**: 低直径高性能网络拓扑
- **InfiniBand**: 高性能计算网络标准

### GPU通信
- **NCCL**: NVIDIA集合通信库
- **All-Reduce**: 分布式训练核心通信原语
- **PCIe**: GPU与CPU间的高速通信接口

## 🤝 贡献

欢迎提交Issues和Pull Requests来改进这个工具！

## 📄 许可证

本项目基于SimAI框架，遵循相应的开源许可证。
