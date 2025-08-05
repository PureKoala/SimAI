# MoE框架通信计算重叠分析配置

这个配置套件专门为MoE (Mixture of Experts) 框架设计，重点支持通信和计算交叠的详细分析。

## 📁 配置文件结构

```
myMoE/
├── moe_system_config.json          # 系统配置（包含重叠策略）
├── moe_busbw.yaml                  # MoE优化的通信带宽配置  
├── moe_network_config.json         # 网络拓扑和重叠优化配置
├── moe_workload_overlap.txt        # MoE工作负载（支持重叠分析）
├── run_moe_overlap_analysis.sh     # 重叠分析执行脚本
└── README.md                       # 本文档
```

## 🎯 重叠分析特性

### 1. 多重叠策略支持
- **none**: 无重叠基准测试
- **conservative**: 保守重叠策略（低风险）
- **aggressive**: 激进重叠策略（推荐）
- **max**: 最大重叠策略（高性能，高内存开销）

### 2. 粒度控制
- **计算粒度**: `kernel` | `layer`
- **通信粒度**: `message` | `tensor` | `packet`

### 3. MoE专用优化
- 专家发现与Token路由重叠
- 专家计算与梯度通信重叠
- 负载均衡感知的重叠调度
- 动态重叠阈值调整

## ⚙️ 关键配置参数

### 系统配置 (moe_system_config.json)

```json
{
  "scheduling": {
    "overlap_strategy": "aggressive",      // 重叠策略
    "compute_granularity": "kernel",       // 计算粒度
    "communication_granularity": "message" // 通信粒度
  },
  
  "overlap-configuration": {
    "overlap_ratios": {
      "TP": 0.85,    // Tensor并行重叠效率
      "DP": 0.75,    // Data并行重叠效率  
      "EP": 0.90,    // Expert并行重叠效率（最高）
      "PP": 0.65     // Pipeline并行重叠效率
    }
  },
  
  "moe-specific": {
    "num_experts": 128,                    // 专家数量
    "experts_per_gpu": 16,                 // 每GPU专家数
    "top_k": 2,                           // Top-K路由
    "expert_parallelism_degree": 8         // 专家并行度
  }
}
```

### 通信带宽配置 (moe_busbw.yaml)

```yaml
# 针对MoE优化的通信带宽，包含重叠效率
EP:
  alltoall: 42                    # MoE最关键的通信模式
  overlap_efficiency: 0.90        # EP层重叠效率最高

# 重叠参数
overlap_parameters:
  compute_intensity_factor:
    medium: 0.8                   # 中等计算强度最优
    high: 0.9                     # 计算密集时重叠效率最高
    
  message_size_overlap:
    medium_msg: 0.85              # 1MB-64MB最佳重叠区间
```

### 网络配置 (moe_network_config.json)

```json
{
  "moe-network-optimization": {
    "expert-placement-aware": true,        // 专家放置感知
    "token-routing-optimization": true,    // Token路由优化
    "adaptive-routing": true               // 自适应路由
  },
  
  "overlap-network-features": {
    "pipeline-support": {
      "enabled": true,
      "depth": 8,                         // 流水线深度
      "chunk-size": "16MB"                // 块大小
    }
  }
}
```

## 🚀 快速开始

### 1. 执行重叠分析

```bash
cd /home/bytedance/SimAI/myMoE
./run_moe_overlap_analysis.sh
```

### 2. 单独测试特定策略

```bash
# 测试激进重叠策略
$SIMAI_ROOT/bin/SimAI_analytical \
    --workload-configuration="moe_workload_overlap.txt" \
    --system-configuration="moe_system_config.json" \
    --comm-group-configuration="moe_busbw.yaml" \
    --network-configuration="moe_network_config.json" \
    --dp-overlap-ratio=0.75 \
    --ep-overlap-ratio=0.90 \
    --tp-overlap-ratio=0.85 \
    --pp-overlap-ratio=0.65
```

## 📊 分析输出

### 性能指标
- **总执行时间**: 包含重叠优化的端到端时间
- **重叠效率**: 实际重叠时间占理论最大重叠时间的比例
- **专家利用率**: 专家计算资源的利用效率
- **网络利用率**: 网络通信资源的利用效率

### 可视化报告
- `moe_overlap_analysis.png`: 性能对比图表
- `moe_overlap_analysis_report.md`: 详细分析报告

## 🔧 高级配置

### 1. 自定义重叠策略

修改 `moe_system_config.json` 中的重叠参数:

```json
{
  "communication-overlap-parameters": {
    "kernel-level-overlap": {
      "overlap-threshold": "1MB",         // 重叠阈值
      "max-concurrent-kernels": 4        // 最大并发kernel数
    },
    
    "message-level-overlap": {
      "chunk-size": "64MB",              // 消息分块大小
      "pipeline-stages": 4               // 流水线阶段数
    }
  }
}
```

### 2. MoE专用工作负载

`moe_workload_overlap.txt` 包含典型的MoE通信模式:

```
# 专家发现和Token路由 - 关键重叠点
moe_expert_discovery_1  -1  262144  ALLTOALL_EP  16777216  1  NONE  0  131072  NONE  0  100
moe_token_routing_1     -1  524288  ALLTOALL     33554432  262144  ALLTOALL_EP  16777216  262144  NONE  0  100

# 专家计算与通信重叠
moe_expert_compute_1    -1  4194304  ALLGATHER  268435456  2097152  REDUCESCATTER  268435456  2097152  NONE  0  100
```

### 3. 动态调优

启用自适应重叠优化:

```yaml
# 在 moe_busbw.yaml 中
dynamic_optimization:
  adaptive_overlap: true              # 启用自适应重叠
  runtime_profiling: true             # 启用运行时分析
  auto_tuning: true                   # 启用自动调优
  overlap_threshold: "4MB"            # 动态重叠阈值
```

## 📈 性能优化建议

### 1. 重叠策略选择
- **开发阶段**: 使用 `conservative` 策略保证稳定性
- **生产环境**: 使用 `aggressive` 策略获得最佳性能
- **性能极限测试**: 使用 `max` 策略

### 2. 粒度优化
- **计算粒度**: `kernel` 级别提供更细粒度的重叠控制
- **通信粒度**: `message` 级别在大多数场景下最优

### 3. MoE特有优化
- 确保专家负载均衡以提高重叠效率
- 优化Token路由算法减少通信开销
- 使用专家放置策略减少跨节点通信

## 🐛 故障排除

### 常见问题

1. **重叠效率低**
   - 检查计算通信比例是否合理
   - 确认消息大小在最优重叠区间 (1MB-64MB)
   - 验证专家负载是否均衡

2. **内存不足**
   - 减少 `overlap_buffer_size`
   - 降低 `pipeline_depth`
   - 使用 `conservative` 重叠策略

3. **性能下降**
   - 检查网络拥塞情况
   - 验证专家放置策略
   - 调整重叠阈值

### 调试选项

```bash
# 启用详细日志
export SIMAI_LOG_LEVEL=DEBUG
export NCCL_DEBUG=INFO

# 内存使用监控
export SIMAI_MEMORY_MONITORING=true
```

## 📚 参考文献

1. **MoE模型**: Switch Transformer, GLaM, PaLM-2
2. **通信优化**: NCCL, Horovod, FairScale
3. **重叠技术**: TensorFlow, PyTorch, DeepSpeed

---

**注意**: 本配置基于H100集群和最新的MoE框架设计，针对您提供的重叠策略进行了优化。建议根据具体硬件环境调整相关参数。
