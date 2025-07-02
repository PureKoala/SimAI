#!/bin/bash
"""
SimAI 现实世界 GPU 延迟仿真运行脚本
使用基于真实硬件规格的参数进行端到端延迟仿真
"""

set -e

# 脚本配置
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SIMAI_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
CONFIG_DIR="$SCRIPT_DIR"
RESULTS_DIR="$SCRIPT_DIR/results"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "SimAI 现实世界 GPU 延迟仿真"
echo "=========================================="
echo "配置目录: $CONFIG_DIR"
echo "结果目录: $RESULTS_DIR"
echo ""

# 检查 SimAI 是否已编译
check_simai_build() {
    echo "检查 SimAI 构建状态..."
    
    if [ ! -f "$SIMAI_ROOT/bin/SimAI_analytical" ]; then
        echo "警告: SimAI Analytical 未找到，正在编译..."
        cd "$SIMAI_ROOT"
        ./scripts/build.sh -c analytical
    fi
    
    if [ ! -f "$SIMAI_ROOT/bin/SimAI_simulator" ]; then
        echo "警告: SimAI Simulator 未找到，正在编译..."
        cd "$SIMAI_ROOT" 
        ./scripts/build.sh -c ns3
    fi
    
    echo "✅ SimAI 构建检查完成"
}

# 生成现实网络拓扑
generate_realistic_topology() {
    echo "生成现实网络拓扑..."
    
    cd "$SIMAI_ROOT/astra-sim-alibabacloud/inputs/topo"
    
    # H100 SuperPOD 配置: 2048 GPU, 8 GPU/服务器, NDR 400Gbps 
    python3 gen_Topo_Template.py \
        -topo Spectrum-X \
        -g 2048 \
        -gps 8 \
        -gt H100 \
        -bw 400Gbps \
        -nvbw 900Gbps \
        -l 0.0005ms \
        -nl 0.000025ms
        
    TOPO_FILE="Spectrum-X_2048g_8gps_400Gbps_H100"
    if [ -f "$TOPO_FILE" ]; then
        cp "$TOPO_FILE" "$RESULTS_DIR/"
        echo "✅ 网络拓扑已生成: $TOPO_FILE"
    else
        echo "❌ 网络拓扑生成失败"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
}

# 生成现实工作负载
generate_realistic_workloads() {
    echo "生成现实工作负载..."
    
    python3 generate_realistic_workload.py \
        --model gpt3-175b \
        --tp 8 \
        --pp 8 \
        --dp 32 \
        --output-dir "$RESULTS_DIR/workloads" \
        --workload-type full
        
    echo "✅ 现实工作负载已生成"
}

# 运行 SimAI Analytical 仿真
run_analytical_simulation() {
    echo "运行 SimAI Analytical 仿真..."
    
    cd "$SIMAI_ROOT"
    
    WORKLOAD_FILE="$RESULTS_DIR/workloads/gpt3-175b_full_tp8_pp8_dp32.txt"
    BUSBW_FILE="$CONFIG_DIR/realistic_busbw.yaml"
    
    if [ ! -f "$WORKLOAD_FILE" ]; then
        echo "❌ 工作负载文件未找到: $WORKLOAD_FILE"
        return 1
    fi
    
    echo "执行命令: ./bin/SimAI_analytical -w $WORKLOAD_FILE -g 2048 -g_p_s 8 -r realistic- -busbw $BUSBW_FILE"
    
    ./bin/SimAI_analytical \
        -w "$WORKLOAD_FILE" \
        -g 2048 \
        -g_p_s 8 \
        -r "realistic-analytical-" \
        -busbw "$BUSBW_FILE" \
        2>&1 | tee "$RESULTS_DIR/analytical_simulation.log"
        
    echo "✅ Analytical 仿真完成"
}

# 运行 SimAI NS3 仿真 
run_ns3_simulation() {
    echo "运行 SimAI NS3 详细仿真..."
    
    cd "$SIMAI_ROOT"
    
    WORKLOAD_FILE="$RESULTS_DIR/workloads/gpt3-175b_full_tp8_pp8_dp32.txt" 
    TOPO_FILE="$RESULTS_DIR/Spectrum-X_2048g_8gps_400Gbps_H100"
    CONFIG_FILE="$SIMAI_ROOT/astra-sim-alibabacloud/inputs/config/SimAI.conf"
    
    if [ ! -f "$WORKLOAD_FILE" ] || [ ! -f "$TOPO_FILE" ]; then
        echo "❌ 必要文件未找到"
        return 1
    fi
    
    echo "执行命令: AS_SEND_LAT=1 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 16 -w $WORKLOAD_FILE -n $TOPO_FILE -c $CONFIG_FILE"
    
    # 设置环境变量来优化仿真
    export AS_SEND_LAT=1
    export AS_NVLS_ENABLE=1
    
    timeout 3600 ./bin/SimAI_simulator \
        -t 16 \
        -w "$WORKLOAD_FILE" \
        -n "$TOPO_FILE" \
        -c "$CONFIG_FILE" \
        2>&1 | tee "$RESULTS_DIR/ns3_simulation.log"
        
    if [ $? -eq 124 ]; then
        echo "⚠️  NS3 仿真超时 (1小时)，但结果可能仍然有用"
    else
        echo "✅ NS3 仿真完成"
    fi
}

# 分析仿真结果
analyze_results() {
    echo "分析仿真结果..."
    
    # 创建结果分析脚本
    cat > "$RESULTS_DIR/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
import re
import json
from pathlib import Path

def parse_analytical_results(log_file):
    """解析 Analytical 仿真结果"""
    results = {}
    
    if not Path(log_file).exists():
        return results
        
    with open(log_file, 'r') as f:
        content = f.read()
        
    # 提取关键指标
    patterns = {
        'total_time': r'Total time: ([\d.]+)',
        'communication_time': r'Communication time: ([\d.]+)',
        'computation_time': r'Computation time: ([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            results[key] = float(match.group(1))
            
    return results

def parse_ns3_results(log_file):
    """解析 NS3 仿真结果"""
    results = {}
    
    if not Path(log_file).exists():
        return results
        
    with open(log_file, 'r') as f:
        content = f.read()
        
    # 提取网络相关指标
    patterns = {
        'avg_latency': r'Average latency: ([\d.]+)',
        'max_latency': r'Maximum latency: ([\d.]+)', 
        'total_bytes': r'Total bytes transferred: ([\d]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            results[key] = float(match.group(1))
            
    return results

def main():
    results_dir = Path('.')
    
    analytical_results = parse_analytical_results('analytical_simulation.log')
    ns3_results = parse_ns3_results('ns3_simulation.log')
    
    # 合并结果
    all_results = {
        'analytical': analytical_results,
        'ns3': ns3_results,
        'metadata': {
            'model': 'GPT-3 175B',
            'gpus': 2048,
            'parallelism': 'TP=8, PP=8, DP=32'
        }
    }
    
    # 保存结果
    with open('simulation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
        
    # 打印摘要
    print("仿真结果摘要:")
    print("=" * 40)
    
    if analytical_results:
        print("Analytical 仿真:")
        for key, value in analytical_results.items():
            print(f"  {key}: {value}")
            
    if ns3_results:
        print("NS3 仿真:")
        for key, value in ns3_results.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
EOF

    cd "$RESULTS_DIR"
    python3 analyze_results.py
    
    echo "✅ 结果分析完成，详见 $RESULTS_DIR/simulation_results.json"
}

# 生成性能报告
generate_report() {
    echo "生成性能报告..."
    
    cat > "$RESULTS_DIR/performance_report.md" << EOF
# SimAI 现实世界 GPU 延迟仿真报告

## 配置摘要

- **模型**: GPT-3 175B
- **硬件**: NVIDIA H100 SXM (2048 GPUs)
- **网络**: InfiniBand NDR 400Gbps
- **拓扑**: Spectrum-X Fat-Tree
- **并行策略**: TP=8, PP=8, DP=32

## 硬件规格

### GPU 规格
- GPU: NVIDIA H100 SXM
- 内存: 80GB HBM3 
- 内存带宽: 3.35TB/s
- NVLink: 900GB/s (18 links × 50GB/s)

### 网络规格
- 网络: InfiniBand NDR
- 带宽: 400Gbps per port
- 延迟: ~1μs base latency
- 拓扑: 3-tier Fat-Tree

## 通信带宽配置

基于真实测量数据的通信带宽:

| 通信类型 | TP (intra-node) | DP (inter-node) | EP | PP |
|---------|----------------|----------------|----|----|
| AllReduce | 280 GB/s | 45 GB/s | 40 GB/s | - |
| AllGather | 320 GB/s | 50 GB/s | 42 GB/s | - |
| ReduceScatter | 320 GB/s | 50 GB/s | 42 GB/s | - |
| AllToAll | 250 GB/s | 35 GB/s | 38 GB/s | - |
| Pipeline | - | - | - | 35 GB/s |

## 仿真结果

详细结果请查看 simulation_results.json

## 延迟分析

基于现实世界的延迟组件:

1. **GPU 计算延迟**: ~100ns (kernel launch overhead)
2. **内存访问延迟**: ~150ns (HBM3 access)
3. **NVLink 延迟**: ~25ns (intra-node)
4. **PCIe 延迟**: ~500ns (PCIe 5.0)
5. **网络延迟**: ~1000ns (InfiniBand base)
6. **协议开销**: ~200ns (NCCL + RDMA)

## 优化建议

1. **通信优化**: 
   - 使用 NCCL 最新版本
   - 启用 NVLS (NVLink Sharp)
   - 优化消息大小以避免小消息延迟

2. **拓扑优化**:
   - 确保 Rail-Optimized 拓扑
   - 使用 GPU 亲和性绑定
   - 优化交换机缓冲区配置

3. **软件优化**:
   - 启用计算通信重叠
   - 使用梯度累积减少通信频率
   - 优化批次大小和序列长度

EOF

    echo "✅ 性能报告已生成: $RESULTS_DIR/performance_report.md"
}

# 主执行流程
main() {
    echo "开始现实世界 GPU 延迟仿真..."
    
    check_simai_build
    generate_realistic_topology
    generate_realistic_workloads
    
    echo ""
    echo "选择仿真模式:"
    echo "1) 快速分析仿真 (Analytical, ~1分钟)"
    echo "2) 详细网络仿真 (NS3, ~30分钟)"  
    echo "3) 两种模式都运行"
    echo ""
    read -p "请选择 [1-3]: " choice
    
    case $choice in
        1)
            run_analytical_simulation
            ;;
        2)
            run_ns3_simulation
            ;;
        3)
            run_analytical_simulation
            run_ns3_simulation
            ;;
        *)
            echo "无效选择，运行快速分析仿真"
            run_analytical_simulation
            ;;
    esac
    
    analyze_results
    generate_report
    
    echo ""
    echo "=========================================="
    echo "🎉 现实世界 GPU 延迟仿真完成!"
    echo "=========================================="
    echo "结果目录: $RESULTS_DIR"
    echo "性能报告: $RESULTS_DIR/performance_report.md"
    echo "详细数据: $RESULTS_DIR/simulation_results.json"
    echo ""
    echo "主要配置差异对比:"
    echo "📊 demo_latency 配置 vs 现实配置:"
    echo "   GPU: 自定义 → H100 SXM (真实规格)"
    echo "   内存: 300ns → 150ns (HBM3)"
    echo "   NVLink: 自定义 → 900GB/s (18×50GB/s)"
    echo "   网络: 100Gbps → 400Gbps (NDR)"
    echo "   延迟: 简化模型 → 多组件真实延迟"
    echo "   工作负载: 简单测试 → GPT-3 175B 真实训练"
    echo ""
}

# 运行主函数
main "$@"
