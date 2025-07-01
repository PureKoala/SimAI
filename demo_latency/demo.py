#!/usr/bin/env python3
"""
SimAI GPU延迟测试演示脚本
Demonstration script for SimAI GPU latency testing

展示如何使用SimAI进行GPU网络延迟分析
"""

import os
import sys
import time
from pathlib import Path

def print_banner():
    """打印横幅"""
    print("=" * 60)
    print("     SimAI GPU网络延迟仿真测试演示")
    print("=" * 60)
    print()

def print_step(step_num, description):
    """打印步骤"""
    print(f"步骤 {step_num}: {description}")
    print("-" * 40)

def demo_basic_usage():
    """演示基本用法"""
    print_banner()
    
    print_step(1, "检查环境和依赖")
    print("检查Python环境...")
    print(f"Python版本: {sys.version}")
    
    try:
        import pandas, numpy, matplotlib, seaborn
        print("✅ 所有依赖包已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: make install-deps")
        return
    
    print("\n" + "="*50)
    time.sleep(2)
    
    print_step(2, "创建配置文件")
    from simai_latency_simulator import create_default_config
    config_file = create_default_config()
    print(f"✅ 配置文件已创建: {config_file}")
    
    print("\n" + "="*50)
    time.sleep(2)
    
    print_step(3, "运行简单延迟测试")
    from simai_latency_simulator import SimAILatencySimulator
    
    # 创建仿真器
    simulator = SimAILatencySimulator(config_file)
    print(f"✅ 仿真器已初始化")
    print(f"   网络后端: {simulator.network_backend}")
    print(f"   仿真器: {simulator.simulator_binary}")
    
    # 生成测试工作负载
    print("\n生成测试工作负载...")
    workload_file = simulator.generate_workload_file("all_reduce", 65536, 2)
    print(f"✅ 工作负载文件: {workload_file}")
    
    # 生成网络配置
    print("\n生成网络配置...")
    network_config = simulator.generate_network_config("fat_tree", 100, 1000)
    print(f"✅ 网络配置文件: {network_config}")
    
    print("\n" + "="*50)
    time.sleep(2)
    
    print_step(4, "运行仿真")
    print("运行延迟仿真...")
    result = simulator.run_simulation(workload_file, network_config)
    
    latency_comp = result['latency_components']
    print(f"✅ 仿真完成!")
    print(f"   总延迟: {latency_comp.total_ns:.1f} ns ({latency_comp.total_ns/1000:.2f} μs)")
    print(f"   延迟组件:")
    print(f"     GPU计算: {latency_comp.gpu_compute_ns:.1f} ns")
    print(f"     内存访问: {latency_comp.memory_access_ns:.1f} ns")
    print(f"     PCIe传输: {latency_comp.pcie_transfer_ns:.1f} ns")
    print(f"     网络传播: {latency_comp.network_propagation_ns:.1f} ns")
    print(f"     交换机处理: {latency_comp.switch_processing_ns:.1f} ns")
    print(f"     协议开销: {latency_comp.protocol_overhead_ns:.1f} ns")
    print(f"     序列化: {latency_comp.serialization_ns:.1f} ns")
    
    print("\n" + "="*50)
    time.sleep(2)
    
    print_step(5, "演示批量测试")
    print("运行小规模批量测试...")
    
    # 运行小规模延迟扫描
    results_df = simulator.run_latency_sweep(
        comm_types=["all_reduce", "p2p"],
        data_sizes=[1024, 4096, 16384],
        topologies=["fat_tree"],
        num_iterations=2
    )
    
    print(f"✅ 批量测试完成!")
    print(f"   总测试数: {len(results_df)}")
    print(f"   平均延迟: {results_df['total_latency_us'].mean():.2f} μs")
    print(f"   最小延迟: {results_df['total_latency_us'].min():.2f} μs")
    print(f"   最大延迟: {results_df['total_latency_us'].max():.2f} μs")
    
    print("\n" + "="*50)
    time.sleep(2)
    
    print_step(6, "分析结果")
    analysis = simulator.analyze_results(results_df)
    
    print("延迟组件分解:")
    for comp, data in analysis['component_breakdown'].items():
        comp_name = comp.replace('_ns', '').replace('_', ' ').title()
        print(f"  {comp_name}: {data['avg_ns']:.1f} ns ({data['percentage']:.1f}%)")
    
    print("\n按通信类型分析:")
    for comm_type, data in analysis['by_comm_type'].items():
        print(f"  {comm_type}: {data['avg_latency_us']:.2f} μs (±{data['std_latency_us']:.2f})")
    
    print("\n" + "="*50)
    time.sleep(2)
    
    print_step(7, "保存结果")
    output_file = simulator.save_results(results_df, analysis)
    print(f"✅ 结果已保存: {output_file}")
    
    # 生成图表
    try:
        simulator.generate_plots(results_df)
        print("✅ 图表已生成: results/plots/")
    except Exception as e:
        print(f"⚠️  图表生成失败: {e}")
    
    print("\n" + "="*50)
    print("🎉 演示完成!")
    print("\n下一步:")
    print("1. 查看结果文件: results/")
    print("2. 运行完整测试: make full-test")
    print("3. 自定义配置: 编辑 configs/simai_latency_config.ini")
    print("4. 查看文档: README.md")

def demo_advanced_usage():
    """演示高级用法"""
    print_banner()
    
    print("高级用法演示")
    print("=" * 30)
    
    print("\n1. 自定义延迟模型:")
    print("   编辑 configs/simai_latency_config.ini 中的 [latency_model] 部分")
    
    print("\n2. 网络拓扑对比:")
    print("   make topology-test")
    
    print("\n3. 数据大小扫描:")
    print("   make size-sweep-test")
    
    print("\n4. 自定义测试:")
    print("   make custom-test COMM_TYPES='all_reduce' DATA_SIZES='1024 4096'")
    
    print("\n5. Python API使用:")
    print("""
   from simai_latency_simulator import SimAILatencySimulator
   
   simulator = SimAILatencySimulator()
   workload = simulator.generate_workload_file("all_reduce", 65536)
   network = simulator.generate_network_config("fat_tree")
   result = simulator.run_simulation(workload, network)
   """)

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "advanced":
        demo_advanced_usage()
    else:
        demo_basic_usage()

if __name__ == "__main__":
    main()
