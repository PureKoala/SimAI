#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行Qwen3-30B在4个H800 GPU上的SimAI仿真
整合所有步骤并执行完整的仿真流程
"""

import os
import time
import json
import matplotlib
# 设置matplotlib使用Agg后端，避免需要GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gen_4_H800_ibgda_topo import generate_4_h800_ibgda_topology
from generate_qwen3_30b_workload import generate_qwen3_30b_workload

# 设置中文字体 - 使用更通用的字体设置
try:
    # 尝试设置中文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] + plt.rcParams['font.sans-serif']
except:
    # 如果字体设置失败，使用默认字体
    print("警告：无法设置中文字体，将使用默认字体")
    pass

def run_simulation():
    """
    运行Qwen3-30B在4个H800 GPU上的SimAI仿真
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
    
    # 确保输出目录存在
    os.makedirs("simulation_results", exist_ok=True)
    
    # 步骤4：运行仿真
    print("\n步骤3：运行仿真...")
    
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
    simulator_path = os.path.join(simai_dir, "bin")
    
    # 确保仿真器存在
    if not os.path.exists(simulator_path):
        print(f"错误：找不到仿真器 {simulator_path}")
        return
    
    simulation_command = f"{simulator_path}/SimAI_simulator --workload {workload_file} --network {topo_file} --output simulation_results"
    print(f"执行命令：{simulation_command}")
    
    start_time = time.time()
    
    # 执行仿真命令
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
        # 检查结果文件是否存在
        results_file = "simulation_results/summary.json"
        if not os.path.exists(results_file):
            print(f"警告：结果文件 {results_file} 不存在，创建示例结果进行演示")
            create_example_simulation_results()
        
        # 读取仿真结果
        with open(results_file, "r", encoding='utf-8') as f:
            results = json.load(f)
        
        # 提取关键指标
        total_time = results.get("total_time_ms", 0) / 1000  # 转换为秒
        compute_time = results.get("compute_time_ms", 0) / 1000
        communication_time = results.get("communication_time_ms", 0) / 1000
        overlap_time = results.get("overlap_time_ms", 0) / 1000
        
        # 计算实际时间（考虑重叠）
        actual_time = total_time - overlap_time
        
        # 打印关键指标
        print("\n仿真结果摘要:")
        print(f"总执行时间: {total_time:.4f}秒")
        print(f"计算时间: {compute_time:.4f}秒 ({compute_time/total_time*100:.2f}%)")
        print(f"通信时间: {communication_time:.4f}秒 ({communication_time/total_time*100:.2f}%)")
        print(f"重叠时间: {overlap_time:.4f}秒 ({overlap_time/total_time*100:.2f}%)")
        print(f"实际执行时间: {actual_time:.4f}秒")
        
        # 计算有效带宽利用率
        if "network_stats" in results:
            total_data_transferred = results["network_stats"].get("total_data_transferred_bytes", 0) / (1024**3)  # 转换为GB
            effective_bandwidth = total_data_transferred / total_time  # GB/s
            print(f"总传输数据量: {total_data_transferred:.2f} GB")
            print(f"有效带宽: {effective_bandwidth:.2f} GB/s")
            
            # 通信操作统计
            if "communication_operations" in results["network_stats"]:
                comm_ops = results["network_stats"]["communication_operations"]
                print("\n通信操作统计:")
                for op, count in comm_ops.items():
                    print(f"  {op}: {count}次")
        
        # GPU利用率统计
        if "gpu_stats" in results:
            print("\nGPU利用率统计:")
            for gpu in results["gpu_stats"]:
                gpu_id = gpu.get("gpu_id", 0)
                compute_util = gpu.get("compute_utilization", 0) * 100
                memory_util = gpu.get("memory_utilization", 0) * 100
                print(f"  GPU {gpu_id}: 计算利用率 {compute_util:.1f}%, 内存利用率 {memory_util:.1f}%")
        
        # 创建可视化
        create_visualizations(results)
        
    except Exception as e:
        print(f"分析结果时出错: {e}")

def create_visualizations(results):
    """
    创建可视化图表
    """
    try:
        # 创建输出目录
        os.makedirs("simulation_results/figures", exist_ok=True)
        
        # 提取关键指标
        total_time = results.get("total_time_ms", 0) / 1000
        compute_time = results.get("compute_time_ms", 0) / 1000
        communication_time = results.get("communication_time_ms", 0) / 1000
        overlap_time = results.get("overlap_time_ms", 0) / 1000
        
        # 1. 时间分布饼图
        plt.figure(figsize=(10, 6))
        
        # 调整数据以避免重复计算重叠时间
        pie_data = [
            compute_time - overlap_time,  # 纯计算时间
            communication_time - overlap_time,  # 纯通信时间
            overlap_time  # 重叠时间
        ]
        
        labels = ['纯计算时间', '纯通信时间', '重叠时间']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.1, 0, 0)  # 突出计算时间
        
        plt.pie(pie_data, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Qwen3-30B推理时间分布')
        
        # 保存图表
        plt.savefig("simulation_results/figures/time_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. GPU利用率条形图
        if "gpu_stats" in results:
            plt.figure(figsize=(10, 6))
            
            gpu_ids = []
            compute_utils = []
            memory_utils = []
            
            for gpu in results["gpu_stats"]:
                gpu_ids.append(f"GPU {gpu.get('gpu_id', 0)}")
                compute_utils.append(gpu.get("compute_utilization", 0) * 100)
                memory_utils.append(gpu.get("memory_utilization", 0) * 100)
            
            x = np.arange(len(gpu_ids))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            rects1 = ax.bar(x - width/2, compute_utils, width, label='计算利用率')
            rects2 = ax.bar(x + width/2, memory_utils, width, label='内存利用率')
            
            ax.set_ylabel('利用率 (%)')
            ax.set_title('GPU利用率统计')
            ax.set_xticks(x)
            ax.set_xticklabels(gpu_ids)
            ax.legend()
            
            # 添加数值标签
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3点垂直偏移
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            fig.tight_layout()
            
            # 保存图表
            plt.savefig("simulation_results/figures/gpu_utilization.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 通信与计算重叠图
        plt.figure(figsize=(12, 6))
        
        # 创建一个简化的重叠时间线图
        # X轴是时间，Y轴表示不同的操作类型
        
        # 示例数据 - 实际应用中应从仿真结果中提取
        operations = [
            {"name": "Attention Forward", "start": 0, "duration": 0.8, "type": "compute"},
            {"name": "AllReduce", "start": 0.5, "duration": 0.6, "type": "communication"},
            {"name": "MoE Forward", "start": 1.2, "duration": 1.0, "type": "compute"},
            {"name": "AllToAll", "start": 1.5, "duration": 0.8, "type": "communication"},
            # 可以添加更多操作
        ]
        
        # 绘制时间线
        y_positions = {}
        current_y = 0
        
        for op in operations:
            if op["type"] not in y_positions:
                y_positions[op["type"]] = current_y
                current_y += 1
            
            y = y_positions[op["type"]]
            color = 'tab:blue' if op["type"] == "compute" else 'tab:orange'
            
            plt.barh(y, op["duration"], left=op["start"], height=0.5, 
                    color=color, alpha=0.7)
            
            # 添加标签
            plt.text(op["start"] + op["duration"]/2, y, op["name"], 
                    ha='center', va='center', color='black', fontsize=8)
        
        # 设置Y轴标签
        plt.yticks(list(y_positions.values()), list(y_positions.keys()))
        
        plt.xlabel('时间 (秒)')
        plt.title('通信与计算重叠时间线')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.savefig("simulation_results/figures/overlap_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n已生成可视化图表:")
        print("  - simulation_results/figures/time_distribution.png")
        print("  - simulation_results/figures/gpu_utilization.png")
        print("  - simulation_results/figures/overlap_timeline.png")
        
    except Exception as e:
        print(f"创建可视化时出错: {e}")

if __name__ == "__main__":
    run_simulation()