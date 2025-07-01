#!/usr/bin/env python3
"""
SimAI GPU网络延迟仿真测试工具
SimAI GPU Network Latency Simulation Tool

基于SimAI框架进行纳秒级精度的GPU网络通信延迟仿真分析
使用astra-sim + ns-3进行精确的网络延迟建模

作者：SimAI团队
版本：1.0
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import subprocess
import configparser
from datetime import datetime
import argparse

# Configure matplotlib for English output only
def setup_matplotlib_fonts():
    """Setup matplotlib fonts for English output"""
    # Use standard English fonts
    font_candidates = [
        'DejaVu Sans',      # Linux default font
        'Liberation Sans',   # Open source font
        'Arial',            # Standard font
        'Helvetica',        # Mac default
        'sans-serif'        # Fallback
    ]
    
    # Set font parameters
    matplotlib.rcParams['font.sans-serif'] = font_candidates
    matplotlib.rcParams['axes.unicode_minus'] = False  # Properly display minus signs
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    print("✅ Matplotlib configured for English output")

# Initialize font settings
setup_matplotlib_fonts()

# 添加SimAI路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'aicb'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'astra-sim-alibabacloud'))

@dataclass
class LatencyComponents:
    """延迟组件数据结构"""
    gpu_compute_ns: float = 0.0          # GPU计算延迟 (纳秒)
    memory_access_ns: float = 0.0        # 内存访问延迟 (纳秒)
    pcie_transfer_ns: float = 0.0        # PCIe传输延迟 (纳秒)
    network_propagation_ns: float = 0.0  # 网络传播延迟 (纳秒)
    switch_processing_ns: float = 0.0    # 交换机处理延迟 (纳秒)
    protocol_overhead_ns: float = 0.0    # 协议开销 (纳秒)
    serialization_ns: float = 0.0        # 序列化延迟 (纳秒)
    
    @property
    def total_ns(self) -> float:
        """计算总延迟"""
        return (self.gpu_compute_ns + self.memory_access_ns + 
                self.pcie_transfer_ns + self.network_propagation_ns + 
                self.switch_processing_ns + self.protocol_overhead_ns + 
                self.serialization_ns)

class SimAILatencySimulator:
    """SimAI延迟仿真器"""
    
    def __init__(self, config_file: str = "configs/simai_latency_config.ini"):
        self.config_file = config_file
        self.config = self._load_config()
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 仿真参数
        self.network_backend = self.config.get('simulation', 'network_backend', fallback='ns3')
        self.simulator_binary = self._find_simulator_binary()
        
        # Latency model parameters (nanosecond level)
        self.base_latencies = {
            'gpu_compute': float(self.config.get('latency_model', 'gpu_compute_ns', fallback='100')),
            'memory_access': float(self.config.get('latency_model', 'memory_access_ns', fallback='300')),
            'pcie_setup_latency': float(self.config.get('latency_model', 'pcie_setup_latency_ns', fallback='3000')),
            'pcie_bandwidth_gbps': float(self.config.get('latency_model', 'pcie_effective_bandwidth_gbps', fallback='45')),
            'network_base': float(self.config.get('latency_model', 'network_base_ns', fallback='1000')),
            'switch_processing': float(self.config.get('latency_model', 'switch_processing_ns', fallback='500')),
            'protocol_overhead': float(self.config.get('latency_model', 'protocol_overhead_ns', fallback='200')),
        }
        
    def _load_config(self) -> configparser.ConfigParser:
        """加载配置文件"""
        config = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            config.read(self.config_file)
        return config
        
    def _find_simulator_binary(self) -> str:
        """查找SimAI仿真器二进制文件"""
        possible_paths = [
            "../bin/SimAI_simulator",
            "../astra-sim-alibabacloud/build/astra_ns3/build/scratch/SimAI",
            "../astra-sim-alibabacloud/build/simai_phy/build/scratch/SimAI_phy",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
                
        # If binary file not found, use simulation mode
        print("Warning: SimAI simulator binary not found, using mathematical model simulation")
        return "simulation_mode"

    def generate_workload_file(self, comm_type: str, data_size_bytes: int, 
                             num_gpus: int = 2) -> str:
        """生成工作负载文件"""
        workload_dir = Path("workloads")
        workload_dir.mkdir(exist_ok=True)
        
        workload_file = workload_dir / f"latency_test_{comm_type}_{data_size_bytes}B_{num_gpus}gpu.txt"
        
        # 生成工作负载描述
        if comm_type == "all_reduce":
            workload_content = self._generate_allreduce_workload(data_size_bytes, num_gpus)
        elif comm_type == "p2p":
            workload_content = self._generate_p2p_workload(data_size_bytes, num_gpus)
        elif comm_type == "broadcast":
            workload_content = self._generate_broadcast_workload(data_size_bytes, num_gpus)
        else:
            raise ValueError(f"Unsupported communication type: {comm_type}")
            
        with open(workload_file, 'w') as f:
            f.write(workload_content)
            
        return str(workload_file)

    def _generate_allreduce_workload(self, data_size: int, num_gpus: int) -> str:
        """生成All-Reduce工作负载"""
        return f"""# All-Reduce Latency Test Workload
# Data size: {data_size} bytes, GPUs: {num_gpus}
0: COMP(1000,FWD,{data_size})
1: COMP(1000,FWD,{data_size}) DEP(0)
2: ALLREDUCE({data_size},ALL) DEP(1)
3: COMP(500,BWD,{data_size}) DEP(2)
"""

    def _generate_p2p_workload(self, data_size: int, num_gpus: int) -> str:
        """生成点对点通信工作负载"""
        return f"""# Point-to-Point Latency Test Workload  
# Data size: {data_size} bytes, GPUs: {num_gpus}
0: COMP(500,FWD,{data_size})
1: SEND({data_size},0,1) DEP(0)
2: RECV({data_size},1,0) DEP(1)
3: COMP(500,BWD,{data_size}) DEP(2)
"""

    def _generate_broadcast_workload(self, data_size: int, num_gpus: int) -> str:
        """生成广播通信工作负载"""
        return f"""# Broadcast Latency Test Workload
# Data size: {data_size} bytes, GPUs: {num_gpus}
0: COMP(500,FWD,{data_size})
1: BROADCAST({data_size},0,ALL) DEP(0)
2: COMP(500,BWD,{data_size}) DEP(1)
"""

    def generate_network_config(self, topology: str = "fat_tree", 
                               link_bw_gbps: int = 100,
                               link_latency_ns: int = 1000) -> str:
        """生成网络配置文件"""
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / f"network_{topology}_{link_bw_gbps}gbps.json"
        
        if topology == "fat_tree":
            network_config = {
                "topology-name": "Fat-Tree",
                "topology-type": "fat-tree",
                "dimensions": [4, 4, 4],  # K=4 fat-tree
                "link-bandwidth": f"{link_bw_gbps}GB/s",
                "link-latency": f"{link_latency_ns}ns",
                "router-latency": "100ns",
                "hca-latency": "200ns",
                "packet-size": "1KB"
            }
        elif topology == "dragonfly":
            network_config = {
                "topology-name": "Dragonfly",
                "topology-type": "dragonfly",
                "dimensions": [2, 2, 2, 2],
                "link-bandwidth": f"{link_bw_gbps}GB/s", 
                "link-latency": f"{link_latency_ns}ns",
                "router-latency": "150ns",
                "hca-latency": "200ns"
            }
        else:
            raise ValueError(f"Unsupported topology type: {topology}")
            
        with open(config_file, 'w') as f:
            json.dump(network_config, f, indent=2)
            
        return str(config_file)

    def run_simulation(self, workload_file: str, network_config: str, 
                      system_config: str = None) -> Dict:
        """运行仿真"""
        if self.simulator_binary == "simulation_mode":
            return self._run_mathematical_simulation(workload_file, network_config)
        else:
            return self._run_astra_simulation(workload_file, network_config, system_config)

    def _run_mathematical_simulation(self, workload_file: str, network_config: str) -> Dict:
        """使用数学模型进行仿真（当没有实际仿真器时）"""
        print(f"Using mathematical model simulation: {workload_file}")
        
        # 解析工作负载文件获取数据大小
        with open(workload_file, 'r') as f:
            content = f.read()
        
        # 简单解析获取数据大小 (这里使用正则表达式会更好)
        import re
        data_sizes = re.findall(r'(\d+)', os.path.basename(workload_file))
        data_size_bytes = int(data_sizes[0]) if data_sizes else 1024
        
        # 加载网络配置
        with open(network_config, 'r') as f:
            net_config = json.load(f)
            
        link_bw_str = net_config.get('link-bandwidth', '100GB/s')
        link_bw_gbps = float(re.findall(r'(\d+)', link_bw_str)[0])
        
        link_latency_str = net_config.get('link-latency', '1000ns')
        link_latency_ns = float(re.findall(r'(\d+)', link_latency_str)[0])
        
        # 计算延迟组件
        latency = LatencyComponents()
        
        # GPU计算延迟
        latency.gpu_compute_ns = self.base_latencies['gpu_compute']
        
        # Memory access latency (related to data size)
        latency.memory_access_ns = self.base_latencies['memory_access'] * (data_size_bytes / 1024)
        
        # PCIe transfer latency (setup + transfer time)
        data_size_gb = data_size_bytes / (1024**3)
        pcie_setup_latency = self.base_latencies['pcie_setup_latency']
        pcie_transfer_time = (data_size_gb * 1e9) / (self.base_latencies['pcie_bandwidth_gbps'] * 1e9) * 1e9  # Convert to ns
        latency.pcie_transfer_ns = pcie_setup_latency + pcie_transfer_time
        
        # 网络传播延迟
        latency.network_propagation_ns = link_latency_ns
        
        # 交换机处理延迟
        latency.switch_processing_ns = self.base_latencies['switch_processing']
        
        # 协议开销
        latency.protocol_overhead_ns = self.base_latencies['protocol_overhead']
        
        # 序列化延迟 (与数据大小相关)
        latency.serialization_ns = 8000 * data_size_gb / link_bw_gbps  # 基于带宽的传输时间
        
        # 添加一些随机性以模拟真实变化
        noise_factor = 0.05  # 5%的随机噪声
        for attr in ['gpu_compute_ns', 'memory_access_ns', 'pcie_transfer_ns', 
                     'network_propagation_ns', 'switch_processing_ns', 'protocol_overhead_ns']:
            current_value = getattr(latency, attr)
            noise = np.random.normal(0, current_value * noise_factor)
            setattr(latency, attr, max(0, current_value + noise))
        
        return {
            'latency_components': latency,
            'total_latency_ns': latency.total_ns,
            'data_size_bytes': data_size_bytes,
            'network_config': net_config,
            'simulation_type': 'mathematical_model'
        }

    def _run_astra_simulation(self, workload_file: str, network_config: str, 
                             system_config: str = None) -> Dict:
        """运行Astra-sim仿真"""
        print(f"Running Astra-sim simulation: {self.simulator_binary}")
        
        # 构建命令行参数
        cmd = [
            self.simulator_binary,
            "--workload-configuration", workload_file,
            "--network-configuration", network_config,
            "--system-configuration", system_config or "configs/default_system.json",
            "--compute-scale", "1",
            "--comm-scale", "1"
        ]
        
        try:
            # 运行仿真
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Simulation failed: {result.stderr}")
                return self._run_mathematical_simulation(workload_file, network_config)
            
            # Parse simulation output
            return self._parse_astra_output(result.stdout, workload_file, network_config)
            
        except subprocess.TimeoutExpired:
            print("Simulation timeout, using mathematical model")
            return self._run_mathematical_simulation(workload_file, network_config)
        except Exception as e:
            print(f"Simulation error: {e}, using mathematical model")
            return self._run_mathematical_simulation(workload_file, network_config)

    def _parse_astra_output(self, output: str, workload_file: str, network_config: str) -> Dict:
        """解析Astra-sim输出"""
        # 这里需要根据实际的Astra-sim输出格式进行解析
        # 目前使用模拟解析
        lines = output.split('\n')
        total_latency_ns = 0
        
        for line in lines:
            if 'Total Time' in line or 'Execution Time' in line:
                # 提取延迟数值 (需要根据实际输出格式调整)
                import re
                numbers = re.findall(r'(\d+\.?\d*)', line)
                if numbers:
                    total_latency_ns = float(numbers[0]) * 1e9  # 假设输出是秒，转换为纳秒
                break
        
        if total_latency_ns == 0:
            # If parsing fails, use mathematical model
            return self._run_mathematical_simulation(workload_file, network_config)
        
        # 创建延迟组件分解 (基于经验比例)
        latency = LatencyComponents()
        latency.gpu_compute_ns = total_latency_ns * 0.1
        latency.memory_access_ns = total_latency_ns * 0.15
        latency.pcie_transfer_ns = total_latency_ns * 0.2
        latency.network_propagation_ns = total_latency_ns * 0.25
        latency.switch_processing_ns = total_latency_ns * 0.1
        latency.protocol_overhead_ns = total_latency_ns * 0.1
        latency.serialization_ns = total_latency_ns * 0.1
        
        return {
            'latency_components': latency,
            'total_latency_ns': total_latency_ns,
            'simulation_output': output,
            'simulation_type': 'astra_sim'
        }

    def run_latency_sweep(self, comm_types: List[str], data_sizes: List[int],
                         topologies: List[str], num_iterations: int = 5) -> pd.DataFrame:
        """运行延迟扫描测试"""
        results = []
        
        total_tests = len(comm_types) * len(data_sizes) * len(topologies) * num_iterations
        current_test = 0
        
        for comm_type in comm_types:
            for data_size in data_sizes:
                for topology in topologies:
                    for iteration in range(num_iterations):
                        current_test += 1
                        print(f"Progress: {current_test}/{total_tests} - "
                              f"Communication type: {comm_type}, Data size: {data_size}B, Topology: {topology}")
                        
                        # 生成配置文件
                        workload_file = self.generate_workload_file(comm_type, data_size)
                        network_config = self.generate_network_config(topology)
                        
                        # 运行仿真
                        sim_result = self.run_simulation(workload_file, network_config)
                        
                        # 收集结果
                        latency_comp = sim_result['latency_components']
                        result_row = {
                            'comm_type': comm_type,
                            'data_size_bytes': data_size,
                            'data_size_kb': data_size / 1024,
                            'topology': topology,
                            'iteration': iteration,
                            'total_latency_ns': latency_comp.total_ns,
                            'total_latency_us': latency_comp.total_ns / 1000,
                            'gpu_compute_ns': latency_comp.gpu_compute_ns,
                            'memory_access_ns': latency_comp.memory_access_ns,
                            'pcie_transfer_ns': latency_comp.pcie_transfer_ns,
                            'network_propagation_ns': latency_comp.network_propagation_ns,
                            'switch_processing_ns': latency_comp.switch_processing_ns,
                            'protocol_overhead_ns': latency_comp.protocol_overhead_ns,
                            'serialization_ns': latency_comp.serialization_ns,
                            'simulation_type': sim_result.get('simulation_type', 'unknown'),
                            'timestamp': datetime.now().isoformat()
                        }
                        results.append(result_row)
        
        return pd.DataFrame(results)

    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """分析仿真结果"""
        analysis = {}
        
        # 基本统计
        analysis['summary'] = {
            'total_tests': len(df),
            'comm_types': df['comm_type'].unique().tolist(),
            'data_sizes_kb': sorted(df['data_size_kb'].unique().tolist()),
            'topologies': df['topology'].unique().tolist(),
            'avg_total_latency_us': df['total_latency_us'].mean(),
            'min_total_latency_us': df['total_latency_us'].min(),
            'max_total_latency_us': df['total_latency_us'].max()
        }
        
        # 按通信类型分组分析
        analysis['by_comm_type'] = {}
        for comm_type in df['comm_type'].unique():
            comm_df = df[df['comm_type'] == comm_type]
            analysis['by_comm_type'][comm_type] = {
                'avg_latency_us': comm_df['total_latency_us'].mean(),
                'std_latency_us': comm_df['total_latency_us'].std(),
                'min_latency_us': comm_df['total_latency_us'].min(),
                'max_latency_us': comm_df['total_latency_us'].max()
            }
        
        # 延迟组件分析
        latency_components = [
            'gpu_compute_ns', 'memory_access_ns', 'pcie_transfer_ns',
            'network_propagation_ns', 'switch_processing_ns', 
            'protocol_overhead_ns', 'serialization_ns'
        ]
        
        analysis['component_breakdown'] = {}
        for comp in latency_components:
            analysis['component_breakdown'][comp] = {
                'avg_ns': df[comp].mean(),
                'percentage': (df[comp].mean() / df['total_latency_ns'].mean()) * 100
            }
        
        # 数据大小与延迟的关系
        analysis['latency_vs_size'] = {}
        for data_size in sorted(df['data_size_kb'].unique()):
            size_df = df[df['data_size_kb'] == data_size]
            analysis['latency_vs_size'][f'{data_size}KB'] = {
                'avg_latency_us': size_df['total_latency_us'].mean(),
                'throughput_gb_per_s': (data_size / 1024 / 1024) / (size_df['total_latency_us'].mean() / 1e6)
            }
        
        return analysis

    def generate_plots(self, df: pd.DataFrame, output_dir: str = "results/plots"):
        """Generate analysis plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Define English labels
        labels = {
            'data_size': 'Data Size (KB)',
            'total_latency': 'Total Latency (μs)',
            'latency_vs_size': 'GPU Network Communication Latency vs Data Size',
            'latency_components': 'Average Latency Components Breakdown',
            'topology_comparison': 'Latency Comparison Across Network Topologies',
            'topology': 'Network Topology',
            'components_stacked': 'Latency Components vs Data Size',
            'latency_us': 'Latency (μs)',
            'timeline': 'Communication Timeline Breakdown',
            'time_us': 'Time (μs)',
            'operation': 'Operation'
        }
        
        # Component labels
        component_labels = ['GPU Compute', 'Memory Access', 'PCIe Transfer', 'Network Propagation', 
                           'Switch Processing', 'Protocol Overhead', 'Serialization']
        
        # 1. Latency vs data size
        plt.figure(figsize=(12, 8))
        for comm_type in df['comm_type'].unique():
            comm_df = df[df['comm_type'] == comm_type]
            grouped = comm_df.groupby('data_size_kb')['total_latency_us'].agg(['mean', 'std']).reset_index()
            plt.errorbar(grouped['data_size_kb'], grouped['mean'], yerr=grouped['std'], 
                        marker='o', label=comm_type, capsize=5)
        
        plt.xlabel(labels['data_size'])
        plt.ylabel(labels['total_latency'])
        plt.title(labels['latency_vs_size'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f"{output_dir}/latency_vs_data_size.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Latency components pie chart
        latency_components = [
            'gpu_compute_ns', 'memory_access_ns', 'pcie_transfer_ns',
            'network_propagation_ns', 'switch_processing_ns', 
            'protocol_overhead_ns', 'serialization_ns'
        ]
        
        avg_components = [df[comp].mean() for comp in latency_components]
        
        plt.figure(figsize=(10, 8))
        plt.pie(avg_components, labels=component_labels, autopct='%1.1f%%', startangle=90)
        plt.title(labels['latency_components'])
        plt.savefig(f"{output_dir}/latency_components_pie.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Different topology latency comparison
        if len(df['topology'].unique()) > 1:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x='topology', y='total_latency_us', hue='comm_type')
            plt.xlabel(labels['topology'])
            plt.ylabel(labels['total_latency'])
            plt.title(labels['topology_comparison'])
            plt.yscale('log')
            plt.savefig(f"{output_dir}/topology_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Latency components stacked chart
        plt.figure(figsize=(14, 8))
        
        # Calculate average latency components for each data size
        size_groups = df.groupby('data_size_kb')[latency_components].mean()
        
        # Create stacked bar chart
        bottom = np.zeros(len(size_groups))
        colors = plt.cm.Set3(np.linspace(0, 1, len(latency_components)))
        
        for i, (comp, label) in enumerate(zip(latency_components, component_labels)):
            plt.bar(size_groups.index, size_groups[comp] / 1000,  # Convert to microseconds
                   bottom=bottom, label=label, color=colors[i], alpha=0.8)
            bottom += size_groups[comp] / 1000
        
        plt.xlabel(labels['data_size'])
        plt.ylabel(labels['latency_us'])
        plt.title(labels['components_stacked'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/latency_components_stacked.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. NEW: Timeline breakdown showing communication phases
        self.generate_timeline_plot(df, output_dir, labels, component_labels, latency_components)
        
        print(f"Plots saved to: {output_dir}")

    def generate_timeline_plot(self, df: pd.DataFrame, output_dir: str, labels: dict, 
                              component_labels: list, latency_components: list):
        """Generate timeline plot showing when different operations occur during communication"""
        
        # Create timeline plot for average case
        avg_latencies = [df[comp].mean() / 1000 for comp in latency_components]  # Convert to microseconds
        
        # Calculate cumulative times (start times for each phase)
        start_times = [0]
        for i in range(len(avg_latencies) - 1):
            start_times.append(start_times[-1] + avg_latencies[i])
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Gantt-style timeline
        colors = plt.cm.Set3(np.linspace(0, 1, len(component_labels)))
        y_pos = 0
        
        for i, (label, duration, start_time, color) in enumerate(zip(component_labels, avg_latencies, start_times, colors)):
            ax1.barh(y_pos, duration, left=start_time, height=0.6, 
                    color=color, alpha=0.8, label=label, edgecolor='black', linewidth=0.5)
            
            # Add text annotations for duration
            if duration > 0.1:  # Only show text for significant durations
                ax1.text(start_time + duration/2, y_pos, f'{duration:.1f}μs', 
                        ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel(labels['time_us'])
        ax1.set_ylabel(labels['operation'])
        ax1.set_title(f"{labels['timeline']} - Sequential Operations")
        ax1.set_yticks([y_pos])
        ax1.set_yticklabels(['Communication Flow'])
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Cumulative time progression
        cumulative_times = np.cumsum([0] + avg_latencies)
        
        ax2.step(cumulative_times, range(len(cumulative_times)), where='post', linewidth=2, marker='o')
        
        # Add phase labels
        for i, (label, cum_time) in enumerate(zip(['Start'] + component_labels, cumulative_times)):
            ax2.annotate(label, (cum_time, i), xytext=(5, 0), 
                        textcoords='offset points', va='center', fontsize=9, rotation=45)
        
        ax2.set_xlabel(labels['time_us'])
        ax2.set_ylabel('Completed Operations')
        ax2.set_title('Cumulative Timeline Progress')
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks(range(len(cumulative_times)))
        ax2.set_yticklabels([f'Phase {i}' for i in range(len(cumulative_times))])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/communication_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional plot: Timeline comparison for different communication types
        if len(df['comm_type'].unique()) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            y_positions = {}
            y_current = 0
            
            for comm_type in df['comm_type'].unique():
                comm_df = df[df['comm_type'] == comm_type]
                comm_avg_latencies = [comm_df[comp].mean() / 1000 for comp in latency_components]
                
                # Calculate start times for this communication type
                comm_start_times = [0]
                for i in range(len(comm_avg_latencies) - 1):
                    comm_start_times.append(comm_start_times[-1] + comm_avg_latencies[i])
                
                y_positions[comm_type] = y_current
                
                # Plot each component as a bar
                for i, (label, duration, start_time, color) in enumerate(zip(component_labels, comm_avg_latencies, comm_start_times, colors)):
                    ax.barh(y_current, duration, left=start_time, height=0.4, 
                           color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    # Add duration text for significant components
                    if duration > max(comm_avg_latencies) * 0.05:  # Show text if > 5% of total
                        ax.text(start_time + duration/2, y_current, f'{duration:.1f}', 
                               ha='center', va='center', fontsize=8)
                
                # Add communication type label
                total_time = sum(comm_avg_latencies)
                ax.text(-total_time * 0.1, y_current, comm_type, 
                       ha='right', va='center', fontweight='bold', fontsize=10)
                
                y_current += 1
            
            ax.set_xlabel(labels['time_us'])
            ax.set_ylabel('Communication Type')
            ax.set_title('Timeline Comparison by Communication Type')
            ax.set_yticks(list(y_positions.values()))
            ax.set_yticklabels(list(y_positions.keys()))
            ax.grid(True, alpha=0.3, axis='x')
            
            # Create legend
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=label) 
                             for label, color in zip(component_labels, colors)]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/timeline_comparison_by_type.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Timeline plots generated successfully")

    def save_results(self, df: pd.DataFrame, analysis: Dict, 
                    output_file: str = None) -> str:
        """保存结果"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/simai_latency_results_{timestamp}.json"
        
        # 准备保存的数据
        save_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(df),
                'config_file': self.config_file,
                'network_backend': self.network_backend,
                'simulator_binary': self.simulator_binary
            },
            'analysis': analysis,
            'raw_data': df.to_dict('records')
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # 同时保存CSV格式
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to: {output_file}")
        print(f"CSV data saved to: {csv_file}")
        
        return output_file

def create_default_config():
    """Create default configuration file"""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "simai_latency_config.ini"
    
    config_content = """# SimAI GPU Network Latency Simulation Configuration

[simulation]
# Network backend: ns3, analytical, physical
network_backend = ns3
# Simulator path
simulator_binary = auto
# Enable verbose logging
verbose_logging = true

[latency_model]
# Base latency parameters (nanoseconds)
gpu_compute_ns = 100
memory_access_ns = 300
pcie_gen4_per_gb_ns = 8000
network_base_ns = 1000
switch_processing_ns = 500
protocol_overhead_ns = 200

[test_parameters]
# Default test parameters
comm_types = ["all_reduce", "p2p", "broadcast"]
data_sizes_kb = [1, 4, 16, 64, 256, 1024, 4096]
topologies = ["fat_tree", "dragonfly"]
num_iterations = 5

[network_configs]
# Network configuration
default_bandwidth_gbps = 100
default_latency_ns = 1000
switch_latency_ns = 100
hca_latency_ns = 200

[output]
# Output configuration
save_plots = true
save_csv = true
save_json = true
results_dir = results
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Default configuration file created: {config_file}")
    return str(config_file)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SimAI GPU Network Latency Simulation Tool")
    parser.add_argument("--config", default="configs/simai_latency_config.ini",
                       help="Configuration file path")
    parser.add_argument("--comm-types", nargs='+', 
                       default=["all_reduce", "p2p", "broadcast"],
                       help="Communication types")
    parser.add_argument("--data-sizes", nargs='+', type=int,
                       default=[1024, 4096, 16384, 65536, 262144, 1048576],
                       help="Data sizes in bytes")
    parser.add_argument("--topologies", nargs='+',
                       default=["fat_tree"],
                       help="Network topologies")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of test iterations per configuration")
    parser.add_argument("--create-config", action="store_true",
                       help="Create default configuration file")
    parser.add_argument("--plot-only", 
                       help="Generate plots only from specified results file")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config()
        return
    
    if args.plot_only:
        # Plot-only mode
        if not os.path.exists(args.plot_only):
            print(f"Results file not found: {args.plot_only}")
            return
            
        with open(args.plot_only, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['raw_data'])
        simulator = SimAILatencySimulator(args.config)
        simulator.generate_plots(df)
        return
    
    # Check configuration file
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Use --create-config to create default configuration file")
        return
    
    # Create simulator
    simulator = SimAILatencySimulator(args.config)
    
    print("Starting SimAI GPU Network Latency Simulation...")
    print(f"Configuration file: {args.config}")
    print(f"Simulator: {simulator.simulator_binary}")
    print(f"Communication types: {args.comm_types}")
    print(f"Data sizes: {args.data_sizes} bytes")
    print(f"Network topologies: {args.topologies}")
    print(f"Iterations: {args.iterations}")
    
    # Run latency sweep
    results_df = simulator.run_latency_sweep(
        comm_types=args.comm_types,
        data_sizes=args.data_sizes,
        topologies=args.topologies,
        num_iterations=args.iterations
    )
    
    # Analyze results
    analysis = simulator.analyze_results(results_df)
    
    # Generate plots
    simulator.generate_plots(results_df)
    
    # Save results
    output_file = simulator.save_results(results_df, analysis)
    
    # Print summary
    print("\n" + "="*60)
    print("SimAI GPU Network Latency Simulation Complete")
    print("="*60)
    print(f"Total tests: {analysis['summary']['total_tests']}")
    print(f"Average total latency: {analysis['summary']['avg_total_latency_us']:.2f} μs")
    print(f"Minimum latency: {analysis['summary']['min_total_latency_us']:.2f} μs")
    print(f"Maximum latency: {analysis['summary']['max_total_latency_us']:.2f} μs")
    
    print("\nLatency component breakdown:")
    for comp, data in analysis['component_breakdown'].items():
        comp_name = comp.replace('_ns', '').replace('_', ' ').title()
        print(f"  {comp_name}: {data['avg_ns']:.1f} ns ({data['percentage']:.1f}%)")
    
    print(f"\nDetailed results: {output_file}")

if __name__ == "__main__":
    main()
