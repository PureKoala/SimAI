#!/usr/bin/env python3
import re
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Try to import plotly for interactive timeline charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Only matplotlib timeline will be generated.")

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

def create_timeline_visualization(results, output_dir='.'):
    """创建硬件延迟分解时间线可视化图表"""
    # Create hardware latency breakdown timeline
    create_hardware_latency_timeline(output_dir)
    
    # Create interactive plotly timeline if available
    if PLOTLY_AVAILABLE:
        create_plotly_hardware_timeline(output_dir)

def load_hardware_latency_config(config_file='hardware_latency_config.json'):
    """加载硬件延迟配置文件，如果不存在则使用默认值"""
    config_path = Path(config_file)
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading hardware config: {e}")
            return None
    return None

def save_default_hardware_config(config_file='hardware_latency_config.json'):
    """保存默认硬件延迟配置文件，供用户自定义"""
    # Define default hardware components directly to avoid circular dependency
    default_hardware_components = [
        {
            'component': 'GPU Kernel Launch',
            'latency_us': 5.2,
            'description': 'Time to launch communication kernel on source GPU',
            'category': 'Computation'
        },
        {
            'component': 'GPU Memory Copy',
            'latency_us': 12.8,
            'description': 'Copy data from GPU memory to host buffer',
            'category': 'Memory'
        },
        {
            'component': 'Host Buffer Setup',
            'latency_us': 2.1,
            'description': 'Prepare host memory buffer for transfer',
            'category': 'Memory'
        },
        {
            'component': 'PCIe Transfer',
            'latency_us': 45.6,
            'description': 'Data transfer over PCIe bus to NIC',
            'category': 'PCIe'
        },
        {
            'component': 'NIC Processing',
            'latency_us': 8.3,
            'description': 'Network interface card packet processing',
            'category': 'NIC'
        },
        {
            'component': 'Ethernet TX',
            'latency_us': 15.7,
            'description': 'Ethernet transmission to network switch',
            'category': 'Network'
        },
        {
            'component': 'Switch Latency',
            'latency_us': 3.4,
            'description': 'Network switch forwarding latency',
            'category': 'Network'
        },
        {
            'component': 'Ethernet RX',
            'latency_us': 12.1,
            'description': 'Ethernet reception at destination',
            'category': 'Network'
        },
        {
            'component': 'Dest NIC Processing',
            'latency_us': 7.9,
            'description': 'Destination NIC packet processing',
            'category': 'NIC'
        },
        {
            'component': 'Dest PCIe Transfer',
            'latency_us': 38.2,
            'description': 'PCIe transfer to destination GPU',
            'category': 'PCIe'
        },
        {
            'component': 'Dest GPU Memory Copy',
            'latency_us': 11.4,
            'description': 'Copy data to destination GPU memory',
            'category': 'Memory'
        },
        {
            'component': 'GPU Kernel Complete',
            'latency_us': 3.8,
            'description': 'Complete communication kernel on dest GPU',
            'category': 'Computation'
        }
    ]
    
    default_config = {
        "description": "GPU-to-GPU communication hardware latency breakdown configuration",
        "hardware_components": default_hardware_components,
        "notes": [
            "All latency values are in microseconds (μs)",
            "These values are examples based on H100 + NDR InfiniBand setup",
            "Modify these values based on your actual hardware measurements",
            "You can add or remove components as needed for your specific setup"
        ],
        "measurement_tips": [
            "Use NVIDIA Nsight Systems to measure GPU kernel times",
            "Use PCIe bandwidth tools to measure bus transfer latency",
            "Use network ping tools to measure network round-trip times", 
            "Consider using NCCL profiling tools for end-to-end measurements"
        ]
    }
    
    config_path = Path(config_file)
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Default hardware latency config saved to: {config_path}")
        print("You can edit this file to customize latency values for your hardware setup.")

def get_hardware_latency_breakdown():
    """获取GPU到GPU通信的硬件延迟分解数据
    
    Returns a list of hardware components with their latency contributions.
    These values can be measured or estimated based on your specific hardware setup.
    """
    # Try to load from config file first
    config = load_hardware_latency_config()
    if config and 'hardware_components' in config:
        return config['hardware_components']
    
    # Example latency breakdown for GPU-to-GPU communication
    # These values should be measured or configured based on your actual hardware
    hardware_components = [
        {
            'component': 'GPU Kernel Launch',
            'latency_us': 5.2,
            'description': 'Time to launch communication kernel on source GPU',
            'category': 'Computation'
        },
        {
            'component': 'GPU Memory Copy',
            'latency_us': 12.8,
            'description': 'Copy data from GPU memory to host buffer',
            'category': 'Memory'
        },
        {
            'component': 'Host Buffer Setup',
            'latency_us': 2.1,
            'description': 'Prepare host memory buffer for transfer',
            'category': 'Memory'
        },
        {
            'component': 'PCIe Transfer',
            'latency_us': 45.6,
            'description': 'Data transfer over PCIe bus to NIC',
            'category': 'PCIe'
        },
        {
            'component': 'NIC Processing',
            'latency_us': 8.3,
            'description': 'Network interface card packet processing',
            'category': 'NIC'
        },
        {
            'component': 'Ethernet TX',
            'latency_us': 15.7,
            'description': 'Ethernet transmission to network switch',
            'category': 'Network'
        },
        {
            'component': 'Switch Latency',
            'latency_us': 3.4,
            'description': 'Network switch forwarding latency',
            'category': 'Network'
        },
        {
            'component': 'Ethernet RX',
            'latency_us': 12.1,
            'description': 'Ethernet reception at destination',
            'category': 'Network'
        },
        {
            'component': 'Dest NIC Processing',
            'latency_us': 7.9,
            'description': 'Destination NIC packet processing',
            'category': 'NIC'
        },
        {
            'component': 'Dest PCIe Transfer',
            'latency_us': 38.2,
            'description': 'PCIe transfer to destination GPU',
            'category': 'PCIe'
        },
        {
            'component': 'Dest GPU Memory Copy',
            'latency_us': 11.4,
            'description': 'Copy data to destination GPU memory',
            'category': 'Memory'
        },
        {
            'component': 'GPU Kernel Complete',
            'latency_us': 3.8,
            'description': 'Complete communication kernel on dest GPU',
            'category': 'Computation'
        }
    ]
    
    return hardware_components

def create_hardware_latency_timeline(output_dir):
    """使用matplotlib创建硬件延迟分解时间线图表"""
    hardware_data = get_hardware_latency_breakdown()
    
    if not hardware_data:
        return
    
    # Create figure with two subplots: waterfall chart and horizontal timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Define colors for different hardware categories
    category_colors = {
        'Computation': '#FF6B6B',  # Red
        'Memory': '#4ECDC4',       # Teal
        'PCIe': '#45B7D1',         # Blue
        'NIC': '#96CEB4',          # Green
        'Network': '#FECA57'       # Yellow
    }
    
    # === Subplot 1: Waterfall Chart ===
    # Calculate cumulative latencies for waterfall effect
    cumulative_latency = 0
    positions = []
    widths = []
    colors = []
    labels = []
    
    for component in hardware_data:
        positions.append(cumulative_latency)
        widths.append(component['latency_us'])
        colors.append(category_colors.get(component['category'], '#CCCCCC'))
        labels.append(f"{component['component']}\n{component['latency_us']:.1f}μs")
        cumulative_latency += component['latency_us']
    
    # Create waterfall bars
    bars = ax1.barh(range(len(hardware_data)), widths, left=positions, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add labels to bars
    for i, (bar, label) in enumerate(zip(bars, labels)):
        width = bar.get_width()
        x_pos = bar.get_x() + width / 2
        ax1.text(x_pos, i, label, ha='center', va='center', 
                fontsize=8, fontweight='bold', wrap=True)
    
    # Customize waterfall chart
    ax1.set_xlabel('Cumulative Latency (μs)', fontsize=12)
    ax1.set_ylabel('Hardware Components', fontsize=12)
    ax1.set_title('GPU-to-GPU Communication Latency Breakdown (Waterfall View)', 
                 fontsize=14, fontweight='bold')
    ax1.set_yticks(range(len(hardware_data)))
    ax1.set_yticklabels([comp['component'] for comp in hardware_data])
    ax1.grid(axis='x', alpha=0.3)
    
    # Add total latency annotation
    total_latency = sum(comp['latency_us'] for comp in hardware_data)
    ax1.axvline(x=total_latency, color='red', linestyle='--', linewidth=2)
    ax1.text(total_latency + 5, len(hardware_data)/2, 
             f'Total: {total_latency:.1f}μs', rotation=90, 
             va='center', ha='left', fontweight='bold', color='red')
    
    # === Subplot 2: Sequential Timeline ===
    # Create timeline showing sequential execution
    timeline_start = 0
    timeline_bars = []
    
    for i, component in enumerate(hardware_data):
        duration = component['latency_us']
        color = category_colors.get(component['category'], '#CCCCCC')
        
        # Create horizontal bar showing sequential execution
        bar = ax2.barh(0, duration, left=timeline_start, 
                      color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add component label
        if duration > 5:  # Only label if bar is wide enough
            ax2.text(timeline_start + duration/2, 0, 
                    f"{component['component']}\n{duration:.1f}μs", 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        timeline_start += duration
        timeline_bars.append(bar)
    
    # Customize timeline chart
    ax2.set_xlabel('Time Progress (μs)', fontsize=12)
    ax2.set_ylabel('Timeline', fontsize=12)
    ax2.set_title('Sequential Hardware Latency Timeline', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.grid(axis='x', alpha=0.3)
    
    # Add phase separators and labels
    phase_boundaries = {}
    current_time = 0
    for component in hardware_data:
        category = component['category']
        if category not in phase_boundaries:
            phase_boundaries[category] = []
        phase_boundaries[category].append((current_time, component['latency_us']))
        current_time += component['latency_us']
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black') 
                      for color in category_colors.values()]
    legend_labels = list(category_colors.keys())
    
    fig.legend(legend_elements, legend_labels, 
              loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=len(legend_labels))
    
    plt.tight_layout()
    
    # Save the chart
    output_path = Path(output_dir) / 'hardware_latency_timeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Hardware latency timeline saved to: {output_path}")
    
    plt.close()

def create_plotly_hardware_timeline(output_dir):
    """使用Plotly创建交互式硬件延迟分解图表"""
    if not PLOTLY_AVAILABLE:
        return
        
    hardware_data = get_hardware_latency_breakdown()
    
    if not hardware_data:
        return
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(hardware_data)
    
    # Define colors for categories
    category_colors = {
        'Computation': '#FF6B6B',
        'Memory': '#4ECDC4',
        'PCIe': '#45B7D1',
        'NIC': '#96CEB4',
        'Network': '#FECA57'
    }
    
    # Create waterfall chart using Plotly
    fig = go.Figure()
    
    # Calculate cumulative values for waterfall effect
    cumulative = 0
    for i, component in enumerate(hardware_data):
        fig.add_trace(go.Waterfall(
            name=component['category'],
            orientation="h",
            measure=["relative"] * len(hardware_data),
            y=[component['component']],
            x=[component['latency_us']],
            text=[f"{component['latency_us']:.1f}μs"],
            textposition="inside",
            connector={"visible": True, "line": {"width": 2, "color": "black"}},
            increasing={"marker": {"color": category_colors.get(component['category'], '#CCCCCC')}},
            hovertemplate=f"<b>{component['component']}</b><br>" +
                         f"Latency: {component['latency_us']:.1f}μs<br>" +
                         f"Category: {component['category']}<br>" +
                         f"Description: {component['description']}<extra></extra>"
        ))
    
    # Create a cleaner version using horizontal bar chart
    fig2 = go.Figure()
    
    # Add bars for each component
    cumulative_time = 0
    for component in hardware_data:
        fig2.add_trace(go.Bar(
            name=component['component'],
            x=[component['latency_us']],
            y=[component['component']],
            orientation='h',
            marker=dict(
                color=category_colors.get(component['category'], '#CCCCCC'),
                line=dict(color='black', width=1)
            ),
            text=f"{component['latency_us']:.1f}μs",
            textposition='inside',
            hovertemplate=f"<b>{component['component']}</b><br>" +
                         f"Latency: {component['latency_us']:.1f}μs<br>" +
                         f"Category: {component['category']}<br>" +
                         f"Description: {component['description']}<extra></extra>"
        ))
    
    # Update layout for the bar chart
    total_latency = sum(comp['latency_us'] for comp in hardware_data)
    
    fig2.update_layout(
        title={
            'text': 'GPU-to-GPU Communication Hardware Latency Breakdown',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Latency (microseconds)',
        yaxis_title='Hardware Components',
        showlegend=False,
        height=600,
        width=1200,
        plot_bgcolor='white',
        annotations=[
            dict(
                x=total_latency * 0.8,
                y=len(hardware_data) * 0.1,
                text=f"Total End-to-End Latency: {total_latency:.1f}μs",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                font=dict(size=14, color="red"),
                bgcolor="white",
                bordercolor="red",
                borderwidth=1
            )
        ]
    )
    
    # Add grid and styling
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig2.update_yaxes(showgrid=False)
    
    # Save interactive chart
    output_path = Path(output_dir) / 'hardware_latency_timeline_interactive.html'
    fig2.write_html(str(output_path))
    print(f"Interactive hardware latency timeline saved to: {output_path}")
    
    # Also save as static image
    static_path = Path(output_dir) / 'hardware_latency_timeline_plotly.png'
    try:
        fig2.write_image(str(static_path), width=1200, height=600)
        print(f"Static plotly chart saved to: {static_path}")
    except Exception as e:
        print(f"Could not save static plotly image: {e}")
        print("Note: Install kaleido package for static image export: pip install kaleido")

def create_performance_comparison_chart(results, output_dir='.'):
    """创建性能对比图表"""
    if not results:
        return
    
    # Extract performance metrics
    metrics = {}
    
    if 'analytical' in results and results['analytical']:
        analytical = results['analytical']
        metrics['Analytical'] = {
            'Total Time': analytical.get('total_time', 0),
            'Computation Time': analytical.get('computation_time', 0),
            'Communication Time': analytical.get('communication_time', 0)
        }
    
    if 'ns3' in results and results['ns3']:
        ns3 = results['ns3']
        metrics['NS3'] = {
            'Avg Latency': ns3.get('avg_latency', 0),
            'Max Latency': ns3.get('max_latency', 0),
            'Total Bytes': ns3.get('total_bytes', 0) / 1e6  # Convert to MB
        }
    
    if not metrics:
        print("No performance metrics available for comparison")
        return
    
    # Create comparison bar chart
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    for i, (sim_type, sim_metrics) in enumerate(metrics.items()):
        ax = axes[i]
        
        metric_names = list(sim_metrics.keys())
        metric_values = list(sim_metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=colors[:len(metric_names)], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom')
        
        ax.set_title(f'{sim_type} Simulation Metrics', fontweight='bold')
        ax.set_ylabel('Value')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir) / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance comparison chart saved to: {output_path}")
    
    plt.close()

def main():
    results_dir = Path('.')
    
    # 生成默认硬件延迟配置文件（如果不存在）
    save_default_hardware_config()
    
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
    
    # 创建硬件延迟分解可视化
    print("正在生成硬件延迟分解可视化图表...")
    try:
        create_timeline_visualization(all_results, results_dir)
        create_performance_comparison_chart(all_results, results_dir)
        print("硬件延迟分解图表生成成功!")
    except Exception as e:
        print(f"生成图表时出错: {e}")
        
    # 打印摘要
    print("\n仿真结果摘要:")
    print("=" * 40)
    
    if analytical_results:
        print("Analytical 仿真:")
        for key, value in analytical_results.items():
            print(f"  {key}: {value}")
            
    if ns3_results:
        print("NS3 仿真:")
        for key, value in ns3_results.items():
            print(f"  {key}: {value}")
    
    # 打印硬件延迟分解摘要
    hardware_data = get_hardware_latency_breakdown()
    total_latency = sum(comp['latency_us'] for comp in hardware_data)
    print(f"\n硬件延迟分解摘要:")
    print("=" * 40)
    print(f"总端到端延迟: {total_latency:.1f} μs")
    
    # 按类别分组显示
    from collections import defaultdict
    category_totals = defaultdict(float)
    for comp in hardware_data:
        category_totals[comp['category']] += comp['latency_us']
    
    print("按硬件类别分组:")
    for category, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (total / total_latency) * 100
        print(f"  {category}: {total:.1f} μs ({percentage:.1f}%)")
    
    # 打印生成的文件列表
    print(f"\n生成的文件:")
    print("=" * 40)
    generated_files = [
        'simulation_results.json',
        'hardware_latency_timeline.png',
        'performance_comparison.png',
        'hardware_latency_config.json'
    ]
    
    if PLOTLY_AVAILABLE:
        generated_files.extend([
            'hardware_latency_timeline_interactive.html',
            'hardware_latency_timeline_plotly.png'
        ])
    
    for file in generated_files:
        file_path = results_dir / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (未生成)")

if __name__ == "__main__":
    main()
