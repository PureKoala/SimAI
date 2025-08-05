#!/usr/bin/env python3
"""
MoE重叠效率分析脚本
分析不同重叠策略的性能差异
"""

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_simai_log(log_file):
    """解析SimAI日志文件，提取性能指标"""
    metrics = {}
    
    if not os.path.exists(log_file):
        return metrics
        
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取关键性能指标
    patterns = {
        'total_time': r'Total time: ([\d.]+)',
        'compute_time': r'Compute time: ([\d.]+)',
        'communication_time': r'Communication time: ([\d.]+)',
        'overlap_efficiency': r'Overlap efficiency: ([\d.]+)%',
        'throughput': r'Throughput: ([\d.]+)',
        'expert_utilization': r'Expert utilization: ([\d.]+)%',
        'network_utilization': r'Network utilization: ([\d.]+)%'
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[metric] = float(match.group(1))
    
    return metrics

def analyze_overlap_strategies(results_dir):
    """分析不同重叠策略的效果"""
    results = []
    
    # 遍历所有结果文件
    for log_file in Path(results_dir).glob("moe_overlap_*_analytical.log"):
        filename = log_file.stem
        parts = filename.split('_')
        
        if len(parts) >= 5:
            strategy = parts[2]
            comp_gran = parts[3] 
            comm_gran = parts[4]
            
            metrics = parse_simai_log(str(log_file))
            metrics.update({
                'strategy': strategy,
                'compute_granularity': comp_gran,
                'communication_granularity': comm_gran,
                'config': f"{strategy}-{comp_gran}-{comm_gran}"
            })
            
            results.append(metrics)
    
    return pd.DataFrame(results)

def generate_overlap_analysis_report(df, output_dir):
    """生成重叠分析报告"""
    
    if df.empty:
        print("警告: 没有找到有效的分析结果")
        return
    
    # 创建分析图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MoE框架通信计算重叠分析报告', fontsize=16)
    
    # 1. 重叠策略性能对比
    if 'total_time' in df.columns:
        strategy_perf = df.groupby('strategy')['total_time'].mean()
        axes[0,0].bar(strategy_perf.index, strategy_perf.values)
        axes[0,0].set_title('不同重叠策略的总执行时间')
        axes[0,0].set_ylabel('时间 (ms)')
        
    # 2. 重叠效率对比
    if 'overlap_efficiency' in df.columns:
        overlap_eff = df.groupby('strategy')['overlap_efficiency'].mean()
        axes[0,1].bar(overlap_eff.index, overlap_eff.values)
        axes[0,1].set_title('重叠效率对比')
        axes[0,1].set_ylabel('重叠效率 (%)')
    
    # 3. 计算通信时间分解
    if 'compute_time' in df.columns and 'communication_time' in df.columns:
        strategies = df['strategy'].unique()
        comp_times = [df[df['strategy']==s]['compute_time'].mean() for s in strategies]
        comm_times = [df[df['strategy']==s]['communication_time'].mean() for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        axes[1,0].bar(x - width/2, comp_times, width, label='计算时间')
        axes[1,0].bar(x + width/2, comm_times, width, label='通信时间')
        axes[1,0].set_title('计算vs通信时间分解')
        axes[1,0].set_ylabel('时间 (ms)')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(strategies)
        axes[1,0].legend()
    
    # 4. 资源利用率对比
    if 'expert_utilization' in df.columns and 'network_utilization' in df.columns:
        expert_util = df.groupby('strategy')['expert_utilization'].mean()
        network_util = df.groupby('strategy')['network_utilization'].mean()
        
        x = np.arange(len(expert_util))
        width = 0.35
        
        axes[1,1].bar(x - width/2, expert_util.values, width, label='专家利用率')
        axes[1,1].bar(x + width/2, network_util.values, width, label='网络利用率')
        axes[1,1].set_title('资源利用率对比')
        axes[1,1].set_ylabel('利用率 (%)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(expert_util.index)
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/moe_overlap_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成详细报告
    report_file = f"{output_dir}/moe_overlap_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# MoE框架通信计算重叠分析报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
        
        f.write("## 分析概要\n\n")
        f.write(f"- 测试配置数量: {len(df)}\n")
        f.write(f"- 重叠策略: {', '.join(df['strategy'].unique())}\n")
        f.write(f"- 计算粒度: {', '.join(df['compute_granularity'].unique())}\n")
        f.write(f"- 通信粒度: {', '.join(df['communication_granularity'].unique())}\n\n")
        
        f.write("## 性能指标对比\n\n")
        
        # 按策略分组的统计
        strategy_stats = df.groupby('strategy').agg({
            'total_time': 'mean',
            'overlap_efficiency': 'mean',
            'expert_utilization': 'mean',
            'network_utilization': 'mean'
        }).round(2)
        
        f.write("### 按重叠策略分组\n\n")
        f.write(strategy_stats.to_markdown())
        f.write("\n\n")
        
        # 最优配置推荐
        if 'total_time' in df.columns:
            best_config = df.loc[df['total_time'].idxmin()]
            f.write("## 最优配置推荐\n\n")
            f.write(f"- **最优策略**: {best_config['strategy']}\n")
            f.write(f"- **计算粒度**: {best_config['compute_granularity']}\n")
            f.write(f"- **通信粒度**: {best_config['communication_granularity']}\n")
            f.write(f"- **总执行时间**: {best_config['total_time']:.2f} ms\n")
            if 'overlap_efficiency' in best_config:
                f.write(f"- **重叠效率**: {best_config['overlap_efficiency']:.1f}%\n")
        
        f.write("\n## 优化建议\n\n")
        f.write("1. **激进重叠策略**通常能获得最好的性能，但需要注意内存开销\n")
        f.write("2. **Kernel级计算粒度**比Layer级粒度有更好的重叠机会\n")
        f.write("3. **Message级通信粒度**在大多数场景下是最优选择\n")
        f.write("4. **专家负载均衡**对MoE模型的重叠效率至关重要\n")
    
    print(f"✓ 分析报告已生成: {report_file}")
    print(f"✓ 分析图表已生成: {output_dir}/moe_overlap_analysis.png")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python analyze_overlap_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    print("分析MoE重叠策略结果...")
    df = analyze_overlap_strategies(results_dir)
    
    if not df.empty:
        print(f"找到 {len(df)} 个分析结果")
        generate_overlap_analysis_report(df, results_dir)
    else:
        print("警告: 没有找到有效的分析结果")
