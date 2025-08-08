#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H800 MoE推理性能分析脚本
专门针对H800硬件和推理工作负载的通信计算重叠分析

硬件配置:
- 4x NVIDIA H800 GPUs
- PCIe 5.0 (35 GB/s per GPU)
- 400Gbps Ethernet网络
- 无NVLink连接

模型配置:
- Qwen3-235B: 84层, 128专家, Top-8路由
- Qwen3-30B: 48层, 128专家, Top-8路由
- Phi-mini-MoE: 32层, 16专家, Top-2路由

并行策略: DP=2, EP=2 (仅数据并行+专家并行)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class H800MoEAnalyzer:
    def __init__(self):
        # H800硬件特性
        self.h800_specs = {
            'memory_bandwidth': 3.35,  # TB/s HBM3
            'compute_performance': 989,  # TFlops FP16
            'pcie_bandwidth': 35,  # GB/s PCIe 5.0 x16
            'num_gpus': 4,
            'network_bandwidth': 50,  # GB/s (400Gbps)
            'memory_size': 80  # GB HBM3
        }
        
        # 模型配置
        self.models = {
            'qwen3_235b': {
                'layers': 84,
                'hidden_size': 12288,
                'num_experts': 128,
                'top_k': 8,
                'expert_size': 98304000,  # ~94MB
                'batch_size': 8,
                'sequence_length': 4096
            },
            'qwen3_30b': {
                'layers': 48,
                'hidden_size': 7168,
                'num_experts': 128,
                'top_k': 8,
                'expert_size': 15728640,  # ~15MB
                'batch_size': 8,
                'sequence_length': 4096
            },
            'phi_mini_moe': {
                'layers': 32,
                'hidden_size': 4096,
                'num_experts': 16,
                'top_k': 2,
                'expert_size': 8388608,  # ~8MB
                'batch_size': 8,
                'sequence_length': 4096
            }
        }
        
        # 通信模式和重叠效率
        self.communication_patterns = {
            'expert_routing': {
                'description': '专家路由决策',
                'overlap_potential': 0.9,  # 高重叠潜力
                'bottleneck': 'network'
            },
            'expert_computation': {
                'description': '专家计算与交换',
                'overlap_potential': 0.75,  # 中等重叠潜力
                'bottleneck': 'memory'
            },
            'dp_allreduce': {
                'description': '数据并行激活聚合',
                'overlap_potential': 0.7,  # 中等重叠潜力
                'bottleneck': 'network'
            }
        }
    
    def calculate_computation_time(self, model_name, layer_type):
        """计算H800上的计算时间"""
        model = self.models[model_name]
        h800 = self.h800_specs
        
        if layer_type == 'attention':
            # 注意力机制计算时间 (基于FLOPs)
            batch_size = model['batch_size']
            seq_len = model['sequence_length']
            hidden_size = model['hidden_size']
            
            # 简化的注意力FLOPs计算
            attention_flops = 4 * batch_size * seq_len * hidden_size * hidden_size
            compute_time = attention_flops / (h800['compute_performance'] * 1e12)
            return compute_time * 1000  # 转换为毫秒
            
        elif layer_type == 'expert':
            # 专家计算时间
            expert_flops = model['expert_size'] * model['batch_size'] * model['top_k']
            compute_time = expert_flops / (h800['compute_performance'] * 1e12)
            return compute_time * 1000
            
        return 0
    
    def calculate_communication_time(self, model_name, comm_type, message_size):
        """计算通信时间，考虑H800硬件限制"""
        h800 = self.h800_specs
        
        if comm_type == 'expert_routing':
            # 专家路由通信 - 主要通过网络
            bandwidth = h800['network_bandwidth']
            latency = 5e-3  # 5us 网络延迟
            
        elif comm_type == 'expert_exchange':
            # 专家数据交换 - 受PCIe限制
            bandwidth = h800['pcie_bandwidth']
            latency = 1e-3  # 1us PCIe延迟
            
        elif comm_type == 'dp_allreduce':
            # 数据并行AllReduce - 网络限制
            bandwidth = h800['network_bandwidth']
            latency = 10e-3  # 10us 集合通信延迟
            
        else:
            bandwidth = h800['network_bandwidth']
            latency = 5e-3
        
        transfer_time = message_size / bandwidth  # GB/s -> 秒
        total_time = (transfer_time + latency) * 1000  # 转换为毫秒
        
        return total_time
    
    def analyze_overlap_efficiency(self, model_name):
        """分析特定模型的通信计算重叠效率"""
        model = self.models[model_name]
        results = {
            'model': model_name,
            'layers': [],
            'total_time': 0,
            'overlap_savings': 0,
            'efficiency_metrics': {}
        }
        
        for layer_idx in range(model['layers']):
            layer_analysis = {
                'layer_id': layer_idx + 1,
                'attention': {},
                'moe': {},
                'total_layer_time': 0
            }
            
            # 注意力层分析
            attention_compute = self.calculate_computation_time(model_name, 'attention')
            attention_comm = self.calculate_communication_time(
                model_name, 'dp_allreduce', 64  # MB
            )
            
            attention_overlap = min(attention_compute, attention_comm) * \
                              self.communication_patterns['dp_allreduce']['overlap_potential']
            
            attention_total = attention_compute + attention_comm - attention_overlap
            
            layer_analysis['attention'] = {
                'compute_time': attention_compute,
                'communication_time': attention_comm,
                'overlap_time': attention_overlap,
                'total_time': attention_total
            }
            
            # MoE层分析
            expert_compute = self.calculate_computation_time(model_name, 'expert')
            expert_routing_comm = self.calculate_communication_time(
                model_name, 'expert_routing', 16  # MB
            )
            expert_exchange_comm = self.calculate_communication_time(
                model_name, 'expert_exchange', 256  # MB
            )
            
            routing_overlap = min(expert_compute * 0.3, expert_routing_comm) * \
                            self.communication_patterns['expert_routing']['overlap_potential']
            
            exchange_overlap = min(expert_compute * 0.7, expert_exchange_comm) * \
                             self.communication_patterns['expert_computation']['overlap_potential']
            
            total_moe_comm = expert_routing_comm + expert_exchange_comm
            total_moe_overlap = routing_overlap + exchange_overlap
            moe_total = expert_compute + total_moe_comm - total_moe_overlap
            
            layer_analysis['moe'] = {
                'compute_time': expert_compute,
                'routing_comm_time': expert_routing_comm,
                'exchange_comm_time': expert_exchange_comm,
                'total_comm_time': total_moe_comm,
                'total_overlap': total_moe_overlap,
                'total_time': moe_total
            }
            
            layer_analysis['total_layer_time'] = attention_total + moe_total
            results['layers'].append(layer_analysis)
            results['total_time'] += layer_analysis['total_layer_time']
            results['overlap_savings'] += attention_overlap + total_moe_overlap
        
        # 计算效率指标
        total_compute = sum(layer['attention']['compute_time'] + 
                          layer['moe']['compute_time'] 
                          for layer in results['layers'])
        
        total_communication = sum(layer['attention']['communication_time'] + 
                                layer['moe']['total_comm_time']
                                for layer in results['layers'])
        
        theoretical_total = total_compute + total_communication
        actual_total = results['total_time']
        
        results['efficiency_metrics'] = {
            'total_compute_time': total_compute,
            'total_communication_time': total_communication,
            'theoretical_sequential_time': theoretical_total,
            'actual_overlapped_time': actual_total,
            'overlap_efficiency': results['overlap_savings'] / total_communication if total_communication > 0 else 0,
            'speedup_ratio': theoretical_total / actual_total if actual_total > 0 else 1,
            'compute_utilization': total_compute / actual_total if actual_total > 0 else 0
        }
        
        return results
    
    def compare_models(self):
        """比较所有模型的性能"""
        model_results = {}
        
        for model_name in self.models.keys():
            print(f"分析模型: {model_name}")
            model_results[model_name] = self.analyze_overlap_efficiency(model_name)
        
        return model_results
    
    def visualize_overlap_analysis(self, results):
        """可视化重叠分析结果"""
        models = list(results.keys())
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('H800 MoE推理性能分析 - 通信计算重叠', fontsize=16, fontweight='bold')
        
        # 1. 总体性能对比
        ax1 = axes[0, 0]
        metrics = ['total_compute_time', 'total_communication_time', 'actual_overlapped_time']
        metric_names = ['计算时间', '通信时间', '实际执行时间']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results[model]['efficiency_metrics'][metric] / 1000 for model in models]  # 转换为秒
            ax1.bar(x + i * width, values, width, label=metric_names[i])
        
        ax1.set_xlabel('模型')
        ax1.set_ylabel('时间 (秒)')
        ax1.set_title('各模型执行时间对比')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 重叠效率对比
        ax2 = axes[0, 1]
        efficiency_metrics = ['overlap_efficiency', 'speedup_ratio', 'compute_utilization']
        efficiency_names = ['重叠效率', '加速比', '计算利用率']
        
        for i, metric in enumerate(efficiency_metrics):
            values = [results[model]['efficiency_metrics'][metric] for model in models]
            ax2.bar(x + i * width, values, width, label=efficiency_names[i])
        
        ax2.set_xlabel('模型')
        ax2.set_ylabel('效率')
        ax2.set_title('重叠效率指标对比')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 层级时间分布 (以Qwen3-235B为例)
        ax3 = axes[1, 0]
        sample_model = 'qwen3_235b'
        if sample_model in results:
            layers = results[sample_model]['layers'][:10]  # 前10层
            layer_ids = [f"L{layer['layer_id']}" for layer in layers]
            
            attention_times = [layer['attention']['total_time'] for layer in layers]
            moe_times = [layer['moe']['total_time'] for layer in layers]
            
            ax3.bar(layer_ids, attention_times, label='注意力层')
            ax3.bar(layer_ids, moe_times, bottom=attention_times, label='MoE层')
            
            ax3.set_xlabel('层号')
            ax3.set_ylabel('时间 (毫秒)')
            ax3.set_title(f'{sample_model} 前10层时间分布')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 硬件瓶颈分析
        ax4 = axes[1, 1]
        bottleneck_data = {
            'Network\n(Expert Routing)': [],
            'Memory\n(Expert Compute)': [],
            'PCIe\n(Data Exchange)': []
        }
        
        for model in models:
            total_routing = sum(layer['moe']['routing_comm_time'] 
                              for layer in results[model]['layers'])
            total_expert_compute = sum(layer['moe']['compute_time'] 
                                     for layer in results[model]['layers'])
            total_exchange = sum(layer['moe']['exchange_comm_time'] 
                               for layer in results[model]['layers'])
            
            bottleneck_data['Network\n(Expert Routing)'].append(total_routing / 1000)
            bottleneck_data['Memory\n(Expert Compute)'].append(total_expert_compute / 1000)
            bottleneck_data['PCIe\n(Data Exchange)'].append(total_exchange / 1000)
        
        bottleneck_names = list(bottleneck_data.keys())
        x = np.arange(len(models))
        width = 0.25
        
        for i, bottleneck in enumerate(bottleneck_names):
            ax4.bar(x + i * width, bottleneck_data[bottleneck], width, 
                   label=bottleneck)
        
        ax4.set_xlabel('模型')
        ax4.set_ylabel('时间 (秒)')
        ax4.set_title('H800硬件瓶颈分析')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/bytedance/SimAI/myMoE/h800_moe_overlap_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results):
        """生成详细的分析报告"""
        report = []
        report.append("H800 MoE推理性能分析报告")
        report.append("=" * 50)
        report.append("")
        
        report.append("硬件配置:")
        report.append(f"- GPU: 4x NVIDIA H800")
        report.append(f"- 内存带宽: {self.h800_specs['memory_bandwidth']} TB/s")
        report.append(f"- 计算性能: {self.h800_specs['compute_performance']} TFlops (FP16)")
        report.append(f"- PCIe带宽: {self.h800_specs['pcie_bandwidth']} GB/s")
        report.append(f"- 网络带宽: {self.h800_specs['network_bandwidth']} GB/s")
        report.append("")
        
        for model_name, result in results.items():
            report.append(f"模型: {model_name}")
            report.append("-" * 30)
            
            metrics = result['efficiency_metrics']
            report.append(f"层数: {len(result['layers'])}")
            report.append(f"总计算时间: {metrics['total_compute_time']:.2f} ms")
            report.append(f"总通信时间: {metrics['total_communication_time']:.2f} ms")
            report.append(f"实际执行时间: {metrics['actual_overlapped_time']:.2f} ms")
            report.append(f"重叠节省时间: {result['overlap_savings']:.2f} ms")
            report.append(f"重叠效率: {metrics['overlap_efficiency']:.3f}")
            report.append(f"加速比: {metrics['speedup_ratio']:.3f}")
            report.append(f"计算利用率: {metrics['compute_utilization']:.3f}")
            report.append("")
        
        # 优化建议
        report.append("H800优化建议:")
        report.append("-" * 30)
        report.append("1. 专家路由优化: 利用高网络带宽进行异步路由决策")
        report.append("2. 内存管理: 充分利用80GB HBM3进行专家缓存")
        report.append("3. PCIe优化: 实现专家数据的流水线传输")
        report.append("4. 批量大小调优: 在内存限制下最大化计算密度")
        report.append("5. 推理特化: 消除训练相关的梯度同步开销")
        
        report_text = "\n".join(report)
        
        # 保存报告
        with open('/home/bytedance/SimAI/myMoE/h800_moe_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text

def main():
    print("开始H800 MoE推理性能分析...")
    
    analyzer = H800MoEAnalyzer()
    results = analyzer.compare_models()
    
    print("\n生成可视化图表...")
    analyzer.visualize_overlap_analysis(results)
    
    print("\n生成分析报告...")
    report = analyzer.generate_report(results)
    print(report)
    
    print("\n分析完成!")
    print("输出文件:")
    print("- 图表: /home/bytedance/SimAI/myMoE/h800_moe_overlap_analysis.png")
    print("- 报告: /home/bytedance/SimAI/myMoE/h800_moe_analysis_report.txt")

if __name__ == "__main__":
    main()
