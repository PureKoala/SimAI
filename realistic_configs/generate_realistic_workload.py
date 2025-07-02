#!/usr/bin/env python3
"""
现实世界 GPU 集群工作负载生成器
基于真实的大语言模型训练通信模式

参考模型：
- GPT-3 175B (Transformer)
- LLaMA 65B 
- PaLM 540B
- Megatron-LM 训练模式
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    parameters: int          # 参数数量
    layers: int             # 层数
    hidden_size: int        # 隐藏层大小
    attention_heads: int    # 注意力头数
    vocab_size: int         # 词汇表大小
    sequence_length: int    # 序列长度

@dataclass
class ParallelismConfig:
    """并行策略配置"""
    tensor_parallel: int    # TP
    pipeline_parallel: int  # PP
    data_parallel: int      # DP
    expert_parallel: int    # EP (MoE)

# 预定义的现实模型配置
REALISTIC_MODELS = {
    "gpt3-175b": ModelConfig(
        name="GPT-3 175B",
        parameters=175_000_000_000,
        layers=96,
        hidden_size=12288,
        attention_heads=96,
        vocab_size=50400,
        sequence_length=2048
    ),
    "llama-65b": ModelConfig(
        name="LLaMA 65B", 
        parameters=65_000_000_000,
        layers=80,
        hidden_size=8192,
        attention_heads=64,
        vocab_size=32000,
        sequence_length=2048
    ),
    "palm-540b": ModelConfig(
        name="PaLM 540B",
        parameters=540_000_000_000,
        layers=118,
        hidden_size=18432,
        attention_heads=48,
        vocab_size=256000,
        sequence_length=2048
    )
}

class RealisticWorkloadGenerator:
    """现实工作负载生成器"""
    
    def __init__(self, model_config: ModelConfig, parallel_config: ParallelismConfig):
        self.model = model_config
        self.parallel = parallel_config
        self.total_gpus = parallel_config.tensor_parallel * parallel_config.pipeline_parallel * parallel_config.data_parallel
        
    def calculate_communication_sizes(self) -> Dict[str, int]:
        """计算各种通信的数据大小"""
        
        # 单精度浮点数大小 (4 bytes)
        dtype_size = 4
        
        # Attention 层通信大小
        attention_hidden = self.model.hidden_size * self.model.sequence_length * dtype_size
        attention_qkv = attention_hidden * 3  # Q, K, V
        attention_output = attention_hidden
        
        # MLP 层通信大小 
        mlp_intermediate = self.model.hidden_size * 4 * self.model.sequence_length * dtype_size
        mlp_output = attention_hidden
        
        # Embedding 通信大小
        embedding_size = self.model.vocab_size * self.model.hidden_size * dtype_size
        
        # 梯度 AllReduce 大小 (per layer)
        layer_params = self.model.hidden_size * self.model.hidden_size * 4 * dtype_size  # 简化估算
        
        return {
            "attention_allgather": attention_qkv,
            "attention_reducescatter": attention_output, 
            "mlp_allgather": mlp_intermediate,
            "mlp_reducescatter": mlp_output,
            "embedding_allreduce": embedding_size,
            "gradient_allreduce": layer_params,
            "pipeline_p2p": attention_hidden,
            "optimizer_allreduce": layer_params * self.model.layers // self.parallel.data_parallel
        }
        
    def generate_transformer_layer_workload(self, layer_id: int = 0) -> str:
        """生成 Transformer 层的通信工作负载"""
        
        comm_sizes = self.calculate_communication_sizes()
        
        workload = f"""# Realistic Transformer Layer {layer_id} Workload
# Model: {self.model.name}
# Parallelism: TP={self.parallel.tensor_parallel} PP={self.parallel.pipeline_parallel} DP={self.parallel.data_parallel}
# Total GPUs: {self.total_gpus}

HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {self.parallel.tensor_parallel} ep: {self.parallel.expert_parallel} pp: {self.parallel.pipeline_parallel} vpp: 1 ga: 1 all_gpus: {self.total_gpus} checkpoints: 0 checkpoint_initiates: 0

{len(self.get_communication_operations())}
"""
        
        # 添加通信操作
        operations = self.get_communication_operations()
        for op in operations:
            workload += f"{op}\n"
            
        return workload
    
    def get_communication_operations(self) -> List[str]:
        """获取通信操作序列"""
        
        comm_sizes = self.calculate_communication_sizes()
        operations = []
        
        # Forward pass communications
        operations.extend([
            # Attention layer forward
            f"attention_qkv_allgather  -1  2000  ALLGATHER  {comm_sizes['attention_allgather']}  1  NONE  0  1000  NONE  0  100",
            f"attention_compute       -1  1000  NONE       0                                    1000  NONE  0  1000  NONE  0  100", 
            f"attention_output_rs     -1  2000  REDUCESCATTER  {comm_sizes['attention_reducescatter']}  1  NONE  0  1000  NONE  0  100",
            
            # MLP layer forward  
            f"mlp_up_allgather       -1  2000  ALLGATHER  {comm_sizes['mlp_allgather']}  1  NONE  0  1000  NONE  0  100",
            f"mlp_compute            -1  3000  NONE       0                             3000  NONE  0  1000  NONE  0  100",
            f"mlp_down_reducescatter -1  2000  REDUCESCATTER  {comm_sizes['mlp_reducescatter']}  1  NONE  0  1000  NONE  0  100",
        ])
        
        # Backward pass communications
        operations.extend([
            # Gradient communications
            f"mlp_grad_allgather     -1  2000  ALLGATHER  {comm_sizes['mlp_allgather']}  1  NONE  0  1000  NONE  0  100",
            f"mlp_grad_reducescatter -1  2000  REDUCESCATTER  {comm_sizes['mlp_reducescatter']}  1  NONE  0  1000  NONE  0  100",
            f"attention_grad_ag      -1  2000  ALLGATHER  {comm_sizes['attention_allgather']}  1  NONE  0  1000  NONE  0  100", 
            f"attention_grad_rs      -1  2000  REDUCESCATTER  {comm_sizes['attention_reducescatter']}  1  NONE  0  1000  NONE  0  100",
            
            # Weight gradient AllReduce (Data Parallel)
            f"weight_grad_allreduce  -1  1  ALLREDUCE  {comm_sizes['gradient_allreduce']}  1  NONE  0  1  NONE  0  100",
        ])
        
        return operations
        
    def generate_full_training_step(self) -> str:
        """生成完整训练步骤的工作负载"""
        
        comm_sizes = self.calculate_communication_sizes()
        
        workload = f"""# Realistic Full Training Step Workload  
# Model: {self.model.name}
# Forward + Backward + Optimizer Step
# Parallelism: TP={self.parallel.tensor_parallel} PP={self.parallel.pipeline_parallel} DP={self.parallel.data_parallel}

HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {self.parallel.tensor_parallel} ep: {self.parallel.expert_parallel} pp: {self.parallel.pipeline_parallel} vpp: 1 ga: 32 all_gpus: {self.total_gpus} checkpoints: 0 checkpoint_initiates: 0

{self.model.layers * 8 + 3}
"""
        
        # Embedding layer
        workload += f"embedding_forward     -1  1000  ALLREDUCE  {comm_sizes['embedding_allreduce']}  1  NONE  0  1000  NONE  0  100\n"
        
        # All transformer layers
        for layer in range(self.model.layers):
            layer_ops = self.get_communication_operations()
            for op in layer_ops:
                workload += f"{op}\n"
                
        # Final gradient synchronization
        workload += f"final_grad_sync      -1  1  ALLREDUCE  {comm_sizes['optimizer_allreduce']}  1  NONE  0  1  NONE  0  100\n"
        workload += f"optimizer_step       -1  5000  NONE  0  5000  NONE  0  1  NONE  0  100\n"
        
        return workload

def main():
    parser = argparse.ArgumentParser(description="Generate realistic GPU workloads for SimAI")
    parser.add_argument("--model", choices=list(REALISTIC_MODELS.keys()), 
                       default="gpt3-175b", help="Model configuration")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=8, help="Pipeline parallel size") 
    parser.add_argument("--dp", type=int, default=32, help="Data parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--output-dir", default="realistic_workloads", help="Output directory")
    parser.add_argument("--workload-type", choices=["layer", "full"], default="full",
                       help="Generate single layer or full training step")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get model and parallelism configs
    model_config = REALISTIC_MODELS[args.model]
    parallel_config = ParallelismConfig(
        tensor_parallel=args.tp,
        pipeline_parallel=args.pp, 
        data_parallel=args.dp,
        expert_parallel=args.ep
    )
    
    # Generate workload
    generator = RealisticWorkloadGenerator(model_config, parallel_config)
    
    if args.workload_type == "layer":
        workload = generator.generate_transformer_layer_workload()
        filename = f"{args.model}_layer_tp{args.tp}_pp{args.pp}_dp{args.dp}.txt"
    else:
        workload = generator.generate_full_training_step()
        filename = f"{args.model}_full_tp{args.tp}_pp{args.pp}_dp{args.dp}.txt"
    
    # Save workload
    output_file = output_dir / filename
    with open(output_file, 'w') as f:
        f.write(workload)
        
    print(f"Generated realistic workload: {output_file}")
    print(f"Model: {model_config.name}")
    print(f"Total GPUs: {generator.total_gpus}")
    print(f"Communication sizes: {generator.calculate_communication_sizes()}")

if __name__ == "__main__":
    main()
