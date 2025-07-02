# Hardware Latency Timeline Analysis

## Overview

This enhanced analysis tool now provides detailed visualization of **GPU-to-GPU communication hardware latency breakdown**. Instead of showing SimAI's simulation process phases, it visualizes the actual hardware components that contribute to end-to-end communication latency between GPUs.

## Features

### 1. Hardware Latency Breakdown Timeline

The tool generates comprehensive visualizations showing how latency is distributed across different hardware components:

- **Computation**: GPU kernel launch and completion
- **Memory**: GPU memory copy operations and host buffer setup
- **PCIe**: Data transfer over PCIe bus
- **NIC**: Network interface card processing
- **Network**: Ethernet transmission, switch latency, and reception

### 2. Visualization Types

#### Matplotlib Timeline (Static)
- **Waterfall Chart**: Shows cumulative latency breakdown
- **Sequential Timeline**: Shows the temporal progression of hardware operations
- **Category Color Coding**: Different colors for each hardware category
- **Total Latency Annotation**: Red line showing total end-to-end latency

#### Plotly Interactive Timeline (if available)
- **Interactive Horizontal Bar Chart**: Hover to see detailed component information
- **Responsive Design**: Zoom, pan, and explore the data
- **Custom Tooltips**: Detailed descriptions of each hardware component

### 3. Customizable Hardware Configuration

The tool uses a JSON configuration file (`hardware_latency_config.json`) that allows you to:

- **Modify Latency Values**: Update latencies based on your actual hardware measurements
- **Add/Remove Components**: Customize the hardware pipeline for your specific setup
- **Change Descriptions**: Update component descriptions for clarity

## Hardware Components Analyzed

### Default Configuration (H100 + NDR InfiniBand)

| Component | Latency (μs) | Category | Description |
|-----------|-------------|----------|-------------|
| GPU Kernel Launch | 5.2 | Computation | Time to launch communication kernel on source GPU |
| GPU Memory Copy | 12.8 | Memory | Copy data from GPU memory to host buffer |
| Host Buffer Setup | 2.1 | Memory | Prepare host memory buffer for transfer |
| PCIe Transfer | 45.6 | PCIe | Data transfer over PCIe bus to NIC |
| NIC Processing | 8.3 | NIC | Network interface card packet processing |
| Ethernet TX | 15.7 | Network | Ethernet transmission to network switch |
| Switch Latency | 3.4 | Network | Network switch forwarding latency |
| Ethernet RX | 12.1 | Network | Ethernet reception at destination |
| Dest NIC Processing | 7.9 | NIC | Destination NIC packet processing |
| Dest PCIe Transfer | 38.2 | PCIe | PCIe transfer to destination GPU |
| Dest GPU Memory Copy | 11.4 | Memory | Copy data to destination GPU memory |
| GPU Kernel Complete | 3.8 | Computation | Complete communication kernel on dest GPU |

**Total End-to-End Latency**: ~166.5 μs

### Latency Distribution by Category

1. **PCIe** (50.3%): 83.8 μs - Largest bottleneck
2. **Network** (18.7%): 31.2 μs - Network infrastructure overhead  
3. **Memory** (15.8%): 26.3 μs - Memory operations
4. **NIC** (9.8%): 16.2 μs - Network interface processing
5. **Computation** (5.4%): 9.0 μs - GPU kernel overhead

## Usage

### Basic Usage

```bash
cd /home/bytedance/SimAI/realistic_configs/results
python analyze_results.py
```

This will generate:
- `hardware_latency_timeline.png` - Static matplotlib visualization
- `hardware_latency_timeline_interactive.html` - Interactive plotly chart (if available)
- `hardware_latency_config.json` - Configuration file for customization

### Customizing Hardware Configuration

1. **Edit the configuration file**:
```bash
nano hardware_latency_config.json
```

2. **Modify latency values** based on your hardware setup:
```json
{
  "hardware_components": [
    {
      "component": "GPU Kernel Launch",
      "latency_us": 5.2,
      "description": "Time to launch communication kernel on source GPU",
      "category": "Computation"
    },
    // ... modify other components as needed
  ]
}
```

3. **Re-run the analysis**:
```bash
python analyze_results.py
```

## Measurement Guidelines

### How to Measure Each Component

#### GPU Computation
- **Tool**: NVIDIA Nsight Systems
- **Metric**: CUDA kernel launch and completion times
- **Command**: `nsys profile --stats=true your_application`

#### Memory Operations
- **Tool**: NVIDIA Nsight Compute
- **Metric**: Memory bandwidth and latency measurements
- **Command**: `ncu --metrics dram__bytes.sum your_application`

#### PCIe Transfer
- **Tool**: PCIe bandwidth testing utilities
- **Metric**: Bus transfer latency per data size
- **Command**: Use `nvidia-smi topo -m` to analyze topology

#### Network Components
- **Tool**: Network ping and iperf3
- **Metric**: Round-trip time measurements
- **Command**: `ping -c 100 destination_host`

#### NCCL Profiling
- **Tool**: NCCL built-in profiling
- **Environment**: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`
- **Analysis**: End-to-end communication timing

## Best Practices

### 1. Hardware-Specific Measurements
- Measure latencies on your actual hardware setup
- Consider different data sizes (small vs. large messages)
- Account for network topology (switch hops, cable lengths)

### 2. Workload-Specific Analysis
- Different ML operations may have different latency patterns
- Consider AllReduce vs. AllGather vs. Point-to-Point patterns
- Measure both intra-node and inter-node communication

### 3. System State Considerations
- Measure under realistic load conditions
- Account for thermal throttling and power management
- Consider network congestion effects

## Integration with SimAI

This hardware latency breakdown complements SimAI simulation results by providing:

1. **Bottom-up Analysis**: Hardware-level latency breakdown
2. **Top-down Validation**: Compare with SimAI simulation outputs
3. **Optimization Guidance**: Identify hardware bottlenecks
4. **Configuration Tuning**: Optimize system parameters based on measurements

## Output Files

### Generated Visualizations

1. **hardware_latency_timeline.png**
   - Two-panel matplotlib chart
   - Waterfall view (top) + Sequential timeline (bottom)
   - High-resolution (300 DPI) for publications

2. **hardware_latency_timeline_interactive.html**
   - Interactive plotly visualization
   - Hover tooltips with detailed information
   - Responsive design for web viewing

3. **hardware_latency_config.json**
   - Editable configuration file
   - Includes measurement tips and notes
   - Version-controlled for reproducibility

### Console Output

The tool provides detailed console output including:
- Hardware latency breakdown summary
- Category-wise analysis with percentages
- File generation status
- Configuration loading status

## Troubleshooting

### Common Issues

1. **Plotly not available**
   - Install with: `pip install plotly pandas kaleido`
   - Static matplotlib version will still work

2. **Configuration file errors**
   - Check JSON syntax validity
   - Ensure all required fields are present
   - Reset by deleting and re-running

3. **Matplotlib rendering issues**
   - Install matplotlib with: `pip install matplotlib`
   - For headless environments: `export MPLBACKEND=Agg`

### Validation

To validate your hardware measurements:
1. Compare total latency with NCCL benchmarks
2. Cross-check with vendor specifications
3. Validate against micro-benchmark results
4. Compare with similar hardware setups in literature

## Future Enhancements

Planned improvements include:
- Support for different communication patterns (AllReduce, AllGather, etc.)
- Multi-GPU topology visualization
- Automated measurement integration
- Comparison with theoretical peak performance
- Integration with real-time monitoring tools
