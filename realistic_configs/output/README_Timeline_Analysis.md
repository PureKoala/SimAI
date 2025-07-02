# SimAI Results Analysis with Timeline Visualization

This enhanced analysis tool provides comprehensive visualization capabilities for SimAI simulation results, including interactive timeline charts to help analyze simulation performance and execution phases.

## Features

### 📊 Timeline Visualization
- **Matplotlib Timeline**: Static timeline chart showing simulation phases and durations
- **Interactive Plotly Timeline**: Interactive Gantt chart with hover details and zoom capabilities
- **Performance Comparison**: Bar charts comparing metrics between different simulation types

### 🔍 Analysis Capabilities
- **Analytical Simulation Results**: Parse total time, computation time, communication time
- **NS3 Network Simulation Results**: Parse latency metrics, throughput, and network statistics
- **Automated Timeline Generation**: Create visual timeline from simulation logs
- **Multi-format Output**: Generate both static PNG images and interactive HTML charts

## Installation

1. **Set up Python virtual environment**:
   ```bash
   cd /home/bytedance/SimAI/realistic_configs/results
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Analysis
```bash
source venv/bin/activate
python analyze_results.py
```

This will:
1. Parse `analytical_simulation.log` and `ns3_simulation.log`
2. Generate timeline visualizations
3. Create performance comparison charts
4. Output summary statistics

### Generated Files

| File | Description |
|------|-------------|
| `simulation_results.json` | Complete results in JSON format |
| `simulation_timeline.png` | Static timeline chart (Matplotlib) |
| `simulation_timeline_interactive.html` | Interactive timeline chart (Plotly) |
| `performance_comparison.png` | Performance metrics comparison |

## Timeline Chart Features

### 📈 Matplotlib Timeline (`simulation_timeline.png`)
- **Phase Visualization**: Different colors for Setup, Computation, Communication, Network phases
- **Duration Labels**: Shows exact timing for each phase
- **Grid Lines**: Easy-to-read timeline with time scales
- **Legend**: Color-coded phase identification

### 🎯 Interactive Plotly Timeline (`simulation_timeline_interactive.html`)
- **Hover Details**: Click on bars to see detailed information
- **Zoom & Pan**: Interactive navigation through timeline
- **Color Coding**: Consistent phase coloring with Matplotlib version
- **Responsive Design**: Works on desktop and mobile browsers

### 📊 Performance Comparison (`performance_comparison.png`)
- **Side-by-side Metrics**: Compare Analytical vs NS3 simulation results
- **Value Labels**: Exact values displayed on each bar
- **Multiple Metrics**: Total time, latency, throughput comparisons

## Log File Format

### Analytical Simulation Log
The script expects patterns like:
```
Total time: 45.67
Communication time: 12.34
Computation time: 28.92
```

### NS3 Simulation Log
The script expects patterns like:
```
Average latency: 0.125
Maximum latency: 0.567
Total bytes transferred: 2048576000
```

## Customization

### Timeline Events
Timeline events are automatically generated from simulation metrics. You can customize the timeline by modifying the `extract_timeline_data()` function:

```python
def extract_timeline_data(results):
    """Extract timeline events from simulation results"""
    timeline_events = []
    
    # Add custom phases here
    timeline_events.append({
        'task': 'Custom Phase',
        'start': start_time,
        'finish': end_time,
        'phase': 'Custom',
        'duration': duration
    })
    
    return timeline_events
```

### Colors and Styling
Modify phase colors in the visualization functions:

```python
phase_colors = {
    'Setup': '#FF6B6B',        # Red
    'Computation': '#4ECDC4',   # Teal
    'Communication': '#45B7D1', # Blue
    'Network': '#96CEB4',       # Green
    'Custom': '#FECA57'         # Yellow
}
```

## Example Output

### Console Output
```
正在生成时间线可视化图表...
Matplotlib timeline saved to: simulation_timeline.png
Interactive Plotly timeline saved to: simulation_timeline_interactive.html
Performance comparison chart saved to: performance_comparison.png
时间线图表生成成功!

仿真结果摘要:
========================================
Analytical 仿真:
  total_time: 45.67
  communication_time: 12.34
  computation_time: 28.92
NS3 仿真:
  avg_latency: 0.125
  max_latency: 0.567
  total_bytes: 2048576000.0

生成的文件:
========================================
  ✓ simulation_results.json
  ✓ simulation_timeline.png
  ✓ performance_comparison.png
  ✓ simulation_timeline_interactive.html
```

### JSON Results
```json
{
  "analytical": {
    "total_time": 45.67,
    "communication_time": 12.34,
    "computation_time": 28.92
  },
  "ns3": {
    "avg_latency": 0.125,
    "max_latency": 0.567,
    "total_bytes": 2048576000.0
  },
  "metadata": {
    "model": "GPT-3 175B",
    "gpus": 2048,
    "parallelism": "TP=8, PP=8, DP=32"
  }
}
```

## Troubleshooting

### Missing Dependencies
If you see import errors:
```bash
pip install matplotlib plotly pandas numpy kaleido
```

### Virtual Environment Issues
If venv creation fails:
```bash
sudo apt install python3.12-venv
```

### Static Image Generation
For Plotly static images, you may need Chrome:
```bash
plotly_get_chrome
```

## Advanced Usage

### Batch Processing
To analyze multiple simulation runs:

```python
import glob
from pathlib import Path

for log_dir in glob.glob('*/results/'):
    os.chdir(log_dir)
    # Run analysis
    subprocess.run(['python', 'analyze_results.py'])
```

### Custom Metrics
Add new metrics to parsing functions:

```python
def parse_analytical_results(log_file):
    patterns = {
        'total_time': r'Total time: ([\d.]+)',
        'memory_usage': r'Memory usage: ([\d.]+)TB',
        'peak_memory': r'Peak memory: ([\d.]+)TB',
        # Add your custom patterns here
    }
```

## Integration

This analysis tool integrates seamlessly with:
- **SimAI Simulation Framework**: Automatically parses standard SimAI log formats
- **CI/CD Pipelines**: Generate reports in automated testing workflows
- **Research Workflows**: Batch process multiple simulation configurations
- **Performance Monitoring**: Track simulation performance over time

## Contributing

To extend the analysis capabilities:
1. Add new parsing patterns in `parse_*_results()` functions
2. Extend timeline event generation in `extract_timeline_data()`
3. Customize visualization styles in chart creation functions
4. Update this documentation with new features

## Dependencies

- **matplotlib>=3.5.0**: Static chart generation
- **plotly>=5.0.0**: Interactive visualization
- **pandas>=1.3.0**: Data manipulation
- **numpy>=1.21.0**: Numerical computations
- **kaleido>=0.2.1**: Static image export (optional)

---

**Note**: This tool is designed specifically for SimAI simulation logs but can be easily adapted for other simulation frameworks by modifying the parsing patterns and timeline generation logic.
