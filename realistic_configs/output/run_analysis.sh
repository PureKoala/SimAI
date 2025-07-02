#!/bin/bash

# Hardware Latency Timeline Analysis Runner
# This script sets up the environment and runs the hardware latency analysis

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Hardware Latency Timeline Analysis"
echo "====================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Setting up virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/pyvenv.cfg" ] || ! pip show matplotlib >/dev/null 2>&1; then
    echo "📋 Installing required packages..."
    pip install -r requirements.txt
    echo "✅ Packages installed"
fi

# Run the analysis
echo "📊 Running hardware latency timeline analysis..."
python analyze_results.py

# Check if files were generated
echo ""
echo "📁 Generated Files:"
echo "=================="

files=("simulation_results.json" "hardware_latency_timeline.png" "performance_comparison.png" "hardware_latency_timeline_interactive.html" "hardware_latency_config.json")

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✅ $file ($size)"
    else
        echo "  ❌ $file (missing)"
    fi
done

echo ""
echo "🎉 Hardware latency analysis completed!"
echo "📝 Open 'hardware_latency_timeline_interactive.html' in your browser for interactive timeline"
echo "🖼️  View 'hardware_latency_timeline.png' for detailed static charts"
echo "⚙️  Edit 'hardware_latency_config.json' to customize your hardware setup"
echo "📚 See 'README_Hardware_Timeline_Analysis.md' for detailed documentation"
