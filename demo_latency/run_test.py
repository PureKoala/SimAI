#!/usr/bin/env python3
"""
SimAI GPU延迟测试快速运行脚本
Quick runner for SimAI GPU latency testing
"""

import subprocess
import sys
import os

def run_quick_test():
    """运行快速延迟测试"""
    cmd = [
        sys.executable, "simai_latency_simulator.py",
        "--comm-types", "all_reduce", "p2p",
        "--data-sizes", "1024", "4096", "16384", "65536",
        "--topologies", "fat_tree",
        "--iterations", "3"
    ]
    
    print("运行SimAI GPU网络延迟快速测试...")
    print("命令:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("\n测试完成！检查results/目录查看结果。")
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")
    except KeyboardInterrupt:
        print("\n测试被用户中断")

def run_comprehensive_test():
    """运行全面延迟测试"""
    cmd = [
        sys.executable, "simai_latency_simulator.py",
        "--comm-types", "all_reduce", "p2p", "broadcast",
        "--data-sizes", "1024", "4096", "16384", "65536", "262144", "1048576",
        "--topologies", "fat_tree", "dragonfly", 
        "--iterations", "5"
    ]
    
    print("运行SimAI GPU网络延迟全面测试...")
    print("命令:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("\n测试完成！检查results/目录查看结果。")
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")
    except KeyboardInterrupt:
        print("\n测试被用户中断")

def create_config():
    """创建配置文件"""
    cmd = [sys.executable, "simai_latency_simulator.py", "--create-config"]
    subprocess.run(cmd, check=True)

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python run_test.py quick      # 快速测试")
        print("  python run_test.py full       # 全面测试")
        print("  python run_test.py config     # 创建配置文件")
        return
    
    mode = sys.argv[1]
    
    if mode == "quick":
        run_quick_test()
    elif mode == "full":
        run_comprehensive_test()
    elif mode == "config":
        create_config()
    else:
        print(f"未知模式: {mode}")

if __name__ == "__main__":
    main()
