#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成4个H800 GPU的全连接拓扑配置
使用IBGDA 400Gbps网络，不使用NVLink
"""

import json
import os

def generate_4_h800_ibgda_topology():
    """
    生成4个H800 GPU的全连接拓扑，使用IBGDA 400Gbps网络，不使用NVLink
    
    返回:
        str: 生成的拓扑配置文件路径
    """
    # IBGDA 400Gbps网络参数
    ibgda_bandwidth = 400 * 1024 * 1024 * 1024 / 8  # 400Gbps转换为字节/秒
    ibgda_latency = 1.5e-6  # 1.5微秒延迟
    
    # 创建4个节点（每个节点一个GPU）
    nodes = []
    for i in range(4):
        node = {
            "id": i,
            "type": "gpu",
            "name": f"H800_GPU_{i}",
            "nics": [
                {
                    "id": 0,
                    "type": "ibgda",
                    "name": f"nic_{i}",
                    "bandwidth": ibgda_bandwidth,
                    "latency": ibgda_latency
                }
            ]
        }
        nodes.append(node)
    
    # 创建全连接拓扑的链接
    links = []
    link_id = 0
    for i in range(4):
        for j in range(i+1, 4):
            link = {
                "id": link_id,
                "name": f"link_{i}_{j}",
                "src_node": i,
                "src_nic": 0,
                "dst_node": j,
                "dst_nic": 0,
                "bandwidth": ibgda_bandwidth,
                "latency": ibgda_latency
            }
            links.append(link)
            link_id += 1
    
    # 创建拓扑配置
    topology = {
        "name": "4_H800_IBGDA_400Gbps_FullyConnected",
        "nodes": nodes,
        "links": links
    }
    
    # 确保输出目录存在
    os.makedirs("topology", exist_ok=True)
    
    # 将拓扑配置写入JSON文件
    topo_file = "topology/4_h800_ibgda_topo.json"
    with open(topo_file, "w") as f:
        json.dump(topology, f, indent=4)
    
    print(f"已生成4个H800 GPU的全连接拓扑配置文件：{topo_file}")
    return topo_file

if __name__ == "__main__":
    generate_4_h800_ibgda_topology()