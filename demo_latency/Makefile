# SimAI GPU网络延迟仿真测试 Makefile

PYTHON := python3
CONFIG_FILE := configs/simai_latency_config.ini

# 默认目标
.PHONY: help
help:
	@echo "SimAI GPU网络延迟仿真测试"
	@echo ""
	@echo "可用目标:"
	@echo "  quick-test     - 快速延迟测试 (3分钟)"
	@echo "  full-test      - 完整延迟测试 (10分钟)"
	@echo "  custom-test    - 自定义测试"
	@echo "  demo          - 运行演示程序"
	@echo "  demo-advanced - 高级用法演示"
	@echo "  create-config  - 创建默认配置文件"
	@echo "  install-deps   - 安装Python依赖"
	@echo "  clean         - 清理结果文件"
	@echo "  plot-last     - 绘制最近结果的图表"
	@echo "  check-env     - 检查环境"
	@echo ""
	@echo "示例:"
	@echo "  make quick-test"
	@echo "  make full-test"
	@echo "  make custom-test COMM_TYPES='all_reduce p2p' DATA_SIZES='1024 4096'"

# 快速测试
.PHONY: quick-test
quick-test:
	@echo "开始SimAI GPU网络延迟快速测试..."
	$(PYTHON) run_test.py quick

# 完整测试
.PHONY: full-test  
full-test:
	@echo "开始SimAI GPU网络延迟完整测试..."
	$(PYTHON) run_test.py full

# 自定义测试参数
COMM_TYPES := all_reduce p2p
DATA_SIZES := 1024 4096 16384 65536
TOPOLOGIES := fat_tree
ITERATIONS := 3

.PHONY: custom-test
custom-test:
	@echo "开始自定义延迟测试..."
	$(PYTHON) simai_latency_simulator.py \
		--comm-types $(COMM_TYPES) \
		--data-sizes $(DATA_SIZES) \
		--topologies $(TOPOLOGIES) \
		--iterations $(ITERATIONS)

# 创建配置文件
.PHONY: create-config
create-config:
	$(PYTHON) run_test.py config

# 安装依赖
.PHONY: install-deps
install-deps:
	$(PYTHON) -m pip install -r requirements.txt

# 清理结果
.PHONY: clean
clean:
	rm -rf results/*
	rm -rf workloads/*
	rm -rf configs/network_*.json
	@echo "结果文件已清理"

# 绘制最近结果的图表
.PHONY: plot-last
plot-last:
	@LAST_RESULT=$$(ls -t results/simai_latency_results_*.json 2>/dev/null | head -1); \
	if [ -n "$$LAST_RESULT" ]; then \
		echo "绘制图表: $$LAST_RESULT"; \
		$(PYTHON) simai_latency_simulator.py --plot-only "$$LAST_RESULT"; \
	else \
		echo "未找到结果文件"; \
	fi

# 演示模式
.PHONY: demo
demo:
	@echo "运行SimAI GPU延迟测试演示..."
	$(PYTHON) demo.py

# 高级演示
.PHONY: demo-advanced
demo-advanced:
	$(PYTHON) demo.py advanced

# 检查环境
.PHONY: check-env
check-env:
	@echo "检查Python环境..."
	$(PYTHON) --version
	@echo "检查依赖包..."
	@$(PYTHON) -c "import pandas, numpy, matplotlib, seaborn; print('✅ 所有依赖包已安装')" 2>/dev/null || \
	(echo "❌ 缺少依赖包，请运行: make install-deps"; exit 1)

# 显示结果摘要
.PHONY: show-results
show-results:
	@echo "最近的测试结果:"
	@ls -lt results/*.json 2>/dev/null | head -5 || echo "未找到结果文件"

# 详细测试（包含所有选项）
.PHONY: detailed-test
detailed-test:
	$(PYTHON) simai_latency_simulator.py \
		--comm-types all_reduce p2p broadcast \
		--data-sizes 1024 4096 16384 65536 262144 1048576 4194304 \
		--topologies fat_tree dragonfly \
		--iterations 5

# 网络拓扑对比测试
.PHONY: topology-test
topology-test:
	$(PYTHON) simai_latency_simulator.py \
		--comm-types all_reduce \
		--data-sizes 65536 \
		--topologies fat_tree dragonfly \
		--iterations 10

# 数据大小扫描测试
.PHONY: size-sweep-test
size-sweep-test:
	$(PYTHON) simai_latency_simulator.py \
		--comm-types all_reduce \
		--data-sizes 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 \
		--topologies fat_tree \
		--iterations 3

.DEFAULT_GOAL := help
