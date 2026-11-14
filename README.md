# AssetOpsBench 独立复现

面向工业资产管理任务的多 Agent 协同研究代码。研究背景为离散制造业；本仓库实验数据为公开工业资产样例，不代表完整离散制造产线。

基于项目内 2026-11-14 实验备份整理，独立实现单 Agent、Agent-as-Tool 与 Plan-Execute 三种策略。本次整理没有克隆或下载上游 GitHub 代码，也没有启动训练或收费模型推理。

> 这是受数据和算力约束的独立复现，不是 AssetOpsBench 官方代码，也不宣称完整复现论文的系统与分数。数据来源及方法差异见 [复现边界](docs/REPRODUCTION.md)。

## 安装与离线验证

要求 Python 3.12+。在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
python scripts/smoke_test.py
```

冒烟脚本使用明确标记为合成的本地样例和固定脚本后端，检查“运行 → 轨迹 → 静态判分”链路，不联网、不训练。其通过不表示语言模型任务能力。

## 数据覆盖与模型推理

```bash
aob-repro validate-data --write-table results/coverage.csv
aob-repro run-suite --backend scripted --limit 1 --out results/scripted
```

公开场景没有标准答案时，静态判分会跳过，不能将该输出作为任务成功率。用于真实模型实验时，先启动自己的 OpenAI 兼容服务，再设置环境变量。以下示例为 PowerShell：

```powershell
$env:AOB_BASE_URL = "http://localhost:8000/v1"
$env:AOB_API_KEY = "<本机服务所需密钥>"
aob-repro run-suite --backend openai --model qwen3.5-9b --only-ready --strategy agent_as_tool --out results/agent_as_tool
```

`--strategy` 可选 `single_agent`、`agent_as_tool`、`plan_execute`。CLI 默认启用当前六域共 44 个工具；历史 15/21 工具实验需要显式关闭扩展域，见复现说明。不要混用不同工具集的结果。

评估独立执行；使用语言模型判官会调用外部 API，可能产生费用：

```powershell
$env:AOB_JUDGE_BASE_URL = "https://api.deepseek.com"
$env:AOB_JUDGE_MODEL = "deepseek-chat"
$env:AOB_JUDGE_API_KEY = "<在本机配置>"
aob-repro evaluate --trajectories results/agent_as_tool --judge --out results/reports
```

参数详情：`aob-repro --help`、`aob-repro run-suite --help`、`aob-repro evaluate --help`。示例环境文件不会自动加载。

## 仓库结构

```text
src/aob_repro/     契约、数据、工具、后端、编排、轨迹及评估
tests/           离线单元测试
scripts/          离线端到端冒烟验证
data/            既有备份中的规范化数据、场景、自建知识库及合成工单
docs/            架构、数据来源、复现边界及验证记录
.github/workflows/ 自动格式检查、测试和冒烟验证
```

本机验证：135 项测试通过，Ruff 检查与离线冒烟通过；详见 [验证记录](docs/VALIDATION.md)。

详见 [架构与流程图](docs/ARCHITECTURE.md)、[数据说明](data/README.md)、[开发规范](CONTRIBUTING.md)。已有实验原始轨迹、服务器备份、模型权重、私钥与个人文档不纳入此代码仓库。

## 论文与来源

研究对象：*AssetOpsBench: A Real-World Evaluation Benchmark for AI-Driven Task Automation in Industrial Asset Management*。

上游项目标识：[IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench)。本仓库保留现有数据来源说明；上游数据的权利与使用条件不因本次整理而改变。未替第三方数据声明新的授权许可。
