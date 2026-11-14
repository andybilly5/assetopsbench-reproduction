"""编排层：论文 §4.3 的两条链 + 单 Agent 基线。

三种策略共用 `agents/react_loop.py` 的循环，差别只在"谁决定下一步"：

single_agent
    一个循环吃全部工具（论文附录的单 Agent 基线）。

agent_as_tool
    领域 Agent（各自一个 `ReActLoop`，各自白名单）注册成上层 supervisor 的工具。
    见 `agent_as_tool.py`。

plan_execute
    Planner 产出 DAG，Executor 按拓扑序执行，Reviewer 判定，Synthesizer 汇总。
    见 `plan_execute.py`。

装配统一走 `wiring.build_orchestrator`，保证三种策略只在编排方式上不同，
工具集 / 采样参数 / 判分器一律相同。
"""

from __future__ import annotations

from .agent_as_tool import AgentAsToolOrchestrator, DomainAgent, build_domain_agents
from .plan_execute import PlanError, PlanExecuteOrchestrator, PlanStep, check_acyclic, parse_plan
from .wiring import (
    STRATEGIES,
    build_orchestrator,
    build_toolset,
    distractor_registry_from_env,
    merged_registry,
    sensor_directory,
)

__all__ = [
    "STRATEGIES",
    "AgentAsToolOrchestrator",
    "DomainAgent",
    "PlanError",
    "PlanExecuteOrchestrator",
    "PlanStep",
    "build_domain_agents",
    "build_orchestrator",
    "build_toolset",
    "distractor_registry_from_env",
    "check_acyclic",
    "merged_registry",
    "parse_plan",
    "sensor_directory",
]
