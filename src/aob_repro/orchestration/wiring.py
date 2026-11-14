"""编排装配：把工具集、领域 Agent 与三种策略接起来。

一处装配、三处复用，目的是让"策略"真的是唯一变量——工具集、判分器、采样参数
都不能在策略之间偷偷变化。
"""

from __future__ import annotations

import os
from typing import Any

from ..data.repositories import AssetRepo, TelemetryRepo
from ..tools.distractors import build_distractor_tools
from ..tools.fmsr import FmsrKnowledge, build_fmsr_tools
from ..tools.iot import build_iot_tools
from ..tools.registry import ToolRegistry
from ..tools.tsfm import build_tsfm_tools
from ..tools.utilities import build_utilities_tools
from ..tools.vibration import build_vibration_tools
from ..tools.wo import WorkorderRepo, build_wo_tools
from .agent_as_tool import AgentAsToolOrchestrator, build_domain_agents
from .plan_execute import PlanExecuteOrchestrator

STRATEGIES = ("single_agent", "agent_as_tool", "plan_execute")

#: distractor 消融开关：不设或为空 → 关闭；整数 N → 加载前 N 个占位噪声工具。
DISTRACTOR_ENV = "AOB_DISTRACTOR_N"


def sensor_directory(assets: AssetRepo, telemetry: TelemetryRepo) -> dict[str, list[str]]:
    """{asset_id.lower(): [sensor, ...]}，供 FMSR 判断"本机到底有哪些测点"。"""
    out: dict[str, list[str]] = {}
    for site in assets.sites():
        for row in assets.assets(site):
            aid = row["asset_id"].strip().lower()
            try:
                out[aid] = telemetry.sensors(site, row["asset_id"])
            except Exception:  # noqa: BLE001 —— 无读数的资产留空表
                out[aid] = []
    return out


def build_toolset(
    assets: AssetRepo,
    telemetry: TelemetryRepo,
    kb: FmsrKnowledge | None = None,
    with_fmsr: bool = True,
    with_tsfm: bool = True,
    with_utilities: bool = True,
    with_wo: bool = True,
    with_vibration: bool = True,
) -> tuple[ToolRegistry, ToolRegistry, ToolRegistry, ToolRegistry, ToolRegistry, ToolRegistry]:
    """返回 (iot, fmsr, tsfm, utilities, wo, vibration) 六个注册表，与领域 Agent 的域划分一致。

    `with_fmsr` / `with_tsfm` / `with_utilities` / `with_wo` / `with_vibration` 只为
    消融保留：关掉某个域时必须返回空注册表，而不是让别的域的工具顶替，否则消融会
    悄悄改变工具面。
    """
    iot = ToolRegistry(build_iot_tools(assets, telemetry))
    fmsr = ToolRegistry([])
    if with_fmsr:
        fmsr = ToolRegistry(
            build_fmsr_tools(
                assets, kb or FmsrKnowledge(), available_sensors=sensor_directory(assets, telemetry)
            )
        )
    tsfm = ToolRegistry(build_tsfm_tools(assets, telemetry)) if with_tsfm else ToolRegistry([])
    utilities = (
        ToolRegistry(build_utilities_tools(assets.root)) if with_utilities else ToolRegistry([])
    )
    wo = ToolRegistry(build_wo_tools(assets, WorkorderRepo())) if with_wo else ToolRegistry([])
    vibration = (
        ToolRegistry(build_vibration_tools(assets, telemetry))
        if with_vibration
        else ToolRegistry([])
    )
    return iot, fmsr, tsfm, utilities, wo, vibration


def distractor_registry_from_env() -> ToolRegistry:
    """按 `AOB_DISTRACTOR_N` 构造占位噪声注册表；默认关闭返回空注册表。

    未设变量 / 空串 → 0 个；非整数 → 抛 ValueError（不静默退回 0，避免开关被
    误配成"关闭"而污染消融口径）。distractor 不属于任何真实域，不进覆盖表。
    """
    raw = os.environ.get(DISTRACTOR_ENV, "").strip()
    if not raw:
        return ToolRegistry([])
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(f"{DISTRACTOR_ENV} must be an integer, got {raw!r}") from exc
    return ToolRegistry(build_distractor_tools(n))


def merged_registry(*registries: ToolRegistry) -> ToolRegistry:
    """单 Agent 基线用的合并注册表。"""
    reg = ToolRegistry([])
    import copy

    for r in registries:
        for spec in r._specs.values():  # noqa: SLF001
            reg.register(copy.deepcopy(spec))
    return reg


def build_orchestrator(
    strategy: str,
    *,
    iot: ToolRegistry,
    fmsr: ToolRegistry,
    tsfm: ToolRegistry | None = None,
    utilities: ToolRegistry | None = None,
    wo: ToolRegistry | None = None,
    vibration: ToolRegistry | None = None,
    backend: Any,
    sub_steps: int = 4,
    supervisor_steps: int = 4,
    review: bool = True,
) -> Any:
    """按策略返回带 `.run(scenario, recorder)` 的对象。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    domains = {"iot": iot, "fmsr": fmsr}
    if tsfm is not None and len(tsfm):
        domains["tsfm"] = tsfm
    if utilities is not None and len(utilities):
        domains["utilities"] = utilities
    if wo is not None and len(wo):
        domains["wo"] = wo
    if vibration is not None and len(vibration):
        domains["vibration"] = vibration

    if strategy == "agent_as_tool":
        agents = build_domain_agents(domains, backend, max_steps=sub_steps)
        return AgentAsToolOrchestrator(agents, backend, max_steps=supervisor_steps)

    if strategy == "plan_execute":
        agents = build_domain_agents(domains, backend, max_steps=sub_steps)
        if os.environ.get("AOB_PE_ROSTER") == "1":
            # 消融用：把完整 agent/工具清单告知子 Agent，消除"能力边界类问题无人能答"的盲区
            roster = "\n".join(f"- {a.name} owns: {', '.join(a.registry.names())}" for a in agents)
            note = (
                "You are one of several domain agents working for a supervisor. "
                "The full agent/tool surface of the system is:\n"
                + roster
                + "\nIf the question asks what the system can do, answer using this list; "
                "do not claim a capability you cannot see here."
            )
            for a in agents:
                a.extra_prompt = note
        return PlanExecuteOrchestrator(agents, backend, review=review)

    raise ValueError("single_agent 由 cli 直接走 ReActLoop，不在此装配")
