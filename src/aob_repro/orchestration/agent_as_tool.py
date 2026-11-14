"""Agent-As-Tool 编排：领域 Agent 注册成 supervisor 的工具。

论文 §4.3 的第一条链。实现要点与取舍：

1. **白名单隔离**：每个领域 Agent 只持有自己那一组工具，IoT Agent 看不见 fmsr.*，
   反之亦然。这样"调错工具"能在轨迹里被定位到具体 Agent，而不是笼统归为模型无能。
2. **观察压缩**：返回给 supervisor 的是领域 Agent 的终态答复，不是原始 payload。
   否则 supervisor 的上下文会被遥测读数撑爆，实验变成"测上下文长度"。
3. **调用计数**：每个 Agent 被调用几次、每次几步，都写进轨迹。论文的失败分析
   （§5.6）依赖这一点，缺了它就只能写"模型不行"。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from ..agents.react_loop import ReActLoop
from ..contracts import RunResult, Scenario
from ..tools.registry import ToolRegistry, ToolSpec
from ..trace.recorder import TrajectoryRecorder

SUPERVISOR_PROMPT = (
    "You are the supervisor of an industrial asset operations team. "
    "You do not have direct data access. Your only tools are domain agents: "
    "each agent owns its own tool set and returns a written answer. "
    "Pick the right agent(s), give each a self-contained task description "
    "(include the asset id and the time window), and then write the final answer "
    "by combining what the agents reported. If an agent reports that data or a "
    "capability is unavailable, say so in the final answer instead of inventing it."
)

DOMAIN_PROMPT = {
    "iot": (
        "You are the IoT data agent. You can only read the asset catalog and sensor "
        "telemetry through your tools. Answer the task with the values you actually "
        "retrieved; if a window has no readings, say the window is empty."
    ),
    "fmsr": (
        "You are the failure-mode agent. Use the failure-mode tools to answer; "
        "never invent failure mode names that the tools did not return. "
        "When a monitoring modality is missing locally, state that explicitly."
    ),
    "wo": (
        "You are the workorder agent. The workorder records are a self-built synthetic "
        "database (not upstream workorder records). Answer only with workorder ids, "
        "statuses, counts, parts or failure-mode mappings that the wo.* tools return; "
        "never invent a workorder number or spare part."
    ),
    "vibration": (
        "You are the vibration agent. This deployment has no installed vibration "
        "measurement point. Every vib.* tool returns DataUnavailable; report that "
        "explicitly and suggest the available local sensors or fmsr modality check "
        "instead of producing any spectrum, bearing frequency or RMS value."
    ),
    "tsfm": (
        "You are the time-series agent. Use tsfm.models / tsfm.aitasks to answer any "
        "question about supported models, checkpoints, context windows or task types, "
        "and tsfm.forecast / tsfm.tsad / tsfm.impute / tsfm.resample / tsfm.finetune "
        "to work with actual sensor series. Never name a model or capability that the "
        "tools did not return, and say when weights are unavailable locally."
    ),
}


@dataclass(slots=True)
class DomainAgent:
    """一个可被 supervisor 当工具调用的领域 Agent。"""

    name: str
    domain: str
    registry: ToolRegistry
    backend: Any
    max_steps: int = 4
    system_prompt: str = ""
    extra_prompt: str = ""
    calls: int = 0
    total_steps: int = 0
    last_answer: str = ""

    def run(self, task: str) -> dict[str, Any]:
        self.calls += 1  # 入口即计数：调用失败也要算一次尝试
        scenario = Scenario(
            id=f"{self.name}-sub-{uuid.uuid4().hex[:6]}", text=task, type=self.domain
        )
        base = self.system_prompt or DOMAIN_PROMPT.get(self.domain, "")
        prompt = (base + "\n" + self.extra_prompt).strip() if self.extra_prompt else base
        loop = ReActLoop(
            registry=self.registry,
            backend=self.backend,
            max_steps=self.max_steps,
            system_prompt=prompt,
        )
        result = loop.run(scenario)
        self.total_steps += result.steps
        self.last_answer = result.answer
        return {
            "agent": self.name,
            "domain": self.domain,
            "answer": result.answer,
            "steps": result.steps,
            "tool_calls": result.tool_calls,
            "unique_tools": result.unique_tools,
            "stop_reason": result.stop_reason,
        }

    def to_tool_spec(self) -> ToolSpec:
        def handler(task: str) -> dict[str, Any]:
            return self.run(task)

        return ToolSpec(
            name=f"agent.{self.name}",
            description=(
                f"Delegate a task to the {self.domain.upper()} domain agent. "
                f"It owns these tools: {', '.join(self.registry.names())}. "
                "Give a self-contained task: asset id, sensors, time window, and what to report."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Self-contained instruction for the domain agent.",
                    }
                },
                "required": ["task"],
            },
            handler=handler,
            tags=["agent", self.domain],
            read_only=True,
        )


class AgentAsToolOrchestrator:
    """supervisor ReAct 循环 + 领域 Agent 工具。"""

    def __init__(
        self,
        agents: list[DomainAgent],
        supervisor_backend: Any,
        max_steps: int = 4,
    ) -> None:
        self.agents = {a.name: a for a in agents}
        self.backend = supervisor_backend
        self.max_steps = max_steps
        self.strategy = "agent_as_tool"

    def _supervisor_registry(self) -> ToolRegistry:
        return ToolRegistry([a.to_tool_spec() for a in self.agents.values()])

    def run(self, scenario: Scenario, recorder: TrajectoryRecorder | None = None) -> RunResult:
        registry = self._supervisor_registry()

        if recorder:
            recorder.event(
                0,
                "user",
                text=scenario.text,
                tools=registry.names(),
                agents={n: a.registry.names() for n, a in self.agents.items()},
                strategy=self.strategy,
            )

        loop = ReActLoop(
            registry=registry,
            backend=self.backend,
            max_steps=self.max_steps,
            system_prompt=SUPERVISOR_PROMPT,
        )
        result = loop.run(scenario, recorder=recorder)

        result.strategy = self.strategy
        if recorder:
            recorder.event(
                result.steps,
                "agent_calls",
                counts={n: a.calls for n, a in self.agents.items()},
                steps={n: a.total_steps for n, a in self.agents.items()},
            )
        result.unique_tools = sorted(
            {f"agent.{n}" for n, a in self.agents.items() if a.calls} | set(result.unique_tools)
        )
        if recorder:
            # ReActLoop 已写过一次 .json（当时 strategy 还是 single_agent），这里覆写纠正
            recorder.finish(
                result, extra={"agent_calls": {n: a.calls for n, a in self.agents.items()}}
            )
        return result


def build_domain_agents(
    registries: dict[str, ToolRegistry],
    backend: Any,
    max_steps: int = 4,
    names: dict[str, str] | None = None,
) -> list[DomainAgent]:
    """按 {domain: registry} 建 Agent；名字默认用域名。"""
    labels = names or {}
    return [
        DomainAgent(
            name=labels.get(d, d),
            domain=d,
            registry=r,
            backend=backend,
            max_steps=max_steps,
        )
        for d, r in registries.items()
    ]
