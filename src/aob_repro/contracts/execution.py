"""执行契约：一次运行里流动的所有记录都在这里定义。

设计原则（对应论文 §5.6 的失败分析需求）：只记录**可观察**的动作与结果，
不记录模型内部思维。任何评估都必须能从这些记录复算出来。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

DecisionKind = Literal["tool", "final"]
StopReason = Literal["final", "max_steps", "budget", "error"]


@dataclass(slots=True)
class ToolCall:
    """模型发起的一次工具调用。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def to_record(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "call_id": self.call_id}


@dataclass(slots=True)
class ToolResult:
    """工具执行结果。失败也是一种结果，不允许抛穿到循环外。"""

    name: str
    ok: bool
    output: Any = None
    error: str = ""
    call_id: str = ""
    duration_ms: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "call_id": self.call_id,
            "duration_ms": round(self.duration_ms, 3),
        }

    def as_observation(self, max_chars: int = 4000) -> str:
        """压成给模型看的一段观察文本，避免超长输出撑爆上下文。"""
        if not self.ok:
            return f"[TOOL_ERROR] {self.name}: {self.error}"
        text = self.output if isinstance(self.output, str) else repr(self.output)
        if len(text) > max_chars:
            text = text[:max_chars] + f"... [truncated, {len(text)} chars]"
        return text


@dataclass(slots=True)
class Decision:
    """模型一次输出：调工具，或给出最终答复。"""

    kind: DecisionKind
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(slots=True)
class TraceEvent:
    """轨迹事件，逐行写 JSONL。"""

    run_id: str
    step: int
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "event": self.event,
            "ts": round(self.ts, 4),
            "payload": self.payload,
        }


@dataclass(slots=True)
class RunResult:
    """一次运行的终态，判分与汇总都吃这个。"""

    run_id: str
    scenario_id: str
    question: str
    answer: str
    strategy: str
    model: str
    steps: int = 0
    tool_calls: int = 0
    unique_tools: list[str] = field(default_factory=list)
    stop_reason: StopReason = "final"
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "question": self.question,
            "answer": self.answer,
            "strategy": self.strategy,
            "model": self.model,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "unique_tools": self.unique_tools,
            "stop_reason": self.stop_reason,
            "ops": {
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "duration_ms": round(self.duration_ms, 3),
            },
        }
