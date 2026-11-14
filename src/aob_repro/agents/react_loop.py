"""Agent 循环。

本模块只实现**一条** ReAct 链，供三种策略复用：
- `single_agent`：一个循环吃全部工具（论文附录 G.3 的单 Agent 基线）
- `agent_as_tool`：把一个循环包成"工具"，交给上层 supervisor 循环
- `plan_execute`：由计划器生成 DAG，再逐节点复用本循环

把循环抽成公共件，是为了让"策略"成为唯一实验变量——这一点在写报告时必须能
经得起追问。
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from ..contracts import Decision, RunResult, Scenario
from ..tools.registry import ToolRegistry
from ..trace.recorder import TrajectoryRecorder

DEFAULT_SYSTEM_PROMPT = (
    "You are an industrial asset operations assistant. "
    "Use the provided tools to retrieve real data; never invent values. "
    "If a tool reports an error, read it and adjust rather than guessing. "
    "Cite the values you retrieved in your final answer, and if the data is "
    "insufficient, say so explicitly instead of producing a spurious result."
)

_TOOL_MARKER_RE = re.compile(r"<\|tool_call\|>\s*(?:\[[^\]]*\])?")


def _strip_tool_markers(text: str) -> str:
    """剥掉残留的工具调用标记（如 <|tool_call|>[]），只保留自然语言正文。"""
    if not text:
        return ""
    return _TOOL_MARKER_RE.sub("", text).strip()


#: M8 行为层取证开关：=1 时把每轮实际发给 vLLM 的 messages 摘要写进轨迹。
TRACE_MESSAGES_ENV = "AOB_TRACE_MESSAGES"
#: M9 非法工具名告警日志路径（测试可用环境变量指到临时文件）。
ILLEGAL_LOG_ENV = "AOB_ILLEGAL_TOOL_LOG"
DEFAULT_ILLEGAL_LOG = "/data/logs/illegal_toolname.log"
ILLEGAL_NAME_MAX_LEN = 40


def _messages_snapshot(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每轮 messages 的取证摘要：role + content 前 200 字，不回传完整上下文。"""
    return [
        {"role": str(m.get("role", "")), "content": str(m.get("content") or "")[:200]}
        for m in messages
    ]


def _illegal_name_reasons(name: Any, registry: ToolRegistry) -> list[str]:
    """M9：判定非法工具名的三类条件；返回命中原因列表（空 = 合法）。"""
    reasons: list[str] = []
    if not isinstance(name, str) or not name:
        reasons.append("empty_name")
        return reasons
    if name not in registry.names():
        reasons.append("not_registered")
    if any(ch.isspace() for ch in name):
        reasons.append("contains_whitespace")
    if len(name) > ILLEGAL_NAME_MAX_LEN:
        reasons.append("name_too_long")
    return reasons


def _warn_illegal_tool_name(
    recorder: TrajectoryRecorder | None,
    scenario_id: str,
    run_id: str,
    step: int,
    name: str,
    reasons: list[str],
) -> None:
    """M9：写轨迹事件 + 追加日志；告警不改变原有失败/通过判定。"""
    truncated = (name or "")[:80]
    payload: dict[str, Any] = {
        "kind": "WARN_ILLEGAL_TOOL_NAME",
        "scenario_id": scenario_id,
        "run_id": run_id,
        "tool_name": truncated,
        "reasons": reasons,
    }
    if recorder:
        recorder.event(step, "warn_illegal_tool_name", **payload)
    log_path = Path(os.environ.get(ILLEGAL_LOG_ENV, DEFAULT_ILLEGAL_LOG))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {**payload, "name": truncated, "step": step, "ts": round(time.time(), 4)}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # 日志写失败不抛穿：告警是旁路取证，不得影响工具调度与判分。
        pass


class ReActLoop:
    """决策 → 调工具 → 观察 → 直到给出终态或触顶。"""

    def __init__(
        self,
        registry: ToolRegistry,
        backend: Any,
        max_steps: int = 8,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.registry = registry
        self.backend = backend
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def run(self, scenario: Scenario, recorder: TrajectoryRecorder | None = None) -> RunResult:
        run_id = recorder.run_id if recorder else f"run-{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()

        import os as _os

        if _os.environ.get("AOB_NO_SYSTEM") == "1":
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": scenario.text},
            ]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": scenario.text},
            ]
        tools = self.registry.openai_tools()
        _fs_path = os.environ.get("AOB_FEWSHOT_FILE") or ""
        if _fs_path and os.path.exists(_fs_path):
            _demo = io.open(_fs_path, encoding="utf-8").read()
            messages[-1]["content"] = _demo + str(scenario.text)

        answer = ""
        stop_reason = "max_steps"
        steps = 0
        total_calls = 0
        used: set[str] = set()
        tokens_in = tokens_out = 0
        nudges = 0
        trace_messages = os.environ.get(TRACE_MESSAGES_ENV) == "1"

        if recorder:
            recorder.event(0, "user", text=scenario.text, tools=self.registry.names())

        for step in range(1, self.max_steps + 1):
            steps = step
            decision: Decision = self.backend.decide(messages, tools)

            # M8：取证开关开启时，把本轮实际发给 vLLM 的 messages 摘要写入轨迹；
            # 默认关闭，关闭时轨迹格式与现有判分完全不受影响。
            if recorder and trace_messages:
                recorder.event(
                    step,
                    "llm_messages",
                    messages=_messages_snapshot(messages),
                )

            usage = getattr(self.backend, "last_usage", {}) or {}
            tokens_in += int(usage.get("prompt_tokens", 0))
            tokens_out += int(usage.get("completion_tokens", 0))

            if decision.kind == "final":
                raw = decision.text or ""
                cleaned = _strip_tool_markers(raw)
                if (not cleaned) and ("<|tool_call|>" in raw) and nudges < 2:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": "No tool call is needed. Answer the question directly in natural language. Do not output tool-call syntax.",
                        }
                    )
                    nudges += 1
                    continue
                answer = cleaned
                stop_reason = "final"
                if recorder:
                    recorder.event(step, "final", text=answer)
                break

            # 回填 assistant 的 tool_calls，保证下一轮上下文与协议一致
            messages.append(
                {
                    "role": "assistant",
                    "content": decision.text or None,
                    "tool_calls": [
                        {
                            "id": c.call_id or f"call_{step}_{i}",
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": _dump_args(c.arguments),
                            },
                        }
                        for i, c in enumerate(decision.tool_calls)
                    ],
                }
            )
            if recorder:
                recorder.event(
                    step,
                    "decision",
                    text=decision.text,
                    tool_calls=[c.to_record() for c in decision.tool_calls],
                )

            for call in decision.tool_calls:
                # M9：非法工具名旁路告警；不改变后续 dispatch 的失败/通过判定。
                reasons = _illegal_name_reasons(call.name, self.registry)
                if reasons:
                    _warn_illegal_tool_name(recorder, scenario.id, run_id, step, call.name, reasons)
                result = self.registry.dispatch(call)
                total_calls += 1
                used.add(call.name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id or f"call_{step}",
                        "name": call.name,
                        "content": result.as_observation(),
                    }
                )
                if recorder:
                    recorder.event(step, "tool_result", result=result.to_record())

        duration_ms = (time.perf_counter() - started) * 1000
        result = RunResult(
            run_id=run_id,
            scenario_id=scenario.id,
            question=scenario.text,
            answer=answer,
            strategy="single_agent",
            model=getattr(self.backend, "model", "unknown"),
            steps=steps,
            tool_calls=total_calls,
            unique_tools=sorted(used),
            stop_reason=stop_reason,  # type: ignore[arg-type]
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
        )
        if recorder:
            recorder.finish(result)
        return result


def _dump_args(args: dict[str, Any]) -> str:
    import json

    return json.dumps(args, ensure_ascii=False)


def run_single(
    scenario: Scenario,
    registry: ToolRegistry,
    backend: Any,
    max_steps: int = 8,
    out_dir: str = "results/trajectories",
) -> RunResult:
    """便捷入口：建recorder → 跑 → 落盘。"""
    run_id = f"{scenario.id}-{uuid.uuid4().hex[:8]}"
    rec = TrajectoryRecorder(out_dir, run_id)
    loop = ReActLoop(registry=registry, backend=backend, max_steps=max_steps)
    return loop.run(scenario, recorder=rec)
