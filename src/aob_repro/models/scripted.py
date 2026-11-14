"""脚本后端：离线跑通流程、做回归测试，不花一分钱、不需要 GPU。

用途有三个：
1. 单测里跑 ReAct 循环的完整路径；
2. 用固定脚本验证"工具调用 → 观察 → 终态"的装配是否正确；
3. 做 harness 自身的回归——改了编排代码后，用同一套脚本确认行为没变。

注意：脚本后端产出的**分数没有任何研究意义**，它只证明机器在转。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..contracts import Decision, ToolCall

ScriptItem = Decision | Callable[[list[dict[str, Any]]], Decision]


class ScriptedBackend:
    """按预设脚本依次返回决策。"""

    def __init__(self, script: Sequence[ScriptItem], model: str = "scripted") -> None:
        self._script: list[ScriptItem] = list(script)
        self._cursor = 0
        self.model = model
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self.calls: list[list[dict[str, Any]]] = []

    def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Decision:
        self.calls.append(list(messages))
        if self._cursor >= len(self._script):
            return Decision(kind="final", text="[scripted] script exhausted")
        item = self._script[self._cursor]
        self._cursor += 1
        if callable(item):
            return item(messages)
        return item

    def reset(self) -> None:
        """跑新场景前必须归零：脚本是有状态的，跨场景复用会让后续场景拿到空脚本。"""
        self._cursor = 0
        self.calls.clear()
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    # -- 便捷构造 ------------------------------------------------------- #
    @staticmethod
    def tool(name: str, **arguments: Any) -> Decision:
        return Decision(
            kind="tool", tool_calls=[ToolCall(name=name, arguments=arguments, call_id="s1")]
        )

    @staticmethod
    def final(text: str) -> Decision:
        return Decision(kind="final", text=text)
