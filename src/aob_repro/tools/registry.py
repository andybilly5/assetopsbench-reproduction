"""工具注册与调度。

论文 §4.3 里"工具"是可枚举、可校验的动作集合。这里的实现刻意保持极简：
参数校验只覆盖 JSON Schema 里最常用的几种约束（required / type / enum），
不引入 jsonschema 依赖——**校验的松紧本身就是一个实验变量**，写死在代码里
才能被审计。

失败语义：任何工具异常都转成 `ok=False` 的 `ToolResult` 交回给模型，
不抛穿到循环外。因为"模型看到了错误还继续乱猜"正是要研究的失败模式。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..contracts import ToolCall, ToolResult

Handler = Callable[..., Any]

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


class UnknownTool(LookupError):
    """模型点名了不存在的工具。"""


class ParameterError(ValueError):
    """参数不满足 schema。"""


@dataclass(slots=True)
class ToolSpec:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    tags: list[str] = field(default_factory=list)
    read_only: bool = True
    #: 来源说明（论文明列 / 语义推断 / 自建实现等），供审计与覆盖表核对
    provenance: str = ""

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI tools 数组里的一项。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, args: dict[str, Any]) -> None:
        """按 schema 校验参数，不通过抛 `ParameterError`。"""
        if not isinstance(args, dict):
            raise ParameterError("arguments must be a JSON object")

        props: dict[str, Any] = self.parameters.get("properties", {})
        required: list[str] = self.parameters.get("required", [])

        missing = [k for k in required if k not in args]
        if missing:
            raise ParameterError(
                f"missing required parameter(s): {', '.join(missing)}; "
                f"expected: {', '.join(required)}"
            )

        unexpected = [k for k in args if k not in props]
        if unexpected:
            raise ParameterError(
                f"unknown parameter(s): {', '.join(unexpected)}; "
                f"allowed: {', '.join(sorted(props))}"
            )

        for key, value in args.items():
            spec = props.get(key, {})
            expected_type = spec.get("type")
            if expected_type in _TYPE_MAP:
                # bool 是 int 的子类，必须单独排除，否则 true 会被当成 1 通过
                if expected_type in ("integer", "number") and isinstance(value, bool):
                    raise ParameterError(f"parameter {key!r} expects {expected_type}, got boolean")
                if not isinstance(value, _TYPE_MAP[expected_type]):
                    raise ParameterError(
                        f"parameter {key!r} expects {expected_type}, got {type(value).__name__}"
                    )
            enum = spec.get("enum")
            if enum is not None and value not in enum:
                raise ParameterError(f"parameter {key!r} must be one of {enum}, got {value!r}")


class ToolRegistry:
    """工具白名单 + 调度器。"""

    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._specs[spec.name] = spec

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise UnknownTool(
                f"unknown tool {name!r}; available: {', '.join(self.names())}"
            ) from exc

    def openai_tools(self) -> list[dict[str, Any]]:
        return [self._specs[n].to_openai_schema() for n in self.names()]

    def dispatch(self, call: ToolCall) -> ToolResult:
        """执行一次调用，永远返回 ToolResult。"""
        started = time.perf_counter()
        try:
            spec = self.get(call.name)
        except UnknownTool as exc:
            return ToolResult(
                name=call.name,
                ok=False,
                error=str(exc),
                call_id=call.call_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        try:
            spec.validate(call.arguments)
            output = spec.handler(**call.arguments)
            return ToolResult(
                name=call.name,
                ok=True,
                output=output,
                call_id=call.call_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except ParameterError as exc:
            return ToolResult(
                name=call.name,
                ok=False,
                error=f"invalid_arguments: {exc}",
                call_id=call.call_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 —— 工具失败一律转成观察
            return ToolResult(
                name=call.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                call_id=call.call_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
