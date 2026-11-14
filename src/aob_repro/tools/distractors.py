"""Distractor 消融工具（形态合法、语义无关，默认不加载）。

这 6 个工具的名字与参数形态仿照各业务域，但在本复现里**没有对应真实数据源**：
handler 只返回确定的"数据不可得"占位观察，不参与任何正确答案的推导。

用途：消融"工具面噪声"——检验模型是否会被无关工具/噪声观察带偏。开关见
`orchestration.wiring` 里的环境变量 `AOB_DISTRACTOR_N`；**默认关闭**（不设该
变量时工具面与未加本功能时完全一致）。

边界：

- distractor 不计入资产覆盖表的工具数（覆盖表只统计真实域工具）；
- 返回内容不引用任何 `data/` 下真实记录，避免污染失败归因；
- 这些名字是消融占位，不属于论文 Table 4 的真实工具清单。
"""

from __future__ import annotations

from typing import Any

from .registry import ToolRegistry, ToolSpec

DISTRACTOR_PROVENANCE = (
    "ablation-only distractor: morphologically valid but semantically unrelated "
    "placeholder; backed by no real source data and excluded from the asset "
    "coverage table."
)

_UNAVAILABLE_REASON = (
    "distractor: this data category is not tracked in this reproduction; "
    "this observation must not be used to derive the answer."
)


def _unavailable(tool: str) -> dict[str, Any]:
    return {
        "tool": tool,
        "available": False,
        "data": None,
        "reason": _UNAVAILABLE_REASON,
    }


def _spec(name: str, description: str, props: dict[str, Any]) -> ToolSpec:
    def handler(**kwargs: Any) -> dict[str, Any]:
        return _unavailable(name)

    return ToolSpec(
        name=name,
        description=description,
        parameters={"type": "object", "properties": props, "required": []},
        handler=handler,
        tags=["distractor", "ablation"],
        provenance=DISTRACTOR_PROVENANCE,
    )


def _all_specs() -> list[ToolSpec]:
    return [
        _spec(
            "iot.alarm_history",
            "Placeholder: would list past alarm events for an asset (no real source).",
            {"site_name": {"type": "string"}, "asset_id": {"type": "string"}},
        ),
        _spec(
            "wo.open_tickets",
            "Placeholder: would list open work-order tickets (no real source).",
            {"status": {"type": "string"}},
        ),
        _spec(
            "vibration.spectrum",
            "Placeholder: would return a vibration spectrum FFT (no real source).",
            {"asset_id": {"type": "string"}, "window": {"type": "string"}},
        ),
        _spec(
            "iot.operator_notes",
            "Placeholder: would read free-text operator notes (no real source).",
            {"asset_id": {"type": "string"}},
        ),
        _spec(
            "wo.spare_parts",
            "Placeholder: would list spare-parts inventory for an asset class (no source).",
            {"asset_class": {"type": "string"}},
        ),
        _spec(
            "tsfm.weather_feed",
            "Placeholder: would fetch an external weather feed (no real source).",
            {"site_name": {"type": "string"}},
        ),
    ]


def build_distractor_tools(n: int = 6) -> list[ToolSpec]:
    """返回前 `n` 个 distractor；`n` 必须在 0..6 之间。"""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"distractor count must be an integer >= 0, got {n!r}")
    specs = _all_specs()
    if n > len(specs):
        raise ValueError(f"only {len(specs)} distractors are defined, got n={n}")
    return specs[:n]


def build_distractor_registry(n: int = 6) -> ToolRegistry:
    return ToolRegistry(build_distractor_tools(n))
