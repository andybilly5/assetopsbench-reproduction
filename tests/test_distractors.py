"""Distractor 消融开关单测：默认关闭、开启 6 个、返回占位噪声、不计入真实域。"""

import pytest

from aob_repro.contracts import ToolCall
from aob_repro.orchestration.wiring import distractor_registry_from_env
from aob_repro.tools.distractors import (
    DISTRACTOR_PROVENANCE,
    build_distractor_registry,
)

DISTRACTOR_NAMES = [
    "iot.alarm_history",
    "wo.open_tickets",
    "vibration.spectrum",
    "iot.operator_notes",
    "wo.spare_parts",
    "tsfm.weather_feed",
]


def test_distractors_default_off(monkeypatch):
    monkeypatch.delenv("AOB_DISTRACTOR_N", raising=False)
    registry = distractor_registry_from_env()
    assert len(registry) == 0


def test_distractors_six_when_enabled(monkeypatch):
    monkeypatch.setenv("AOB_DISTRACTOR_N", "6")
    registry = distractor_registry_from_env()
    assert len(registry) == 6
    assert registry.names() == sorted(DISTRACTOR_NAMES)


def test_distractor_handler_returns_unavailable_placeholder():
    registry = build_distractor_registry(6)
    for name in registry.names():
        res = registry.dispatch(ToolCall(name=name, arguments={}))
        # 形态合法但不可用；占位观察不得带出真实数据
        assert res.ok is True  # 调用本身成功，观察里标注不可得
        assert res.output["available"] is False
        assert res.output["data"] is None
        assert "distractor" in res.output["reason"]
        spec = registry.get(name)
        assert "distractor" in spec.tags
        assert "excluded from the asset coverage table" in spec.provenance
        assert DISTRACTOR_PROVENANCE


def test_distractor_count_capped_and_invalid_rejected():
    with pytest.raises(ValueError):
        build_distractor_registry(7)
    with pytest.raises(ValueError):
        build_distractor_registry(-1)


def test_invalid_env_value_raises(monkeypatch):
    monkeypatch.setenv("AOB_DISTRACTOR_N", "abc")
    with pytest.raises(ValueError, match="AOB_DISTRACTOR_N"):
        distractor_registry_from_env()
