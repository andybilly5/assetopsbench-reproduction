"""FMSR 工具与知识库单测。

重点不是"能返回什么"，而是三件容易出错的事：
1. 故障名模糊匹配能命中，且不会把无关故障凑上来；
2. "本机没有振动测点"这类能力边界必须答得出来（FMSR 场景 110 就考这个）；
3. 缺失的知识库要抛异常，不能静默返回空列表。
"""

import json

import pytest

from aob_repro.data.repositories import AssetRepo, DataRoot
from aob_repro.tools.fmsr import FmsrKnowledge, KnowledgeUnavailable, build_fmsr_tools
from aob_repro.tools.registry import ToolRegistry, UnknownTool

ASSETS = "site,asset_id,asset_class\nMAIN,Chiller 6,Chiller\nMAIN,hyd_1,Pump\n"
SENSORS = (
    "site,asset_id,sensor,unit\n"
    "MAIN,Chiller 6,Power Input,kW\n"
    "MAIN,Chiller 6,Supply Temperature,F\n"
    "MAIN,Chiller 6,Return Temperature,F\n"
    "MAIN,Chiller 6,Condenser Water Flow,gpm\n"
    "MAIN,Chiller 6,Run Status,\n"
    "MAIN,hyd_1,Power Input,kW\n"
)
READINGS = (
    "site,asset_id,sensor,ts,value\n"
    "MAIN,Chiller 6,Power Input,2020-06-01T00:00:00,100\n"
    "MAIN,hyd_1,Power Input,2020-06-01T00:00:00,50\n"
)

KB = {
    "sensor_classes": {
        "temperature": ["Supply Temperature", "Return Temperature"],
        "power": ["Power Input"],
        "flow": ["Condenser Water Flow"],
        "status": ["Run Status"],
        "vibration": [],
    },
    "failure_modes": [
        {
            "name": "Compressor Overheating",
            "description": "d",
            "detected_by": ["Power Input", "Supply Temperature"],
            "sensor_classes": ["power", "temperature"],
            "temporal": [{"sensor": "Power Input", "signature": "rises"}],
            "monitoring_plan": ["p"],
            "priority_sensor": "Power Input",
        },
        {
            "name": "Condenser Water Flow Reduction",
            "description": "d",
            "detected_by": ["Condenser Water Flow"],
            "sensor_classes": ["flow"],
            "temporal": [{"sensor": "Condenser Water Flow", "signature": "drops"}],
            "monitoring_plan": ["p"],
            "priority_sensor": "Condenser Water Flow",
        },
    ],
    "symptom_causes": [
        {
            "sensor": "Power Input",
            "direction": "drops",
            "candidates": [
                {"failure_mode": "Compressor Overheating", "confidence": "low", "note": "n"}
            ],
        },
    ],
}


@pytest.fixture()
def env(tmp_path):
    (tmp_path / "assets.csv").write_text(ASSETS, encoding="utf-8")
    (tmp_path / "sensors.csv").write_text(SENSORS, encoding="utf-8")
    (tmp_path / "readings.csv").write_text(READINGS, encoding="utf-8")
    kb_path = tmp_path / "kb.json"
    kb_path.write_text(json.dumps(KB), encoding="utf-8")
    root = DataRoot.resolve(tmp_path)
    assets = AssetRepo(root)
    available = {
        "chiller 6": [
            "Power Input",
            "Supply Temperature",
            "Return Temperature",
            "Condenser Water Flow",
            "Run Status",
        ],
        "hyd_1": ["Power Input"],
    }
    tools = build_fmsr_tools(assets, FmsrKnowledge(kb_path), available_sensors=available)
    return ToolRegistry(tools), assets, kb_path


def test_registers_four_tools(env):
    reg, _, _ = env
    assert reg.names() == [
        "fmsr.failure_modes",
        "fmsr.failures_by_sensor",
        "fmsr.sensors_for_failure",
        "fmsr.symptom_cause",
    ]


def test_failure_modes_returns_catalogue(env):
    reg, _, _ = env
    from aob_repro.contracts import ToolCall

    res = reg.dispatch(ToolCall(name="fmsr.failure_modes", arguments={"asset_id": "Chiller 6"}))
    assert res.ok
    assert res.output["count"] == 2
    assert res.output["available_sensors"]


def test_sensor_class_expands_to_class_members(env):
    reg, _, _ = env
    from aob_repro.contracts import ToolCall

    res = reg.dispatch(
        ToolCall(
            name="fmsr.failures_by_sensor",
            arguments={"asset_id": "Chiller 6", "sensor": "temperature sensors"},
        )
    )
    assert res.ok
    assert res.output["matched_as"] == "temperature (class)"
    assert [f["failure_mode"] for f in res.output["failure_modes"]] == ["Compressor Overheating"]


def test_missing_modality_is_reported(env):
    reg, _, _ = env
    from aob_repro.contracts import ToolCall

    res = reg.dispatch(
        ToolCall(
            name="fmsr.failures_by_sensor",
            arguments={"asset_id": "Chiller 6", "sensor": "vibration"},
        )
    )
    assert res.ok
    assert res.output["monitoring_possible"] is False
    assert "vibration" in res.output["reason"]


def test_failure_name_fuzzy_match(env):
    reg, _, _ = env
    from aob_repro.contracts import ToolCall

    res = reg.dispatch(
        ToolCall(
            name="fmsr.sensors_for_failure",
            arguments={"asset_id": "Chiller 6", "failure_mode": "condenser water flow reduction"},
        )
    )
    assert res.ok
    assert res.output["resolved"] is True
    assert res.output["failure_mode"] == "Condenser Water Flow Reduction"


def test_unknown_failure_returns_candidates_not_guess(env):
    reg, _, _ = env
    from aob_repro.contracts import ToolCall

    res = reg.dispatch(
        ToolCall(
            name="fmsr.sensors_for_failure",
            arguments={"asset_id": "Chiller 6", "failure_mode": "Pump Seal Leak"},
        )
    )
    assert res.ok
    assert res.output["resolved"] is False
    assert "known_failure_modes" in res.output


def test_symptom_cause_by_direction(env):
    reg, _, _ = env
    from aob_repro.contracts import ToolCall

    res = reg.dispatch(
        ToolCall(
            name="fmsr.symptom_cause",
            arguments={"asset_id": "Chiller 6", "sensor": "Power Input", "direction": "drops"},
        )
    )
    assert res.ok
    assert res.output["matched"] is True
    assert res.output["candidates"][0]["failure_mode"] == "Compressor Overheating"


def test_unknown_asset_is_tool_error(env):
    reg, _, _ = env
    from aob_repro.contracts import ToolCall

    res = reg.dispatch(ToolCall(name="fmsr.failure_modes", arguments={"asset_id": "Chiller 99"}))
    assert res.ok is False
    assert "unknown asset" in res.error


def test_missing_kb_raises(tmp_path):
    with pytest.raises(KnowledgeUnavailable):
        FmsrKnowledge(tmp_path / "nope.json")


def test_unknown_tool_name_raises(env):
    reg, _, _ = env
    with pytest.raises(UnknownTool):
        reg.get("fmsr.nope")
