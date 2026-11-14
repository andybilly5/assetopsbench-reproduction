"""端到端单测：种子数据 → 工具 → ReAct 循环 → 轨迹落盘。

这条测试是"机器在转"的最小证明：不联网、不需要 GPU、不依赖上游仓库。
改了编排或工具层之后，它必须仍然通过。
"""

import json

import pytest

from aob_repro.agents.react_loop import run_single
from aob_repro.contracts import Scenario
from aob_repro.data.repositories import AssetRepo, DataRoot, TelemetryRepo
from aob_repro.models.scripted import ScriptedBackend
from aob_repro.tools.iot import build_registry

ASSETS = "site,asset_id,asset_class\nMAIN,Chiller 6,Chiller\nMAIN,Chiller 9,Chiller\n"
SENSORS = (
    "site,asset_id,sensor,unit\n"
    "MAIN,Chiller 6,tonnage,ton\n"
    "MAIN,Chiller 6,supply_temp,F\n"
    "MAIN,Chiller 9,tonnage,ton\n"
)
READINGS = (
    "site,asset_id,sensor,ts,value\n"
    "MAIN,Chiller 6,tonnage,2020-06-01,100.5\n"
    "MAIN,Chiller 6,tonnage,2020-06-02,101.2\n"
    "MAIN,Chiller 9,tonnage,2020-06-01,88.0\n"
)


@pytest.fixture()
def registry(tmp_path):
    for name, content in (
        ("assets.csv", ASSETS),
        ("sensors.csv", SENSORS),
        ("readings.csv", READINGS),
    ):
        (tmp_path / name).write_text(content, encoding="utf-8")
    root = DataRoot(tmp_path)
    return build_registry(AssetRepo(root), TelemetryRepo(root))


def test_tool_output_is_real(registry):
    from aob_repro.contracts import ToolCall

    r = registry.dispatch(ToolCall(name="iot.sites", arguments={}))
    assert r.ok and r.output == ["MAIN"]

    r2 = registry.dispatch(
        ToolCall(name="iot.sensors", arguments={"site_name": "MAIN", "asset_id": "Chiller 6"})
    )
    assert r2.ok and "tonnage" in r2.output


def test_missing_asset_becomes_observation(registry):
    from aob_repro.contracts import ToolCall

    r = registry.dispatch(
        ToolCall(
            name="iot.history",
            arguments={"site_name": "MAIN", "asset_id": "Chiller 99", "sensor": "tonnage"},
        )
    )
    assert r.ok is False
    assert "DataUnavailable" in r.error


def test_react_loop_records_trajectory(registry, tmp_path):
    scenario = Scenario.from_record({"id": "1", "text": "What IoT sites are available?"})
    backend = ScriptedBackend(
        [
            ScriptedBackend.tool("iot.sites"),
            ScriptedBackend.final('{"sites": ["MAIN"]}'),
        ]
    )
    out = tmp_path / "trajectories"
    result = run_single(scenario, registry, backend, max_steps=4, out_dir=str(out))

    assert result.stop_reason == "final"
    assert result.steps == 2
    assert result.tool_calls == 1
    assert result.unique_tools == ["iot.sites"]
    assert "MAIN" in result.answer

    files = sorted(p.name for p in out.iterdir())
    assert any(f.endswith(".jsonl") for f in files)
    assert any(f.endswith(".json") for f in files)

    traj = json.loads((out / f"{result.run_id}.json").read_text(encoding="utf-8"))
    assert traj["scenario_id"] == "1"
    assert traj["ops"]["duration_ms"] >= 0


def test_max_steps_guard(registry, tmp_path):
    scenario = Scenario.from_record({"id": "2", "text": "loop forever"})
    # 永远要求再调工具：没有终态时必须以 max_steps 收尾，而不是挂死
    backend = ScriptedBackend([ScriptedBackend.tool("iot.sites") for _ in range(10)])
    result = run_single(scenario, registry, backend, max_steps=3, out_dir=str(tmp_path))
    assert result.stop_reason == "max_steps"
    assert result.steps == 3


def test_scripted_exhaustion_terminates(registry, tmp_path):
    scenario = Scenario.from_record({"id": "3", "text": "x"})
    backend = ScriptedBackend([])
    result = run_single(scenario, registry, backend, max_steps=4, out_dir=str(tmp_path))
    assert result.stop_reason == "final"  # 脚本耗尽时返回 final 文本，不挂死
