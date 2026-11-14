"""M8：AOB_TRACE_MESSAGES 行为层取证开关单测。

关闭（默认）→ 轨迹不含 llm_messages 事件；开启 → 每轮一条，条数与 decide 轮数一致。
"""

import json

import pytest

from aob_repro.agents.react_loop import TRACE_MESSAGES_ENV, run_single
from aob_repro.contracts import Scenario
from aob_repro.data.repositories import AssetRepo, DataRoot, TelemetryRepo
from aob_repro.models.scripted import ScriptedBackend
from aob_repro.tools.iot import build_registry

ASSETS = "site,asset_id,asset_class\nMAIN,Chiller 6,Chiller\n"
SENSORS = "site,asset_id,sensor,unit\nMAIN,Chiller 6,tonnage,ton\n"
READINGS = "site,asset_id,sensor,ts,value\nMAIN,Chiller 6,tonnage,2020-06-01,120.0\n"


@pytest.fixture()
def registry(tmp_path):
    (tmp_path / "assets.csv").write_text(ASSETS, encoding="utf-8")
    (tmp_path / "sensors.csv").write_text(SENSORS, encoding="utf-8")
    (tmp_path / "readings.csv").write_text(READINGS, encoding="utf-8")
    root = DataRoot(tmp_path)
    return build_registry(AssetRepo(root), TelemetryRepo(root))


def _events(out_dir, run_id):
    lines = (out_dir / f"{run_id}.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_trace_messages_off_by_default(registry, tmp_path, monkeypatch):
    monkeypatch.delenv(TRACE_MESSAGES_ENV, raising=False)
    scenario = Scenario.from_record({"id": "t-off", "text": "list sites"})
    backend = ScriptedBackend(
        [
            ScriptedBackend.tool("iot.sites"),
            ScriptedBackend.final('{"sites": ["MAIN"]}'),
        ]
    )
    result = run_single(scenario, registry, backend, max_steps=4, out_dir=str(tmp_path))
    events = _events(tmp_path, result.run_id)
    assert not any(e["event"] == "llm_messages" for e in events)


def test_trace_messages_on_matches_rounds(registry, tmp_path, monkeypatch):
    monkeypatch.setenv(TRACE_MESSAGES_ENV, "1")
    scenario = Scenario.from_record({"id": "t-on", "text": "list sites"})
    backend = ScriptedBackend(
        [
            ScriptedBackend.tool("iot.sites"),
            ScriptedBackend.final('{"sites": ["MAIN"]}'),
        ]
    )
    result = run_single(scenario, registry, backend, max_steps=4, out_dir=str(tmp_path))
    events = _events(tmp_path, result.run_id)
    trace = [e for e in events if e["event"] == "llm_messages"]
    # 每轮 decide 一条；脚本两轮（tool + final）
    assert len(trace) == len(backend.calls) == 2
    for e in trace:
        assert e["payload"]["messages"]
        for m in e["payload"]["messages"]:
            assert m["role"] in {"system", "user", "assistant", "tool"}
            assert len(m["content"]) <= 200
