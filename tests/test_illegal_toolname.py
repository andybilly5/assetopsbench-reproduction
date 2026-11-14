"""M9：非法工具名告警单测。

三类触发条件：不在注册表 / 含空格 / 长度 > 40。告警写轨迹事件并追加日志，
但不改变原有失败/通过判定。
"""

import json

import pytest

from aob_repro.agents.react_loop import ILLEGAL_LOG_ENV, run_single
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


def _run(registry, tmp_path, monkeypatch, illegal_name):
    log = tmp_path / "illegal_toolname.log"
    monkeypatch.setenv(ILLEGAL_LOG_ENV, str(log))
    scenario = Scenario.from_record({"id": "t-illegal", "text": "call a bad tool"})
    backend = ScriptedBackend(
        [
            ScriptedBackend.tool(illegal_name),
            ScriptedBackend.final("done"),
        ]
    )
    result = run_single(scenario, registry, backend, max_steps=4, out_dir=str(tmp_path))
    events = [
        json.loads(line)
        for line in (tmp_path / f"{result.run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    warns = [e for e in events if e["event"] == "warn_illegal_tool_name"]
    log_lines = (
        [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if log.exists()
        else []
    )
    return result, warns, log_lines


def test_unknown_name_warns(registry, tmp_path, monkeypatch):
    result, warns, log_lines = _run(registry, tmp_path, monkeypatch, "nope.missing")
    assert len(warns) == 1
    assert "not_registered" in warns[0]["payload"]["reasons"]
    assert warns[0]["payload"]["scenario_id"] == "t-illegal"
    assert len(log_lines) == 1 and log_lines[0]["name"] == "nope.missing"
    # 判定不受影响：照常收尾，非法调用仍被记为一次工具调用
    assert result.stop_reason == "final"
    assert result.tool_calls == 1


def test_name_with_space_warns(registry, tmp_path, monkeypatch):
    result, warns, log_lines = _run(registry, tmp_path, monkeypatch, "iot sites")
    assert len(warns) == 1
    reasons = warns[0]["payload"]["reasons"]
    assert "contains_whitespace" in reasons
    assert "not_registered" in reasons
    assert log_lines[0]["name"] == "iot sites"
    assert result.tool_calls == 1


def test_name_too_long_warns(registry, tmp_path, monkeypatch):
    long_name = "v" * 41
    result, warns, log_lines = _run(registry, tmp_path, monkeypatch, long_name)
    assert len(warns) == 1
    assert "name_too_long" in warns[0]["payload"]["reasons"]
    # 原串截断到 80 字；此处 41 < 80，全保留
    assert len(warns[0]["payload"]["tool_name"]) == 41
    assert log_lines[0]["name"] == long_name
    assert result.tool_calls == 1


def test_name_truncated_to_80_chars(registry, tmp_path, monkeypatch):
    long_name = "w" * 120
    _, warns, log_lines = _run(registry, tmp_path, monkeypatch, long_name)
    assert len(warns[0]["payload"]["tool_name"]) == 80
    assert len(log_lines[0]["name"]) == 80


def test_legal_name_has_no_warning(registry, tmp_path, monkeypatch):
    log = tmp_path / "illegal_toolname.log"
    monkeypatch.setenv(ILLEGAL_LOG_ENV, str(log))
    scenario = Scenario.from_record({"id": "t-legal", "text": "list sites"})
    backend = ScriptedBackend(
        [
            ScriptedBackend.tool("iot.sites"),
            ScriptedBackend.final("done"),
        ]
    )
    result = run_single(scenario, registry, backend, max_steps=4, out_dir=str(tmp_path))
    events = [
        json.loads(line)
        for line in (tmp_path / f"{result.run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert not any(e["event"] == "warn_illegal_tool_name" for e in events)
    assert not log.exists() or log.read_text(encoding="utf-8") == ""
