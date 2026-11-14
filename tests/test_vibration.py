"""Vibration 域单测：8 个工具注册可枚举；无测点时统一 DataUnavailable，不编造数值。"""

import pytest

from aob_repro.contracts import ToolCall
from aob_repro.data.repositories import AssetRepo, DataRoot, TelemetryRepo
from aob_repro.tools.vibration import build_vibration_registry

ASSETS = "site,asset_id,asset_class\nMAIN,Chiller 6,Chiller\n"
SENSORS = "site,asset_id,sensor,unit\nMAIN,Chiller 6,Tonnage,ton\n"
READINGS = "site,asset_id,sensor,ts,value\nMAIN,Chiller 6,Tonnage,2020-06-01T00:00:00,120.0\n"


@pytest.fixture()
def registry(tmp_path):
    (tmp_path / "assets.csv").write_text(ASSETS, encoding="utf-8")
    (tmp_path / "sensors.csv").write_text(SENSORS, encoding="utf-8")
    (tmp_path / "readings.csv").write_text(READINGS, encoding="utf-8")
    root = DataRoot(tmp_path)
    return build_vibration_registry(AssetRepo(root), TelemetryRepo(root))


def test_vibration_registers_eight_tools(registry):
    assert len(registry) == 8
    assert all(n.startswith("vib.") for n in registry.names())


def test_vibration_directory_enumerable(registry):
    schemas = registry.openai_tools()
    assert len(schemas) == 8
    assert {s["function"]["name"] for s in schemas} == set(registry.names())


@pytest.mark.parametrize(
    "name,args",
    [
        ("vib.list_points", {}),
        ("vib.spectrum", {"asset_id": "Chiller 6", "point": "DE-1"}),
        ("vib.bearing_frequencies", {"asset_id": "Chiller 6", "point": "DE-1"}),
        ("vib.envelope_rms", {"asset_id": "Chiller 6", "point": "DE-1"}),
        ("vib.accel_rms", {"asset_id": "Chiller 6", "point": "DE-1"}),
        ("vib.threshold_check", {"asset_id": "Chiller 6", "point": "DE-1"}),
        ("vib.series_by_point", {"asset_id": "Chiller 6", "point": "DE-1"}),
        ("vib.active_alerts", {"asset_id": "Chiller 6"}),
    ],
)
def test_vibration_returns_data_unavailable_with_alternatives(registry, name, args):
    r = registry.dispatch(ToolCall(name=name, arguments=args))
    assert r.ok is False
    assert "DataUnavailable" in r.error
    # 替代测点/数据源建议必须出现
    assert "Available local sensors" in r.error
    assert "iot.history" in r.error or "fmsr" in r.error
    # 不得编造频谱/RMS/特征频率数值：错误串里不应出现结构化数值结果
    assert "spectrum" not in (r.output or {}) if isinstance(r.output, dict) else True


def test_vibration_no_fabricated_numeric_output(registry):
    r = registry.dispatch(
        ToolCall(name="vib.accel_rms", arguments={"asset_id": "Chiller 6", "point": "DE-1"})
    )
    assert r.ok is False
    assert r.output is None
