"""utilities 域工具单测：注册面、目录枚举、参数校验、provenance 与消融开关。

全部走本地 tmp_path 数据，不联网、不读远端 results/ 与 docs/。
"""

import json

import pytest

from aob_repro.contracts import ToolCall
from aob_repro.data.repositories import AssetRepo, DataRoot, TelemetryRepo
from aob_repro.orchestration.wiring import build_toolset, merged_registry
from aob_repro.tools.utilities import (
    UTILITIES_PROVENANCE,
    build_utilities_registry,
)

ASSETS = "site,asset_id,asset_class\nMAIN,Chiller 6,Chiller\n"
SENSORS = "site,asset_id,sensor,unit\nMAIN,Chiller 6,tonnage,ton\n"
READINGS = "site,asset_id,sensor,ts,value\nMAIN,Chiller 6,tonnage,2020-06-01T00:00:00,120.0\n"

UTILITY_NAMES = [
    "jsonreader",
    "currentdatetime",
    "csvreader",
    "textsearch",
    "unitconvert",
    "calculator",
]


def _data_root(tmp_path):
    (tmp_path / "assets.csv").write_text(ASSETS, encoding="utf-8")
    (tmp_path / "sensors.csv").write_text(SENSORS, encoding="utf-8")
    (tmp_path / "readings.csv").write_text(READINGS, encoding="utf-8")
    (tmp_path / "sample.json").write_text(
        json.dumps({"greeting": "hello", "count": 3}, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "sample.csv").write_text(
        "site,asset_id\nMAIN,Chiller 6\nOTHER,Pump 1\n", encoding="utf-8"
    )
    (tmp_path / "notes.txt").write_text(
        "first line\nsecond line mentions Chiller 6\nthird line\n", encoding="utf-8"
    )
    return DataRoot.resolve(tmp_path)


def _call(registry, name, arguments):
    return registry.dispatch(ToolCall(name=name, arguments=arguments))


# ① 注册成功且数量为 6
def test_utilities_registry_has_six_tools(tmp_path):
    root = _data_root(tmp_path)
    registry = build_utilities_registry(root)
    assert len(registry) == 6
    assert registry.names() == sorted(UTILITY_NAMES)


# ② 工具目录 / catalog 可枚举这 6 个
def test_catalog_lists_all_six(tmp_path):
    root = _data_root(tmp_path)
    registry = build_utilities_registry(root)
    catalog = [item["function"]["name"] for item in registry.openai_tools()]
    assert sorted(catalog) == sorted(UTILITY_NAMES)
    assert all(item["type"] == "function" for item in registry.openai_tools())


# ③ 未知 / 缺失参数调用报明确错误，不静默返回空
def test_missing_required_parameter_rejected(tmp_path):
    root = _data_root(tmp_path)
    registry = build_utilities_registry(root)
    res = _call(registry, "calculator", {})
    assert res.ok is False
    assert "missing required" in res.error

    res2 = _call(registry, "csvreader", {"file": "sample.csv", "bogus": 1})
    assert res2.ok is False
    assert "unknown parameter" in res2.error


def test_missing_file_raises_clear_error(tmp_path):
    root = _data_root(tmp_path)
    registry = build_utilities_registry(root)
    res = _call(registry, "jsonreader", {"file": "nope.json"})
    assert res.ok is False
    assert "missing data file" in res.error


# ④ provenance 可验证：2 个来自论文 Table 4 + 4 个语义推断 + 自建不等同上游
def test_provenance_markers_present(tmp_path):
    root = _data_root(tmp_path)
    registry = build_utilities_registry(root)
    assert "Table 4" in UTILITIES_PROVENANCE
    assert "upstream" in UTILITIES_PROVENANCE.lower()
    for name in registry.names():
        spec = registry.get(name)
        assert spec.provenance
        assert "Table 4" in spec.provenance
        assert "NOT equivalent" in spec.provenance
    # 名字归属标记：jsonreader / currentdatetime 来自论文，其余 4 个推断
    for name in ("jsonreader", "currentdatetime"):
        assert name in UTILITIES_PROVENANCE


# 功能冒烟：6 个工具都能在本地确定性跑通
def test_tools_run_against_local_data(tmp_path):
    root = _data_root(tmp_path)
    registry = build_utilities_registry(root)

    r = _call(registry, "jsonreader", {"file": "sample.json", "key": "greeting"})
    assert r.ok and r.output["data"] == "hello"

    r = _call(registry, "currentdatetime", {})
    assert r.ok and r.output["timezone"] == "UTC" and "iso" in r.output

    r = _call(
        registry, "csvreader", {"file": "sample.csv", "where_column": "site", "where_value": "MAIN"}
    )
    assert r.ok and r.output["count"] == 1 and r.output["rows"][0]["asset_id"] == "Chiller 6"

    r = _call(registry, "textsearch", {"file": "notes.txt", "query": "chiller 6"})
    assert r.ok and r.output["match_count"] == 1 and r.output["matches"][0]["line_number"] == 2

    r = _call(
        registry, "unitconvert", {"value": 0, "from_unit": "celsius", "to_unit": "fahrenheit"}
    )
    assert r.ok and r.output["result"] == pytest.approx(32.0)

    r = _call(registry, "calculator", {"expression": "2 + 3 * 4"})
    assert r.ok and r.output["result"] == 14.0


def test_calculator_rejects_unsafe_expression(tmp_path):
    root = _data_root(tmp_path)
    registry = build_utilities_registry(root)
    r = _call(registry, "calculator", {"expression": "__import__('os').listdir('/')"})
    assert r.ok is False


# ⑤ 接线：默认 44（iot4+fmsr4+tsfm7+utilities6+wo15+vib8）；
# --no-wo --no-vibration 回到 21；--no-utilities --no-wo --no-vibration 回到 15。
def test_toolset_counts_default_and_no_utilities(tmp_path):
    root = _data_root(tmp_path)
    assets, telemetry = AssetRepo(root), TelemetryRepo(root)

    iot, fmsr, tsfm, utilities, wo, vibration = build_toolset(assets, telemetry)
    assert len(utilities) == 6 and len(wo) == 15 and len(vibration) == 8
    assert len(iot) == 4 and len(fmsr) == 4 and len(tsfm) == 7
    default_merged = merged_registry(iot, fmsr, tsfm, utilities, wo, vibration)
    assert len(default_merged) == 44

    iot2, fmsr2, tsfm2, utilities2, wo2, vib2 = build_toolset(
        assets, telemetry, with_wo=False, with_vibration=False
    )
    assert len(wo2) == 0 and len(vib2) == 0
    no_wo_vib_merged = merged_registry(iot2, fmsr2, tsfm2, utilities2, wo2, vib2)
    assert len(no_wo_vib_merged) == 21

    iot3, fmsr3, tsfm3, utilities3, wo3, vib3 = build_toolset(
        assets, telemetry, with_utilities=False, with_wo=False, with_vibration=False
    )
    assert len(utilities3) == 0 and len(wo3) == 0 and len(vib3) == 0
    no_util_merged = merged_registry(iot3, fmsr3, tsfm3, utilities3, wo3, vib3)
    assert len(no_util_merged) == 15
