"""WO 工单域单测：注册面、检索/统计/归类/SLA/备件/映射、provenance 可验证。

数据源是自建合成工单库 `data/wo/synthetic_workorders.json`；本文件不读上游工单记录。
"""

import json
from pathlib import Path

import pytest

from aob_repro.contracts import ToolCall
from aob_repro.data.repositories import AssetRepo, DataRoot
from aob_repro.tools.wo import WorkorderRepo, build_wo_registry

ASSETS = "site,asset_id,asset_class\nMAIN,Chiller 6,Chiller\nMAIN,hyd_1,Pump\n"
SENSORS = (
    "site,asset_id,sensor,unit\n"
    "MAIN,Chiller 6,Tonnage,ton\n"
    "MAIN,hyd_1,FS1_VolumeFlow_l_per_min_10Hz,L/min\n"
)
READINGS = (
    "site,asset_id,sensor,ts,value\n"
    "MAIN,Chiller 6,Tonnage,2020-06-01T00:00:00,120.0\n"
    "MAIN,hyd_1,FS1_VolumeFlow_l_per_min_10Hz,2020-06-01T00:00:00,30.0\n"
)

WO_DB = Path("data/wo/synthetic_workorders.json")
PROVENANCE = Path("data/wo/PROVENANCE.json")


@pytest.fixture(scope="module")
def repo():
    return WorkorderRepo()


@pytest.fixture()
def registry(tmp_path, repo):
    (tmp_path / "assets.csv").write_text(ASSETS, encoding="utf-8")
    (tmp_path / "sensors.csv").write_text(SENSORS, encoding="utf-8")
    (tmp_path / "readings.csv").write_text(READINGS, encoding="utf-8")
    assets = AssetRepo(DataRoot(tmp_path))
    return build_wo_registry(assets, repo)


def _call(registry, name, **args):
    return registry.dispatch(ToolCall(name=name, arguments=args))


# ① 注册面与目录枚举
def test_wo_registers_fifteen_tools(registry):
    assert len(registry) == 15
    names = registry.names()
    assert names[0].startswith("wo.")
    assert "wo.by_asset" in names and "wo.to_failure_mode" in names


def test_wo_directory_enumerable(registry):
    schemas = registry.openai_tools()
    assert len(schemas) == 15
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["function"]["name"].startswith("wo.")


# ② 合成工单检索
def test_wo_list_filters_by_status(registry):
    r = _call(registry, "wo.list", status="closed")
    assert r.ok and r.output["count"] >= 1
    assert all(w["status"] == "closed" for w in r.output["workorders"])


def test_wo_by_asset_is_isolated(registry):
    r = _call(registry, "wo.by_asset", asset_id="Chiller 6")
    assert r.ok and r.output["count"] >= 1
    assert all(w["asset_id"] == "Chiller 6" for w in r.output["workorders"])


def test_wo_search_returns_matches(registry):
    r = _call(registry, "wo.search", keyword="temperature")
    assert r.ok and r.output["count"] >= 1


def test_wo_unknown_id_becomes_observation(registry):
    r = _call(registry, "wo.get", wo_id="WO-2099-999")
    assert r.ok is False
    assert "DataUnavailable" in r.error


# ③ 统计 / 归类 / SLA / 停机 / 备件 / 映射
def test_wo_status_and_priority_summaries(registry):
    s = _call(registry, "wo.status_summary")
    p = _call(registry, "wo.priority_summary")
    assert s.ok and sum(s.output["counts_by_status"].values()) == s.output["total"]
    assert p.ok and sum(p.output["counts_by_priority"].values()) == p.output["total"]


def test_wo_defect_classify(registry, repo):
    wid = repo.workorders[0]["wo_id"]
    r = _call(registry, "wo.defect_classify", wo_id=wid)
    assert r.ok
    assert r.output["defect_category"]
    assert r.output["related_sensor"]


def test_wo_sla_status_values(registry, repo):
    wid = repo.workorders[0]["wo_id"]
    r = _call(registry, "wo.sla_status", wo_id=wid)
    assert r.ok and r.output["sla_status"] in {"on_time", "overdue", "open"}


def test_wo_downtime_total(registry):
    r = _call(registry, "wo.downtime", asset_id="Chiller 6")
    assert r.ok and r.output["downtime_hours_total"] >= 0


def test_wo_parts_linked(registry, repo):
    wid = next(w["wo_id"] for w in repo.workorders if w["parts"])
    r = _call(registry, "wo.parts_linked", wo_id=wid)
    assert r.ok and r.output["parts"]


def test_wo_to_failure_mode(registry, repo):
    wid = next(w["wo_id"] for w in repo.workorders if w["failure_mode"])
    r = _call(registry, "wo.to_failure_mode", wo_id=wid)
    assert r.ok and r.output["failure_mode"]


# ④ provenance 可验证
def test_wo_provenance_fields_present():
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert prov["source_origin"] == "self-built"
    assert prov["is_upstream_assetopsbench_workorders"] is False
    assert prov["workorder_count"] == 26
    assert "4.2K" in prov["upstream_workorder_count_reference"]


def test_wo_db_count_matches_provenance(repo):
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert len(repo.workorders) == prov["workorder_count"]
    assert WO_DB.is_file()
