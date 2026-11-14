"""TSFM 工具面单测。

盯三件事，而不是"能跑"：
1. 目录类工具必须能列出模型与任务（TSFM 场景问的就是这个，答不出即 0 分）；
2. 计算类工具的数字必须来自真实读数，且缺口不得被静默填补；
3. 未知模型 / 未知资产 / 缺目录都要显式失败，不能返回伪造结果。
"""

import json
from pathlib import Path

import pytest

from aob_repro.data.repositories import AssetRepo, DataRoot, DataUnavailable, TelemetryRepo
from aob_repro.tools.tsfm import (
    CatalogUnavailable,
    _seasonal_baseline,
    _to_grid,
    build_tsfm_tools,
)

ASSETS = "site,asset_id,asset_class\nMAIN,Chiller 6,Chiller\n"
SENSORS = "site,asset_id,sensor,unit\nMAIN,Chiller 6,tonnage,ton\nMAIN,Chiller 6,Run Status,state\n"

# 96 步 = 24 小时；再造 8 个缺口，方便验证"长缺口不插补"
_READINGS = ["site,asset_id,sensor,ts,value"]
for i in range(200):
    if 120 <= i < 128:
        continue  # 8 步缺口，超过 max_gap_steps=3
    _READINGS.append(
        f"MAIN,Chiller 6,tonnage,2020-06-{1 + i // 96:02d}T{(i % 96) // 4:02d}:{(i % 4) * 15:02d}:00,{100 + (i % 96)}"
    )
for i in range(200):
    _READINGS.append(
        f"MAIN,Chiller 6,Run Status,2020-06-{1 + i // 96:02d}T{(i % 96) // 4:02d}:{(i % 4) * 15:02d}:00,{1 if i % 96 > 20 else 0}"
    )
READINGS = "\n".join(_READINGS) + "\n"


@pytest.fixture()
def tools(tmp_path: Path):
    (tmp_path / "assets.csv").write_text(ASSETS, encoding="utf-8")
    (tmp_path / "sensors.csv").write_text(SENSORS, encoding="utf-8")
    (tmp_path / "readings.csv").write_text(READINGS, encoding="utf-8")
    root = DataRoot.resolve(tmp_path)
    specs = build_tsfm_tools(AssetRepo(root), TelemetryRepo(root))
    return {s.name: s for s in specs}


def test_registers_seven_tools(tools):
    assert set(tools) == {
        "tsfm.aitasks",
        "tsfm.models",
        "tsfm.forecast",
        "tsfm.tsad",
        "tsfm.impute",
        "tsfm.resample",
        "tsfm.finetune",
    }


def test_models_lists_checkpoints(tools):
    out = tools["tsfm.models"].handler()
    assert out["count"] >= 5
    assert all(m["checkpoint"] for m in out["models"])
    # 能力边界题必须能从工具里读出模型名
    assert any("chronos" in m["checkpoint"] for m in out["models"])


def test_aitasks_lists_task_ids(tools):
    out = tools["tsfm.aitasks"].handler()
    ids = {t["task_id"] for t in out["tasks"]}
    assert {"forecasting", "anomaly_detection", "imputation", "resampling"} <= ids


def test_forecast_returns_real_points_and_mae(tools):
    out = tools["tsfm.forecast"].handler("Chiller 6", "tonnage", 24)
    assert out["resolved"] is True
    assert len(out["forecast"]) == 24
    assert out["backtest_mae"] >= 0
    assert out["missing_points"] >= 0
    # 不得声称跑的是上游 checkpoint
    assert "not the output of the named checkpoint" in out["execution_note"]


def test_forecast_rejects_unknown_model(tools):
    out = tools["tsfm.forecast"].handler("Chiller 6", "tonnage", 8, "gpt-forecast")
    assert out["resolved"] is False
    assert "chronos_t5_small" in out["known_models"]


def test_forecast_unknown_asset_raises(tools):
    with pytest.raises(DataUnavailable):
        tools["tsfm.forecast"].handler("Chiller 99", "tonnage", 8)


def test_tsad_reports_no_false_positives_on_flat_series(tools):
    out = tools["tsfm.tsad"].handler("Chiller 6", "tonnage", 96, 3.0)
    assert out["anomaly_count"] == 0  # 单调爬升的合成序列没有离群点
    assert out["points_checked"] > 100


def test_impute_marks_long_gaps_not_imputed(tools):
    out = tools["tsfm.impute"].handler("Chiller 6", "tonnage", 3)
    assert out["long_gaps"] >= 1
    assert "left as missing" in out["policy"]


def test_resample_infers_operating_ratio(tools):
    out = tools["tsfm.resample"].handler("Chiller 6", "Run Status")
    assert out["resolved"] is True
    assert 0 < out["operating_ratio"] < 1  # 合成序列里 20/96 为停机
    assert out["operating_periods"]
    assert out["cadence_minutes"] == 15


def test_resample_refuses_other_cadence(tools):
    out = tools["tsfm.resample"].handler("Chiller 6", "Run Status", 60)
    assert out["resolved"] is False
    assert out["supported_freq_minutes"] == 15


def test_grid_keeps_gap_as_none():
    from datetime import datetime, timedelta

    base = datetime(2020, 6, 1)
    series = [(base, 1.0), (base + timedelta(minutes=30), 2.0)]
    grid, values = _to_grid(series)
    assert values[1] is None  # 中间 15 分钟缺口不被插值
    assert len(grid) == 3


def test_seasonal_baseline_needs_min_points():
    with pytest.raises(DataUnavailable):
        _seasonal_baseline([1.0, 2.0], 8)


def test_missing_catalog_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AOB_TSFM_CATALOG", str(tmp_path / "nope.json"))
    (tmp_path / "assets.csv").write_text(ASSETS, encoding="utf-8")
    (tmp_path / "sensors.csv").write_text(SENSORS, encoding="utf-8")
    (tmp_path / "readings.csv").write_text(READINGS, encoding="utf-8")
    root = DataRoot.resolve(tmp_path)
    with pytest.raises(CatalogUnavailable):
        build_tsfm_tools(AssetRepo(root), TelemetryRepo(root))


def test_catalog_provenance_is_declared():
    cat = json.loads(Path("data/kb/tsfm_catalog.json").read_text(encoding="utf-8"))
    assert "非上游" in cat["provenance"]["source"]
    assert "本地基线" in cat["provenance"]["execution_note"]
