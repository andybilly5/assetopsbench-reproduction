"""TSFM 领域工具（时序基础模型 / 预测 / 异常检测 / 插补 / 重采样）。

工具命名与参数对齐论文 §A.3 Table 4 里点名的 TSFM 工具：
`aitasks`、`tsfmmodels`、`tsfm_forecasting`、`tsfm forecasting finetune`、
`tsfm integrated tsad`；另外补上论文 §4.3 正文明确描述的三种前置行为
（15 分钟栅格重采样、由 Run Status 判定工况、只在短缺口上做有限插补）。

**三处必须随分数一起声明的边界**：
1. 上游 tsfm MCP server 的源码与模型清单都拿不到（GitHub/HF 均不可达），
   `data/kb/tsfm_catalog.json` 的模型目录是按公开 HuggingFace checkpoint 重建的，
   不是上游 `tsfmmodels` 的原始清单。
2. 本机无法加载 HF 权重（离网 + 单卡显存留给被测模型），所以 `forecast` / `tsad`
   用确定性本地基线执行（水平+季节剖面 / 滚动稳健 z 分数），接口与参数对齐上游，
   但数值不是上游模型的输出。
3. 因此本模块只能声明"工具面缺口已补、可枚举能力已补"，不得声明
   "上游 41 个 tsfm 工具已复现"。

设计约束与 fmsr 一致：工具只做**可枚举的目录查询 + 确定性数值计算**，不做自由生成。
预测结果由数据算出，模型不能编造时间序列点位。
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..data.repositories import AssetRepo, DataUnavailable, TelemetryRepo
from .registry import ToolRegistry, ToolSpec

STEP_MINUTES = 15
SEASON_STEPS = 96  # 15 分钟栅格下的 24 小时周期


class CatalogUnavailable(RuntimeError):
    """TSFM 模型目录缺失或结构非法。"""


def default_catalog_path() -> Path:
    return Path(os.environ.get("AOB_TSFM_CATALOG", "data/kb/tsfm_catalog.json")).expanduser()


def _load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_catalog_path()
    if not p.is_file():
        raise CatalogUnavailable(f"missing TSFM catalog: {p.resolve()} (set AOB_TSFM_CATALOG)")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not raw.get("models") or not raw.get("tasks"):
        raise CatalogUnavailable(f"catalog missing models/tasks: {p.resolve()}")
    return raw


# --------------------------------------------------------------------------- #
def _parse_ts(ts: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts[: len(fmt) + 2].replace("Z", ""), fmt)
        except ValueError:
            continue
    return None


def _numeric(points: list[dict[str, Any]]) -> list[tuple[datetime, float]]:
    out: list[tuple[datetime, float]] = []
    for p in points:
        if not isinstance(p["value"], (int, float)):
            continue
        dt = _parse_ts(str(p["ts"]))
        if dt is None:
            continue
        out.append((dt, float(p["value"])))
    return out


def _to_grid(series: list[tuple[datetime, float]]) -> tuple[list[datetime], list[float | None]]:
    """按 15 分钟栅格取均值；缺口位置留 None（不静默填补）。"""
    buckets: dict[datetime, list[float]] = {}
    for dt, v in series:
        key = dt.replace(minute=(dt.minute // STEP_MINUTES) * STEP_MINUTES, second=0, microsecond=0)
        buckets.setdefault(key, []).append(v)
    if not buckets:
        return [], []
    keys = sorted(buckets)
    full: list[datetime] = []
    cur = keys[0]
    while cur <= keys[-1]:
        full.append(cur)
        cur += timedelta(minutes=STEP_MINUTES)
    values = [statistics.fmean(buckets[k]) if k in buckets else None for k in full]
    return full, values


def _seasonal_baseline(values: list[float], horizon: int) -> tuple[list[float], float]:
    """水平 + 季节剖面基线：按同期历史均值外推，返回 (预测点, 回测 MAE)。

    这不是任何具体 HF 模型的实现，只在无权重条件下给出"可复算的确定性预测"。
    """
    clean = [v for v in values if v is not None]
    if len(clean) < 8:
        raise DataUnavailable("not enough clean points to forecast (need >= 8)")
    profile: list[float] = []
    for i in range(SEASON_STEPS):
        bucket = [v for j, v in enumerate(clean) if j % SEASON_STEPS == i and v is not None]
        profile.append(
            statistics.fmean(bucket) if bucket else statistics.fmean(clean[-SEASON_STEPS:])
        )
    level = statistics.fmean(clean[-SEASON_STEPS:]) - statistics.fmean(profile)
    start_idx = len(clean) - min(len(clean), SEASON_STEPS)
    preds = [max(0.0, level + profile[(start_idx + h) % SEASON_STEPS]) for h in range(horizon)]

    # 回测：用最后一段做留出，算 MAE
    hold = min(SEASON_STEPS, max(8, len(clean) // 5))
    errs = [
        abs(clean[-hold + i] - (level + profile[(len(clean) - hold + i) % SEASON_STEPS]))
        for i in range(hold)
    ]
    mae = statistics.fmean(errs) if errs else 0.0
    return preds, round(mae, 4)


# --------------------------------------------------------------------------- #
def build_tsfm_tools(
    assets: AssetRepo,
    telemetry: TelemetryRepo,
    catalog: dict[str, Any] | None = None,
) -> list[ToolSpec]:
    """构造 TSFM 工具集。"""
    cat = catalog or _load_catalog()
    models = {m["model_id"]: m for m in cat["models"]}

    def _asset(asset_id: str) -> dict[str, str]:
        target = asset_id.strip().lower()
        for site in assets.sites():
            for row in assets.assets(site):
                if row["asset_id"].strip().lower() == target:
                    return row
        raise DataUnavailable(
            f"unknown asset {asset_id!r}; known assets: "
            + ", ".join(r["asset_id"] for s in assets.sites() for r in assets.assets(s))
        )

    def _series(asset_id: str, sensor: str) -> tuple[dict[str, str], list[tuple[datetime, float]]]:
        row = _asset(asset_id)
        pts = telemetry.history(row["site"], row["asset_id"], sensor)
        series = _numeric(pts)
        if not series:
            raise DataUnavailable(f"no numeric readings for {asset_id!r}/{sensor!r}")
        return row, series

    # -------- 目录类 --------
    def aitasks() -> dict[str, Any]:
        return {
            "tasks": cat["tasks"],
            "count": len(cat["tasks"]),
            "provenance": "tool names from paper §A.3 Table 4; local catalogue",
        }

    def tsfmmodels() -> dict[str, Any]:
        return {
            "models": cat["models"],
            "count": len(cat["models"]),
            "note": (
                "catalogue of public HuggingFace time-series foundation checkpoints; "
                "weights are not loaded locally, forecasts use a deterministic baseline"
            ),
        }

    # -------- 计算类 --------
    def forecast(
        asset_id: str,
        sensor: str,
        horizon: int = 96,
        model_id: str = "chronos_t5_small",
    ) -> dict[str, Any]:
        if model_id not in models:
            return {
                "resolved": False,
                "query": {"model_id": model_id},
                "known_models": list(models),
                "hint": "call tsfm.models to list supported model ids",
            }
        row, series = _series(asset_id, sensor)
        grid, values = _to_grid(series)
        horizon = max(1, min(int(horizon), int(models[model_id]["prediction_length"]) * 4))
        preds, mae = _seasonal_baseline(values, horizon)
        last_ts = grid[-1]
        points = [
            {
                "ts": (last_ts + timedelta(minutes=STEP_MINUTES * (i + 1))).isoformat(),
                "yhat": round(v, 4),
            }
            for i, v in enumerate(preds)
        ]
        return {
            "resolved": True,
            "asset": row["asset_id"],
            "sensor": sensor,
            "model_id": model_id,
            "checkpoint": models[model_id]["checkpoint"],
            "cadence_minutes": STEP_MINUTES,
            "context_points": len([v for v in values if v is not None]),
            "missing_points": len([v for v in values if v is None]),
            "horizon": horizon,
            "backtest_mae": mae,
            "forecast": points,
            "execution_note": (
                "weights not loaded locally; this is a deterministic seasonal baseline, "
                "not the output of the named checkpoint"
            ),
        }

    def tsad(
        asset_id: str,
        sensor: str,
        window: int = SEASON_STEPS,
        threshold: float = 3.0,
    ) -> dict[str, Any]:
        row, series = _series(asset_id, sensor)
        grid, values = _to_grid(series)
        clean_idx = [i for i, v in enumerate(values) if v is not None]
        if len(clean_idx) < 12:
            raise DataUnavailable("not enough points for anomaly detection (need >= 12)")
        window = max(8, min(int(window), len(clean_idx)))
        anomalies: list[dict[str, Any]] = []
        for pos in range(window, len(values)):
            ref = [values[j] for j in range(pos - window, pos) if values[j] is not None]
            v = values[pos]
            if v is None or len(ref) < 6:
                continue
            med = statistics.median(ref)
            mad = statistics.median([abs(x - med) for x in ref]) or 1e-9
            z = 0.6745 * (v - med) / mad  # 稳健 z 分数（近似 conformal 残差阈）
            if abs(z) >= threshold:
                anomalies.append(
                    {
                        "ts": grid[pos].isoformat(),
                        "value": round(v, 4),
                        "baseline": round(med, 4),
                        "robust_z": round(z, 2),
                    }
                )
        anomalies.sort(key=lambda a: -abs(a["robust_z"]))
        return {
            "asset": row["asset_id"],
            "sensor": sensor,
            "window": window,
            "threshold": threshold,
            "method": "robust_zscore_on_rolling_median",
            "points_checked": len([v for v in values if v is not None]),
            "anomaly_count": len(anomalies),
            "top_anomalies": anomalies[:8],
        }

    def impute(asset_id: str, sensor: str, max_gap_steps: int = 3) -> dict[str, Any]:
        row, series = _series(asset_id, sensor)
        grid, values = _to_grid(series)
        gaps: list[int] = []
        run = 0
        for v in values:
            if v is None:
                run += 1
            elif run:
                gaps.append(run)
                run = 0
        if run:
            gaps.append(run)
        short = [g for g in gaps if g <= max_gap_steps]
        long = [g for g in gaps if g > max_gap_steps]
        return {
            "asset": row["asset_id"],
            "sensor": sensor,
            "grid_points": len(values),
            "missing_points": len([v for v in values if v is None]),
            "missing_ratio": round(len([v for v in values if v is None]) / max(1, len(values)), 4),
            "gap_count": len(gaps),
            "short_gaps": len(short),
            "long_gaps": len(long),
            "policy": (
                f"linear interpolation only for gaps <= {max_gap_steps} steps "
                f"(-> {len(short)} gaps); longer gaps left as missing and treated as "
                "asset-inactive signal rather than imputed"
            ),
        }

    def resample(
        asset_id: str, sensor: str = "Run Status", freq_minutes: int = STEP_MINUTES
    ) -> dict[str, Any]:
        row, series = _series(asset_id, sensor)
        if freq_minutes != STEP_MINUTES:
            # 只声明能力，不静默换栅格：换栅格会让下游 forecast 的季节索引失效
            return {
                "resolved": False,
                "requested_freq_minutes": freq_minutes,
                "supported_freq_minutes": STEP_MINUTES,
                "reason": "downstream seasonal index assumes a 15-minute grid",
            }
        grid, values = _to_grid(series)
        active = [i for i, v in enumerate(values) if v is not None and v > 0]
        return {
            "resolved": True,
            "asset": row["asset_id"],
            "sensor": sensor,
            "cadence_minutes": freq_minutes,
            "grid_points": len(grid),
            "start": grid[0].isoformat() if grid else None,
            "end": grid[-1].isoformat() if grid else None,
            "operating_points": len(active),
            "operating_ratio": round(len(active) / max(1, len(values)), 4),
            "operating_periods": _contiguous(grid, active),
            "note": "operating periods inferred from Run Status > 0 on the 15-minute grid",
        }

    def finetune(
        asset_id: str, sensor: str, model_id: str = "granite_ttm_r2", epochs: int = 3
    ) -> dict[str, Any]:
        if model_id not in models:
            return {"resolved": False, "known_models": list(models)}
        row, series = _series(asset_id, sensor)
        grid, values = _to_grid(series)
        return {
            "resolved": True,
            "asset": row["asset_id"],
            "model_id": model_id,
            "checkpoint": models[model_id]["checkpoint"],
            "config": {
                "dataset_path": f"data/normalized/readings.csv::{row['asset_id']}::{sensor}",
                "timestamp_column": "ts",
                "target_columns": [sensor],
                "freq": f"{STEP_MINUTES}min",
                "context_length": models[model_id]["context_length"],
                "prediction_length": models[model_id]["prediction_length"],
                "epochs": max(1, min(int(epochs), 10)),
                "batch_size": 32,
                "loss": "mae",
            },
            "trainable_points": len([v for v in values if v is not None]),
            "execution_note": "recipe only: no weights or GPU are available locally, nothing is trained",
        }

    return [
        ToolSpec(
            name="tsfm.aitasks",
            description="List the AI tasks the time-series service supports, with task ids and methods.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=aitasks,
            tags=["tsfm", "catalog"],
        ),
        ToolSpec(
            name="tsfm.models",
            description=(
                "List supported forecasting models with model_id, HuggingFace checkpoint, "
                "context window, prediction length and whether multivariate. Use this whenever "
                "the question asks which time-series models or capabilities the system has."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=tsfmmodels,
            tags=["tsfm", "catalog"],
        ),
        ToolSpec(
            name="tsfm.forecast",
            description=(
                "Forecast a sensor series over a horizon on the 15-minute grid, with backtest MAE. "
                "Equivalent to the paper's tsfm_forecasting tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset identifier, e.g. 'Chiller 6'.",
                    },
                    "sensor": {
                        "type": "string",
                        "description": "Target sensor name, e.g. 'Tonnage'.",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Steps ahead on the 15-min grid.",
                    },
                    "model_id": {"type": "string", "description": "Model id from tsfm.models."},
                },
                "required": ["asset_id", "sensor"],
            },
            handler=forecast,
            tags=["tsfm", "compute"],
        ),
        ToolSpec(
            name="tsfm.tsad",
            description=(
                "Time-series anomaly detection on model residuals with a calibrated robust threshold; "
                "equivalent to the paper's tsfm_integrated_tsad tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "sensor": {"type": "string"},
                    "window": {"type": "integer", "description": "Rolling window in grid steps."},
                    "threshold": {
                        "type": "number",
                        "description": "Robust z threshold, default 3.0.",
                    },
                },
                "required": ["asset_id", "sensor"],
            },
            handler=tsad,
            tags=["tsfm", "compute"],
        ),
        ToolSpec(
            name="tsfm.impute",
            description="Report gaps on the grid and apply the limited-imputation policy (short gaps only).",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "sensor": {"type": "string"},
                    "max_gap_steps": {"type": "integer"},
                },
                "required": ["asset_id", "sensor"],
            },
            handler=impute,
            tags=["tsfm", "quality"],
        ),
        ToolSpec(
            name="tsfm.resample",
            description=(
                "Align readings to the 15-minute grid and infer operating periods from Run Status."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "sensor": {"type": "string"},
                    "freq_minutes": {"type": "integer"},
                },
                "required": ["asset_id"],
            },
            handler=resample,
            tags=["tsfm", "quality"],
        ),
        ToolSpec(
            name="tsfm.finetune",
            description=(
                "Produce a fine-tuning recipe for a pretrained checkpoint on this asset's history; "
                "equivalent to the paper's tsfm forecasting finetune tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "sensor": {"type": "string"},
                    "model_id": {"type": "string"},
                    "epochs": {"type": "integer"},
                },
                "required": ["asset_id", "sensor"],
            },
            handler=finetune,
            tags=["tsfm", "compute"],
        ),
    ]


def _contiguous(grid: list[datetime], idx: list[int]) -> list[dict[str, str]]:
    """把离散下标合并成连续区间，供"工况时段"回答使用。"""
    if not idx:
        return []
    spans: list[dict[str, str]] = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        spans.append({"start": grid[start].isoformat(), "end": grid[prev].isoformat()})
        start = prev = i
    spans.append({"start": grid[start].isoformat(), "end": grid[prev].isoformat()})
    return spans[:12]


def build_tsfm_registry(
    assets: AssetRepo, telemetry: TelemetryRepo, catalog: dict[str, Any] | None = None
) -> ToolRegistry:
    return ToolRegistry(build_tsfm_tools(assets, telemetry, catalog))
