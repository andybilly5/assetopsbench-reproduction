"""Vibration（振动）领域工具——语义重建，本机无振动测点。

## provenance（随任何引用本模块产出的分数一起声明）

论文 Table 4 只给工具面数量、未列振动工具名；本机本地数据面**没有振动测点**
（3 资产 / 45 传感器 / 136,805 读数，测点目录见 `data/normalized/sensors.csv`）。
本模块按振动分析工作流语义补齐 8 个工具，但在无测点条件下**统一抛 `DataUnavailable`**，
并给出可用替代测点 / 替代数据源建议；**任何情况下不编造频谱、轴承特征频率或 RMS 数值**。

这些实现是本仓库自建的接口骨架，不等同于上游振动 MCP server；接口对齐的是
"可枚举能力"，不是上游数值。
"""

from __future__ import annotations

from typing import Any

from ..data.repositories import AssetRepo, DataUnavailable, TelemetryRepo
from .registry import ToolRegistry, ToolSpec


def _available_sensors(
    assets: AssetRepo, telemetry: TelemetryRepo, asset_id: str | None
) -> dict[str, list[str]]:
    """列出本地真实测点目录，作为"替代测点"建议的依据。"""
    out: dict[str, list[str]] = {}
    for site in assets.sites():
        for row in assets.assets(site):
            aid = row["asset_id"]
            if asset_id and aid.strip().lower() != asset_id.strip().lower():
                continue
            try:
                out[aid] = telemetry.sensors(site, aid)
            except Exception:  # noqa: BLE001 —— 无读数资产留空表
                out[aid] = []
    return out


def _raise_unavailable(
    tool: str,
    assets: AssetRepo,
    telemetry: TelemetryRepo,
    asset_id: str | None,
    detail: str,
) -> None:
    alternatives = _available_sensors(assets, telemetry, asset_id)
    raise DataUnavailable(
        f"[vibration] {tool}: no vibration measurement point exists locally; {detail}. "
        f"Available local sensors by asset: {alternatives}. "
        "Alternative data sources: iot.history for the listed sensors, "
        "fmsr.sensors_for_failure to check which modalities are missing locally; "
        "do not infer any spectrum, bearing frequency or RMS from this answer."
    )


def build_vibration_tools(assets: AssetRepo, telemetry: TelemetryRepo) -> list[ToolSpec]:
    """构造 8 个振动工具；全部在无测点时返回 DataUnavailable。"""

    def list_points(asset_id: str | None = None) -> dict[str, Any]:
        _raise_unavailable(
            "vib.list_points", assets, telemetry, asset_id, "vibration point directory is empty"
        )
        return {}  # pragma: no cover —— 上面必抛

    def spectrum(asset_id: str, point: str) -> dict[str, Any]:
        _raise_unavailable(
            "vib.spectrum",
            assets,
            telemetry,
            asset_id,
            f"requested point {point!r} has no waveform or FFT record",
        )
        return {}  # pragma: no cover

    def bearing_frequencies(
        asset_id: str, point: str, bearing_type: str = "deep_groove"
    ) -> dict[str, Any]:
        _raise_unavailable(
            "vib.bearing_frequencies",
            assets,
            telemetry,
            asset_id,
            f"no bearing geometry or waveform for point {point!r}; "
            "BPFO/BPFI/BSF/FTF cannot be computed",
        )
        return {}  # pragma: no cover

    def envelope_rms(asset_id: str, point: str) -> dict[str, Any]:
        _raise_unavailable(
            "vib.envelope_rms",
            assets,
            telemetry,
            asset_id,
            f"no envelope-demodulated signal for point {point!r}",
        )
        return {}  # pragma: no cover

    def accel_rms(asset_id: str, point: str) -> dict[str, Any]:
        _raise_unavailable(
            "vib.accel_rms",
            assets,
            telemetry,
            asset_id,
            f"no acceleration time series for point {point!r}",
        )
        return {}  # pragma: no cover

    def threshold_check(asset_id: str, point: str) -> dict[str, Any]:
        _raise_unavailable(
            "vib.threshold_check",
            assets,
            telemetry,
            asset_id,
            f"no baseline or alarm threshold record for point {point!r}",
        )
        return {}  # pragma: no cover

    def series_by_point(asset_id: str, point: str) -> dict[str, Any]:
        _raise_unavailable(
            "vib.series_by_point",
            assets,
            telemetry,
            asset_id,
            f"vibration time series for point {point!r} is absent",
        )
        return {}  # pragma: no cover

    def active_alerts(asset_id: str | None = None) -> dict[str, Any]:
        _raise_unavailable(
            "vib.active_alerts",
            assets,
            telemetry,
            asset_id,
            "no vibration alert records are installed locally",
        )
        return {}  # pragma: no cover

    return [
        ToolSpec(
            name="vib.list_points",
            description="List installed vibration measurement points. Locally unavailable: "
            "returns DataUnavailable with the local sensor directory as alternatives.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                },
                "required": [],
            },
            handler=list_points,
            tags=["vibration", "catalog"],
            provenance="self-built skeleton; no local vibration points",
        ),
        ToolSpec(
            name="vib.spectrum",
            description="FFT spectrum for one vibration point. Locally unavailable: DataUnavailable.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "point": {"type": "string"},
                },
                "required": ["asset_id", "point"],
            },
            handler=spectrum,
            tags=["vibration", "spectrum"],
            provenance="self-built skeleton; no local waveform",
        ),
        ToolSpec(
            name="vib.bearing_frequencies",
            description="Bearing characteristic frequencies (BPFO/BPFI/BSF/FTF). Locally unavailable.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "point": {"type": "string"},
                    "bearing_type": {"type": "string"},
                },
                "required": ["asset_id", "point"],
            },
            handler=bearing_frequencies,
            tags=["vibration", "bearing"],
            provenance="self-built skeleton; no bearing geometry",
        ),
        ToolSpec(
            name="vib.envelope_rms",
            description="Envelope-demodulated RMS for one point. Locally unavailable.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "point": {"type": "string"},
                },
                "required": ["asset_id", "point"],
            },
            handler=envelope_rms,
            tags=["vibration", "bearing"],
            provenance="self-built skeleton; no envelope signal",
        ),
        ToolSpec(
            name="vib.accel_rms",
            description="Overall acceleration RMS for one point. Locally unavailable.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "point": {"type": "string"},
                },
                "required": ["asset_id", "point"],
            },
            handler=accel_rms,
            tags=["vibration", "metric"],
            provenance="self-built skeleton; no acceleration series",
        ),
        ToolSpec(
            name="vib.threshold_check",
            description="Compare a vibration metric against its alarm threshold. Locally unavailable.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "point": {"type": "string"},
                },
                "required": ["asset_id", "point"],
            },
            handler=threshold_check,
            tags=["vibration", "alert"],
            provenance="self-built skeleton; no local threshold",
        ),
        ToolSpec(
            name="vib.series_by_point",
            description="Vibration time series for one point. Locally unavailable.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "point": {"type": "string"},
                },
                "required": ["asset_id", "point"],
            },
            handler=series_by_point,
            tags=["vibration", "series"],
            provenance="self-built skeleton; no local series",
        ),
        ToolSpec(
            name="vib.active_alerts",
            description="Active vibration alerts, optionally per asset. Locally unavailable.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                },
                "required": [],
            },
            handler=active_alerts,
            tags=["vibration", "alert"],
            provenance="self-built skeleton; no local alert records",
        ),
    ]


def build_vibration_registry(assets: AssetRepo, telemetry: TelemetryRepo) -> ToolRegistry:
    return ToolRegistry(build_vibration_tools(assets, telemetry))
