"""IoT 领域工具。

对齐上游 `iot` MCP server 的核心读工具：sites / assets / sensors / history。
工具名用 `<server>.<tool>` 前缀，便于将来把 FMSR、TSFM、WO 并列接入同一注册表。
"""

from __future__ import annotations

from typing import Any

from ..data.repositories import AssetRepo, TelemetryRepo
from .registry import ToolRegistry, ToolSpec

_SITE_PROP = {
    "type": "string",
    "description": "Site identifier, e.g. MAIN.",
}


def build_iot_tools(assets: AssetRepo, telemetry: TelemetryRepo) -> list[ToolSpec]:
    """构造 IoT 工具集。"""

    def sites() -> list[str]:
        return assets.sites()

    def list_assets(site_name: str) -> list[dict[str, Any]]:
        return assets.assets(site_name)

    def sensors(site_name: str, asset_id: str) -> list[str]:
        return telemetry.sensors(site_name, asset_id)

    def history(
        site_name: str,
        asset_id: str,
        sensor: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        values = telemetry.history(site_name, asset_id, sensor, start_date, end_date)
        return {
            "site": site_name,
            "asset_id": asset_id,
            "sensor": sensor,
            "count": len(values),
            "start": values[0]["ts"],
            "end": values[-1]["ts"],
            "values": values,
        }

    return [
        ToolSpec(
            name="iot.sites",
            description="List all sites known to the IoT service.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=sites,
            tags=["iot", "catalog"],
        ),
        ToolSpec(
            name="iot.assets",
            description="List all assets located at a given site.",
            parameters={
                "type": "object",
                "properties": {"site_name": _SITE_PROP},
                "required": ["site_name"],
            },
            handler=list_assets,
            tags=["iot", "catalog"],
        ),
        ToolSpec(
            name="iot.sensors",
            description="List the sensors that actually report measurements for an asset.",
            parameters={
                "type": "object",
                "properties": {
                    "site_name": _SITE_PROP,
                    "asset_id": {
                        "type": "string",
                        "description": "Asset identifier, e.g. 'Chiller 6'.",
                    },
                },
                "required": ["site_name", "asset_id"],
            },
            handler=sensors,
            tags=["iot", "catalog"],
        ),
        ToolSpec(
            name="iot.history",
            description=(
                "Return sensor readings for an asset within an inclusive date window. "
                "Dates are ISO-8601 (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "site_name": _SITE_PROP,
                    "asset_id": {"type": "string", "description": "Asset identifier."},
                    "sensor": {"type": "string", "description": "Sensor/metric name."},
                    "start_date": {"type": "string", "description": "Inclusive lower bound."},
                    "end_date": {"type": "string", "description": "Inclusive upper bound."},
                },
                "required": ["site_name", "asset_id", "sensor"],
            },
            handler=history,
            tags=["iot", "telemetry"],
        ),
    ]


def build_registry(assets: AssetRepo, telemetry: TelemetryRepo) -> ToolRegistry:
    """只装 IoT 工具的注册表；后续按需 merge FMSR/TSFM/WO。"""
    return ToolRegistry(build_iot_tools(assets, telemetry))
