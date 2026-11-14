"""FMSR 领域工具（故障模式与传感器推理）。

上游 `fmsr` MCP server 共 4 个工具，但这 4 个工具的源码在本机拿不到（GitHub 直连超时）。
因此本模块按 18 条 FMSR 场景的实际提问需求重建了 4 个等价能力的工具：

    fmsr.failure_modes        资产有哪些故障模式
    fmsr.failures_by_sensor   某个传感器（或传感器类别）能检出哪些故障
    fmsr.sensors_for_failure  某个故障应该看哪些传感器（含"本机缺失的模态"）
    fmsr.symptom_cause        某个传感器往某方向变化，最可能是哪种故障

**这是自建知识库，不是上游知识库**（见 `data/kb/fmsr_chillers.json` 的 `provenance`）。
凡引用本模块产出的分数，必须同时声明这一偏差。

设计约束：工具只做"查表 + 过滤"，不做自由生成。因为 FMSR 场景的正确答案是
有界的故障清单，把生成权交给模型会让"编造故障名"无法与"查到故障名"区分开。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..data.repositories import AssetRepo, DataUnavailable
from .registry import ToolRegistry, ToolSpec


class KnowledgeUnavailable(RuntimeError):
    """FMSR 知识库文件缺失或结构非法。"""


def default_kb_path() -> Path:
    return Path(os.environ.get("AOB_FMSR_KB", "data/kb/fmsr_chillers.json")).expanduser()


class FmsrKnowledge:
    """故障模式知识库的只读视图。"""

    def __init__(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else default_kb_path()
        if not p.is_file():
            raise KnowledgeUnavailable(
                f"missing FMSR knowledge base: {p.resolve()} (set AOB_FMSR_KB)"
            )
        raw = json.loads(p.read_text(encoding="utf-8"))
        self.sensor_classes: dict[str, list[str]] = raw.get("sensor_classes", {})
        self.failure_modes: list[dict[str, Any]] = raw.get("failure_modes", [])
        self.symptom_causes: list[dict[str, Any]] = raw.get("symptom_causes", [])
        if not self.failure_modes:
            raise KnowledgeUnavailable(f"knowledge base has no failure_modes: {p.resolve()}")

    # ---------------------------------------------------------------- #
    def by_name(self, name: str) -> dict[str, Any] | None:
        """故障名模糊匹配：精确 → 子串 → 词元覆盖。"""
        target = _norm(name)
        for fm in self.failure_modes:
            if _norm(fm["name"]) == target:
                return fm
        for fm in self.failure_modes:
            if target in _norm(fm["name"]) or _norm(fm["name"]) in target:
                return fm
        tokens = set(target.split())
        best, best_score = None, 0.0
        for fm in self.failure_modes:
            ft = set(_norm(fm["name"]).split())
            if not ft:
                continue
            score = len(tokens & ft) / len(ft)
            if score > best_score:
                best, best_score = fm, score
        return best if best_score >= 0.5 else None

    def sensor_class_of(self, sensor: str) -> str | None:
        key = _norm(sensor)
        for cls, members in self.sensor_classes.items():
            ckey = _norm(cls)
            # "temperature sensors" / "temperature" / "vibration sensor" 都要认出来
            if key == ckey or key.rstrip("s") == ckey.rstrip("s"):
                return cls
            if key.startswith(ckey) or key.startswith(ckey.rstrip("s")):
                return cls
            for m in members:
                if key == _norm(m) or key in _norm(m) or _norm(m) in key:
                    return cls
        return None

    def class_sensors(self, cls: str) -> list[str]:
        return list(self.sensor_classes.get(cls, []))

    def symptom(self, sensor: str, direction: str) -> list[dict[str, Any]]:
        key = _norm(sensor)
        out: list[dict[str, Any]] = []
        for row in self.symptom_causes:
            same = (
                _norm(row["sensor"]) == key
                or key in _norm(row["sensor"])
                or _norm(row["sensor"]) in key
            )
            if not same:
                continue
            if direction and _norm(row["direction"]) != _norm(direction):
                continue
            out.extend(row["candidates"])
        return out


def _norm(text: str) -> str:
    return " ".join(str(text).lower().replace("-", " ").replace("_", " ").split())


def _find_asset(assets: AssetRepo, asset_id: str) -> dict[str, str]:
    """按 asset_id 在有权限的 site 里找资产；找不到抛 DataUnavailable。"""
    aid = asset_id.strip().lower()
    for site in assets.sites():
        for row in assets.assets(site):
            if row["asset_id"].strip().lower() == aid:
                return row
    raise DataUnavailable(f"unknown asset {asset_id!r}; known sites: {', '.join(assets.sites())}")


def build_fmsr_tools(
    assets: AssetRepo,
    kb: FmsrKnowledge,
    available_sensors: dict[str, list[str]] | None = None,
) -> list[ToolSpec]:
    """构造 4 个 FMSR 工具。

    `available_sensors` 用来回答"本机到底有哪些测点"，是 FMSR 场景里最常见的
    隐含约束（例如问振动能否检出故障，本机根本没有振动测点）。
    """

    def _sensors(asset_id: str) -> list[str]:
        if available_sensors is None:
            return []
        return available_sensors.get(asset_id.strip().lower(), [])

    def _resolve(asset_id: str) -> dict[str, str]:
        return _find_asset(assets, asset_id)

    def failure_modes(asset_id: str) -> dict[str, Any]:
        asset = _resolve(asset_id)
        modes = [
            {
                "failure_mode": fm["name"],
                "description": fm["description"],
                "detected_by": fm["detected_by"],
                "sensor_classes": fm["sensor_classes"],
            }
            for fm in kb.failure_modes
        ]
        return {
            "asset": asset["asset_id"],
            "asset_class": asset.get("asset_class", ""),
            "count": len(modes),
            "failure_modes": modes,
            "available_sensors": _sensors(asset_id),
        }

    def failures_by_sensor(asset_id: str, sensor: str) -> dict[str, Any]:
        asset = _resolve(asset_id)
        cls = kb.sensor_class_of(sensor)
        matched: list[dict[str, Any]] = []
        for fm in kb.failure_modes:
            if cls and cls in fm["sensor_classes"]:
                hit = [s for s in fm["detected_by"] if kb.sensor_class_of(s) == cls]
                if hit:
                    matched.append({"failure_mode": fm["name"], "matched_sensors": hit})
                continue
            direct = [
                s
                for s in fm["detected_by"]
                if _norm(sensor) == _norm(s) or _norm(sensor) in _norm(s)
            ]
            if direct:
                matched.append({"failure_mode": fm["name"], "matched_sensors": direct})
        if cls and not kb.class_sensors(cls):
            return {
                "asset": asset["asset_id"],
                "query": sensor,
                "matched_as": f"{cls} (class)",
                "monitoring_possible": False,
                "reason": (
                    f"本机 IoT 目录没有 {cls} 类测点，无法据此监测任何故障；"
                    "该判断只代表本地数据面，不代表上游 AssetOpsBench 的工具面"
                ),
                "failure_modes": [],
            }
        return {
            "asset": asset["asset_id"],
            "query": sensor,
            "matched_as": f"{cls} (class)" if cls else "exact sensor name",
            "monitoring_possible": True,
            "count": len(matched),
            "failure_modes": matched,
        }

    def sensors_for_failure(asset_id: str, failure_mode: str) -> dict[str, Any]:
        asset = _resolve(asset_id)
        fm = kb.by_name(failure_mode)
        if fm is None:
            return {
                "asset": asset["asset_id"],
                "query": failure_mode,
                "resolved": False,
                "known_failure_modes": [m["name"] for m in kb.failure_modes],
            }
        avail = _sensors(asset_id)
        usable = [s for s in fm["detected_by"] if s in avail]
        missing: list[str] = []
        for cls in fm["sensor_classes"]:
            if not kb.class_sensors(cls):
                missing.append(f"{cls} (本机无此类测点)")
            elif not any(kb.sensor_class_of(s) == cls for s in avail):
                missing.append(f"{cls} (本机该类测点不在该资产的目录里)")
        return {
            "asset": asset["asset_id"],
            "query": failure_mode,
            "resolved": True,
            "failure_mode": fm["name"],
            "description": fm["description"],
            "relevant_sensors": fm["detected_by"],
            "usable_sensors_local": usable,
            "missing_modalities": missing,
            "priority_sensor": fm.get("priority_sensor", ""),
            "temporal_behavior": fm.get("temporal", []),
            "monitoring_plan": fm.get("monitoring_plan", []),
        }

    def symptom_cause(asset_id: str, sensor: str, direction: str = "rises") -> dict[str, Any]:
        asset = _resolve(asset_id)
        d = _norm(direction)
        if d in ("drop", "decrease", "decreases", "fall", "falls", "down"):
            d = "drops"
        elif d in ("rise", "increase", "increases", "up"):
            d = "rises"
        rows = kb.symptom(sensor, d)
        if not rows:
            return {
                "asset": asset["asset_id"],
                "query": {"sensor": sensor, "direction": d},
                "matched": False,
                "hint": "该测点在该方向没有登记因果候选，可换 direction 或不指定方向",
            }
        return {
            "asset": asset["asset_id"],
            "query": {"sensor": sensor, "direction": d},
            "matched": True,
            "count": len(rows),
            "candidates": [
                {
                    "failure_mode": r["failure_mode"],
                    "confidence": r["confidence"],
                    "evidence_note": r["note"],
                    "temporal_behavior": [
                        t
                        for fm in kb.failure_modes
                        if fm["name"] == r["failure_mode"]
                        for t in fm.get("temporal", [])
                    ],
                }
                for r in rows
            ],
        }

    return [
        ToolSpec(
            name="fmsr.failure_modes",
            description=(
                "List every known failure mode of an asset, with the sensor classes that "
                "can detect each one. Use this when the question asks 'what can go wrong'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset identifier, e.g. 'Chiller 6'.",
                    },
                },
                "required": ["asset_id"],
            },
            handler=failure_modes,
            tags=["fmsr", "knowledge"],
        ),
        ToolSpec(
            name="fmsr.failures_by_sensor",
            description=(
                "Given a sensor name or a sensor class (temperature / power / flow / status / "
                "vibration), list the failure modes detectable from it. Also reports when the "
                "class has no installed sensor locally."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "Asset identifier."},
                    "sensor": {
                        "type": "string",
                        "description": "Sensor name or sensor class, e.g. 'Supply Temperature' or 'temperature sensors'.",
                    },
                },
                "required": ["asset_id", "sensor"],
            },
            handler=failures_by_sensor,
            tags=["fmsr", "detectability"],
        ),
        ToolSpec(
            name="fmsr.sensors_for_failure",
            description=(
                "Given a failure mode, return the sensors that monitor it, which of them are "
                "actually usable on this asset, which modalities are missing locally, the "
                "temporal signature of each sensor and the recommended monitoring plan."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "Asset identifier."},
                    "failure_mode": {
                        "type": "string",
                        "description": "Failure mode name, e.g. 'Compressor Overheating'.",
                    },
                },
                "required": ["asset_id", "failure_mode"],
            },
            handler=sensors_for_failure,
            tags=["fmsr", "diagnosis"],
        ),
        ToolSpec(
            name="fmsr.symptom_cause",
            description=(
                "Given a sensor and the direction it moved (rises / drops), return the ranked "
                "candidate failure modes with confidence and the distinguishing evidence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "Asset identifier."},
                    "sensor": {"type": "string", "description": "Sensor name."},
                    "direction": {
                        "type": "string",
                        "enum": ["rises", "drops"],
                        "description": "Observed direction of change.",
                    },
                },
                "required": ["asset_id", "sensor", "direction"],
            },
            handler=symptom_cause,
            tags=["fmsr", "diagnosis"],
        ),
    ]


def build_fmsr_registry(assets: AssetRepo, kb: FmsrKnowledge | None = None) -> ToolRegistry:
    return ToolRegistry(build_fmsr_tools(assets, kb or FmsrKnowledge()))
