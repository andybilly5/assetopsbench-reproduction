"""WO（工单）领域工具——语义重建，自建最小工单库。

## provenance（随任何引用本模块产出的分数一起声明）

论文 Table 4 只给工具面数量、未列工单工具名；上游 AssetOpsBench 约 4.2K 条工单记录
在本机不可获取。本模块按工单工作流语义补齐 15 个工具，数据源是
`data/wo/synthetic_workorders.json` 的 **26 条合成工单**（生成方式与边界见同目录
`PROVENANCE.json`，`source_origin=self-built`）。

**这些实现是本仓库自建的等价能力，不等同于上游工单 MCP server 的实现**：
工具只做"查表 + 过滤 + 确定性统计"，不做自由生成，不声称数值/字段与上游一致。

## 本地边界

本地数据面只有 3 资产 / 45 传感器 / 136,805 读数，没有上游工单记录；合成工单的
`related_sensor` 来自本地传感器目录，其余工单字段为模板派生。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..data.repositories import AssetRepo, DataUnavailable
from .registry import ToolRegistry, ToolSpec


class WorkorderUnavailable(RuntimeError):
    """工单库文件缺失或结构非法。"""


def default_db_path() -> Path:
    return Path(os.environ.get("AOB_WO_DB", "data/wo/synthetic_workorders.json")).expanduser()


class WorkorderRepo:
    """合成工单库的只读视图。"""

    def __init__(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else default_db_path()
        if not p.is_file():
            raise WorkorderUnavailable(f"missing workorder database: {p.resolve()} (set AOB_WO_DB)")
        raw = json.loads(p.read_text(encoding="utf-8"))
        self.workorders: list[dict[str, Any]] = raw.get("workorders", [])
        if not self.workorders:
            raise WorkorderUnavailable(f"workorder database has no records: {p.resolve()}")
        self._by_id = {w["wo_id"]: w for w in self.workorders}

    def get(self, wo_id: str) -> dict[str, Any]:
        row = self._by_id.get(wo_id.strip())
        if row is None:
            raise DataUnavailable(
                f"unknown workorder {wo_id!r}; known ids: {', '.join(sorted(self._by_id))}"
            )
        return row

    def filter(
        self,
        status: str | None = None,
        priority: str | None = None,
        wtype: str | None = None,
        asset_id: str | None = None,
        keyword: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        out = self.workorders
        if status:
            out = [w for w in out if w["status"] == status.strip().lower()]
        if priority:
            out = [w for w in out if w["priority"] == priority.strip().lower()]
        if wtype:
            out = [w for w in out if w["type"] == wtype.strip().lower()]
        if asset_id:
            aid = asset_id.strip().lower()
            out = [w for w in out if w["asset_id"].strip().lower() == aid]
        if keyword:
            needle = keyword.strip().lower()
            out = [
                w
                for w in out
                if needle in w["symptom"].lower()
                or needle in w["description"].lower()
                or needle in w["wo_id"].lower()
            ]
        if start:
            out = [w for w in out if w["reported_at"] >= start]
        if end:
            out = [w for w in out if w["reported_at"] <= end]
        return out


def _find_asset(assets: AssetRepo, asset_id: str) -> dict[str, str]:
    aid = asset_id.strip().lower()
    for site in assets.sites():
        for row in assets.assets(site):
            if row["asset_id"].strip().lower() == aid:
                return row
    raise DataUnavailable(
        f"unknown asset {asset_id!r}; known: "
        + ", ".join(r["asset_id"] for s in assets.sites() for r in assets.assets(s))
    )


def _sla_status(w: dict[str, Any]) -> str:
    if w["closed_at"] is None:
        return "open"
    return "on_time" if w["closed_at"] <= w["sla_due"] else "overdue"


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for w in rows:
        out[w[key]] = out.get(w[key], 0) + 1
    return dict(sorted(out.items()))


def build_wo_tools(assets: AssetRepo, db: WorkorderRepo | None = None) -> list[ToolSpec]:
    """构造 15 个 WO 工具。"""
    repo = db or WorkorderRepo()

    def _brief(w: dict[str, Any]) -> dict[str, Any]:
        return {
            "wo_id": w["wo_id"],
            "asset_id": w["asset_id"],
            "type": w["type"],
            "status": w["status"],
            "priority": w["priority"],
            "defect_category": w["defect_category"],
            "symptom": w["symptom"],
            "related_sensor": w["related_sensor"],
            "reported_at": w["reported_at"],
            "failure_mode": w["failure_mode"],
        }

    # -------- 检索类 --------
    def list_workorders(
        status: str | None = None,
        priority: str | None = None,
        wtype: str | None = None,
    ) -> dict[str, Any]:
        rows = repo.filter(status=status, priority=priority, wtype=wtype)
        return {"count": len(rows), "workorders": [_brief(w) for w in rows]}

    def get(wo_id: str) -> dict[str, Any]:
        return repo.get(wo_id)

    def by_asset(asset_id: str, status: str | None = None) -> dict[str, Any]:
        asset = _find_asset(assets, asset_id)
        rows = repo.filter(asset_id=asset["asset_id"], status=status)
        return {
            "asset": asset["asset_id"],
            "asset_class": asset["asset_class"],
            "count": len(rows),
            "workorders": [_brief(w) for w in rows],
        }

    def by_status(status: str) -> dict[str, Any]:
        rows = repo.filter(status=status)
        return {"status": status, "count": len(rows), "workorders": [_brief(w) for w in rows]}

    def by_priority(priority: str) -> dict[str, Any]:
        rows = repo.filter(priority=priority)
        return {"priority": priority, "count": len(rows), "workorders": [_brief(w) for w in rows]}

    def search(keyword: str) -> dict[str, Any]:
        rows = repo.filter(keyword=keyword)
        return {"keyword": keyword, "count": len(rows), "workorders": [_brief(w) for w in rows]}

    def by_date_range(start: str, end: str) -> dict[str, Any]:
        rows = repo.filter(start=start, end=end)
        return {
            "window": {"start": start, "end": end},
            "count": len(rows),
            "workorders": [_brief(w) for w in rows],
        }

    # -------- 统计类 --------
    def asset_summary() -> dict[str, Any]:
        return {
            "counts_by_asset": _summary(repo.workorders, "asset_id"),
            "total": len(repo.workorders),
        }

    def status_summary() -> dict[str, Any]:
        return {
            "counts_by_status": _summary(repo.workorders, "status"),
            "total": len(repo.workorders),
        }

    def priority_summary() -> dict[str, Any]:
        return {
            "counts_by_priority": _summary(repo.workorders, "priority"),
            "total": len(repo.workorders),
        }

    # -------- 归类 / SLA / 停机 / 备件 / 故障模式 --------
    def defect_classify(wo_id: str) -> dict[str, Any]:
        w = repo.get(wo_id)
        return {
            "wo_id": w["wo_id"],
            "asset_id": w["asset_id"],
            "defect_category": w["defect_category"],
            "symptom": w["symptom"],
            "related_sensor": w["related_sensor"],
            "evidence_note": "category and sensor come from the self-built synthetic record",
        }

    def sla_status(wo_id: str | None = None) -> dict[str, Any]:
        if wo_id:
            w = repo.get(wo_id)
            return {
                "wo_id": w["wo_id"],
                "sla_due": w["sla_due"],
                "closed_at": w["closed_at"],
                "sla_status": _sla_status(w),
            }
        rows = [repo.get(w["wo_id"]) for w in repo.workorders]
        return {
            "counts": _summary([{"k": _sla_status(w)} for w in rows], "k"),
            "records": [{"wo_id": w["wo_id"], "sla_status": _sla_status(w)} for w in rows],
        }

    def downtime(asset_id: str | None = None) -> dict[str, Any]:
        rows = repo.filter(asset_id=asset_id) if asset_id else repo.workorders
        total = round(sum(w["downtime_hours"] or 0 for w in rows), 2)
        return {
            "asset_id": asset_id,
            "count": len(rows),
            "downtime_hours_total": total,
            "records": [
                {"wo_id": w["wo_id"], "downtime_hours": w["downtime_hours"], "status": w["status"]}
                for w in rows
            ],
        }

    def parts_linked(wo_id: str) -> dict[str, Any]:
        w = repo.get(wo_id)
        return {"wo_id": w["wo_id"], "parts": w["parts"], "parts_count": len(w["parts"])}

    def to_failure_mode(wo_id: str) -> dict[str, Any]:
        w = repo.get(wo_id)
        return {
            "wo_id": w["wo_id"],
            "asset_id": w["asset_id"],
            "failure_mode": w["failure_mode"],
            "related_sensor": w["related_sensor"],
            "defect_category": w["defect_category"],
            "note": "mapping is self-built; cross-check with fmsr tools before claiming a root cause",
        }

    return [
        ToolSpec(
            name="wo.list",
            description="List workorders, optionally filtered by status, priority or type. "
            "Records come from the self-built synthetic workorder database.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "in_progress", "resolved", "closed"],
                    },
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "wtype": {
                        "type": "string",
                        "enum": ["repair", "inspection", "preventive", "emergency"],
                    },
                },
                "required": [],
            },
            handler=list_workorders,
            tags=["wo", "search"],
            provenance="self-built synthetic workorder DB; not upstream 4.2K records",
        ),
        ToolSpec(
            name="wo.get",
            description="Return the full record of one workorder by wo_id.",
            parameters={
                "type": "object",
                "properties": {
                    "wo_id": {"type": "string", "description": "Workorder id, e.g. 'WO-2020-001'."},
                },
                "required": ["wo_id"],
            },
            handler=get,
            tags=["wo", "search"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.by_asset",
            description="List workorders attached to one asset, optionally by status.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["open", "in_progress", "resolved", "closed"],
                    },
                },
                "required": ["asset_id"],
            },
            handler=by_asset,
            tags=["wo", "search"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.by_status",
            description="List workorders in one status (open / in_progress / resolved / closed).",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "in_progress", "resolved", "closed"],
                    },
                },
                "required": ["status"],
            },
            handler=by_status,
            tags=["wo", "search"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.by_priority",
            description="List workorders at one priority (low / medium / high / critical).",
            parameters={
                "type": "object",
                "properties": {
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
                "required": ["priority"],
            },
            handler=by_priority,
            tags=["wo", "search"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.search",
            description="Free-text search over symptom, description and wo_id.",
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                },
                "required": ["keyword"],
            },
            handler=search,
            tags=["wo", "search"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.by_date_range",
            description="List workorders reported within an inclusive ISO-8601 date window.",
            parameters={
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
                "required": ["start", "end"],
            },
            handler=by_date_range,
            tags=["wo", "search"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.asset_summary",
            description="Count workorders per asset.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=asset_summary,
            tags=["wo", "stats"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.status_summary",
            description="Count workorders per status.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=status_summary,
            tags=["wo", "stats"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.priority_summary",
            description="Count workorders per priority.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=priority_summary,
            tags=["wo", "stats"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.defect_classify",
            description="Return the defect category, symptom and related sensor for one workorder.",
            parameters={
                "type": "object",
                "properties": {
                    "wo_id": {"type": "string"},
                },
                "required": ["wo_id"],
            },
            handler=defect_classify,
            tags=["wo", "classification"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.sla_status",
            description="SLA status of one workorder (on_time / overdue / open), or counts across all.",
            parameters={
                "type": "object",
                "properties": {
                    "wo_id": {"type": "string"},
                },
                "required": [],
            },
            handler=sla_status,
            tags=["wo", "sla"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.downtime",
            description="Downtime hours per workorder, optionally aggregated for one asset.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                },
                "required": [],
            },
            handler=downtime,
            tags=["wo", "downtime"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.parts_linked",
            description="Spare parts associated with one workorder.",
            parameters={
                "type": "object",
                "properties": {
                    "wo_id": {"type": "string"},
                },
                "required": ["wo_id"],
            },
            handler=parts_linked,
            tags=["wo", "parts"],
            provenance="self-built synthetic workorder DB",
        ),
        ToolSpec(
            name="wo.to_failure_mode",
            description="Map a workorder to its self-built failure-mode candidate and related sensor.",
            parameters={
                "type": "object",
                "properties": {
                    "wo_id": {"type": "string"},
                },
                "required": ["wo_id"],
            },
            handler=to_failure_mode,
            tags=["wo", "mapping"],
            provenance="self-built synthetic workorder DB",
        ),
    ]


def build_wo_registry(assets: AssetRepo, db: WorkorderRepo | None = None) -> ToolRegistry:
    return ToolRegistry(build_wo_tools(assets, db))
