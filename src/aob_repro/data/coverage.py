"""数据覆盖分析。

这一步回答"我到底能忠实跑多少条场景"，输出的是复现报告里最关键的一张表：
`docs/资产覆盖表.md` 的数据来源。

判定规则（保持可解释，不做推测）：
- `ready`        语句点名的资产全部在随仓库数据里 → 可完整执行
- `blocked`      语句点名的资产有在数据里不存在的 → 工具必然报 DataUnavailable，
                 此时的失败是数据缺失，不是模型能力
- `generic`      语句没有点名具体资产（如 "list all sites"） → 仅靠工具即可判定

资产提及识别分三层，全部显式表驱动：
1. 现有资产 ID 子串匹配（小写 + 空白归一）；
2. 别名表 `ASSET_ALIASES`（别名必须带限定词，禁止裸 `pump 1` 这类会在
   hyd_1 / mp_1 之间交叉歧义的串入表）；
3. 未知资产模式 `UNKNOWN_PATTERNS`（机器族 + 编号/设备码），命中片段原样进
   `missing_assets` 留痕；模式只认"族 + 编号"，泛指词（bare `chillers`）不算点名。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["ready", "blocked", "generic"]


#: 现有资产 ID → 显式别名列表（归一为小写；键需与 inventory 键集一致）。
#: 别名必须携带限定词（hydraulic / metro / chiller）；裸 "pump 1" 禁止入表。
ASSET_ALIASES: dict[str, list[str]] = {
    "hyd_1": ["hydraulic pump 1", "hydraulic pump one", "hydraulicpump1", "hyd 1"],
    "mp_1": ["metro pump 1", "metro pump one", "metropump1"],
    "chiller 6": ["chiller 6"],
}

#: 未知资产显式模式：(族标签, 受限正则)。命中片段归一后原样留痕。
#: 仅匹配"族 + 编号/设备码"；泛指族名（无编号）由调用方归 generic。
UNKNOWN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("chiller", re.compile(r"chiller\s*\d+")),
    ("ahu", re.compile(r"cqpa\s*ahu\s*[0-9a-z]+")),
    ("wind_turbine", re.compile(r"wind\s*turbine(?:\s*\d+)?")),
    ("equipment_code", re.compile(r"cwc\d{5}")),
]


def _norm(text: str) -> str:
    """小写并把连续空白压成单个空格。"""
    return " ".join(text.lower().split())


@dataclass(slots=True)
class Coverage:
    scenario_id: str
    status: Status
    matched_assets: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    scorer: str = "llm_judge"

    def to_record(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "status": self.status,
            "matched_assets": sorted(self.matched_assets),
            "missing_assets": sorted(self.missing_assets),
            "scorer": self.scorer,
        }


def _extract_mentions(
    text: str, inventory_assets: dict[str, Any]
) -> tuple[dict[str, list[str]], list[str]]:
    """抽取语句中的资产提及。

    返回 `(matched, missing)`：
    - `matched`：inventory 内资产 ID → 命中片段列表；
    - `missing`：inventory 外的未知资产片段（去重、保序）。
    """
    norm_text = _norm(text)
    matched: dict[str, list[str]] = {}
    existing_norms = {_norm(aid) for aid in inventory_assets}
    existing_alias_norms = {_norm(al) for aliases in ASSET_ALIASES.values() for al in aliases}

    # 1) 现有资产 ID 直中
    for aid in inventory_assets:
        nid = _norm(aid)
        if nid and nid in norm_text:
            matched.setdefault(aid, []).append(nid)

    # 2) 别名表（仅当别名目标在 inventory 内才采纳）
    for aid, aliases in ASSET_ALIASES.items():
        if aid not in inventory_assets:
            continue
        for alias in aliases:
            nalias = _norm(alias)
            if nalias and nalias in norm_text:
                matched.setdefault(aid, []).append(nalias)

    # 3) 未知资产模式；若片段其实对应现有 ID/别名，跳过（归 matched）
    missing: list[str] = []
    # 按"去空格"键去重：`chiller9` 与 `chiller 9` 是同一未知资产的两种写法。
    seen: dict[str, str] = {}
    for _family, pattern in UNKNOWN_PATTERNS:
        for m in pattern.finditer(norm_text):
            frag = _norm(m.group(0))
            if frag in existing_norms or frag in existing_alias_norms:
                continue
            key = re.sub(r"\s+", "", frag)
            if key not in seen:
                seen[key] = frag
    missing = list(seen.values())

    return matched, missing


def classify(
    scenario_id: str, text: str, inventory: dict[str, Any], has_expected: bool
) -> Coverage:
    """对单条场景给出覆盖判定。

    `inventory` 来自 `AssetRepo.inventory()`：{"assets": {...}, "sensors": {...}}。
    """
    matched, missing = _extract_mentions(text, inventory["assets"])
    scorer = "static_json" if has_expected else "llm_judge"

    if not matched and not missing:
        return Coverage(scenario_id=scenario_id, status="generic", scorer=scorer)

    if missing:
        return Coverage(
            scenario_id=scenario_id,
            status="blocked",
            matched_assets=sorted(matched.keys()),
            missing_assets=sorted(missing),
            scorer=scorer,
        )

    return Coverage(
        scenario_id=scenario_id,
        status="ready",
        matched_assets=sorted(matched.keys()),
        scorer=scorer,
    )


def summarize(coverages: list[Coverage]) -> dict[str, Any]:
    """汇总成漏斗：总数 → ready → 其中可确定性判分。"""
    total = len(coverages)
    ready = [c for c in coverages if c.status == "ready"]
    generic = [c for c in coverages if c.status == "generic"]
    blocked = [c for c in coverages if c.status == "blocked"]
    return {
        "total": total,
        "ready": len(ready),
        "generic": len(generic),
        "blocked": len(blocked),
        "executable": len(ready) + len(generic),
        # 可确定性判分的条数按全部场景统计：generic 场景同样可能有 expected_answer
        "deterministic_scored": len([c for c in coverages if c.scorer == "static_json"]),
        "executable_ratio": round((len(ready) + len(generic)) / total, 4) if total else 0.0,
    }
