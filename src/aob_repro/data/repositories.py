"""数据访问层。

上游 `src/couchdb/scenarios_data/shared/` 里是 JSON/CSV 混合形态，且部分文件是
`*_sample.*`。本复现先在自己的 `normalized/` 三表格式上跑通流程，字段映射另写
（见 `data/README.md`），避免把"数据形态"和"方法"两件事混在一起。

三表：
    assets.csv    site,asset_id,asset_class
    sensors.csv   site,asset_id,sensor,unit
    readings.csv  site,asset_id,sensor,ts,value

任何一张表缺失都不静默返回空结果，而是抛 `DataUnavailable`——静默的空结果会
被错误地解读成"模型没查到"，污染失败归因。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class DataUnavailable(RuntimeError):
    """数据文件缺失或为空。"""


def default_root() -> Path:
    """数据根目录：环境变量 AOB_DATA_DIR 优先，否则 <cwd>/data/normalized。"""
    return Path(os.environ.get("AOB_DATA_DIR", "data/normalized")).expanduser()


@dataclass(slots=True)
class DataRoot:
    path: Path

    @classmethod
    def resolve(cls, path: str | Path | None = None) -> "DataRoot":
        return cls(path=Path(path) if path else default_root())

    def file(self, name: str) -> Path:
        p = self.path / name
        if not p.is_file():
            raise DataUnavailable(
                f"missing data file: {p.resolve()} "
                f"(set AOB_DATA_DIR, or run scripts/prepare_data.py)"
            )
        return p

    def read_csv(self, name: str) -> list[dict[str, str]]:
        p = self.file(name)
        with p.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise DataUnavailable(f"empty data file: {p}")
        return rows

    def read_json(self, name: str) -> Any:
        p = self.file(name)
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)


class AssetRepo:
    """资产与传感器目录查询。"""

    def __init__(self, root: DataRoot) -> None:
        self.root = root
        self._assets = root.read_csv("assets.csv")
        self._sensors = root.read_csv("sensors.csv")

    def sites(self) -> list[str]:
        return sorted({r["site"] for r in self._assets})

    def assets(self, site: str) -> list[dict[str, str]]:
        site_n = site.strip().upper()
        rows = [r for r in self._assets if r["site"].upper() == site_n]
        if not rows:
            raise DataUnavailable(f"no assets for site {site!r}")
        return rows

    def asset(self, site: str, asset_id: str) -> dict[str, str]:
        site_n = site.strip().upper()
        aid_n = asset_id.strip().lower()
        for r in self._assets:
            if r["site"].upper() == site_n and r["asset_id"].lower() == aid_n:
                return r
        raise DataUnavailable(f"unknown asset {asset_id!r} at site {site!r}")

    def inventory(self) -> dict[str, Any]:
        """供覆盖分析使用：全部资产 ID 与其传感器集合。"""
        assets = {r["asset_id"].lower(): r for r in self._assets}
        sensors: dict[str, list[str]] = {aid: [] for aid in assets}
        for r in self._sensors:
            aid = r["asset_id"].lower()
            sensors.setdefault(aid, []).append(r["sensor"])
        return {"assets": assets, "sensors": sensors, "sites": self.sites()}


class TelemetryRepo:
    """时序读数查询。"""

    def __init__(self, root: DataRoot) -> None:
        self.root = root
        self._readings = root.read_csv("readings.csv")

    def sensors(self, site: str, asset_id: str) -> list[str]:
        site_n, aid_n = site.upper(), asset_id.lower()
        vals = sorted(
            {
                r["sensor"]
                for r in self._readings
                if r["site"].upper() == site_n and r["asset_id"].lower() == aid_n
            }
        )
        if not vals:
            raise DataUnavailable(f"no measurements for {asset_id!r} at site {site!r}")
        return vals

    def history(
        self,
        site: str,
        asset_id: str,
        sensor: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        site_n, aid_n, sen = site.upper(), asset_id.lower(), sensor.lower()
        out: list[dict[str, Any]] = []
        for r in self._readings:
            if r["site"].upper() != site_n:
                continue
            if r["asset_id"].lower() != aid_n or r["sensor"].lower() != sen:
                continue
            ts = r["ts"]
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            try:
                value: Any = float(r["value"])
            except (TypeError, ValueError):
                value = r["value"]
            out.append({"ts": ts, "value": value})
        if not out:
            raise DataUnavailable(
                f"no readings for {asset_id!r}/{sensor!r} in window "
                f"[{start or '-inf'}, {end or '+inf'}]"
            )
        return sorted(out, key=lambda x: x["ts"])


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读 JSONL 场景文件，跳过空行。"""
    p = Path(path)
    if not p.is_file():
        raise DataUnavailable(f"missing file: {p.resolve()}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p.name}:{lineno} invalid JSON: {exc}") from exc
    if not rows:
        raise DataUnavailable(f"empty file: {p}")
    return rows


def iter_asset_tokens(texts: Iterable[str]) -> set[str]:
    """把若干段文本切成小写词元，用于在场景语句里做资产名匹配。"""
    tokens: set[str] = set()
    for t in texts:
        for raw in t.replace("/", " ").replace(",", " ").split():
            tk = raw.strip(" .;:()[]'\"%?").lower()
            if len(tk) >= 3:
                tokens.add(tk)
    return tokens
