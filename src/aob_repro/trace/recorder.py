"""轨迹记录。

产出两样东西，二者用 run_id 关联：
- `results/trajectories/<run_id>.jsonl`：逐事件流水，供失败分析
- `results/trajectories/<run_id>.json`：本次运行的轨迹对象（含终态答案），供判分

判分只读 `.json`，分析只读 `.jsonl`，互不干扰。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import RunResult, TraceEvent


@dataclass(slots=True)
class TrajectoryRecorder:
    out_dir: Path
    run_id: str
    _events_path: Path = field(init=False)

    def __init__(self, out_dir: str | Path, run_id: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._events_path = self.out_dir / f"{run_id}.jsonl"

    def _append(self, record: dict[str, Any]) -> None:
        with self._events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def event(self, step: int, name: str, **payload: Any) -> None:
        self._append(
            TraceEvent(run_id=self.run_id, step=step, event=name, payload=payload).to_record()
        )

    def finish(self, result: RunResult, extra: dict[str, Any] | None = None) -> Path:
        path = self.out_dir / f"{self.run_id}.json"
        record = result.to_record()
        if extra:
            record.update(extra)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        return path


def load_trajectories(path: str | Path) -> list[dict[str, Any]]:
    """读单个 `.json` 或目录下所有 `.json`（忽略 `_aggregate.json` 与 `.jsonl`）。"""
    p = Path(path)
    if p.is_file():
        return [json.loads(p.read_text(encoding="utf-8"))]
    if not p.is_dir():
        raise FileNotFoundError(f"no such trajectory path: {p}")
    out: list[dict[str, Any]] = []
    for f in sorted(p.glob("*.json")):
        if f.name.startswith("_"):
            continue
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out
