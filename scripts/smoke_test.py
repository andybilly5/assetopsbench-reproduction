"""在合成夹具上验证运行、轨迹落盘和静态判分；不调用外部模型。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """使用临时输出目录，确保冒烟过程不污染实验结果。"""
    with tempfile.TemporaryDirectory(prefix="aob-smoke-") as temporary:
        output = Path(temporary)
        scenario = ROOT / "examples/smoke/scenario.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "aob_repro.cli",
                "run-one",
                "--scenario",
                str(scenario),
                "--backend",
                "scripted",
                "--data-root",
                str(ROOT / "examples/smoke/data"),
                "--scripted-answer",
                '{"sites":["DEMO"]}',
                "--no-fmsr",
                "--no-tsfm",
                "--no-utilities",
                "--no-wo",
                "--no-vibration",
                "--out",
                str(output / "traces"),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "aob_repro.cli",
                "evaluate",
                "--scenarios",
                str(scenario),
                "--trajectories",
                str(output / "traces"),
                "--out",
                str(output / "reports"),
            ],
            cwd=ROOT,
            check=True,
        )
        report = json.loads((output / "reports/_aggregate.json").read_text(encoding="utf-8"))
        if report["totals"]["scored"] != 1 or report["totals"]["passed"] != 1:
            raise RuntimeError("Offline smoke test did not produce one passing result")
    print("Offline smoke test passed (synthetic fixture; no model capability claim).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
