"""命令行入口。

四个子命令对应复现流水线：
    validate-data   校验场景契约 + 数据覆盖
    run-one         跑单条场景（冒烟）
    run-suite       跑评测集
    evaluate        对已落盘的轨迹做判分并生成汇总报告

设计约束：`run-*` 只负责产出轨迹，`evaluate` 只读轨迹。这样"换判官/换判分器"
不需要重新调用被测模型，便于保持轨迹一致并减少重复推理成本。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .agents.react_loop import run_single
from .contracts import Scenario
from .data.coverage import classify, summarize
from .data.repositories import AssetRepo, DataRoot, TelemetryRepo, load_jsonl
from .evaluation.static_json import score as score_static
from .models.openai_compat import OpenAIBackend
from .models.scripted import ScriptedBackend
from .orchestration import (
    STRATEGIES,
    build_orchestrator,
    build_toolset,
    distractor_registry_from_env,
    merged_registry,
)
from .trace.recorder import TrajectoryRecorder, load_trajectories


# --------------------------------------------------------------------------- #
def _load_scenarios(path: str) -> list[Scenario]:
    """支持 `.jsonl`、`.json`（单对象/数组）与目录。"""
    p = Path(path)
    if p.is_dir():
        out: list[Scenario] = []
        for f in sorted(p.glob("*.json")):
            out.extend(_load_scenarios(str(f)))
        return out
    if p.suffix == ".jsonl":
        return [Scenario.from_record(r) for r in load_jsonl(p)]
    raw = json.loads(p.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else [raw]
    return [Scenario.from_record(r) for r in records]


def _toolset(args: argparse.Namespace):
    root = DataRoot.resolve(args.data_root)
    assets, telemetry = AssetRepo(root), TelemetryRepo(root)
    iot, fmsr, tsfm, utilities, wo, vibration = build_toolset(
        assets,
        telemetry,
        with_fmsr=not getattr(args, "no_fmsr", False),
        with_tsfm=not getattr(args, "no_tsfm", False),
        with_utilities=not getattr(args, "no_utilities", False),
        with_wo=not getattr(args, "no_wo", False),
        with_vibration=not getattr(args, "no_vibration", False),
    )
    # distractor 消融开关：默认空注册表；只有 AOB_DISTRACTOR_N 显式设置才加入。
    distractors = distractor_registry_from_env()
    return iot, fmsr, tsfm, utilities, wo, vibration, distractors, root


def _run_scenario(
    args, scenario: Scenario, iot, fmsr, tsfm, utilities, wo, vibration, distractors, backend
) -> object:
    """按 --strategy 跑一条场景；轨迹落盘口径与 run_single 一致。"""
    strategy = getattr(args, "strategy", "single_agent")
    if strategy == "single_agent":
        return run_single(
            scenario,
            merged_registry(iot, fmsr, tsfm, utilities, wo, vibration, distractors),
            backend,
            max_steps=args.max_steps,
            out_dir=args.out,
        )
    orchestrator = build_orchestrator(
        strategy,
        iot=iot,
        fmsr=fmsr,
        tsfm=tsfm,
        utilities=utilities,
        wo=wo,
        vibration=vibration,
        backend=backend,
        sub_steps=getattr(args, "sub_steps", 4),
        supervisor_steps=getattr(args, "max_steps", 4),
        review=not getattr(args, "no_review", False),
    )
    import uuid

    rec = TrajectoryRecorder(args.out, f"{scenario.id}-{uuid.uuid4().hex[:8]}")
    return orchestrator.run(scenario, recorder=rec)


def _backend(args: argparse.Namespace):
    if args.backend == "scripted":
        # 脚本后端只证明链路可跑；答案可配置，便于验证 evaluate 的判分路径
        return ScriptedBackend(
            [
                ScriptedBackend.tool("iot.sites"),
                ScriptedBackend.final(
                    getattr(args, "scripted_answer", None) or "[scripted] smoke ok"
                ),
            ]
        )
    return OpenAIBackend(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


# --------------------------------------------------------------------------- #
def cmd_validate_data(args: argparse.Namespace) -> int:
    scenarios = _load_scenarios(args.scenarios)
    root = DataRoot.resolve(args.data_root)
    try:
        inventory = AssetRepo(root).inventory()
    except Exception as exc:  # noqa: BLE001
        print(f"[data] 不可用：{exc}")
        print("       先准备 data/normalized/{assets,sensors,readings}.csv（见 data/README.md）")
        return 2

    coverages = [classify(s.id, s.text, inventory, s.has_expected) for s in scenarios]
    agg = summarize(coverages)
    print(json.dumps(agg, ensure_ascii=False, indent=2))

    if args.write_table:
        out = Path(args.write_table)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario_id", "status", "matched_assets", "missing_assets", "scorer"])
            for c in coverages:
                w.writerow(
                    [
                        c.scenario_id,
                        c.status,
                        ";".join(sorted(c.matched_assets)),
                        ";".join(sorted(c.missing_assets)),
                        c.scorer,
                    ]
                )
        print(f"[write] {out}")
    return 0


def cmd_run_one(args: argparse.Namespace) -> int:
    [scenario] = _load_scenarios(args.scenario)
    iot, fmsr, tsfm, utilities, wo, vibration, distractors, _ = _toolset(args)
    result = _run_scenario(
        args, scenario, iot, fmsr, tsfm, utilities, wo, vibration, distractors, _backend(args)
    )
    print(json.dumps(result.to_record(), ensure_ascii=False, indent=2))
    return 0


def cmd_run_suite(args: argparse.Namespace) -> int:
    scenarios = _load_scenarios(args.scenarios)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        scenarios = [s for s in scenarios if s.id in wanted]
    if args.only_ready:
        root = DataRoot.resolve(args.data_root)
        inventory = AssetRepo(root).inventory()
        scenarios = [
            s
            for s in scenarios
            if classify(s.id, s.text, inventory, s.has_expected).status != "blocked"
        ]
    if args.limit:
        scenarios = scenarios[: args.limit]

    iot, fmsr, tsfm, utilities, wo, vibration, distractors, _ = _toolset(args)
    backend = _backend(args)
    results = []
    for s in scenarios:
        if isinstance(backend, ScriptedBackend):
            backend.reset()
        results.append(
            _run_scenario(args, s, iot, fmsr, tsfm, utilities, wo, vibration, distractors, backend)
        )
    print(
        f"[run] {len(results)} scenarios → {args.out} strategy={getattr(args, 'strategy', 'single_agent')}"
    )
    for r in results:
        print(f"  {r.scenario_id}\tsteps={r.steps}\tcalls={r.tool_calls}\t{r.stop_reason}")
    return 0


def _traj_text(run_id: str, traj_dir: str, limit: int = 4000) -> str:
    """取该 run 的轨迹正文尾部，供判官参考；取不到就返回空串。"""
    path = Path(traj_dir) / f"{run_id}.jsonl"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")[-limit:]
    except OSError:
        return ""


def _judge_from_env(judged_model: str):
    from .evaluation.judge import DeepSeekJudge

    return DeepSeekJudge.from_env(judged_model=judged_model or "unknown")


def cmd_evaluate(args: argparse.Namespace) -> int:
    scenarios = {s.id: s for s in _load_scenarios(args.scenarios)}
    records = load_trajectories(args.trajectories)

    rows: list[dict[str, Any]] = []
    skipped = 0
    judge = None
    for rec in records:
        sid = str(rec.get("scenario_id", ""))
        scenario = scenarios.get(sid)
        if scenario is None:
            skipped += 1
            continue
        if scenario.expected_answer is None:
            if not getattr(args, "judge", False):
                skipped += 1
                continue
            if judge is None:
                judge = _judge_from_env(str(rec.get("model") or ""))
            try:
                jr = judge.judge_scenario(
                    scenario,
                    str(rec.get("answer", "")),
                    trajectory=_traj_text(str(rec.get("run_id", "")), args.trajectories),
                )
            except Exception as exc:  # 判官失败不静默吞掉，单列一行
                rows.append(
                    {
                        "scenario_id": sid,
                        "run_id": rec.get("run_id"),
                        "scenario_type": scenario.type or "unknown",
                        "passed": False,
                        "score": None,
                        "error": f"judge_error: {type(exc).__name__}",
                    }
                )
                continue
            rows.append(
                {
                    "scenario_id": sid,
                    "run_id": rec.get("run_id"),
                    "scenario_type": scenario.type or "unknown",
                    "passed": jr.passed,
                    "score": jr.score,
                    "valid_runs": jr.valid_runs,
                    "parse_errors": jr.parse_errors,
                }
            )
            continue
        res = score_static(scenario, rec.get("answer", ""))
        rows.append(
            {
                "scenario_id": sid,
                "run_id": rec.get("run_id"),
                "scenario_type": scenario.type or "unknown",
                "passed": res.passed,
                "score": res.score,
                **{k: v for k, v in res.details.items() if k != "missing_keys"},
            }
        )

    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    by_type: dict[str, list[bool]] = {}
    for row in rows:
        by_type.setdefault(str(row["scenario_type"]), []).append(bool(row["passed"]))

    report = {
        "scorer": "llm_judge" if getattr(args, "judge", False) else "static_json",
        "totals": {
            "scored": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        },
        "by_scenario_type": {
            k: {"total": len(v), "passed": sum(v), "pass_rate": round(sum(v) / len(v), 4)}
            for k, v in by_type.items()
        },
        "skipped_no_ground_truth": skipped,
        "trajectories_read": len(records),
        "results": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "_aggregate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2
        )
    )
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aob-repro", description="AssetOpsBench 精简复现")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--data-root", default=None, help="数据目录，默认 data/normalized")
        sp.add_argument("--out", default="results/trajectories", help="轨迹输出目录")

    v = sub.add_parser("validate-data", help="校验场景契约与数据覆盖")
    v.add_argument("--scenarios", default="data/scenarios/all_utterance.jsonl")
    v.add_argument("--write-table", default=None, help="把覆盖表写成 CSV")
    common(v)
    v.set_defaults(func=cmd_validate_data)

    def harness(sp: argparse.ArgumentParser) -> None:
        """三种策略共用的一组开关。"""
        sp.add_argument(
            "--strategy",
            choices=STRATEGIES,
            default="single_agent",
            help="编排范式：single_agent（单 Agent 基线）/ agent_as_tool / plan_execute",
        )
        sp.add_argument("--sub-steps", type=int, default=4, help="领域 Agent 的步数预算")
        sp.add_argument(
            "--no-review", action="store_true", help="plan_execute 关闭 Reviewer（消融用）"
        )
        sp.add_argument(
            "--no-fmsr", action="store_true", help="不加载 FMSR 工具（复跑旧 iot-only 口径用）"
        )
        sp.add_argument("--no-tsfm", action="store_true", help="不加载 TSFM 工具（消融用）")
        sp.add_argument(
            "--no-utilities", action="store_true", help="不加载 utilities 工具（消融用）"
        )
        sp.add_argument("--no-wo", action="store_true", help="不加载 WO 工单工具（消融用）")
        sp.add_argument(
            "--no-vibration", action="store_true", help="不加载 vibration 工具（消融用）"
        )

    o = sub.add_parser("run-one", help="跑单条场景")
    o.add_argument("--scenario", required=True)
    o.add_argument("--backend", choices=["scripted", "openai"], default="scripted")
    o.add_argument("--model", default="qwen3.5-9b")
    o.add_argument("--base-url", default=None)
    o.add_argument("--temperature", type=float, default=0.0)
    o.add_argument("--max-tokens", type=int, default=2048)
    o.add_argument("--max-steps", type=int, default=8)
    o.add_argument("--scripted-answer", default=None, help="scripted 后端的终态答案")
    harness(o)
    common(o)
    o.set_defaults(func=cmd_run_one)

    s = sub.add_parser("run-suite", help="跑评测集")
    s.add_argument("--scenarios", default="data/scenarios/all_utterance.jsonl")
    s.add_argument("--ids", default=None, help="逗号分隔的场景 ID")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--only-ready", action="store_true", help="跳过被数据覆盖阻塞的场景")
    s.add_argument("--backend", choices=["scripted", "openai"], default="scripted")
    s.add_argument("--model", default="qwen3.5-9b")
    s.add_argument("--base-url", default=None)
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--max-tokens", type=int, default=2048)
    s.add_argument("--max-steps", type=int, default=8)
    s.add_argument("--scripted-answer", default=None, help="scripted 后端的终态答案")
    harness(s)
    common(s)
    s.set_defaults(func=cmd_run_suite)

    e = sub.add_parser("evaluate", help="对已落盘轨迹判分")
    e.add_argument("--trajectories", required=True)
    e.add_argument("--scenarios", default="data/scenarios/all_utterance.jsonl")
    e.add_argument("--out", default="results/reports")
    e.add_argument("--judge", action="store_true", help="无 expected_answer 的场景改用 LLM 判官")
    e.set_defaults(func=cmd_evaluate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
