"""编排层单测：Agent-As-Tool 与 Plan-Execute。

这组测试盯的是论文 §4.3 的四条硬约束，而不是"能跑":
1. 白名单隔离（IoT Agent 看不到 fmsr 工具）；
2. 成环直接报错，不靠步数上限兜；
3. 上游失败 → 下游 skipped，不得拿空值继续编；
4. 计划解析失败要能退回单步计划并留痕。
"""

import json

import pytest

from aob_repro.contracts import Decision, Scenario
from aob_repro.data.repositories import AssetRepo, DataRoot, TelemetryRepo
from aob_repro.models.scripted import ScriptedBackend
from aob_repro.orchestration.agent_as_tool import AgentAsToolOrchestrator, DomainAgent
from aob_repro.orchestration.plan_execute import (
    PlanError,
    PlanExecuteOrchestrator,
    PlanStep,
    check_acyclic,
    parse_plan,
)
from aob_repro.tools.registry import ToolRegistry, ToolSpec

ASSETS = "site,asset_id,asset_class\nMAIN,Chiller 6,Chiller\n"
SENSORS = "site,asset_id,sensor,unit\nMAIN,Chiller 6,tonnage,ton\n"
READINGS = "site,asset_id,sensor,ts,value\nMAIN,Chiller 6,tonnage,2020-06-01T00:00:00,120.0\n"


def _tool(name: str, value: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"fake {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda: value,
    )


# --------------------------------------------------------------------------- #
def test_parse_plan_rejects_cycle():
    steps = [
        PlanStep(id="s1", agent="iot", task="a", depends_on=["s2"]),
        PlanStep(id="s2", agent="fmsr", task="b", depends_on=["s1"]),
    ]
    with pytest.raises(PlanError):
        check_acyclic(steps)


def test_parse_plan_rejects_forward_reference():
    raw = json.dumps({"steps": [{"id": "s1", "agent": "iot", "task": "t", "depends_on": ["s2"]}]})
    with pytest.raises(PlanError):
        parse_plan(raw)


def test_parse_plan_accepts_code_fence():
    raw = (
        "```json\n"
        + json.dumps(
            {
                "steps": [
                    {
                        "id": "s1",
                        "agent": "iot",
                        "task": "look up",
                        "depends_on": [],
                        "expects": "e",
                    }
                ]
            }
        )
        + "\n```"
    )
    steps = parse_plan(raw)
    assert steps[0].agent == "iot"


def test_parse_plan_without_json_raises():
    with pytest.raises(PlanError):
        parse_plan("I think we should call the iot agent first.")


# --------------------------------------------------------------------------- #
def test_agent_as_tool_isolates_registries():
    iot = ToolRegistry([_tool("iot.sites", "MAIN")])
    fmsr = ToolRegistry([_tool("fmsr.failure_modes", "[]")])
    backend = ScriptedBackend([])
    a = DomainAgent(name="iot", domain="iot", registry=iot, backend=backend)
    b = DomainAgent(name="fmsr", domain="fmsr", registry=fmsr, backend=backend)
    assert "fmsr.failure_modes" not in a.to_tool_spec().description
    assert "iot.sites" not in b.to_tool_spec().description


def test_agent_as_tool_supervisor_calls_domain_agent():
    iot = ToolRegistry([_tool("iot.sites", "MAIN")])
    fmsr = ToolRegistry([_tool("fmsr.failure_modes", "[]")])
    # supervisor: 一次 agent.iot 调用，然后终态
    backend = ScriptedBackend(
        [
            ScriptedBackend.tool("agent.iot", task="list sites"),
            ScriptedBackend.final("sites are MAIN"),
        ]
    )
    agents = [
        DomainAgent(
            name="iot",
            domain="iot",
            registry=iot,
            backend=ScriptedBackend(
                [
                    ScriptedBackend.tool("iot.sites"),
                    ScriptedBackend.final("MAIN"),
                ]
            ),
        ),
        DomainAgent(name="fmsr", domain="fmsr", registry=fmsr, backend=ScriptedBackend([])),
    ]
    orch = AgentAsToolOrchestrator(agents, backend, max_steps=3)
    result = orch.run(Scenario(id="t1", text="which sites exist?"))
    assert result.strategy == "agent_as_tool"
    assert result.tool_calls == 1
    assert "agent.iot" in result.unique_tools
    assert agents[0].calls == 1
    assert agents[1].calls == 0


# --------------------------------------------------------------------------- #
def _orch(plan_text: str, sub_scripts: dict[str, list]):
    iot = ToolRegistry([_tool("iot.sites", "MAIN")])
    fmsr = ToolRegistry([_tool("fmsr.failure_modes", "[]")])

    def agent(name, registry):
        return DomainAgent(
            name=name,
            domain=name,
            registry=registry,
            backend=ScriptedBackend(sub_scripts.get(name, [])),
        )

    agents = [agent("iot", iot), agent("fmsr", fmsr)]
    # planner 第一枪返回 plan，之后 reviewer/synth 返回文本
    backend = ScriptedBackend(
        [Decision(kind="final", text=plan_text)]
        + [
            Decision(kind="final", text="OK"),
            Decision(kind="final", text="final answer"),
        ]
        * 4
    )
    return PlanExecuteOrchestrator(agents, backend), agents


def test_plan_execute_runs_dag_in_order():
    plan = json.dumps(
        {
            "steps": [
                {"id": "s1", "agent": "iot", "task": "list sites", "depends_on": []},
                {"id": "s2", "agent": "fmsr", "task": "list failures", "depends_on": ["s1"]},
            ]
        }
    )
    orch, agents = _orch(
        plan,
        {
            "iot": [ScriptedBackend.tool("iot.sites"), ScriptedBackend.final("MAIN")],
            "fmsr": [
                ScriptedBackend.tool("fmsr.failure_modes", asset_id="Chiller 6"),
                ScriptedBackend.final("2 failures"),
            ],
        },
    )
    result = orch.run(Scenario(id="t2", text="q"))
    assert result.tool_calls == 2
    assert result.stop_reason == "final"
    assert agents[0].calls == 1 and agents[1].calls == 1


def test_plan_execute_skips_downstream_on_failure():
    plan = json.dumps(
        {
            "steps": [
                {"id": "s1", "agent": "iot", "task": "list sites", "depends_on": []},
                {"id": "s2", "agent": "fmsr", "task": "list failures", "depends_on": ["s1"]},
            ]
        }
    )

    class Boom(ScriptedBackend):
        def decide(self, messages, tools):
            raise RuntimeError("backend down")

    iot = ToolRegistry([_tool("iot.sites", "MAIN")])
    fmsr = ToolRegistry([_tool("fmsr.failure_modes", "[]")])
    agents = [
        DomainAgent(name="iot", domain="iot", registry=iot, backend=Boom([])),
        DomainAgent(
            name="fmsr",
            domain="fmsr",
            registry=fmsr,
            backend=ScriptedBackend(
                [
                    ScriptedBackend.final("should not run"),
                ]
            ),
        ),
    ]
    backend = ScriptedBackend(
        [Decision(kind="final", text=plan)] + [Decision(kind="final", text="final answer")] * 6
    )
    orch = PlanExecuteOrchestrator(agents, backend, review=False)
    result = orch.run(Scenario(id="t3", text="q"))
    assert agents[0].calls == 1
    assert agents[1].calls == 0  # 上游 failed → 下游 skipped，没被调用
    assert result.stop_reason == "error"


def test_plan_fallback_on_bad_planner_output():
    orch, agents = _orch(
        "sorry, I cannot plan",
        {
            "iot": [
                ScriptedBackend.final("MAIN"),
            ]
        },
    )
    result = orch.run(Scenario(id="t4", text="which sites exist?"))
    assert agents[0].calls == 1  # 退回 iot 单步
    assert result.answer


def test_plan_execute_records_trajectory(tmp_path):
    plan = json.dumps(
        {"steps": [{"id": "s1", "agent": "iot", "task": "list sites", "depends_on": []}]}
    )
    orch, _ = _orch(
        plan, {"iot": [ScriptedBackend.tool("iot.sites"), ScriptedBackend.final("MAIN")]}
    )
    from aob_repro.trace.recorder import TrajectoryRecorder

    rec = TrajectoryRecorder(str(tmp_path), "r1")
    orch.run(Scenario(id="t5", text="q"), recorder=rec)
    lines = (tmp_path / "r1.jsonl").read_text(encoding="utf-8").strip().splitlines()
    events = {json.loads(line)["event"] for line in lines}
    assert {"user", "plan", "step_result", "final"} <= events


def test_wiring_builds_expected_types(tmp_path):
    (tmp_path / "assets.csv").write_text(ASSETS, encoding="utf-8")
    (tmp_path / "sensors.csv").write_text(SENSORS, encoding="utf-8")
    (tmp_path / "readings.csv").write_text(READINGS, encoding="utf-8")
    root = DataRoot.resolve(tmp_path)
    from aob_repro.orchestration.wiring import build_orchestrator, build_toolset

    iot, fmsr, tsfm, utilities, wo, vibration = build_toolset(
        AssetRepo(root), TelemetryRepo(root), with_fmsr=False, with_tsfm=False
    )
    assert len(fmsr) == 0 and len(tsfm) == 0  # 关闭某域时不得有别的域顶替
    with pytest.raises(ValueError):
        build_orchestrator("nope", iot=iot, fmsr=fmsr, tsfm=tsfm, backend=ScriptedBackend([]))
    aat = build_orchestrator(
        "agent_as_tool", iot=iot, fmsr=fmsr, tsfm=tsfm, backend=ScriptedBackend([])
    )
    assert type(aat).__name__ == "AgentAsToolOrchestrator"
