"""Plan-Execute 编排：Planner → Executor → Reviewer → Synthesizer。

论文 §4.3 的第二条链。与 Agent-As-Tool 的差别只在"谁决定下一步"：这里由一次
规划产出 DAG，执行阶段不再反复问模型。

四条硬约束（也是本模块单测覆盖的重点）：
1. 依赖成环直接 `PlanError`，不允许靠步数上限兜住；
2. 某节点失败，其下游标 `skipped`，绝不拿空值继续编结论；
3. Reviewer 的判定必须落轨迹，让"计划错"与"执行错"可分；
4. 计划解析失败时退回单步计划并记 `plan_fallback=True`——格式失败是模型能力
   问题，不该伪装成"计划质量"问题。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..contracts import RunResult, Scenario
from ..trace.recorder import TrajectoryRecorder
from .agent_as_tool import DomainAgent

PLANNER_PROMPT = (
    "You are the planner of an industrial asset operations team. "
    "Break the user question into the smallest number of dependent steps. "
    "Each step is assigned to exactly one domain agent. "
    "Reply with a single JSON object and nothing else, in this schema:\n"
    '{"steps": [{"id": "s1", "agent": "<agent name>", "task": "<self-contained task>", '
    '"depends_on": [], "expects": "<what a correct result looks like>"}]}\n'
    "Rules: ids are s1, s2, ...; depends_on only references earlier ids; "
    "never invent agent names; keep task self-contained (asset id, sensor, window). "
    "If the question needs only one lookup, return one step."
)

REVIEWER_PROMPT = (
    "You are a reviewer. Given one plan step and the agent's result, decide whether the "
    "result actually satisfies what the step expects. Reply exactly 'OK' or "
    "'BAD: <one-line reason>'. Be strict about invented values: if the result states "
    "numbers or failure modes that the step could not have obtained, answer BAD."
)

SYNTH_PROMPT = (
    "You are the synthesizer. Answer the user question using only the step results "
    "provided. Cite the values they contain. If a step was skipped or reported that data "
    "is unavailable, say so explicitly instead of filling the gap."
)


class PlanError(RuntimeError):
    """计划非法：成环、引用不存在的 Agent、id 重复。"""


@dataclass(slots=True)
class PlanStep:
    id: str
    agent: str
    task: str
    depends_on: list[str] = field(default_factory=list)
    expects: str = ""


@dataclass(slots=True)
class StepOutcome:
    id: str
    status: str  # done | failed | skipped
    answer: str = ""
    steps: int = 0
    tool_calls: int = 0
    unique_tools: list[str] = field(default_factory=list)
    review: str = ""
    error: str = ""


def _complete(backend: Any, system: str, user: str) -> str:
    """不挂工具的纯文本调用（规划、复核、汇总都用它）。"""
    decision = backend.decide(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], []
    )
    return decision.text or ""


def parse_plan(text: str) -> list[PlanStep]:
    """从模型输出里抽 DAG；抽不出来抛 `PlanError`。"""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise PlanError(f"planner output has no JSON object: {text[:160]!r}")
    try:
        raw = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlanError(f"planner JSON invalid: {exc}") from exc

    items = raw.get("steps")
    if not isinstance(items, list) or not items:
        raise PlanError("planner returned no steps")

    steps: list[PlanStep] = []
    seen: set[str] = set()
    for item in items:
        sid = str(item.get("id") or "").strip()
        agent = str(item.get("agent") or "").strip()
        task = str(item.get("task") or "").strip()
        if not sid or not agent or not task:
            raise PlanError(f"step missing id/agent/task: {item!r}")
        if sid in seen:
            raise PlanError(f"duplicate step id: {sid}")
        seen.add(sid)
        deps = [str(d) for d in (item.get("depends_on") or [])]
        unknown = [d for d in deps if d not in seen]
        if unknown:
            raise PlanError(f"step {sid} depends on unknown/forward id(s): {unknown}")
        steps.append(
            PlanStep(
                id=sid,
                agent=agent,
                task=task,
                depends_on=deps,
                expects=str(item.get("expects") or ""),
            )
        )
    return steps


def check_acyclic(steps: list[PlanStep]) -> None:
    """Kahn 拓扑排序；排不完说明成环。"""
    graph = {s.id: list(s.depends_on) for s in steps}
    indeg = {sid: len(deps) for sid, deps in graph.items()}
    queue = [sid for sid, d in indeg.items() if d == 0]
    visited = 0
    while queue:
        cur = queue.pop(0)
        visited += 1
        for sid, deps in graph.items():
            if cur in deps:
                indeg[sid] -= 1
                if indeg[sid] == 0:
                    queue.append(sid)
    if visited != len(steps):
        cycled = [sid for sid, d in indeg.items() if d > 0]
        raise PlanError(f"dependency cycle detected among steps: {cycled}")


class PlanExecuteOrchestrator:
    """两条链共用领域 Agent，差别在编排方式。"""

    def __init__(
        self,
        agents: list[DomainAgent],
        backend: Any,
        review: bool = True,
        max_steps: int = 5,
    ) -> None:
        self.agents = {a.name: a for a in agents}
        self.backend = backend
        self.review = review
        self.max_steps = max_steps
        self.strategy = "plan_execute"

    # ---------------------------------------------------------------- #
    def _plan(
        self, scenario: Scenario, recorder: TrajectoryRecorder | None
    ) -> tuple[list[PlanStep], bool]:
        roster = "\n".join(
            f"- {name} (domain {a.domain}); tools: {', '.join(a.registry.names())}"
            for name, a in self.agents.items()
        )
        user = f"Available agents:\n{roster}\n\nUser question:\n{scenario.text}"
        raw = _complete(self.backend, PLANNER_PROMPT, user)
        try:
            steps = parse_plan(raw)
            check_acyclic(steps)
            for s in steps:
                if s.agent not in self.agents:
                    raise PlanError(f"step {s.id} names unknown agent {s.agent!r}")
            if recorder:
                recorder.event(
                    0,
                    "plan",
                    raw=raw[:1200],
                    steps=[
                        {
                            "id": s.id,
                            "agent": s.agent,
                            "task": s.task,
                            "depends_on": s.depends_on,
                            "expects": s.expects,
                        }
                        for s in steps
                    ],
                    fallback=False,
                )
            return steps, False
        except PlanError as exc:
            # 退回单步计划：目的是把"格式失败"和"计划质量差"分开记录
            fallback_agent = self._fallback_agent(scenario)
            if recorder:
                recorder.event(0, "plan_error", error=str(exc), raw=raw[:800])
            steps = [
                PlanStep(
                    id="s1",
                    agent=fallback_agent,
                    task=scenario.text,
                    expects="a complete answer to the user question",
                )
            ]
            if recorder:
                recorder.event(
                    0,
                    "plan",
                    steps=[
                        {
                            "id": "s1",
                            "agent": fallback_agent,
                            "task": scenario.text,
                            "depends_on": [],
                        }
                    ],
                    fallback=True,
                )
            return steps, True

    def _fallback_agent(self, scenario: Scenario) -> str:
        text = (scenario.text or "").lower()
        if any(
            k in text for k in ("failure", "fault", "overheat", "fouling", "refrigerant", "trip")
        ):
            if "fmsr" in self.agents:
                return "fmsr"
        if "iot" in self.agents:
            return "iot"
        return next(iter(self.agents))

    def _review_step(self, step: PlanStep, answer: str) -> str:
        if not self.review or not answer.strip():
            return ""
        verdict = _complete(
            self.backend,
            REVIEWER_PROMPT,
            f"Step: {step.task}\nExpects: {step.expects or 'a correct, grounded result'}\n"
            f"Agent result:\n{answer[:1500]}",
        )
        return verdict.strip()[:200]

    # ---------------------------------------------------------------- #
    def run(self, scenario: Scenario, recorder: TrajectoryRecorder | None = None) -> RunResult:
        run_id = recorder.run_id if recorder else f"run-{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        if recorder:
            recorder.event(
                0,
                "user",
                text=scenario.text,
                tools=sorted(f"agent.{n}" for n in self.agents),
                agents={n: a.registry.names() for n, a in self.agents.items()},
                strategy=self.strategy,
            )

        steps, fallback = self._plan(scenario, recorder)
        outcomes = self._execute(steps, recorder)

        done = [o for o in outcomes if o.status == "done"]
        failed = [o for o in outcomes if o.status != "done"]

        single = len(done) == 1 and len(steps) == 1
        if single:
            answer = done[0].answer  # 单步计划不必再花一次汇总调用
        else:
            answer = self._synthesize(scenario, steps, outcomes, recorder)

        if recorder and single:
            recorder.event(len(steps), "final", text=answer)  # 单步也要留终态事件

        duration_ms = (time.perf_counter() - started) * 1000
        result = RunResult(
            run_id=run_id,
            scenario_id=scenario.id,
            question=scenario.text,
            answer=answer,
            strategy=self.strategy,
            model=getattr(self.backend, "model", "unknown"),
            steps=len(steps),
            tool_calls=sum(o.tool_calls for o in outcomes),
            unique_tools=sorted({t for o in outcomes for t in o.unique_tools}),
            stop_reason="final" if not failed else "error",
            duration_ms=duration_ms,
        )
        if recorder:
            recorder.event(
                len(steps),
                "plan_summary",
                plan_fallback=fallback,
                done=len(done),
                failed=len(failed),
                skipped=[o.id for o in outcomes if o.status == "skipped"],
                reviewed=sum(1 for o in outcomes if o.review),
                bad_reviews=sum(1 for o in outcomes if o.review.upper().startswith("BAD")),
                final=answer[:400],
            )
            recorder.finish(result)
        return result

    def _execute(
        self, steps: list[PlanStep], recorder: TrajectoryRecorder | None
    ) -> list[StepOutcome]:
        outcomes: dict[str, StepOutcome] = {}
        for step in steps:
            blocked = [
                d
                for d in step.depends_on
                if outcomes.get(d) is None or outcomes[d].status != "done"
            ]
            if blocked:
                outcomes[step.id] = StepOutcome(
                    id=step.id, status="skipped", error=f"upstream not done: {blocked}"
                )
                if recorder:
                    recorder.event(
                        step.id,
                        "step_result",
                        step_id=step.id,
                        status="skipped",
                        blocked_by=blocked,
                    )
                continue

            context = ""
            for dep in step.depends_on:
                context += f"\n[result of {dep}]\n{outcomes[dep].answer[:900]}\n"
            task = step.task if not context else f"{step.task}\n{context}"

            agent = self.agents[step.agent]
            try:
                sub = agent.run(task)
                outcome = StepOutcome(
                    id=step.id,
                    status="done" if sub.get("answer", "").strip() else "failed",
                    answer=str(sub.get("answer") or ""),
                    steps=int(sub.get("steps") or 0),
                    tool_calls=int(sub.get("tool_calls") or 0),
                    unique_tools=list(sub.get("unique_tools") or []),
                )
                if outcome.status == "failed":
                    outcome.error = "empty answer"
            except Exception as exc:  # noqa: BLE001 —— 单步失败不能拖垮整条计划
                outcome = StepOutcome(
                    id=step.id, status="failed", error=f"{type(exc).__name__}: {exc}"
                )

            outcome.review = self._review_step(step, outcome.answer)
            outcomes[step.id] = outcome
            if recorder:
                recorder.event(
                    step.id,
                    "step_result",
                    step_id=step.id,
                    agent=step.agent,
                    status=outcome.status,
                    answer=outcome.answer[:1200],
                    steps=outcome.steps,
                    tool_calls=outcome.tool_calls,
                    error=outcome.error,
                )
                if outcome.review:
                    recorder.event(step.id, "review", step_id=step.id, verdict=outcome.review)
        return [outcomes[s.id] for s in steps]

    def _synthesize(
        self,
        scenario: Scenario,
        steps: list[PlanStep],
        outcomes: list[StepOutcome],
        recorder: TrajectoryRecorder | None,
    ) -> str:
        body = ""
        for step, out in zip(steps, outcomes):
            body += (
                f"\n[{step.id} → {step.agent}] status={out.status}\n"
                f"task: {step.task}\nresult: {out.answer[:1200] if out.answer else '(none) '}\n"
                + (f"note: {out.error}\n" if out.error else "")
            )
        try:
            answer = _complete(
                self.backend, SYNTH_PROMPT, f"User question: {scenario.text}\n{body}"
            ).strip()
        except Exception as exc:  # noqa: BLE001
            answer = ""
            if recorder:
                recorder.event(len(steps), "synth_error", error=f"{type(exc).__name__}: {exc}")
        if not answer:
            # 汇总失败时退化为拼接，保证答案非空（fail-open 有记录）
            answer = "\n".join(o.answer for o in outcomes if o.answer).strip()
        if recorder:
            recorder.event(len(steps), "final", text=answer)
        return answer
