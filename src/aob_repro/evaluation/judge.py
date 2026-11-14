"""LLM-As-Judge：DeepSeek 判官（六维 rubric，5 次重复判定）。

## 口径（pass 与 score 两条线严格分离，禁止互相反推）

### pass（是否通过，布尔线）

单次判定（one repeat）的通过条件是**同时**满足：

1. 前五维全部为 `true`：
   - `task_completion`
   - `data_retrieval_accuracy`
   - `generalized_result_verification`
   - `agent_sequence_correct`
   - `clarity_and_justification`
2. `hallucinations` 为 `false`（未出现幻觉）。

一条轨迹判 `repeats` 次（默认 5 次），最终 pass 由**投票通过比例**决定：
有效（成功解析）判定里，`pass_votes / valid > 0.5 + delta` 才判通过；`delta`
取自环境变量 `AOB_JUDGE_PASS_DELTA`，默认 `0.0`。`delta=0` 时严格等价于旧口径
"通过次数严格多于不通过次数"（平局与全失败均判不通过）。**不得**用均值分数
反推 pass，**不得**用 pass 次数充当分数。

### score（得分，数值线）

单次判定得分 = 前五维中为 `true` 的维数 / 5（取值 [0, 1]）；
若该次判定 `hallucinations == true`，总分**扣 0.2**（下界钳到 0.0）。
一条轨迹的最终 score = 全部有效判定单次得分的**算术均值**。
解析失败的判定**不计入分母**（不当 0 分），但会登记到 `parse_errors` 供
上层做失败归因；若全部判定都解析失败，`score = None`、`passed = False`，
并在 `error` 字段标注 `judge_parse_error`。

## 后端与配置

- 复用 `OpenAIBackend`（stdlib HTTP，零第三方依赖）；
- 环境变量：`AOB_JUDGE_BASE_URL`（默认 `https://api.deepseek.com`）、
  `AOB_JUDGE_MODEL`（默认 `deepseek-chat`）、`AOB_JUDGE_API_KEY`；
- temperature 固定 0.3；提示词要求模型只返回严格 JSON 对象；
- 自评审防护：归一化后判官模型与被测模型同名/同源，直接抛
  `SelfJudgingError`，不允许跑评测。

## 可比性声明

原设计（论文/上游）判官为 llama-4-maverick（400B 级 MoE），本复现替换为
DeepSeek `deepseek-chat`；判分的**绝对水位**与论文不可比，只能做同口径下的
模型间相对比较。
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..contracts import Scenario
from ..models.openai_compat import OpenAIBackend

# --------------------------------------------------------------------------- #
# 常量与环境变量名

JUDGE_ENV_BASE_URL = "AOB_JUDGE_BASE_URL"
JUDGE_ENV_MODEL = "AOB_JUDGE_MODEL"
JUDGE_ENV_API_KEY = "AOB_JUDGE_API_KEY"
JUDGE_ENV_PASS_DELTA = "AOB_JUDGE_PASS_DELTA"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_REPEATS = 5
DEFAULT_PASS_DELTA = 0.0

#: 前五维：全部 true 才算单次通过
RUBRIC_FIELDS: tuple[str, ...] = (
    "task_completion",
    "data_retrieval_accuracy",
    "generalized_result_verification",
    "agent_sequence_correct",
    "clarity_and_justification",
)
#: 第六维：幻觉标记；为 true 时扣分且单次不通过
HALLUCINATION_FIELD = "hallucinations"
ALL_FIELDS: tuple[str, ...] = RUBRIC_FIELDS + (HALLUCINATION_FIELD,)

HALLUCINATION_PENALTY = 0.2
MAX_RAW_TEXT_CHARS = 300

_RUBRIC_DESCRIPTIONS: dict[str, str] = {
    "task_completion": "是否完成了用户要求的全部动作（含必要的多步操作）",
    "data_retrieval_accuracy": "检索到的数值/资产/时间窗是否与真实数据一致",
    "generalized_result_verification": "是否对结果做了合理性校验（数据不足时明确指出）",
    "agent_sequence_correct": "工具调用与观察回填的次序是否正确、有无跳步或乱序",
    "clarity_and_justification": "结论是否清晰，并给出了可追溯的依据或引用",
    "hallucinations": "是否出现编造的数值、资产、设备或未被工具返回的结论",
}

_SYSTEM_PROMPT = (
    "You are an impartial industrial-asset operations judge. "
    "Evaluate the assistant's answer against the user question and recorded tool "
    "sequence. You MUST return only one strict JSON object with no prose, no code "
    "fence, no markdown, exactly these six boolean keys:\n"
    + "\n".join(f"- {k}: {_RUBRIC_DESCRIPTIONS[k]}" for k in ALL_FIELDS)
    + '\nOptional key: "rationale" (short string). Booleans must be true/false, '
    "never strings. JSON schema:\n"
    '{"task_completion": true, "data_retrieval_accuracy": true, '
    '"generalized_result_verification": true, "agent_sequence_correct": true, '
    '"clarity_and_justification": true, "hallucinations": false, '
    '"rationale": "..."}'
)


class SelfJudgingError(RuntimeError):
    """判官模型与被测模型同名/同源：禁止自评审。"""


# --------------------------------------------------------------------------- #
# 模型名归一化与同源判定


def _normalize_model_name(name: str) -> str:
    """归一化模型名：小写、去空白、取最后一段、剥掉常见后缀。"""
    if not name:
        return ""
    n = str(name).strip().lower()
    n = n.split("/")[-1]  # 去掉 org/前缀
    n = re.sub(r"\s+", "-", n)
    n = re.sub(r"-(instruct|chat|base|preview|it|q\d+)$", "", n)
    return n


def parse_pass_delta(raw: str | None) -> float:
    """解析 `AOB_JUDGE_PASS_DELTA`；空值回退 0.0，负数 / 非数 / NaN 抛 ValueError。

    delta 只作用在"投票通过比例"阈值上，不参与 score 计算。
    """
    if raw is None or not str(raw).strip():
        return DEFAULT_PASS_DELTA
    text = str(raw).strip()
    try:
        delta = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{JUDGE_ENV_PASS_DELTA} must be a number, got {text!r}") from exc
    if isinstance(delta, bool) or math.isnan(delta) or delta < 0:
        raise ValueError(
            f"{JUDGE_ENV_PASS_DELTA} must be a non-negative finite number, got {text!r}"
        )
    return delta


def is_self_judging(judge_model: str, judged_model: str) -> bool:
    """归一化后判官与被测同名，或一方是另一方带版本/规格后缀的变体，即拦截。

    注意：同族不同尺寸（如 qwen3.5-9b 与 qwen3.5-4b）不在此拦截——它们是
    不同模型；真正要拦的是"同一个模型换个写法/带日期后缀"这种自评审。
    """
    j = _normalize_model_name(judge_model)
    t = _normalize_model_name(judged_model)
    if not j or not t:
        return False
    if j == t:
        return True
    # 变体：deepseek-chat 与 deepseek-chat-2501 / deepseek-chat-v3
    return j.startswith(t) or t.startswith(j)


# --------------------------------------------------------------------------- #
# 结果结构


@dataclass(slots=True)
class JudgeResult:
    """一条轨迹的判官汇总结果（pass 与 score 分别落字段，不混用）。"""

    passed: bool
    score: float | None
    judge_model: str
    judged_model: str
    repeats: int
    valid_runs: int
    pass_votes: int
    fail_votes: int
    runs: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": None if self.score is None else round(self.score, 4),
            "judge_model": self.judge_model,
            "judged_model": self.judged_model,
            "repeats": self.repeats,
            "valid_runs": self.valid_runs,
            "pass_votes": self.pass_votes,
            "fail_votes": self.fail_votes,
            "runs": self.runs,
            "parse_errors": self.parse_errors,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# JSON 解析（宽松抽取，严格校验六维布尔）


def _extract_json_object(text: str) -> dict[str, Any]:
    """从返回文本里抽取首个 JSON 对象；失败抛 ValueError。"""
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("empty answer")

    candidates = [stripped]
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", stripped, re.S)
    if fence:
        candidates.append(fence.group(1).strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    last_err: Exception | None = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"not a JSON object: {last_err}")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _parse_run_output(text: str) -> dict[str, Any]:
    """解析单次判定输出；缺维/类型异常抛 ValueError。"""
    obj = _extract_json_object(text)
    missing = [k for k in ALL_FIELDS if k not in obj]
    if missing:
        raise ValueError(f"missing fields: {','.join(missing)}")
    verdict = {k: _coerce_bool(obj.get(k)) for k in ALL_FIELDS}
    verdict["rationale"] = str(obj.get("rationale") or "")[:200]
    return verdict


def _run_pass(verdict: dict[str, bool]) -> bool:
    return all(verdict[k] for k in RUBRIC_FIELDS) and not verdict[HALLUCINATION_FIELD]


def _run_score(verdict: dict[str, bool]) -> float:
    passed_dims = sum(1 for k in RUBRIC_FIELDS if verdict[k])
    score = passed_dims / len(RUBRIC_FIELDS)
    if verdict[HALLUCINATION_FIELD]:
        score -= HALLUCINATION_PENALTY
    return max(0.0, score)


# --------------------------------------------------------------------------- #
# 判官主体


class DeepSeekJudge:
    """DeepSeek 判官；后端可注入 fake，便于离线单测。"""

    def __init__(
        self,
        backend: Any,
        judge_model: str,
        judged_model: str,
        repeats: int = DEFAULT_REPEATS,
        temperature: float = DEFAULT_TEMPERATURE,
        pass_delta: float = DEFAULT_PASS_DELTA,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        if isinstance(pass_delta, bool) or not isinstance(pass_delta, (int, float)):
            raise ValueError(f"pass_delta must be a number, got {pass_delta!r}")
        if math.isnan(pass_delta) or pass_delta < 0:
            raise ValueError(f"pass_delta must be non-negative finite, got {pass_delta!r}")
        if is_self_judging(judge_model, judged_model):
            raise SelfJudgingError(
                f"self-judging blocked: judge model {judge_model!r} and judged model "
                f"{judged_model!r} are the same/same-source; a judge must be an "
                f"independent model."
            )
        self.backend = backend
        self.judge_model = judge_model
        self.judged_model = judged_model
        self.repeats = repeats
        self.temperature = temperature
        self.pass_delta = float(pass_delta)

    @classmethod
    def from_env(
        cls,
        judged_model: str,
        repeats: int = DEFAULT_REPEATS,
        temperature: float = DEFAULT_TEMPERATURE,
        backend: Any | None = None,
    ) -> "DeepSeekJudge":
        """按环境变量构造判官后端；`backend` 仅用于离线注入 fake。"""
        judge_model = os.environ.get(JUDGE_ENV_MODEL) or DEFAULT_MODEL
        base_url = os.environ.get(JUDGE_ENV_BASE_URL) or DEFAULT_BASE_URL
        api_key = os.environ.get(JUDGE_ENV_API_KEY)
        if backend is None:
            backend = OpenAIBackend(
                model=judge_model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=1024,
            )
        return cls(
            backend=backend,
            judge_model=judge_model,
            judged_model=judged_model,
            repeats=repeats,
            temperature=temperature,
            pass_delta=parse_pass_delta(os.environ.get(JUDGE_ENV_PASS_DELTA)),
        )

    # ------------------------------------------------------------------ #
    def judge(
        self, question: str, answer: str, judged_model: str | None = None, trajectory: str = ""
    ) -> JudgeResult:
        """对一条轨迹答案做 repeats 次判定。"""
        if judged_model is not None and judged_model != self.judged_model:
            if is_self_judging(self.judge_model, judged_model):
                raise SelfJudgingError(f"self-judging blocked for judged model {judged_model!r}")
            self.judged_model = judged_model

        user_content = self._build_user_prompt(
            question=question, answer=answer, trajectory=trajectory
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        runs: list[dict[str, Any]] = []
        parse_errors: list[dict[str, Any]] = []
        pass_votes = fail_votes = 0

        for idx in range(self.repeats):
            decision = self.backend.decide(messages, [])
            raw = getattr(decision, "text", "") or ""
            usage = getattr(self.backend, "last_usage", None) or {}
            run: dict[str, Any] = {
                "index": idx,
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                },
            }
            try:
                verdict = _parse_run_output(raw)
            except (ValueError, TypeError) as exc:
                err_detail = f"judge_parse_error: {exc}"
                run["error"] = err_detail
                run["raw_summary"] = raw[:MAX_RAW_TEXT_CHARS]
                parse_errors.append(run)
                continue

            run.update(verdict)
            run["passed"] = _run_pass(verdict)
            run["score"] = round(_run_score(verdict), 4)
            runs.append(run)
            if run["passed"]:
                pass_votes += 1
            else:
                fail_votes += 1

        valid = len(runs)
        if valid == 0:
            return JudgeResult(
                passed=False,
                score=None,
                judge_model=self.judge_model,
                judged_model=self.judged_model,
                repeats=self.repeats,
                valid_runs=0,
                pass_votes=0,
                fail_votes=0,
                runs=runs,
                parse_errors=parse_errors,
                error="judge_parse_error: all repeats failed to parse",
            )

        # pass 线：投票通过比例阈值；delta 只抬高比例门槛，绝不读 score。
        # delta=0 时 `pass_votes/valid > 0.5` 与旧口径 `pass_votes > fail_votes`
        # 逐位等价（平局 0.5 不满足严格 >，仍判 fail）。
        passed = (pass_votes / valid) > (0.5 + self.pass_delta)
        score = sum(r["score"] for r in runs) / valid

        return JudgeResult(
            passed=passed,
            score=round(score, 4),
            judge_model=self.judge_model,
            judged_model=self.judged_model,
            repeats=self.repeats,
            valid_runs=valid,
            pass_votes=pass_votes,
            fail_votes=fail_votes,
            runs=runs,
            parse_errors=parse_errors,
            error="",
        )

    def judge_scenario(self, scenario: Scenario, answer: str, trajectory: str = "") -> JudgeResult:
        """对契约场景判分；被测模型名取自场景无关的构造参数。"""
        return self.judge(
            question=scenario.text,
            answer=answer,
            trajectory=trajectory,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_user_prompt(question: str, answer: str, trajectory: str) -> str:
        parts = [f"User question:\n{question}", f"Assistant final answer:\n{answer}"]
        if trajectory:
            parts.append(f"Recorded tool sequence:\n{trajectory}")
        return "\n\n".join(parts)
