"""DeepSeek 判官单测：全部使用 fake backend，不联网、不耗 token、不依赖 key。"""

import json

import pytest

from aob_repro.contracts import Decision
from aob_repro.evaluation.judge import (
    ALL_FIELDS,
    DeepSeekJudge,
    SelfJudgingError,
)

JUDGE_MODEL = "deepseek-chat"
JUDGED_MODEL = "qwen3.5-9b"


def _verdict(**overrides: bool) -> str:
    base = {k: True for k in ALL_FIELDS}
    base["hallucinations"] = False
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


class _FakeBackend:
    """按队列依次返回终态文本；记录模型名与 usage 字段。"""

    def __init__(self, texts: list[str], model: str = JUDGE_MODEL) -> None:
        self.texts = list(texts)
        self.model = model
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 2}

    def decide(self, messages: list[dict], tools: list[dict]) -> Decision:
        return Decision(kind="final", text=self.texts.pop(0))


def _make_judge(texts: list[str], judged: str = JUDGED_MODEL, repeats: int = 5) -> DeepSeekJudge:
    return DeepSeekJudge(
        backend=_FakeBackend(texts),
        judge_model=JUDGE_MODEL,
        judged_model=judged,
        repeats=repeats,
    )


# (a) 六维全真 → pass
def test_all_dimensions_true_passes():
    judge = _make_judge([_verdict()] * 5)
    res = judge.judge(question="q", answer="a")
    assert res.passed is True
    assert res.score == 1.0
    assert res.valid_runs == 5
    assert res.parse_errors == []


# (b) 出现幻觉 → fail，且单次得分扣 0.2（1.0 - 0.2 = 0.8）
def test_hallucination_fails_and_penalizes_score():
    judge = _make_judge([_verdict(hallucinations=True)] * 5)
    res = judge.judge(question="q", answer="a")
    assert res.passed is False
    assert res.score == pytest.approx(0.8)
    assert all(r["hallucinations"] is True for r in res.runs)
    assert res.fail_votes == 5


# (c) 某一维为 false → fail，得分 4/5 = 0.8
def test_one_dimension_false_fails():
    judge = _make_judge([_verdict(data_retrieval_accuracy=False)] * 5)
    res = judge.judge(question="q", answer="a")
    assert res.passed is False
    assert res.score == pytest.approx(0.8)
    assert res.fail_votes == 5


# (d) 5 次多数票 3:2：pass 由多数票决定，而非均值反推
def test_pass_follows_majority_vote_not_mean_score():
    texts = [
        _verdict(),  # pass
        _verdict(),  # pass
        _verdict(),  # pass
        _verdict(agent_sequence_correct=False),  # fail
        _verdict(clarity_and_justification=False),  # fail
    ]
    judge = _make_judge(texts)
    res = judge.judge(question="q", answer="a")
    assert res.pass_votes == 3
    assert res.fail_votes == 2
    assert res.passed is True
    # 均值介于两者之间，但 pass 结论只看票数
    assert res.score == pytest.approx((1.0 + 1.0 + 1.0 + 0.8 + 0.8) / 5)


# (e) JSON 解析失败 → judge_parse_error 计入失败归因，不当 0 分
def test_parse_error_is_attributed_not_scored_zero():
    garbage = ["模型今天心情不好，无法给出 JSON"] * 5
    judge = _make_judge(garbage)
    res = judge.judge(question="q", answer="a")
    assert res.passed is False
    assert res.score is None  # 不是 0.0
    assert res.valid_runs == 0
    assert len(res.parse_errors) == 5
    assert res.error.startswith("judge_parse_error")
    assert all("raw_summary" in e for e in res.parse_errors)


# 混合：4 次有效 + 1 次解析失败，失败不计入投票与均分分母
def test_mixed_parse_error_excluded_from_vote_and_mean():
    texts = [_verdict()] * 4 + ["not json"]
    judge = _make_judge(texts)
    res = judge.judge(question="q", answer="a")
    assert res.valid_runs == 4
    assert len(res.parse_errors) == 1
    assert res.passed is True
    assert res.pass_votes == 4
    assert res.score == 1.0


# (f) 自评审拦截：judge 与被测同名/同源 → 抛自定义异常
def test_self_judging_is_blocked():
    with pytest.raises(SelfJudgingError):
        DeepSeekJudge(
            backend=_FakeBackend([]),
            judge_model=JUDGE_MODEL,
            judged_model=JUDGE_MODEL,
        )


def test_self_judging_blocked_on_version_variant():
    with pytest.raises(SelfJudgingError):
        DeepSeekJudge(
            backend=_FakeBackend([]),
            judge_model="deepseek-chat",
            judged_model="deepseek-chat-2501",
        )


def test_self_judging_allows_independent_model():
    # 不同模型不抛错
    judge = DeepSeekJudge(
        backend=_FakeBackend([_verdict()] * 5),
        judge_model=JUDGE_MODEL,
        judged_model="qwen3.5-9b",
    )
    assert judge.judge("q", "a").passed is True


def test_case_insensitive_and_org_prefix_normalized():
    # "Provider/DeepSeek-Chat" 与 "deepseek-chat" 视为同名
    with pytest.raises(SelfJudgingError):
        DeepSeekJudge(
            backend=_FakeBackend([]),
            judge_model="deepseek/deepseek-chat",
            judged_model="DeepSeek-Chat",
        )
