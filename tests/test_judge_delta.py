"""判官 pass 阈值开关 `AOB_JUDGE_PASS_DELTA` 单测。

盯三条不变量：
1. delta=0 时与旧口径 `pass_votes > fail_votes` 逐位等价（平局仍 fail）；
2. delta 只作用在投票通过比例上，绝不读 score；
3. 非法 delta（负数 / 非数 / NaN）显式报错。
"""

import json

import pytest

from aob_repro.contracts import Decision
from aob_repro.evaluation.judge import (
    ALL_FIELDS,
    DeepSeekJudge,
    parse_pass_delta,
)

JUDGE_MODEL = "deepseek-chat"
JUDGED_MODEL = "qwen3.5-9b"


def _verdict(**overrides: bool) -> str:
    base = {k: True for k in ALL_FIELDS}
    base["hallucinations"] = False
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


PASS = _verdict()
FAIL = _verdict(agent_sequence_correct=False)


class _FakeBackend:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 2}

    def decide(self, messages: list[dict], tools: list[dict]) -> Decision:
        return Decision(kind="final", text=self.texts.pop(0))


def _make(texts: list[str], pass_delta: float = 0.0) -> DeepSeekJudge:
    return DeepSeekJudge(
        backend=_FakeBackend(texts),
        judge_model=JUDGE_MODEL,
        judged_model=JUDGED_MODEL,
        repeats=len(texts),
        pass_delta=pass_delta,
    )


def _old_rule(pass_votes: int, fail_votes: int) -> bool:
    """改前口径：通过次数严格多于失败次数。"""
    return pass_votes > fail_votes


# ⑥ delta=0 等价性：多数票 / 平局 / 全失败逐位对齐旧公式
@pytest.mark.parametrize(
    "pass_n,fail_n",
    [(3, 2), (2, 3), (4, 1), (1, 4), (5, 0), (0, 5), (2, 2), (3, 1)],
)
def test_delta_zero_matches_old_majority_rule(pass_n, fail_n):
    texts = [PASS] * pass_n + [FAIL] * fail_n
    judge = _make(texts, pass_delta=0.0)
    res = judge.judge(question="q", answer="a")
    assert res.valid_runs == pass_n + fail_n
    assert res.pass_votes == pass_n and res.fail_votes == fail_n
    assert res.passed == _old_rule(pass_n, fail_n)


def test_delta_raises_threshold_on_pass_ratio():
    # 3/5 = 0.6：delta=0.1 阈值 0.6，严格 > 不成立 → fail
    judge = _make([PASS] * 3 + [FAIL] * 2, pass_delta=0.1)
    res = judge.judge(question="q", answer="a")
    assert res.passed is False
    # 5/5 = 1.0：delta=0.2 阈值 0.7 → 仍通过
    judge2 = _make([PASS] * 5, pass_delta=0.2)
    assert judge2.judge("q", "a").passed is True


def test_delta_does_not_touch_score():
    texts = [PASS] * 3 + [FAIL] * 2
    score0 = _make(texts, 0.0).judge("q", "a").score
    score_high = _make(texts, 0.1).judge("q", "a").score
    assert score0 == pytest.approx(score_high)
    # 分数仍由各维 true 比例决定，与票数结论无关
    assert score0 == pytest.approx((1.0 + 1.0 + 1.0 + 0.8 + 0.8) / 5)


def test_parse_pass_delta_defaults_and_valid_values():
    assert parse_pass_delta(None) == 0.0
    assert parse_pass_delta("") == 0.0
    assert parse_pass_delta("0.1") == pytest.approx(0.1)
    assert parse_pass_delta("0") == 0.0


@pytest.mark.parametrize("bad", ["-0.1", "abc", "nan", "1e-3,2"])
def test_parse_pass_delta_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_pass_delta(bad)


def test_constructor_rejects_negative_delta():
    with pytest.raises(ValueError):
        DeepSeekJudge(
            backend=_FakeBackend([]),
            judge_model=JUDGE_MODEL,
            judged_model=JUDGED_MODEL,
            pass_delta=-0.01,
        )


def test_from_env_reads_delta(monkeypatch):
    monkeypatch.setenv("AOB_JUDGE_PASS_DELTA", "0.1")
    judge = DeepSeekJudge.from_env(JUDGED_MODEL, backend=_FakeBackend([PASS] * 5))
    assert judge.pass_delta == pytest.approx(0.1)
