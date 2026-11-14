"""确定性判分单测。"""

from aob_repro.contracts import Scenario
from aob_repro.evaluation.static_json import compare, parse_structured, score


def _scenario(expected, sid="1"):
    return Scenario.from_record({"id": sid, "text": "q", "expected_answer": expected})


def test_parse_plain_json():
    v, err = parse_structured('{"a": 1}')
    assert err == "" and v == {"a": 1}


def test_parse_fenced_json():
    text = '答案如下：\n```json\n{"sites": ["MAIN"]}\n```\n以上。'
    v, err = parse_structured(text)
    assert err == "" and v == {"sites": ["MAIN"]}


def test_parse_failure_is_flagged_not_scored_as_wrong_answer():
    v, err = parse_structured("MAIN 站点一共有 12 台设备")
    assert err == "parse_error"


def test_exact_match_passes():
    res = score(_scenario({"sites": ["MAIN", "PLANT"]}), '{"sites": ["MAIN", "PLANT"]}')
    assert res.passed is True
    assert res.score == 1.0


def test_list_order_insensitive():
    res = score(_scenario(["Chiller 6", "Chiller 9"]), '["Chiller 9", "Chiller 6"]')
    assert res.passed is True


def test_partial_match_scores_between_zero_and_one():
    res = score(_scenario({"a": 1, "b": 2}), '{"a": 1}')
    assert res.passed is False
    assert 0.0 < res.score < 1.0
    assert res.details["missing_keys"] == ["b"]


def test_numeric_normalization():
    res = score(_scenario({"value": 12.0}), '{"value": 12}')
    assert res.passed is True


def test_no_ground_truth():
    s = Scenario.from_record({"id": "9", "text": "q"})
    res = score(s, "anything")
    assert res.passed is False
    assert "no expected_answer" in res.details["error"]


def test_compare_reports_extra_keys():
    d = compare({"a": 1, "zzz": 9}, {"a": 1})
    assert d["exact_match"] is False
    assert d["extra_keys"] == ["zzz"]
