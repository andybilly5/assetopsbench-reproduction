"""契约层单测：ID 归一化与字段保真。"""

import pytest

from aob_repro.contracts import Scenario


def test_id_coerced_to_str():
    s = Scenario.from_record({"id": 101, "text": "List failure modes of Chiller."})
    assert s.id == "101"
    assert isinstance(s.id, str)


def test_missing_text_rejected():
    with pytest.raises(ValueError):
        Scenario.from_record({"id": 7, "text": "   "})


def test_missing_id_rejected():
    with pytest.raises(ValueError):
        Scenario.from_record({"text": "hello"})


def test_roundtrip_keeps_fields():
    rec = {
        "id": "507",
        "text": "What is the predicted energy consumption for Chiller 9?",
        "type": "TSFM",
        "category": "Analytical",
        "deterministic": False,
        "characteristic_form": "should confirm actions",
        "group": "prospective",
        "entity": "Chiller",
        "note": "source",
    }
    s = Scenario.from_record(rec)
    out = s.to_record()
    for key in ("id", "text", "type", "category", "deterministic", "characteristic_form"):
        assert out[key] == rec[key]


def test_unknown_fields_kept_in_extra():
    s = Scenario.from_record({"id": "1", "text": "x", "custom_field": 42})
    assert s.extra["custom_field"] == 42


def test_scorer_default_and_override():
    s = Scenario.from_record({"id": "2", "text": "x"})
    assert s.effective_scorer() == "llm_judge"
    s2 = Scenario.from_record({"id": "3", "text": "x", "scoring_method": "static_json"})
    assert s2.effective_scorer() == "static_json"
