"""确定性判分（static_json）。

上游 `evaluation/scorers/static_json` 的语义是：把结构化答案从模型输出里解析
出来，与 `expected_answer` 逐键比对，**完全一致才算 pass**，部分正确只体现在
分数与细节里。

本实现刻意保留两条容易被忽略的语义：
1. 解析失败 ≠ 判负分，而是记 `parse_error`——否则"输出格式错"会被算成
   "答案错"，污染失败归因；
2. 列表按多重集比较（顺序无关），因为"列出全部资产"这类问题的正确性
   与输出顺序无关；但重复项会被计数，避免模型靠复制粘贴刷召回。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..contracts import Scenario

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


@dataclass(slots=True)
class ScorerResult:
    scorer: str
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "scorer": self.scorer,
            "passed": self.passed,
            "score": round(self.score, 4),
            "details": self.details,
            "rationale": self.rationale,
        }


def parse_structured(text: str) -> tuple[Any, str]:
    """尽力从自由文本里解析出结构化答案。

    返回 `(value, error)`；解析成功时 `error == ""`。
    """
    if not text or not text.strip():
        return None, "empty_answer"

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)

    for m in _FENCE.finditer(stripped):
        candidates.append(m.group(1).strip())

    # 截取第一个 { 或 [ 到最后一个配对符，处理"答案前后有解释文字"的情况
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if start != -1 and end > start:
            candidates.append(stripped[start : end + 1])

    for cand in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(cand), ""
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
    return None, "parse_error"


def _normalize(value: Any) -> Any:
    """数值统一成 float，字符串去空白；递归处理容器。"""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """把嵌套 dict 压成 `a.b.c -> 标量`；列表整体作为一个叶子值。"""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
        return out
    if isinstance(value, (list, tuple)):
        return {prefix: _canonical_list(value)}
    return {prefix: value}


def _canonical_list(items: Any) -> tuple:
    """列表转成可比较的多重集键（顺序无关，重复计数）。"""
    keys = []
    for item in items:
        if isinstance(item, dict):
            keys.append(json.dumps(_flatten(item), sort_keys=True, ensure_ascii=False))
        else:
            keys.append(json.dumps(item, sort_keys=True, ensure_ascii=False))
    return tuple(sorted(keys))


def compare(predicted: Any, expected: Any) -> dict[str, Any]:
    """逐键比对，返回精度/召回/F1 与缺失、多余键。"""
    p = _flatten(_normalize(predicted))
    e = _flatten(_normalize(expected))

    if isinstance(_normalize(expected), (list, tuple)):
        p = {"": _canonical_list(predicted)}
        e = {"": _canonical_list(expected)}

    matched = {k: p[k] for k in p.keys() & e.keys() if p[k] == e[k]}
    missing = sorted(e.keys() - matched.keys())
    extra = sorted(p.keys() - matched.keys())

    precision = len(matched) / len(p) if p else 0.0
    recall = len(matched) / len(e) if e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "exact_match": not missing and not extra,
        "matched_keys": len(matched),
        "missing_keys": missing,
        "extra_keys": extra,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def score(scenario: Scenario, answer: str) -> ScorerResult:
    """对一条场景的最终答案做确定性判分。"""
    if scenario.expected_answer is None:
        return ScorerResult(
            scorer="static_json",
            passed=False,
            score=0.0,
            details={"error": "scenario has no expected_answer"},
            rationale="无法做确定性判分：该场景未提供 expected_answer",
        )

    parsed, err = parse_structured(answer)
    if err:
        return ScorerResult(
            scorer="static_json",
            passed=False,
            score=0.0,
            details={"error": err},
            rationale=f"答案无法解析为结构化数据（{err}），按失败计",
        )

    details = compare(parsed, scenario.expected_answer)
    return ScorerResult(
        scorer="static_json",
        passed=bool(details["exact_match"]),
        score=float(details["f1"]),
        details=details,
        rationale="完全一致"
        if details["exact_match"]
        else (f"缺 {len(details['missing_keys'])} 键、多 {len(details['extra_keys'])} 键"),
    )
