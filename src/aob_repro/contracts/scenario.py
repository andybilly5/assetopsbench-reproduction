"""场景契约。

对应论文 §4.2：一条场景 = 用户语句 + 期望行为（characteristic_form）+ 可选的结构化
标准答案。字段命名与上游 HuggingFace 数据集保持一致，便于直接读入
`all_utterance.jsonl`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 上游数据集里存在、但本复现不解释的字段，统一收进 `extra`
_KNOWN_FIELDS = {
    "id",
    "text",
    "type",
    "category",
    "deterministic",
    "characteristic_form",
    "expected_answer",
    "scoring_method",
    "group",
    "entity",
    "note",
}


@dataclass(slots=True)
class Scenario:
    """一条评测场景。"""

    id: str
    text: str
    type: str = ""
    characteristic_form: str = ""
    deterministic: bool = False
    expected_answer: Any = None
    scoring_method: str | None = None
    category: str = ""
    group: str = ""
    entity: str = ""
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Scenario":
        """从上游 JSONL 记录构造。

        `id` 在上游存在整数与字符串两种形态，统一强制成字符串——判分与轨迹
        都靠它做 join，类型不一致会导致"有 152 条场景却只匹配上 140 条"。
        """
        if "id" not in record or record["id"] is None:
            raise ValueError("scenario record missing 'id'")
        if "text" not in record or not str(record["text"]).strip():
            raise ValueError(f"scenario {record.get('id')!r} missing 'text'")

        return cls(
            id=str(record["id"]),
            text=str(record["text"]),
            type=str(record.get("type") or ""),
            characteristic_form=str(record.get("characteristic_form") or ""),
            deterministic=bool(record.get("deterministic", False)),
            expected_answer=record.get("expected_answer"),
            scoring_method=record.get("scoring_method"),
            category=str(record.get("category") or ""),
            group=str(record.get("group") or ""),
            entity=str(record.get("entity") or ""),
            note=str(record.get("note") or ""),
            extra={k: v for k, v in record.items() if k not in _KNOWN_FIELDS},
        )

    def to_record(self) -> dict[str, Any]:
        """回写成可序列化字典，`None` 与空串保持原样。"""
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type,
            "category": self.category,
            "deterministic": self.deterministic,
            "characteristic_form": self.characteristic_form,
            "expected_answer": self.expected_answer,
            "scoring_method": self.scoring_method,
            "group": self.group,
            "entity": self.entity,
            "note": self.note,
            **self.extra,
        }

    @property
    def has_expected(self) -> bool:
        """是否具备可做确定性判分的标准答案。"""
        return self.expected_answer is not None

    def effective_scorer(self, default: str = "llm_judge") -> str:
        """场景自带 `scoring_method` 时优先，否则用调用方默认。"""
        return self.scoring_method or default
