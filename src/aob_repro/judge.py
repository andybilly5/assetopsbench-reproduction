"""根级薄转发：判官实现位于 `aob_repro.evaluation.judge`。

保留本模块仅为兼容 `from aob_repro.judge import ...` 的调用路径；
真实实现、口径 docstring 与测试都在 evaluation 包内，不要在此重复逻辑。
"""

from .evaluation.judge import (  # noqa: F401  (re-export)
    ALL_FIELDS,
    HALLUCINATION_FIELD,
    RUBRIC_FIELDS,
    DeepSeekJudge,
    SelfJudgingError,
    is_self_judging,
)

__all__ = [
    "ALL_FIELDS",
    "DeepSeekJudge",
    "HALLUCINATION_FIELD",
    "RUBRIC_FIELDS",
    "SelfJudgingError",
    "is_self_judging",
]
