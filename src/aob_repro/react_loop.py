"""兼容旧导入路径；ReAct 循环的唯一实现位于 agents.react_loop。

避免在两个文件中维护重复逻辑。新代码请直接导入 aob_repro.agents。
"""

from .agents.react_loop import DEFAULT_SYSTEM_PROMPT, ReActLoop, run_single

__all__ = ["DEFAULT_SYSTEM_PROMPT", "ReActLoop", "run_single"]
