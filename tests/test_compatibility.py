"""旧包路径须可导入，防止目录重构后保留无效的相对导入。"""


def test_legacy_react_module_imports():
    from aob_repro import react_loop
    from aob_repro.agents import ReActLoop

    assert react_loop.ReActLoop is ReActLoop
