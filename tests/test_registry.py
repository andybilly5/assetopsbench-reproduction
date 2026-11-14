"""工具注册表单测：参数校验的松紧直接决定"格式失败"能否被单独统计。"""

from aob_repro.contracts import ToolCall
from aob_repro.tools.registry import ToolRegistry, ToolSpec


def _registry() -> ToolRegistry:
    def echo(site_name: str, mode: str = "fast") -> dict:
        return {"site": site_name, "mode": mode}

    return ToolRegistry(
        [
            ToolSpec(
                name="demo.echo",
                description="echo back",
                parameters={
                    "type": "object",
                    "properties": {
                        "site_name": {"type": "string"},
                        "mode": {"type": "string", "enum": ["fast", "slow"]},
                        "limit": {"type": "integer"},
                    },
                    "required": ["site_name"],
                },
                handler=echo,
            )
        ]
    )


def test_unknown_tool_reported_not_raised():
    reg = _registry()
    r = reg.dispatch(ToolCall(name="nope", arguments={}))
    assert r.ok is False
    assert "unknown tool" in r.error


def test_missing_required():
    reg = _registry()
    r = reg.dispatch(ToolCall(name="demo.echo", arguments={}))
    assert r.ok is False
    assert "missing required" in r.error


def test_unexpected_parameter_rejected():
    reg = _registry()
    r = reg.dispatch(ToolCall(name="demo.echo", arguments={"site_name": "MAIN", "bogus": 1}))
    assert r.ok is False
    assert "unknown parameter" in r.error


def test_type_mismatch_and_bool_trap():
    reg = _registry()
    r = reg.dispatch(ToolCall(name="demo.echo", arguments={"site_name": 5}))
    assert r.ok is False
    assert "expects string" in r.error

    # bool 是 int 的子类，必须被拦下，否则 true 会被当 1 接受
    r2 = reg.dispatch(ToolCall(name="demo.echo", arguments={"site_name": "MAIN", "limit": True}))
    assert r2.ok is False
    assert "boolean" in r2.error


def test_enum_enforced():
    reg = _registry()
    r = reg.dispatch(ToolCall(name="demo.echo", arguments={"site_name": "MAIN", "mode": "turbo"}))
    assert r.ok is False
    assert "one of" in r.error


def test_handler_exception_becomes_observation():
    def boom(**kwargs):
        raise RuntimeError("backend down")

    reg = ToolRegistry(
        [
            ToolSpec(
                name="demo.boom",
                description="x",
                parameters={"type": "object", "properties": {}},
                handler=boom,
            )
        ]
    )
    r = reg.dispatch(ToolCall(name="demo.boom", arguments={}))
    assert r.ok is False
    assert "RuntimeError" in r.error
    assert r.as_observation().startswith("[TOOL_ERROR]")


def test_openai_schema_shape():
    reg = _registry()
    schema = reg.openai_tools()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "demo.echo"
    assert "parameters" in schema["function"]
