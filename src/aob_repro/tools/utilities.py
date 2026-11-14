"""通用 utilities 领域工具（本地只读、确定性）。

## provenance（来源声明，随任何引用本模块产出的分数一起声明）

本模块 6 个工具中：

- `jsonreader`、`currentdatetime` 这 2 个名字来自论文 Table 4 明列；
- `csvreader`、`textsearch`、`unitconvert`、`calculator` 这 4 个按通用 utilities
  语义补齐（论文未逐一列名）。

**这些实现是本仓库自建的等价能力，不等同于上游任何 utilities MCP server 的实现**：
上游源码拿不到，这里只对齐"可枚举、只读本地数据、确定性可复算"的接口语义，
不声称数值/行为与上游一致。

## 实现约束

- 只读 `data/` 下本地文件；不联网、不写盘；
- 除 `currentdatetime` 取当前时刻外，全部为纯函数式确定性计算；
- 参数缺失 / 非法 / 文件不可得时抛明确错误，由注册表层转成 `ok=False` 观察，
  不静默返回空结果。
"""

from __future__ import annotations

import ast
import csv
import json
import operator
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..data.repositories import DataRoot, DataUnavailable
from .registry import ToolRegistry, ToolSpec

#: 统一来源字符串：同时被每个工具的 `provenance` 字段与本模块 docstring 引用。
UTILITIES_PROVENANCE = (
    "provenance: 2 tool names (jsonreader, currentdatetime) are listed explicitly "
    "in the paper Table 4; the other 4 (csvreader, textsearch, unitconvert, "
    "calculator) are inferred by general utilities semantics. These implementations "
    "are self-built in this repository and are NOT equivalent to any upstream "
    "implementation."
)

#: 允许被路径工具读取的文件后缀（白名单，避免把数据目录之外的文件暴露给模型）。
_ALLOWED_SUFFIXES = {".csv", ".json", ".jsonl", ".txt"}

_LENGTH_TO_METER = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
    "mi": 1609.344,
}
_MASS_TO_KG = {
    "mg": 0.000001,
    "g": 0.001,
    "kg": 1.0,
    "lb": 0.45359237,
    "oz": 0.028349523125,
}
_TEMPERATURE_UNITS = {"celsius", "fahrenheit", "kelvin"}

#: calculator 允许的 AST 二元 / 一元运算
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _resolve_file(root: DataRoot, name: str) -> Path:
    """把相对路径解析到数据根目录内；拒绝越界路径与白名单外后缀。"""
    if not isinstance(name, str) or not name.strip():
        raise DataUnavailable("file parameter is missing or empty")
    p = Path(name)
    if p.is_absolute():
        raise DataUnavailable(f"absolute paths are not allowed: {name!r}")
    if ".." in p.parts:
        raise DataUnavailable(f"path traversal is not allowed: {name!r}")
    resolved = (root.path / p).resolve()
    root_resolved = root.path.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise DataUnavailable(f"path escapes data root: {name!r}")
    if resolved.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise DataUnavailable(
            f"unsupported file suffix {resolved.suffix!r}; allowed: {sorted(_ALLOWED_SUFFIXES)}"
        )
    if not resolved.is_file():
        raise DataUnavailable(f"missing data file: {resolved}")
    return resolved


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """摄氏作为中间单位做温度换算。"""
    if from_unit == to_unit:
        return value
    if from_unit == "fahrenheit":
        celsius = (value - 32.0) * 5.0 / 9.0
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        celsius = value
    if to_unit == "fahrenheit":
        return celsius * 9.0 / 5.0 + 32.0
    if to_unit == "kelvin":
        return celsius + 273.15
    return celsius


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("booleans are not valid in a numeric expression")
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _safe_calculate(expression: str) -> float:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression is missing or empty")
    if not re.fullmatch(r"[0-9+\-*/().\s%]+", expression):
        raise ValueError("expression may only contain numbers and + - * / // % ( )")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc.msg}") from exc
    return _eval_node(tree)


def build_utilities_tools(root: DataRoot | str | Path | None = None) -> list[ToolSpec]:
    """构造 6 个 utilities 工具；`root` 为数据根目录，默认走环境变量/相对路径。"""
    data_root = root if isinstance(root, DataRoot) else DataRoot.resolve(root)

    def jsonreader(file: str, key: str | None = None) -> dict[str, Any]:
        path = _resolve_file(data_root, file)
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataUnavailable(f"invalid JSON in {file!r}: {exc.msg}") from exc
        if key is not None:
            if not isinstance(payload, dict) or key not in payload:
                raise DataUnavailable(f"top-level key {key!r} not found in {file!r}")
            payload = payload[key]
        return {"file": file, "key": key, "data": payload}

    def currentdatetime() -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "iso": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.time().isoformat(timespec="seconds"),
            "timezone": "UTC",
        }

    def csvreader(
        file: str,
        where_column: str | None = None,
        where_value: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        path = _resolve_file(data_root, file)
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise DataUnavailable(f"CSV has no rows: {file!r}")
        columns = list(rows[0].keys())
        if where_column is not None:
            if where_column not in columns:
                raise DataUnavailable(f"column {where_column!r} not found; columns: {columns}")
            target = (where_value or "").strip().lower()
            rows = [r for r in rows if str(r[where_column]).strip().lower() == target]
            if not rows:
                raise DataUnavailable(
                    f"no rows where {where_column!r} == {where_value!r} in {file!r}"
                )
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            rows = rows[:limit]
        return {"file": file, "columns": columns, "count": len(rows), "rows": rows}

    def textsearch(file: str, query: str, case_sensitive: bool = False) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is missing or empty")
        path = _resolve_file(data_root, file)
        text = path.read_text(encoding="utf-8-sig")
        needle = query if case_sensitive else query.lower()
        matches = [
            {"line_number": idx, "line": line.rstrip("\n")}
            for idx, line in enumerate(text.splitlines(), start=1)
            if needle in (line if case_sensitive else line.lower())
        ]
        if not matches:
            raise DataUnavailable(f"no match for {query!r} in {file!r}")
        return {
            "file": file,
            "query": query,
            "case_sensitive": case_sensitive,
            "match_count": len(matches),
            "matches": matches,
        }

    def unitconvert(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("value must be a number")
        if from_unit == to_unit:
            result = float(value)
            family = "identity"
        elif from_unit in _LENGTH_TO_METER and to_unit in _LENGTH_TO_METER:
            result = value * _LENGTH_TO_METER[from_unit] / _LENGTH_TO_METER[to_unit]
            family = "length"
        elif from_unit in _MASS_TO_KG and to_unit in _MASS_TO_KG:
            result = value * _MASS_TO_KG[from_unit] / _MASS_TO_KG[to_unit]
            family = "mass"
        elif from_unit in _TEMPERATURE_UNITS and to_unit in _TEMPERATURE_UNITS:
            result = _convert_temperature(float(value), from_unit, to_unit)
            family = "temperature"
        else:
            raise ValueError(
                f"unsupported unit pair {from_unit!r} -> {to_unit!r}; supported: "
                f"length {sorted(_LENGTH_TO_METER)}, mass {sorted(_MASS_TO_KG)}, "
                f"temperature {sorted(_TEMPERATURE_UNITS)}"
            )
        return {
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "family": family,
            "result": round(result, 10),
        }

    def calculator(expression: str) -> dict[str, Any]:
        result = _safe_calculate(expression)
        return {"expression": expression, "result": result}

    return [
        ToolSpec(
            name="jsonreader",
            description=(
                "Read a local JSON file under the data directory and return its "
                "parsed contents; optionally return one top-level object key."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Relative path under data/."},
                    "key": {"type": "string", "description": "Optional top-level key."},
                },
                "required": ["file"],
            },
            handler=jsonreader,
            tags=["utilities", "read"],
            provenance=UTILITIES_PROVENANCE,
        ),
        ToolSpec(
            name="currentdatetime",
            description="Return the current UTC date and time as ISO-8601 strings.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=currentdatetime,
            tags=["utilities", "clock"],
            provenance=UTILITIES_PROVENANCE,
        ),
        ToolSpec(
            name="csvreader",
            description=(
                "Read a local CSV file under the data directory; optionally filter "
                "rows by an exact column/value match and cap the row count."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Relative path under data/."},
                    "where_column": {"type": "string"},
                    "where_value": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["file"],
            },
            handler=csvreader,
            tags=["utilities", "read"],
            provenance=UTILITIES_PROVENANCE,
        ),
        ToolSpec(
            name="textsearch",
            description=(
                "Search a local text/CSV/JSON file under the data directory for a "
                "substring; return matching 1-based line numbers and lines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Relative path under data/."},
                    "query": {"type": "string", "description": "Substring to search."},
                    "case_sensitive": {"type": "boolean"},
                },
                "required": ["file", "query"],
            },
            handler=textsearch,
            tags=["utilities", "read"],
            provenance=UTILITIES_PROVENANCE,
        ),
        ToolSpec(
            name="unitconvert",
            description=(
                "Convert a numeric value between supported length, mass or "
                "temperature units. Deterministic local arithmetic only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
            handler=unitconvert,
            tags=["utilities", "math"],
            provenance=UTILITIES_PROVENANCE,
        ),
        ToolSpec(
            name="calculator",
            description=(
                "Evaluate a deterministic numeric arithmetic expression "
                "(+ - * / // % ( )); no variables, functions or side effects."
            ),
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=calculator,
            tags=["utilities", "math"],
            provenance=UTILITIES_PROVENANCE,
        ),
    ]


def build_utilities_registry(root: DataRoot | str | Path | None = None) -> ToolRegistry:
    """只装 utilities 工具的注册表。"""
    return ToolRegistry(build_utilities_tools(root))
