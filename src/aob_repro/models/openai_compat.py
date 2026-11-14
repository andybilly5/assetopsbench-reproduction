"""模型后端：OpenAI 兼容端点（stdlib 实现，零第三方依赖）。

为什么不用官方 SDK：这个复现要在 vLLM、云端 API、离线脚本三种后端之间切换，
只依赖 HTTP 协议可以复用后端接口。通过 `AOB_BASE_URL`、`AOB_API_KEY` 环境变量
配置服务地址和认证，模型名称使用 CLI 的 `--model` 参数传入。

`extra_body` 用于透传 Qwen 的开关，例如：
    {"chat_template_kwargs": {"enable_thinking": False}}   # 关闭 thinking
    {"guided_json": {...}}                                # vLLM 约束解码
这两个开关都会显著改变分数，必须写进运行配置并在报告中声明。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from ..contracts import Decision, ToolCall


class BackendError(RuntimeError):
    """后端调用失败（不可重试或重试耗尽）。"""


class OpenAIBackend:
    """OpenAI `/chat/completions` 兼容后端。"""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url or os.environ.get("AOB_BASE_URL") or "http://localhost:8000/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("AOB_API_KEY") or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    # ------------------------------------------------------------------ #
    def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Decision:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update(self.extra_body)

        data = self._post(payload)
        return self._to_decision(data)

    # ------------------------------------------------------------------ #
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = ""
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {exc.read()[:500].decode('utf-8', 'replace')}"
                if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise BackendError(last_error) from exc
            except urllib.error.URLError as exc:
                last_error = f"connection error: {exc}"
            if attempt < self.max_retries:
                time.sleep(2**attempt)
        raise BackendError(
            f"request to {url} failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _to_decision(self, data: dict[str, Any]) -> Decision:
        usage = data.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }

        choices = data.get("choices") or []
        if not choices:
            raise BackendError(f"no choices in response: {json.dumps(data)[:300]}")
        message = choices[0].get("message") or {}

        raw_calls = message.get("tool_calls") or []
        calls: list[ToolCall] = []
        for idx, raw in enumerate(raw_calls):
            fn = raw.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    # 参数不是合法 JSON 也要记下来：这是"格式遵循"类失败的证据
                    args = {"__raw__": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            calls.append(
                ToolCall(name=name, arguments=args, call_id=raw.get("id") or f"call_{idx}")
            )

        content = message.get("content") or ""
        if calls:
            return Decision(kind="tool", text=content, tool_calls=calls)
        return Decision(kind="final", text=content)
