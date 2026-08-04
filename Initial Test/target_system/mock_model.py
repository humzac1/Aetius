"""A deterministic Agno Model backend that returns scripted responses instead
of calling a real provider. Lets the full pipeline (orchestration, logging,
policy evaluation, and later the statistics module) be developed and tested
with zero API spend.

Grounded against agno==2.6.17's actual contract (verified by reading
agno/models/base.py rather than assumed):
  - Model.invoke(...) is called directly by Model._invoke_with_retry and its
    return value is used as the already-parsed ModelResponse — providers
    normally call a raw SDK inside invoke() and convert with
    _parse_provider_response(), but a mock has no raw SDK response to parse,
    so invoke() builds and returns the ModelResponse directly.
  - assistant_message.tool_calls expects a list of OpenAI-style dicts:
    {"id", "type": "function", "function": {"name", "arguments": <json str>}}.
  - response_usage on ModelResponse must be a MessageMetrics (input_tokens/
    output_tokens/total_tokens), which agno accumulates into the run's
    RunMetrics automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator
from uuid import uuid4

from agno.models.base import Model
from agno.models.message import Message, MessageMetrics
from agno.models.response import ModelResponse


@dataclass
class MockToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class MockStep:
    """One scripted model turn. If tool_calls is non-empty, agno will execute
    those tools and call invoke() again with the results appended to
    messages, advancing to the next step."""

    content: str | None = None
    tool_calls: list[MockToolCall] = field(default_factory=list)


def _estimate_tokens(text: str) -> int:
    # Deterministic, provider-independent stand-in for a real tokenizer.
    # Good enough for exercising token_usage plumbing offline; real token
    # counts come from the Anthropic arm.
    return max(1, len(text) // 4)


class MockModel(Model):
    def __init__(self, agent_label: str, steps: list[MockStep], **kwargs: Any) -> None:
        kwargs.setdefault("id", f"mock-scripted-{agent_label}")
        kwargs.setdefault("name", "MockScripted")
        kwargs.setdefault("provider", "mock")
        super().__init__(**kwargs)
        self.agent_label = agent_label
        self.steps = steps
        self._step_idx = 0

    def _next_step(self) -> MockStep:
        if self._step_idx >= len(self.steps):
            # Script ran out — fall back to a plain closing message rather
            # than crashing an otherwise-working run.
            return MockStep(content="(mock backend: no more scripted steps)")
        step = self.steps[self._step_idx]
        self._step_idx += 1
        return step

    def _build_response(self, messages: list[Message]) -> ModelResponse:
        step = self._next_step()

        prompt_tokens = _estimate_tokens("".join(m.get_content_string() or "" for m in messages))
        completion_tokens = _estimate_tokens(step.content or "") + 15 * len(step.tool_calls)
        usage = MessageMetrics(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        tool_calls = None
        if step.tool_calls:
            tool_calls = [
                {
                    "id": f"call_{uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                }
                for tc in step.tool_calls
            ]

        return ModelResponse(
            role="assistant",
            content=step.content,
            tool_calls=tool_calls,
            response_usage=usage,
        )

    # --- Model abstract interface -----------------------------------

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        messages: list[Message] = kwargs.get("messages") or (args[0] if args else [])
        return self._build_response(messages)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self.invoke(*args, **kwargs)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self.invoke(*args, **kwargs)

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        yield self.invoke(*args, **kwargs)

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        # Unused: invoke() builds the ModelResponse directly since there is
        # no raw provider response to parse. Required to satisfy the
        # abstract base class.
        assert isinstance(response, ModelResponse)
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        assert isinstance(response, ModelResponse)
        return response
