"""
llm.py — a small OpenAI tool-calling agent runtime.

`LLMAgent` wraps an OpenAI chat model and runs the standard agentic loop:

    1. send the conversation (system + task) to the model, advertising the
       agent's tools;
    2. if the model asks to call tools, execute them and feed the results back;
    3. repeat until the model returns a plain-text answer (or the step budget
       is exhausted).

The runtime is generic — it knows nothing about nutrition or risk. The two
agents in `nutrition_agent.py` and `risk_agent.py` are just an LLMAgent with a
domain system prompt and a set of `Tool`s. Tools do the precise/deterministic
work (data fetches, the ML call); the model decides when to call them and turns
the results into language.

The OpenAI client is injectable so the loop can be unit-tested with a fake
client and no network/API key.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("agents.llm")

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")


@dataclass
class Tool:
    """A function the model can call, plus its JSON-schema description."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for the arguments
    fn: Callable[..., Any]

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# Tool that takes no arguments — the common case here, since each tool is bound
# to a fixed user_id at construction time (the model never passes IDs).
NO_ARGS: Dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


class LLMAgent:
    """An OpenAI-backed agent that can call tools to accomplish a task."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Optional[List[Tool]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_steps: int = 6,
        client: Any = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.max_steps = max_steps
        if client is None:
            from openai import OpenAI  # imported lazily: tests can inject a fake

            client = OpenAI()
        self.client = client

    def run(self, task: str) -> str:
        """Run the agent on `task` and return its final natural-language answer."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        tool_schemas = [t.schema() for t in self.tools.values()] or None

        for _ in range(self.max_steps):
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas
            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            messages.append(self._assistant_dict(msg))

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return (msg.content or "").strip()

            for call in tool_calls:
                result = self._dispatch(call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, default=str),
                    }
                )

        # Step budget exhausted — ask once more for a final answer, no tools.
        logger.warning("agent=%s hit max_steps=%d", self.name, self.max_steps)
        messages.append(
            {"role": "user", "content": "Stop using tools and give your final answer now."}
        )
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        return (response.choices[0].message.content or "").strip()

    # ---- internals -------------------------------------------------------
    def _dispatch(self, call: Any) -> Any:
        """Execute one tool call and return a JSON-serialisable result."""
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            return {"error": "could not parse tool arguments as JSON"}

        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'"}

        logger.info("agent=%s calling tool=%s args=%s", self.name, name, args)
        try:
            return tool.fn(**args)
        except Exception as e:  # noqa: BLE001 — surface the error to the model, not the user
            logger.warning("agent=%s tool=%s failed: %s", self.name, name, e)
            return {"error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _assistant_dict(msg: Any) -> Dict[str, Any]:
        """Convert an OpenAI assistant message back into a plain dict for replay.

        Building the dict by hand (rather than dumping the SDK object) keeps the
        conversation payload stable across SDK versions and strips null fields
        the API rejects on the next turn.
        """
        out: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return out
