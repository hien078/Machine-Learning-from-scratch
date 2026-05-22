"""
AI Agent using OpenAI-compatible API (FreeModel).

Implements the agentic loop:
  User message → LLM → tool_calls? → execute tools → feed results → LLM → ... → final text
"""

import json
import requests


class Agent:
    """An AI agent that can autonomously call tools via OpenAI function calling."""

    def __init__(self, api_key, base_url, model, tool_registry=None, system_prompt=None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tool_registry = tool_registry
        self.messages = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _call_api(self):
        """Send current messages to the LLM and return the response."""
        payload = {
            "model": self.model,
            "messages": self.messages,
        }
        if self.tool_registry:
            tools = self.tool_registry.get_tools_json()
            if tools:
                payload["tools"] = tools

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        if not resp.ok:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")

        return resp.json()["choices"][0]["message"]

    def ask(self, user_message, max_turns=10):
        """Send a message and let the agent reason + call tools until it has a final answer."""
        self.messages.append({"role": "user", "content": user_message})

        for turn in range(max_turns):
            assistant_msg = self._call_api()

            # Append assistant message to history
            self.messages.append(assistant_msg)

            # Check if agent wants to call tools
            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                # No tool calls → final text response
                return assistant_msg.get("content", "")

            # Execute each tool call
            for tc in tool_calls:
                func = tc["function"]
                tool_name = func["name"]
                tool_args = func.get("arguments", "{}")

                print(f"   🔧 Calling: {tool_name}({_truncate(tool_args, 80)})")

                if self.tool_registry:
                    result = self.tool_registry.execute(tool_name, tool_args)
                else:
                    result = f"Error: no tool registry configured."

                print(f"   📋 Result: {_truncate(result, 120)}")

                # Feed tool result back to conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        return "(Agent reached max turns without a final answer)"

    def reset(self):
        """Clear conversation history, keeping only the system prompt."""
        system = [m for m in self.messages if m["role"] == "system"]
        self.messages = system


def _truncate(text, max_len):
    text = text.replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text
