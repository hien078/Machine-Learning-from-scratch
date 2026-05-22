from google import genai
from google.genai import types
from typing import List, Callable, Optional

class Agent:
    def __init__(self, name: str, role: str, api_key: str,
                 tools: Optional[List[Callable]] = None,
                 tool_registry=None):
        self.name = name
        self.role = role
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or []
        self.tool_registry = tool_registry

        # Initialize a chat session (this session will automatically retain the full memory/history)
        self.chat_session = self.client.chats.create(
            model='gemini-2.5-flash',
            config={
                'system_instruction': self.role,
                'tools': self.tools,
            }
        )

    def ask(self, message: str, max_turns: int = 10) -> str:
        """Send a message to the Agent; it will remember context and call Tools as needed."""
        response = self.chat_session.send_message(message)

        # Agentic loop: iterate until Gemini returns text (no more tool calls)
        turn = 0
        while response.function_calls and turn < max_turns:
            turn += 1
            parts = []
            for fc in response.function_calls:
                if self.tool_registry:
                    print(f"   🔧 [{self.name}] calls tool: {fc.name}({fc.args})")
                    result = self.tool_registry.execute(fc)
                else:
                    result = f"Error: agent '{self.name}' has no ToolRegistry to execute tools."

                parts.append(types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result}
                ))

            response = self.chat_session.send_message(parts)

        return response.text
