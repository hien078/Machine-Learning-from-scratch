"""
Interactive terminal chat with the AI Agent.

Usage:
    python main.py
"""

import sys
import os

# Fix Vietnamese encoding on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"

from agent import Agent
from tools import registry

API_KEY = "fe_oa_9f4973c54fdc11c161bd56a9d351ddbe256e26581c9570eb"
BASE_URL = "https://api.freemodel.dev"
MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
When a user asks something that requires computation, current information, or file access, use the appropriate tool.
Always respond in the same language the user uses.
Be concise and direct.

IMPORTANT formatting rules:
- NEVER use LaTeX notation (no $, \\(, \\), \\[, \\], \\frac, \\sum, \\beta, etc.)
- Use Unicode math symbols instead: β₀, β₁, ŷ, x̄, ȳ, Σ, ∫, √, ², ³, ×, ÷, ≈, ≠, ≤, ≥, ∈, →
- Use plain text for fractions: (a + b) / c
- Use ** for exponents: x**2, or superscript: x²
- Format code with triple backticks"""

def main():
    agent = Agent(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        tool_registry=registry,
        system_prompt=SYSTEM_PROMPT,
    )

    print("=" * 50)
    print(f"  🤖 AI Agent ({MODEL})")
    print(f"  Tools: {', '.join(t['function']['name'] for t in registry.get_tools_json())}")
    print(f"  Type 'quit' to exit, 'reset' to clear history")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye! 👋")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("🔄 Conversation reset.")
            continue

        print()
        try:
            response = agent.ask(user_input)
            print(f"\n🤖 Agent: {response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
