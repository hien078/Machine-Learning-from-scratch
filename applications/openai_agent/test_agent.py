"""Quick automated test — send 3 prompts to agent, print results."""
import sys
sys.path.insert(0, ".")

from agent import Agent
from tools import registry

API_KEY = "fe_oa_9f4973c54fdc11c161bd56a9d351ddbe256e26581c9570eb"

agent = Agent(
    api_key=API_KEY,
    base_url="https://api.freemodel.dev",
    model="gpt-5.4-mini",
    tool_registry=registry,
    system_prompt="You are a helpful assistant with tools. Be concise.",
)

tests = [
    "What is today's date and time?",
    "Calculate 2**100 using Python",
    "Read the file requirements.txt",
]

for i, prompt in enumerate(tests, 1):
    print(f"\n{'='*50}")
    print(f"TEST {i}: {prompt}")
    print("=" * 50)
    try:
        response = agent.ask(prompt)
        print(f"\n🤖 Agent: {response}")
        agent.reset()
    except Exception as e:
        print(f"\n❌ Error: {e}")

print(f"\n{'='*50}")
print("All tests complete!")
