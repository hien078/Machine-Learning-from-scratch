import os
import sys
import time
from pathlib import Path

# Add project root to path so `from src...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from src.agent import Agent
from src.tools import ToolRegistry

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 1. Initialize the Tool registry
registry = ToolRegistry()

@registry.register
def calculate_tax(salary: int) -> int:
    """Tool for computing income tax. Whenever tax needs to be computed, this function must be called. The default tax rate is 10% of the gross salary."""
    return int(salary * 0.10)

# 2. Create Agent 1: Accountant (equipped with the tax tool)
accountant = Agent(
    name="Accountant",
    role="You are a by-the-book accountant. Whenever the boss asks about tax, you must use the calculate_tax tool — never compute it in your head. Reply politely and concisely.",
    api_key=api_key,
    tools=registry.get_all_functions(),
    tool_registry=registry
)

# 3. Create Agent 2: Boss (no tools, only gives orders)
boss = Agent(
    name="Boss",
    role="You are a demanding boss. Ask the accountant to compute the income tax for an employee. Make up the employee's name and salary (e.g. 15 million, 20 million...). Keep the question short.",
    api_key=api_key
)

# 4. SCENARIO: let the two agents talk to each other!
print("\n🎬 MEETING STARTS")
print("=" * 50)

# Boss's opening (start of the conversation)
boss_message = boss.ask("Tell the accountant to compute the tax for a new employee.")
print(f"👔 [Boss]: {boss_message}")
time.sleep(2) # Pause 2 seconds to feel like a human chat

# Accountant hears the boss, will automatically analyze and call the calculate_tax tool
accountant_reply = accountant.ask(boss_message)
print(f"\n👩‍💼 [Accountant]: {accountant_reply}")
time.sleep(5)

# Boss hears the accountant's report and approves
final_decision = boss.ask(f"The accountant just replied: '{accountant_reply}'. Wrap it up and tell her to go transfer the money.")
print(f"\n👔 [Boss]: {final_decision}")
print("=" * 50)
