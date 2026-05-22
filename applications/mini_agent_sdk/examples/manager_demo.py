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

# ==========================================
# 1. CREATE SUB-AGENTS (EMPLOYEES)
# ==========================================
accountant = Agent(
    name="Accountant",
    role="You are a financial / accounting specialist. Answer questions about money, cost calculations, and taxes professionally. Keep replies extremely concise.",
    api_key=api_key
)

developer = Agent(
    name="Developer",
    role="You are a Python developer. Your job is to write code on request. Return only code, no long-winded explanations.",
    api_key=api_key
)

# ==========================================
# 2. BUILD A "DELEGATION TOOL" FOR THE MANAGER
# ==========================================
registry = ToolRegistry()

@registry.register
def delegate_task(agent_role: str, task_description: str) -> str:
    """
    Use this tool to assign work to sub-agents.
    agent_role must be either 'Accountant' or 'Dev'.
    task_description is a detailed description of the work to assign.
    """
    print(f"\n   🔄 [Routing System]: Manager is calling {agent_role} to do: '{task_description}'...")

    if agent_role == 'Accountant':
        time.sleep(3) # Sleep 3 seconds to avoid API rate-limit errors
        result = accountant.ask(task_description)
        print(f"   👩‍💼 [Accountant replies internally]: {result}")
        return result
    elif agent_role == 'Dev':
        time.sleep(3) # Sleep 3 seconds to avoid API rate-limit errors
        result = developer.ask(task_description)
        print(f"   👨‍💻 [Dev replies internally]: {result}")
        return result
    else:
        return f"Error: no sub-agent has the role {agent_role}."

# ==========================================
# 3. CREATE THE MANAGER AGENT
# ==========================================
manager = Agent(
    name="Manager",
    role="You are a project manager. You NEVER do the work yourself. If a client asks about money/finance, use the delegate_task tool to assign it to 'Accountant'. If they ask about code/software, assign it to 'Dev'. Once you receive the result from a sub-agent, report back to the client politely.",
    api_key=api_key,
    tools=registry.get_all_functions(),
    tool_registry=registry
)

# ==========================================
# 4. RUN A LIVE TEST
# ==========================================
print("\n🏢 THE AI COMPANY IS OPEN FOR BUSINESS")
print("=" * 60)

# Scenario 1: Code request
request_1 = "Write me a Python function that returns the sum of two numbers."
print(f"👤 [Client]: {request_1}")
final_reply_1 = manager.ask(request_1)
print(f"\n👔 [Manager replies to client]:\n{final_reply_1}")

print("\n" + "-" * 60 + "\n")
print("⏳ (Pausing 10 seconds to avoid Google API blocking due to fast bursts...)")
time.sleep(10)

# Scenario 2: Money calculation
request_2 = "I have 100 million VND. If I deposit it in a bank at 6%/year, how much interest do I earn after 1 year?"
print(f"👤 [Client]: {request_2}")
final_reply_2 = manager.ask(request_2)
print(f"\n👔 [Manager replies to client]:\n{final_reply_2}")

print("=" * 60)
