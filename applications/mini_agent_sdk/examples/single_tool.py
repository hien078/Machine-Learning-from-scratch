import os
import sys
from pathlib import Path

# Add project root to path so `from src...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from google import genai
from src.tools import ToolRegistry

# 1. Initialize the Tool Registry
registry = ToolRegistry()

# 2. Create a tool and register it
@registry.register
def get_weather(location: str) -> str:
    """Returns the current weather for a given location."""
    # In real life you would call a weather API (e.g. OpenWeather). Here we mock it.
    if "Hanoi" in location:
        return "Light rain, 22°C."
    elif "Ho Chi Minh" in location:
        return "Sunny and hot, 34°C."
    return "No weather data available for this region."

# 3. Connect to the brain (Gemini)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("✅ SDK initialized and connected to Gemini.")

# 4. Challenge the AI
prompt = "What's the weather in Hanoi today?"
print(f"\n👤 User asks: {prompt}")

# Send the request ALONG WITH the list of tools
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
    config={'tools': registry.get_all_functions()}
)

# 5. Check whether Gemini wants to use a tool
if response.function_calls:
    print("\n🧠 Gemini realized it needs to check the weather, so it requested a Tool!")

    for function_call in response.function_calls:
        print(f"  -> Gemini wants to call: {function_call.name} with args {function_call.args}")

        # Our SDK proceeds to actually run the function
        tool_result = registry.execute(function_call)
        print(f"  -> Result from the function: {tool_result}")
else:
    print("\n🤖 Gemini answered directly (no tool call):")
    print(response.text)
