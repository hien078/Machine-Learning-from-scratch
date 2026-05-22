import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()

# Read the API Key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "your_api_key_here":
    print("❌ Error: you have not set GEMINI_API_KEY in the .env file!")
    exit(1)

print("✅ API Key found. Connecting to Gemini...")

try:
    # Initialize the client connected to Gemini
    client = genai.Client(api_key=api_key)

    # Send the first prompt to gemini-2.5-flash (light, fast, free)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Hello, can you say a short sentence in English?'
    )

    print("\n🤖 Gemini replies:")
    print("-" * 20)
    print(response.text)
    print("-" * 20)
    print("\n🎉 Success! We have connected to the Agent's 'brain'.")

except Exception as e:
    print(f"\n❌ Connection error: {e}")
