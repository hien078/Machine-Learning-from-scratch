# OpenAI-Compatible AI Agent

An AI agent powered by FreeModel API (OpenAI-compatible) with real tool calling.

## How it works

```
User message
    │
    ▼
┌──────────────────────────┐
│   Agent (agentic loop)   │
│                          │
│  User msg → LLM API     │
│         ↓                │
│  tool_calls? ──no──→ return text
│         │yes             │
│         ▼                │
│  Execute tools           │
│         │                │
│  Feed results → LLM     │
│         │                │
│  (repeat until text)     │
└──────────────────────────┘
```

## Tools

| Tool | Description |
|---|---|
| `get_datetime` | Current date & time |
| `run_python` | Execute Python code |
| `web_search` | Search the web (DuckDuckGo) |
| `read_file` | Read text files |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Example

```
🧑 You: What's 2^100?

   🔧 Calling: run_python({"code": "print(2**100)"})
   📋 Result: 1267650600228229401496703205376

🤖 Agent: 2^100 = 1,267,650,600,228,229,401,496,703,205,376

🧑 You: Search for Bitcoin price today

   🔧 Calling: web_search({"query": "Bitcoin price today"})
   📋 Result: 1. Bitcoin Price — ...

🤖 Agent: Bitcoin is currently trading at $XX,XXX.
```

## Files

| File | Description |
|---|---|
| `agent.py` | Agent class with agentic loop |
| `tools.py` | ToolRegistry + 4 built-in tools |
| `main.py` | Interactive terminal chat |
