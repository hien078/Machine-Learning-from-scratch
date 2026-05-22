# Mini Agent SDK

A lightweight Python SDK for building AI agents powered by Google Gemini. Implements the core **agentic loop** — the pattern where an LLM autonomously decides when to call tools, executes them, and continues reasoning until it has a final answer.

## Key Concepts

| Concept | Description |
|---|---|
| **Agent** | An LLM-powered entity with a role, memory, and optional tools |
| **ToolRegistry** | A registry that maps Python functions to tools the LLM can call |
| **Agentic Loop** | The cycle: LLM thinks → calls tool → gets result → thinks again → ... → final text |
| **Multi-Agent** | Multiple agents communicating, with one delegating tasks to others |

## Architecture

```
User
  │
  ▼
┌─────────────────────────────────────────┐
│  Agent (name, role, tools, memory)      │
│  ┌───────────────────────────────────┐  │
│  │         Agentic Loop              │  │
│  │  ┌─────────┐    ┌─────────────┐   │  │
│  │  │ Gemini  │───▶│ Tool calls? │   │  │
│  │  │  API    │    └──────┬──────┘   │  │
│  │  └─────────┘       yes │ no ──▶ return text
│  │       ▲                ▼          │  │
│  │       │        ┌──────────────┐   │  │
│  │       └────────│ ToolRegistry │   │  │
│  │                │  .execute()  │   │  │
│  │                └──────────────┘   │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

For a deeper dive, see [docs/architecture.md](docs/architecture.md).

## Quickstart

### 1. Setup

```bash
# Create and activate virtualenv
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 3. Run an example

```bash
# Basic connection test
python examples/basic_connection.py

# Agent with tool calling
python examples/single_tool.py

# Two agents talking to each other
python examples/multi_agent.py

# Manager delegating to specialized agents
python examples/manager_demo.py
```

## API Reference

### `Agent(name, role, api_key, tools=None, tool_registry=None)`

Creates an agent with an LLM backend and optional tool-use capability.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Display name for the agent |
| `role` | `str` | System instruction defining the agent's persona |
| `api_key` | `str` | Google Gemini API key |
| `tools` | `list[Callable]` | Python functions the agent can call (from `registry.get_all_functions()`) |
| `tool_registry` | `ToolRegistry` | Registry that executes tool calls |

#### `agent.ask(message: str, max_turns: int = 10) -> str`

Send a message to the agent. It will autonomously call tools if needed, looping up to `max_turns` times before returning a text response.

### `ToolRegistry()`

A registry that converts Python functions into LLM-callable tools.

#### `@registry.register`

Decorator that registers a function as a tool. The function's name, signature, and docstring are automatically exposed to the LLM.

```python
registry = ToolRegistry()

@registry.register
def calculate_tax(salary: int) -> int:
    """Calculate income tax at 10% rate."""
    return int(salary * 0.10)
```

#### `registry.get_all_functions() -> list[Callable]`

Returns all registered functions (pass this to `Agent(tools=...)`).

#### `registry.execute(function_call) -> str`

Executes a function call received from the Gemini API and returns the result as a string.

## Project Structure

```
mini_agent_sdk/
├── README.md               # This file
├── requirements.txt        # Dependencies
├── .env.example            # API key template
├── src/
│   ├── __init__.py
│   ├── agent.py            # Agent class with agentic loop
│   └── tools.py            # ToolRegistry with @register decorator
├── examples/
│   ├── basic_connection.py # Minimal Gemini API test
│   ├── single_tool.py      # One agent + one tool
│   ├── multi_agent.py      # Two agents communicating
│   └── manager_demo.py     # Manager delegating to specialists
└── docs/
    └── architecture.md     # Detailed architecture explanation
```

## Examples Overview

| Example | What it demonstrates |
|---|---|
| `basic_connection.py` | Minimal API call — verifies your key works |
| `single_tool.py` | Agent receives a question, decides to call `get_weather()`, returns enriched answer |
| `multi_agent.py` | Boss agent asks Accountant agent to calculate tax; Accountant uses `calculate_tax` tool |
| `manager_demo.py` | Manager agent delegates to Dev or Accountant based on the request type (tool = another agent) |

## Dependencies

- `google-genai` — Google Gemini SDK
- `python-dotenv` — Load `.env` files
