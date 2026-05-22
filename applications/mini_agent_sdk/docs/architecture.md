# Architecture

## Core Pattern: The Agentic Loop

The fundamental pattern in this SDK is the **agentic loop** — the LLM doesn't just answer questions, it autonomously decides when to use tools and loops until it has enough information to respond.

### Flow

```
agent.ask("What's the weather in Hanoi?")
        │
        ▼
┌─ Turn 1 ──────────────────────────────────┐
│  Send message to Gemini API               │
│  Gemini returns: function_call(            │
│    name="get_weather",                     │
│    args={"location": "Hanoi"}              │
│  )                                         │
│                                            │
│  SDK executes: get_weather("Hanoi")        │
│  Result: "Light rain, 22°C."               │
│                                            │
│  Send result back to Gemini as             │
│  FunctionResponse                          │
└────────────────────────────────────────────┘
        │
        ▼
┌─ Turn 2 ──────────────────────────────────┐
│  Gemini now has the tool result            │
│  No more function_calls                   │
│  Returns final text:                       │
│  "The weather in Hanoi is light rain,      │
│   22°C."                                   │
└────────────────────────────────────────────┘
        │
        ▼
  Return text to caller
```

### Why `max_turns`?

The loop has a safety limit (`max_turns=10` by default). Without it, a misconfigured tool could cause infinite loops — the LLM calls a tool, gets an unhelpful result, calls it again, etc.

## Components

### Agent (`src/agent.py`)

Responsibilities:
- Holds a **Gemini chat session** (which automatically manages conversation memory)
- Defines the **system instruction** (role/persona)
- Implements the **agentic loop** in `ask()`
- Delegates tool execution to the `ToolRegistry`

Key design decision: the `chat_session` is created once in `__init__` and reused across calls. This means the agent **remembers** all previous interactions within a single process lifetime.

### ToolRegistry (`src/tools.py`)

Responsibilities:
- **Register** Python functions via the `@register` decorator
- **Look up** functions by name when the LLM requests a tool call
- **Execute** functions with the arguments provided by the LLM
- **Return** results as strings

The decorator pattern means any Python function with type hints and a docstring can become a tool:

```python
@registry.register
def my_tool(param: str) -> str:
    """Description that the LLM reads to decide when to use this tool."""
    return do_something(param)
```

The LLM sees:
- Function **name** → decides which tool to call
- **Docstring** → understands what the tool does and when to use it
- **Parameter types** → knows what arguments to provide

## Multi-Agent Patterns

### Pattern 1: Peer Communication (`multi_agent.py`)

```
Boss Agent ──message──▶ Accountant Agent
     ▲                       │
     │                       │ (uses calculate_tax tool)
     └────── response ───────┘
```

Two independent agents, each with their own chat session. One agent's output is fed as input to the other. Simple message passing.

### Pattern 2: Manager Delegation (`manager_demo.py`)

```
User
  │
  ▼
Manager Agent
  │ (has delegate_task tool)
  │
  ├──▶ Accountant Agent (finance questions)
  │
  └──▶ Developer Agent (code questions)
```

The key insight: **an agent can be another agent's tool**. The `delegate_task` function wraps calls to sub-agents, so the Manager doesn't know it's talking to other LLMs — it just sees a tool that returns answers.

This is a powerful pattern because:
- The Manager decides **who** to delegate to based on the question
- Each sub-agent has its own **specialized role** and memory
- The Manager can **synthesize** results from multiple sub-agents
- It scales: add more specialists by registering more tools

## Memory Model

Currently, memory is **session-scoped**:
- Each `Agent` creates a `chat_session` on init
- The session accumulates all messages (user + assistant + tool calls)
- Memory dies when the process exits

Future improvement: persist conversation history to disk (JSON/SQLite) so agents can resume across sessions.

## Limitations

| Limitation | Impact | Future fix |
|---|---|---|
| No memory persistence | History lost on exit | Save/load to JSON |
| Synchronous only | Blocks on each API call | `asyncio` support |
| No streaming | Wait for full response | Streaming API |
| Hardcoded model | Always `gemini-2.5-flash` | Config file |
| `print()` for logging | Not production-ready | `logging` module |
| No retry/backoff | API rate limits crash | Exponential backoff |
