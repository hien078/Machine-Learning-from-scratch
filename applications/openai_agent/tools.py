"""
Tool Registry + Built-in Tools for OpenAI-compatible Agent.

Each tool = Python function + JSON schema (OpenAI function calling format).
"""

import json
import subprocess
import sys
from datetime import datetime


class ToolRegistry:
    """Registry that maps tool names to Python functions + their JSON schemas."""

    def __init__(self):
        self._functions = {}  # name -> callable
        self._schemas = {}    # name -> OpenAI tool schema

    def register(self, func=None, *, schema=None):
        """Decorator to register a function as a tool with its schema."""
        def decorator(f):
            name = f.__name__
            self._functions[name] = f
            if schema:
                self._schemas[name] = {
                    "type": "function",
                    "function": schema,
                }
            return f
        if func is not None:
            return decorator(func)
        return decorator

    def get_tools_json(self):
        """Return list of tool schemas for OpenAI API."""
        return list(self._schemas.values())

    def execute(self, name, arguments_str):
        """Execute a tool by name with JSON arguments string. Returns result string."""
        if name not in self._functions:
            return f"Error: tool '{name}' not found."
        try:
            args = json.loads(arguments_str) if arguments_str else {}
            result = self._functions[name](**args)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {e}"


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

registry = ToolRegistry()


@registry.register(schema={
    "name": "get_datetime",
    "description": "Get current date, time, and day of week.",
    "parameters": {"type": "object", "properties": {}, "required": []},
})
def get_datetime():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")


@registry.register(schema={
    "name": "run_python",
    "description": "Execute a Python code snippet and return its stdout output. Use for math, data processing, or any computation.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Use print() to output results.",
            },
        },
        "required": ["code"],
    },
})
def run_python(code: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output += f"\n[STDERR] {result.stderr.strip()}"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out (15s limit)."


@registry.register(schema={
    "name": "web_search",
    "description": "Search the web and return top results. Use for current events, prices, weather, facts, etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
        },
        "required": ["query"],
    },
})
def web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['body']}\n   URL: {r['href']}")
        return "\n\n".join(lines)
    except ImportError:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {e}"


@registry.register(schema={
    "name": "read_file",
    "description": "Read contents of a text file. Returns the file content as string.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
        },
        "required": ["path"],
    },
})
def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 5000:
            return content[:5000] + f"\n\n... [truncated, total {len(content)} chars]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"
