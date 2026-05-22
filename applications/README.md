# Applications

Project-style code that **uses** ML/AI rather than teaches a concept. Separated from the numbered learning modules (`01_supervised_learning/`, …) because the purpose is different:

| | Learning modules | Applications |
|---|---|---|
| **Goal** | Understand the algorithm | Build something with it |
| **Format** | 5 notebooks per algorithm | `src/` + `examples/` + `tests/` |
| **From-scratch?** | Yes (NumPy implementation) | No (use the best library) |
| **Audience** | Future-me, learning | Future-me, shipping |

---

## Subprojects

* [`mini_agent_sdk/`](mini_agent_sdk/) — Lightweight AI agent SDK powered by Google Gemini. Single-tool and multi-agent examples; manager pattern.
* [`openai_agent/`](openai_agent/) — OpenAI Agents SDK examples and tooling.

---

## Why these moved here

Originally located at `06_mini_agent_sdk/` and `07_openai_agent/` alongside the learning modules. The numbering implied they were the next learning step after RL, but they are concept-using projects rather than concept-teaching modules. Keeping them next to learning modules muddied the difference between "I want to *understand* X" and "I want to *use* X".
