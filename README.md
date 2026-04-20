
# smashingLearning

smashingLearning: local LLM pipeline — dataset → fine-tune → serve → minimal agents.

Setup (fast)

```bash
git clone https://github.com/ConardLi/easy-dataset.git
git clone https://github.com/hiyouga/LLaMA-Factory.git

# easy-dataset (dev)
cd easy-dataset && npm install && npm run start   # http://localhost:1717

# llma-factory (fine-tune)
cd ../LLaMA-Factory && pip install -r requirements.txt
```

Quick operations

- Prepare training data with `easy-dataset` and export JSONL.
- Fine-tune via `LLaMA-Factory` (see repo for params).
- Serve model locally (Ollama or any http-compatible server).

Notes

- See `STATE_OF_ART_RECAP.md` and `STATE_OF_ART_RECAP_IT.md` for current recommendations (RAG, agent-loop, swarm).
- Replace model names with your served model identifiers.

Ollama & Testing

See `ollamaDocs/` for official Ollama documentation (core API, tool calling, streaming, web search).

Test local models with `ollama_test_llms.py`:

```bash
cd ollamaExperiments
python ollama_test_llms.py                        # print to stdout
python ollama_test_llms.py > ollama_test_llms.log 2>&1  # log to file
```

**What it tests**

Each test is independent. All except `tools` work on any model, including models that don't support function calling.

| Flag | What it tests | Passes if |
|---|---|---|
| `reasoning` | Distributed caching system-design question | Response mentions eviction/coherence/consistency keywords |
| `code` | Generate a `factorial(n)` function | Code is extracted and executed — `factorial(5)==120` |
| `finance` | BUY/SELL/HOLD analysis of AAPL | Response starts with one of the three words |
| `stream` | Streaming token generation | >5 chunks received, >30 words total |
| `security` | Identify reentrancy bug in a Solidity contract | Response mentions "reentrancy" or "check-effects" |
| `tools` | **4 scenarios** (see below) | Model invokes the expected tools |

**Tools test scenarios**

Requires a model that supports Ollama function-calling (e.g. `llama3.1`, `mistral`, `qwen2.5`). Each scenario runs independently.

| Scenario | Prompt | Expected |
|---|---|---|
| `tools/calc` | Complex arithmetic expression | Calls `calculate_expression` |
| `tools/weather` | "What's the weather in Paris?" | Calls `get_weather` |
| `tools/multi-tool` | "Weather and time in Tokyo?" | Calls both `get_weather` + `get_current_time` (≥2 calls) |
| `tools/search` | "Search for LLM prompt engineering tips" | Calls `search_web`, summarizes results |

The tools runner uses a full agentic loop: model calls tool → receives result → can call more tools → produces final answer.

**Available mock tools** (deterministic, no external calls):
- `calculate_expression(expression)` — safe AST-based math (no `eval`)
- `get_weather(city)` — mock weather data
- `get_current_time(timezone)` — mock UTC time
- `search_web(query)` — mock search results

**Config** — edit `CONFIG` at the top of the file:

```python
CONFIG = {
    "models": ["phi3:3.8b", "llama3.1:8b"],  # empty list → all available models
    "reasoning": True,
    "code":      True,
    "tools":     True,   # skip if model doesn't support function calling
    "finance":   True,
    "stream":    True,
    "security":  True,
}
```

**Output format**

```
MODEL: llama3.1:8b
  [PASS] reasoning     1243ms  Response mentions LRU, TTL, consistency...
  [PASS] code          876ms   function executed correctly: factorial(5)=120
  [PASS] finance       654ms   HOLD — growth is moderate but FCF is strong...
  [PASS] stream        3201ms  chunks=187 words=312 ~97 words/s
  [PASS] security      891ms   Reentrancy vulnerability — attacker calls...
  [PASS] tools/calc    1102ms  tool_calls=1 result=16226741...
  [FAIL] tools/multi-tool 2341ms  tool_calls=1 (expected ≥2)...
```

Summary table at the end shows PASS/FAIL/SKIP per test per model.

Resources

See `resources_sorted.md` and `resources_extendex.md` for 2026 tech stack: AI frameworks, data tools, trading bots, e-commerce platforms, automation, and more.

---

