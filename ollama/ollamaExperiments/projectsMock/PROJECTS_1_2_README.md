# Antirez-Style Projects: Minimal LLM Applications

Two working, minimal implementations using Ollama. Antirez-style: clean, direct, no bloat.

## Project 1: Minimal RAG Document QA

**What it does**: Load documents, search semantically, answer questions with LLM context.

### Setup

```bash
pip install txtai ollama
python project_1_rag_qa.py
```

**First run**:
- Creates `./docs/` folder with 3 sample documents (Python, Distributed Systems, Caching)
- Indexes them using all-MiniLM-L6-v2 embeddings
- Runs 3 demo queries
- Drops into interactive mode

**Interactive mode**:
```
Q: What is Python used for?
A: [LLM-generated answer]
[0.45s]

Q: quit
```

### How it works

```
Load docs → Index embeddings → Query → Semantic search (top-2)
→ Build context → LLM answer → Print
```

**Key code**:
```python
qa = RAGQA(doc_folder="./docs")
answer, elapsed = qa.answer("What is Python?")
print(answer)  # LLM response with document context
```

### Add your documents

Drop `.txt` files in `./docs/` folder. Each file is embedded and searchable.

---

## Project 2: Agent Loop PRD-Driven Sequential Execution

**What it does**: Execute tasks sequentially per PRD, update progress, restart-safe.

### Setup

```bash
pip install ollama
python project_2_agent_loop.py
```

**First run**:
- Creates `prd.json` with sample goal + 4 tasks
- Creates `progress.json` (empty)
- Loops through tasks, executes each via LLM
- Updates `progress.json` after each task
- Prints summary

**Output**:
```
[AGENT-LOOP] Goal: Design and document a simple caching layer...
[AGENT-LOOP] Tasks: 4

[TASK 1] Select eviction policy
         Choose between LRU, LFU, or FIFO...
         ✓ [2.15s]
         LRU is optimal for web workloads because...

[TASK 2] Design cache API
         Define get(key), set(key, value, ttl)...
         ✓ [1.89s]
         API should include methods: get, set, invalidate...

...

SUMMARY
Completed: 4
Failed: 0
Total time: 8.23s (0.1 min)
Progress saved to: progress.json
```

### How it works

```
Load PRD + progress → Loop:
  Find next pending task
  Execute task via LLM
  Save result to progress.json
  Continue
```

**Key code**:
```python
loop = AgentLoop(prd_file="prd.json", progress_file="progress.json")
loop.run()  # Executes all pending tasks, saves state
```

### Restart execution

To restart from beginning:
```bash
python project_2_agent_loop.py --reset
```

This deletes `progress.json`, so loop restarts fresh.

### Customize PRD

Edit `prd.json`:
```json
{
  "goal": "Your project goal here",
  "tasks": [
    {"name": "Task 1", "desc": "Do this..."},
    {"name": "Task 2", "desc": "Then do this..."}
  ]
}
```

Re-run with `--reset` to use new tasks.

---

## Common Patterns

### Project 1: Extend RAG

```python
qa = RAGQA()

# Add custom documents
Path("./docs/mydoc.txt").write_text("Important content...")

# Search + answer
answer, t = qa.answer("Your question")

# Batch queries
for q in ["Q1", "Q2", "Q3"]:
    print(qa.answer(q))
```

### Project 2: Extend Agent Loop

```python
prd = {
    "goal": "Your goal",
    "tasks": [
        {"name": "Step 1", "desc": "Do X"},
        {"name": "Step 2", "desc": "Do Y"}
    ]
}
Path("prd.json").write_text(json.dumps(prd, indent=2))

loop = AgentLoop()
loop.run()

# Check progress
progress = json.loads(Path("progress.json").read_text())
for t in progress["tasks"]:
    print(f"{t['name']}: {t['status']}")
```

---

## Requirements

- **Python 3.9+**
- **Ollama running** locally (http://localhost:11434)
- **Models**: `phi4:latest` (tested, ~14B, but any chat model works)
- **Dependencies**:
  - `ollama` (Python client)
  - `txtai` (Project 1 only, for embeddings + search)

## Models

Tested and working:
- `phi4:latest` (14B, balanced quality/speed)
- `glm-4.7-flash:latest` (30B MoE, very fast)
- `mistral:latest` (7B, lightweight)
- `devstral-small-2:latest` (24B, good for reasoning)

Any chat model works. Adjust model name in code:
```python
RAGQA(model="glm-4.7-flash:latest")
AgentLoop(model="mistral:latest")
```

---

## Performance

**Project 1 (RAG-QA)**:
- Setup: ~1-2s (embeddings)
- Query: ~0.5-3s (depends on model)
- Memory: ~200-500MB (embeddings + model cached in Ollama)

**Project 2 (Agent Loop)**:
- Per task: ~2-8s (LLM inference)
- 4 tasks: ~8-30s total
- Memory: same as project 1

---

## Troubleshooting

**"Connection refused"**:
- Ollama not running. Start it: `ollama serve`

**"Model not found: phi4:latest"**:
- Pull it: `ollama pull phi4`
- Or use a different model: `ollama list`

**Embedding not found**:
- Download happens on first run. Takes ~30s for all-MiniLM-L6-v2 (~100MB)

**"permission denied" on `progress.json`**:
- Delete it: `rm progress.json`

---

## Philosophy

**Antirez-style**:
- Minimal dependencies (2: ollama + txtai)
- No configuration files (uses defaults)
- Direct, readable code (no abstractions beyond needed)
- Works out-of-the-box (auto-creates sample data)
- Easy to extend (touch `./docs/` or edit `prd.json`)

Both projects fit in <200 lines, including comments.

---

## Next Steps

1. Run both: `python project_1_rag_qa.py` → `python project_2_agent_loop.py`
2. Add your documents to `./docs/`
3. Edit `prd.json` for custom tasks
4. Combine: use RAG results as PRD input for Agent Loop
5. Deploy: Ollama runs anywhere (Docker, cloud, local)

---

**Author**: Antirez-inspired minimal implementations
**License**: MIT
