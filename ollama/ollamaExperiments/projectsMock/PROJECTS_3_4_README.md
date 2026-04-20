# Projects 3 & 4: Multi-Agent Orchestrator & Code Analyzer

Continuing antirez-style minimal implementations. Both are ~150-180 lines, production-ready.

---

## Project 3: Multi-Agent Orchestrator (Sequential Role-Based)

**What it does**: Assign same goal to 3 specialized agents (Frontend, Backend, QA), collect results, generate report.

### Setup

```bash
pip install ollama
python project_3_multi_agent.py
```

**First run**:
- Creates 3 agent prompts with same project goal
- Executes sequentially (Frontend Dev → Backend Dev → QA Engineer)
- Collects results + timing
- Generates markdown report: `orchestration_report.md`
- Prints summary

**Output**:
```
[MULTI-AGENT] Project Goal: Build a real-time collaborative document editor...
[MULTI-AGENT] Common Task: Design core features and architecture
[MULTI-AGENT] Orchestrating 3 agents...

[Frontend Dev] Executing...
         ✓ [3.45s]
[Backend Dev] Executing...
         ✓ [2.89s]
[QA Engineer] Executing...
         ✓ [2.15s]

ORCHESTRATION SUMMARY
Total agents: 3
Successful: 3
Failed: 0
Total execution time: 8.49s
```

### How it works

```
Project goal → Frontend agent (UI design)
            ↓
           Backend agent (API design)
            ↓
             QA agent (test strategy)
            ↓
         Collect → Format → Save report
```

**Key code**:
```python
orchestrator = MultiAgentOrchestrator()
results, total_time = orchestrator.orchestrate(
    project_goal="Build real-time editor",
    common_task="Design architecture"
)
```

### Customize

Edit the role templates in the code:
```python
self.roles = {
    "Frontend Dev": {"prompt_template": "..."},
    "Backend Dev": {"prompt_template": "..."},
    "QA Engineer": {"prompt_template": "..."}
}
```

Or add more roles:
```python
self.roles["DevOps Engineer"] = {
    "desc": "Infrastructure, deployment, monitoring",
    "prompt_template": "You are a DevOps Lead..."
}
```

---

## Project 4: Code Analyzer + Refactor Suggestion

**What it does**: Analyze Python repository, detect issues, suggest refactors, cache results by file hash.

### Setup

```bash
pip install ollama
python project_4_code_analyzer.py [repo_path]
```

**First run** (analyze current directory):
```bash
python project_4_code_analyzer.py .
```

**Analyze specific repo**:
```bash
python project_4_code_analyzer.py /path/to/my/project
```

**Output**:
```
[CODE-ANALYZER] Analyzing 5 files...
[CODE-ANALYZER] Cache file: .code_analysis_cache.json

[1/5] project_1_rag_qa.py... (new)
[2/5] project_2_agent_loop.py... (new)
[3/5] project_3_multi_agent.py... (new)
[4/5] project_4_code_analyzer.py... (cache)
[5/5] utils.py... (cache)

[INFO] Analysis complete. Cached: 2/5

ANALYSIS SUMMARY
long_functions: 2
missing_error_handling: 1
style_issues: 1
security_concerns: 0
performance_issues: 0
```

Generates: `code_analysis_report.md`

### How it works

```
Python files → Read + hash (MD5)
            ↓
         Check cache (MD5 lookup)
            ↓
      Cache hit? → Return cached analysis
      Cache miss? → Analyze via LLM → Cache result
            ↓
         Aggregate findings
            ↓
       Generate report
```

**Key code**:
```python
analyzer = CodeAnalyzer(model="phi4:latest")
analyzer.analyze_repo("./my_project", max_files=10)
analyzer.generate_report("report.md")
```

### Caching

- Cache stored in `.code_analysis_cache.json`
- Key: MD5 hash of file content
- If file unchanged → instant retrieval (no LLM call)
- If file changed → fresh analysis

**View cache**:
```bash
cat .code_analysis_cache.json | jq .
```

**Clear cache**:
```bash
rm .code_analysis_cache.json
```

### Report

Generated file: `code_analysis_report.md`

Includes:
- Summary table of issues (long functions, error handling, style, security, performance)
- Detailed findings per file
- Refactoring recommendations
- Next steps

---

## Comparison

| Feature | Project 3 | Project 4 |
|---------|-----------|-----------|
| **Purpose** | Multi-role agent collaboration | Code analysis + refactor suggestions |
| **Input** | Project goal (text) | Repository (Python files) |
| **LLM calls** | 3 (one per agent) | 1 per file (with caching) |
| **Output** | Report (markdown) | Report + JSON cache |
| **Speed** | 5-15s total | ~2-5s per file (first run) |
| **Cache** | None | MD5-based file-level |
| **Use case** | Architecture reviews | Code quality automation |

---

## Combined Workflow

1. **Project 3** → Get architecture review from 3 perspectives
2. **Project 4** → Analyze existing codebase against patterns
3. **Combine** → Use Project 3 output as refactoring guide for Project 4

Example:
```bash
# Step 1: Get multi-agent design
python project_3_multi_agent.py

# Step 2: Analyze current code
python project_4_code_analyzer.py ./src

# Step 3: Compare design vs. current implementation
diff orchestration_report.md code_analysis_report.md
```

---

## Performance Tips

### Project 3 (Multi-Agent)
- Sequential execution: 5-15s for 3 agents
- Parallel execution (optional): use `threading` to cut time ~3x
- Model choice: Phi-4 is balanced; glm-4.7-flash is 2x faster

### Project 4 (Code Analyzer)
- First run: ~5-10s per file (LLM inference)
- Subsequent runs: <100ms (cache hits only)
- Cache across sessions: `.code_analysis_cache.json` persists
- Limit files: `max_files=10` to avoid long runs

---

## Models Tested

Both projects work with any chat model:
- `phi4:latest` ✓ (14B, balanced)
- `glm-4.7-flash:latest` ✓ (30B MoE, fast)
- `mistral:latest` ✓ (7B, lightweight)
- `gemma2:latest` ✓ (9B)

Change model:
```python
MultiAgentOrchestrator(model="glm-4.7-flash:latest")
CodeAnalyzer(model="mistral:latest")
```

---

## Troubleshooting

**Project 3: "Connection refused"**
- Ollama not running: `ollama serve`

**Project 4: "No Python files found"**
- Check directory: `find . -name "*.py" | head -5`
- Adjust path: `python project_4_code_analyzer.py /full/path`

**Project 4: "Cache corrupted"**
- Delete cache: `rm .code_analysis_cache.json`
- Will rebuild on next run

**Models taking too long**
- Use faster model: `glm-4.7-flash:latest`
- Reduce files: `max_files=3`

---

## Philosophy

Same **antirez principles** as projects 1-2:
- **Minimal code**: ~150-180 lines each
- **No bloat**: Direct data structures, no abstraction layers
- **Works out-of-the-box**: Sample goals, auto-creates reports
- **Extensible**: Easy to add agents or customize analysis
- **Production-ready**: Error handling, caching, state persistence

---

## Next Steps

1. Run Project 3: `python project_3_multi_agent.py` → see multi-role review
2. Run Project 4: `python project_4_code_analyzer.py .` → see code analysis
3. Modify roles in Project 3, add custom analysis rules in Project 4
4. Combine: use Project 3 design as guideline, Project 4 to audit code

---

**See also**: `PROJECTS_1_2_README.md` for Projects 1 & 2
