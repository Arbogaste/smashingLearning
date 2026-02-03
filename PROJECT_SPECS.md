# 10 Antirez-Style Projects: Technical Specifications

English design document for 10 concrete projects. Each spec is minimal, realistic, and ready for implementation. No mock, no fantasy—actual code patterns.

---

## Project 1: Minimal RAG Document QA

**Goal**: Load plain-text documents, answer questions using semantic relevance.

**Scope**:
- Load `.txt` files from `/docs` folder (5-50 files, max 100KB each)
- Embed documents using Ollama (via txtai or simple chunking)
- Answer user queries by retrieving top-2 relevant chunks + LLM answer
- No vector DB: in-memory storage only

**Architecture**:
```
User query → embed query → cosine similarity search → top-2 chunks 
→ build prompt with context → Ollama (phi4:latest) → answer
```

**Key Decision**: Use txtai library (all-in-one embeddings + search) instead of custom vector DB. Single dependency, zero setup.

**Success Criteria**:
- Q&A accuracy on 10 test questions: >70% relevant
- Response time: <5s per query (including LLM inference)
- No external service required (fully local)

**Code Pattern**:
```python
from txtai.embeddings import Embeddings
from ollama import Client

# Init once
embeddings = Embeddings({"content": True, "path": "sentence-transformers/all-MiniLM-L6-v2"})
documents = [{"id": i, "text": content} for i, content in enumerate(docs)]
embeddings.index(documents)

# Query
results = embeddings.search(query, limit=2)
context = "\n".join([r["text"] for r in results])
answer = Client().chat(...)
```

**Stack**: Python 3.9+, txtai, ollama (client)
**Time to implement**: 2-3 hours

---

## Project 2: Agent Loop PRD-Driven Sequential Execution

**Goal**: Execute tasks sequentially per PRD, update state, long-run autonomy.

**Scope**:
- Read `prd.json`: goal, requirements, success metrics
- Read `progress.json`: task list with status (pending/completed)
- For each pending task: generate prompt (PRD + task) → execute → log result
- Update `progress.json` after each task
- Repeat until all tasks done

**Architecture**:
```
Loop:
  1. Read progress.json → find next pending task
  2. Build prompt: PRD + current task description
  3. Call Ollama (phi4:latest) → get result
  4. Parse result (naive: first 150 chars as summary)
  5. Update progress.json: task.status = "completed", task.result = summary
  6. Print status
  7. Repeat
```

**Key Decision**: PRD + progress.json as single source of truth. No async, no parallelism—sequential guarantees state consistency.

**Success Criteria**:
- Execute 5-task project without manual intervention
- progress.json updated correctly after each task
- No context-rot: each task prompt includes full PRD (self-contained)
- Handles Ollama timeout gracefully (retry once)

**Code Pattern**:
```python
import json
from ollama import Client

prd = json.load(open("prd.json"))
progress = json.load(open("progress.json"))
client = Client()

for task in progress["tasks"]:
    if task["status"] == "completed":
        continue
    
    prompt = f"PRD:\n{prd['goal']}\n\nTask: {task['name']}\nDescription: {task['desc']}"
    try:
        r = client.chat(model="phi4:latest", messages=[{"role": "user", "content": prompt}], stream=False)
        task["result"] = r['message']['content'][:150]
        task["status"] = "completed"
    except:
        task["status"] = "error"
    
    json.dump(progress, open("progress.json", "w"), indent=2)
    print(f"✓ {task['name']}")
```

**JSON Format**:
```json
{
  "goal": "Build a CLI tool for X",
  "tasks": [
    {"name": "Design CLI interface", "desc": "...", "status": "pending"},
    {"name": "Implement core logic", "desc": "...", "status": "pending"}
  ]
}
```

**Stack**: Python 3.9+, ollama (client), stdlib (json)
**Time to implement**: 1-2 hours

---

## Project 3: Multi-Agent Orchestrator (Parallel Role-Based)

**Goal**: Assign specialized tasks to 3 agents in parallel, collect results.

**Scope**:
- Define 3 roles: Frontend Dev, Backend Dev, QA Engineer
- Each role gets same project goal + role-specific task
- Execute all 3 in parallel (via threading or sequential mock)
- Collect results and format report

**Architecture**:
```
Project goal → Role 1 prompt → Ollama → result 1
            ↘ Role 2 prompt → Ollama → result 2
            ↘ Role 3 prompt → Ollama → result 3
Aggregate results → print report
```

**Key Decision**: Sequential execution (not true parallel) to avoid context pollution. Each agent gets 1 LLM call, no back-and-forth.

**Success Criteria**:
- 3 agents execute without deadlock/timeout
- Each agent's output is distinct and role-appropriate
- Execution time: <30s total (sequential LLM calls)
- Results format: dict with role → output

**Code Pattern**:
```python
from ollama import Client
import time

class Orchestrator:
    def __init__(self):
        self.client = Client()
        self.roles = {
            "Frontend Dev": "Design UI/UX and layout for the app",
            "Backend Dev": "Design API endpoints and database schema",
            "QA Engineer": "Plan test strategy and edge cases"
        }
    
    def execute(self, project_goal):
        results = {}
        for role, task_desc in self.roles.items():
            prompt = f"Role: {role}\nGoal: {project_goal}\nTask: {task_desc}"
            t0 = time.time()
            r = self.client.chat(model="glm-4.7-flash:latest", 
                                messages=[{"role": "user", "content": prompt}],
                                stream=False)
            elapsed = time.time() - t0
            results[role] = {
                "output": r['message']['content'][:200],
                "time": elapsed
            }
        return results
```

**Stack**: Python 3.9+, ollama (client), threading (optional)
**Time to implement**: 1-2 hours

---

## Project 4: Code Analyzer + Refactor Suggestion

**Goal**: Analyze a git repository (code structure, patterns, issues), suggest refactors.

**Scope**:
- Clone or read repo from disk (max 10 Python files)
- For each file: extract code → send to Phi-4 (analysis) → cache result
- Aggregate findings: common patterns, anti-patterns, quality issues
- Generate refactor suggestions (text report, no code generation)

**Architecture**:
```
Repo → Read files (*.py) → For each file:
  Cache key = MD5(code)
  If cached: use cached analysis
  Else: Ollama analyze → cache result
→ Aggregate insights → Generate report
```

**Key Decision**: Cache by MD5 hash (avoid re-analyzing same code). No RAG, just sequential file analysis.

**Success Criteria**:
- Analyze 10 files in <30s (with cache hits)
- Detect at least 3 common anti-patterns (e.g., large functions, missing error handling)
- Cache working correctly (skip re-analysis on reruns)
- Report readable (markdown-like format)

**Code Pattern**:
```python
from pathlib import Path
from ollama import Client
import hashlib
import json

class CodeAnalyzer:
    def __init__(self):
        self.client = Client()
        self.cache = {}  # MD5 hash → analysis
    
    def analyze_file(self, filepath):
        code = Path(filepath).read_text()
        fhash = hashlib.md5(code.encode()).hexdigest()
        
        if fhash in self.cache:
            return self.cache[fhash]
        
        prompt = f"Analyze this Python code for patterns and issues:\n{code[:1000]}"
        r = self.client.chat(model="phi4:latest", 
                            messages=[{"role": "user", "content": prompt}],
                            stream=False)
        
        result = r['message']['content']
        self.cache[fhash] = result
        return result
    
    def analyze_repo(self, repo_path):
        py_files = list(Path(repo_path).glob("**/*.py"))[:10]
        results = {}
        for f in py_files:
            results[f.name] = self.analyze_file(str(f))
        return results
```

**Stack**: Python 3.9+, ollama (client), pathlib, hashlib
**Time to implement**: 2-3 hours

---

## Project 5: Semantic Web Scraper (Mock Pattern)

**Goal**: Given URL + query, use LLM to determine what data to extract (mock scraping logic).

**Scope**:
- Input: URL + natural language query (e.g., "extract product names and prices")
- LLM (GLM-4.7) understands query and URL structure
- Mock extraction: return simulated structured data (JSON)
- No actual web fetching (design pattern only)

**Architecture**:
```
URL + Query → LLM prompt (understand structure) → LLM suggests extraction schema
→ Mock data (return pre-defined JSON) → User gets simulated result
```

**Key Decision**: No Firecrawl/Puppeteer—just LLM as decision engine for what to extract. Actual scraping done separately (Selenium/Playwright).

**Success Criteria**:
- LLM correctly interprets 5 different URLs/queries
- Output schema is valid JSON
- Mock data is plausible (e.g., realistic product names)
- Execution time: <3s per query

**Code Pattern**:
```python
from ollama import Client
import json

class SemanticScraper:
    def __init__(self):
        self.client = Client()
    
    def understand_extraction(self, url, query):
        prompt = f"URL: {url}\nQuery: {query}\nWhat JSON schema should we extract?"
        r = self.client.chat(model="glm-4.7-flash:latest",
                            messages=[{"role": "user", "content": prompt}],
                            stream=False)
        return r['message']['content']
    
    def scrape(self, url, query):
        schema_suggestion = self.understand_extraction(url, query)
        
        # Mock data (in production: actual scraping)
        mock_data = {
            "url": url,
            "query": query,
            "schema_understood": schema_suggestion[:100],
            "data": [{"name": "Item 1", "price": 99.99}]
        }
        return mock_data
```

**Stack**: Python 3.9+, ollama (client), json
**Time to implement**: 1 hour

---

## Project 6: Trading Strategy Backtest + Multi-Agent Advisor

**Goal**: Analyze historical trading data, generate strategy signals via GLM-4.7 + Phi-4, backtest results.

**Scope**:
- Load CSV: OHLCV data (open, high, low, close, volume) for 1 stock, 1 year
- GLM-4.7 (Analyst): given price data → generate trading signals (buy/sell/hold)
- Phi-4 (Validator): review analyst signals for risk/bias
- Backtrader: run backtest with final signals
- Report: returns, sharpe ratio, max drawdown

**Architecture**:
```
CSV → Parse OHLCV
    → GLM-4.7 analyze (last 50 bars) → signals
    → Phi-4 validate signals
    → Backtrader execute signals
    → Calculate metrics
    → Report
```

**Key Decision**: No deep ML—just LLM as signal generator + validator. Backtest is deterministic (reproducible).

**Success Criteria**:
- Backtest runs without crash
- Generated signals are plausible (not all buy/sell)
- Validation catches obvious bad signals
- Report shows: total return, sharpe ratio, drawdown
- Execution time: <10s

**Code Pattern**:
```python
from ollama import Client
import pandas as pd
import backtrader as bt

class TradingAdvisor:
    def __init__(self):
        self.analyst_client = Client()
        self.validator_client = Client()
    
    def generate_signals(self, ohlcv_df):
        # Last 50 bars to analyst
        data_str = ohlcv_df.tail(50).to_string()
        prompt = f"Given this price data:\n{data_str}\nGenerate: buy, sell, or hold signal"
        r = self.analyst_client.chat(model="glm-4.7-flash:latest",
                                     messages=[{"role": "user", "content": prompt}],
                                     stream=False)
        signal = r['message']['content'][:50]
        
        # Validate
        val_prompt = f"Is this a reasonable trading signal? {signal}\nYes/No + reason"
        vr = self.validator_client.chat(model="phi4:latest",
                                        messages=[{"role": "user", "content": val_prompt}],
                                        stream=False)
        
        return signal, vr['message']['content'][:50]
    
    def backtest(self, ohlcv_df, signals):
        # Use Backtrader to execute signals
        # Returns: final portfolio value, metrics
        pass
```

**Stack**: Python 3.9+, ollama (client), pandas, backtrader
**Time to implement**: 3-4 hours

---

## Project 7: Knowledge Base with Semantic Search + QA

**Goal**: Build in-memory KB, add documents programmatically, answer questions via semantic search.

**Scope**:
- Define 5-10 documents (strings or loaded from files)
- Simple semantic search: embed query + find top-2 docs (txtai)
- Answer question using LLM + retrieved docs
- No persistent storage (RAM only)

**Architecture**:
```
Add docs → Index in txtai
Query → Embed + search
Results → Build context
Context + query → LLM answer
```

**Key Decision**: Use txtai (single dependency, no external DB). Documents stay in memory for session.

**Success Criteria**:
- 10 documents indexed in <1s
- Query response time: <2s (including LLM)
- Retrieved docs are relevant to query
- Answers are grounded in retrieved docs

**Code Pattern**:
```python
from txtai.embeddings import Embeddings
from ollama import Client
import json

class KnowledgeBase:
    def __init__(self):
        self.embeddings = Embeddings({"content": True, "path": "sentence-transformers/all-MiniLM-L6-v2"})
        self.docs = []
    
    def add_doc(self, title, content):
        self.docs.append({"id": len(self.docs), "title": title, "text": content})
    
    def index(self):
        self.embeddings.index(self.docs)
    
    def answer(self, query):
        results = self.embeddings.search(query, limit=2)
        context = "\n".join([r["text"] for r in results])
        
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        r = Client().chat(model="phi4:latest",
                         messages=[{"role": "user", "content": prompt}],
                         stream=False)
        return r['message']['content']
```

**Stack**: Python 3.9+, txtai, ollama (client)
**Time to implement**: 1-2 hours

---

## Project 8: Automated Code Reviewer (Multi-Model Router)

**Goal**: Review code (PR-like), get analysis + cross-validation, output report.

**Scope**:
- Input: code file or code snippet
- Phi-4: initial analysis (quality, style, logic)
- GLM-4.7: cross-check + suggest improvements
- Output: combined report (markdown)

**Architecture**:
```
Code → Phi-4 analyze (logical issues)
     → GLM-4.7 review (style + improvements)
     → Merge insights
     → Format report
```

**Key Decision**: Route different models by task (Phi-4 = lightweight reasoning, GLM-4.7 = fast language understanding). No tool calling.

**Success Criteria**:
- Both models analyze same code without conflict
- Report identifies at least 2 distinct issues
- Report is markdown-formatted
- Execution time: <10s

**Code Pattern**:
```python
from ollama import Client

class CodeReviewer:
    def __init__(self):
        self.client = Client()
    
    def analyze_with_phi4(self, code):
        prompt = f"Review this code for logic issues, bugs, edge cases:\n{code[:1500]}"
        r = self.client.chat(model="phi4:latest",
                            messages=[{"role": "user", "content": prompt}],
                            stream=False)
        return r['message']['content']
    
    def review_with_glm(self, code):
        prompt = f"Review this code for style, clarity, best practices:\n{code[:1500]}"
        r = self.client.chat(model="glm-4.7-flash:latest",
                            messages=[{"role": "user", "content": prompt}],
                            stream=False)
        return r['message']['content']
    
    def generate_report(self, code):
        phi4_analysis = self.analyze_with_phi4(code)
        glm_analysis = self.review_with_glm(code)
        
        report = f"""# Code Review Report

## Logic & Correctness (Phi-4)
{phi4_analysis[:300]}

## Style & Best Practices (GLM-4.7)
{glm_analysis[:300]}
"""
        return report
```

**Stack**: Python 3.9+, ollama (client)
**Time to implement**: 1-2 hours

---

## Project 9: Finance Multi-Agent Advisor

**Goal**: Analyze asset (stock/crypto), get recommendation from specialist agents + cross-validation.

**Scope**:
- Input: asset data (ticker, price, P/E, market cap, financials)
- GLM-4.7 (Analyst): analyze metrics, generate buy/sell/hold + rationale
- Phi-4 (Validator): review analyst recommendation for bias, suggest alternatives
- Output: final recommendation + reasoning

**Architecture**:
```
Asset data → GLM-4.7 analyze → recommendation + rationale
          → Phi-4 validate → critique + alternative view
          → Merge both → final decision + confidence
```

**Key Decision**: Sequential, not parallel. Analyst first, then validator reviews analyst output. This mirrors real trading workflow (analyst → compliance).

**Success Criteria**:
- Recommendation is clear (buy/sell/hold + confidence score 1-10)
- Validator catches at least 1 potential bias
- Final recommendation is balanced (not always bullish/bearish)
- Execution time: <8s

**Code Pattern**:
```python
from ollama import Client
import json

class FinanceAdvisor:
    def __init__(self):
        self.analyst = "glm-4.7-flash:latest"
        self.validator = "phi4:latest"
        self.client = Client()
    
    def analyze(self, asset_data):
        data_str = json.dumps(asset_data, indent=2)
        prompt = f"""Analyze this asset as a senior analyst:
{data_str}

Provide: 1) Key metrics interpretation, 2) Risk/opportunity, 3) BUY/SELL/HOLD + confidence (1-10)"""
        
        r = self.client.chat(model=self.analyst,
                            messages=[{"role": "user", "content": prompt}],
                            stream=False)
        return r['message']['content']
    
    def validate(self, analysis):
        prompt = f"""Review this investment analysis for bias/errors:
{analysis[:500]}

Provide: 1) Any missed risks, 2) Alternative view, 3) Final confidence adjustment"""
        
        r = self.client.chat(model=self.validator,
                            messages=[{"role": "user", "content": prompt}],
                            stream=False)
        return r['message']['content']
    
    def recommend(self, asset_data):
        analysis = self.analyze(asset_data)
        validation = self.validate(analysis)
        
        return {
            "asset": asset_data.get("ticker"),
            "analyst_view": analysis[:200],
            "validator_critique": validation[:200],
            "final": "Ready for decision"
        }
```

**Stack**: Python 3.9+, ollama (client), json
**Time to implement**: 2 hours

---

## Project 10: Document Processor (Extract + Structure)

**Goal**: Read documents (PDF/MD/TXT), extract text + entities, structure into JSON.

**Scope**:
- Load: `.pdf` (via pdfplumber), `.md`, `.txt` files
- Extract: title, main sections, key entities (named entities, important phrases)
- Structure: JSON output with hierarchical sections
- No external APIs (local processing only)

**Architecture**:
```
File → Extract raw text (pdfplumber or pathlib)
    → Parse structure (headings, paragraphs)
    → LLM identify entities + key phrases
    → Build JSON output
    → Save result
```

**Key Decision**: Use pdfplumber (PDF extraction), pathlib (text files). LLM for semantic analysis, not structure parsing.

**Success Criteria**:
- Extract text from 3 file types (PDF, MD, TXT)
- Identify at least 5 key entities per document
- JSON output is valid and navigable
- Execution time: <5s per file

**Code Pattern**:
```python
from pathlib import Path
from ollama import Client
import json

class DocumentProcessor:
    def __init__(self):
        self.client = Client()
    
    def extract_text(self, filepath):
        if filepath.endswith(".txt") or filepath.endswith(".md"):
            return Path(filepath).read_text()
        elif filepath.endswith(".pdf"):
            import pdfplumber
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages[:5]:  # First 5 pages
                    text += page.extract_text()
            return text
        return None
    
    def analyze_content(self, text):
        prompt = f"""Extract from this document:
1) Title/main topic
2) Key sections (list)
3) Important entities/names
4) Key takeaways

Text: {text[:1000]}"""
        
        r = self.client.chat(model="phi4:latest",
                            messages=[{"role": "user", "content": prompt}],
                            stream=False)
        return r['message']['content']
    
    def process(self, filepath):
        text = self.extract_text(filepath)
        analysis = self.analyze_content(text)
        
        return {
            "filename": Path(filepath).name,
            "text_length": len(text),
            "analysis": analysis[:500]
        }
```

**Stack**: Python 3.9+, ollama (client), pdfplumber (optional), pathlib
**Time to implement**: 2-3 hours

---

## Summary Table

| Project | Complexity | Dependencies | Learning | Time |
|---------|-----------|--------------|----------|------|
| 1. Minimal RAG | Easy | txtai, ollama | RAG fundamentals | 2h |
| 2. Agent Loop | Medium | ollama | Agent loop pattern | 1h |
| 3. Multi-Agent Orchestrator | Medium | ollama, threading | Parallel design | 2h |
| 4. Code Analyzer | Medium | ollama, hashlib | Repo analysis | 3h |
| 5. Web Scraper | Easy | ollama | Tool-calling pattern | 1h |
| 6. Trading Backtest | Hard | ollama, backtrader, pandas | Finance domain | 4h |
| 7. Knowledge Base | Easy | txtai, ollama | RAG + QA | 2h |
| 8. Code Reviewer | Medium | ollama | Multi-model routing | 2h |
| 9. Finance Advisor | Medium | ollama | Domain expert agents | 2h |
| 10. Document Processor | Medium | pdfplumber, ollama | Document extraction | 3h |

---

**Next Step**: Choose one project. Provide project number, and I'll deliver:
- Full working code (antirez-style, no bloat)
- Example data / test cases
- Setup instructions
- README with usage

