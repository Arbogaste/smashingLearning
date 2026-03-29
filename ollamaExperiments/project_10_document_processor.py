#!/usr/bin/env python3
"""
Project 10: Document Processor (Extract + Structure)
Read PDF/MD/TXT, extract text, identify entities and key sections via LLM, output JSON.
Antirez-style: minimal, direct, no bloat.

Usage:
    python project_10_document_processor.py file.pdf
    python project_10_document_processor.py file.txt --out results.json
    python project_10_document_processor.py --dir ./docs --out batch_results.json
    python project_10_document_processor.py --demo   (creates sample files and processes them)

Setup:
    pip install ollama
    pip install pdfplumber   # only for PDF support
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from ollama import Client

MODEL = "phi4:latest"

EXTRACT_PROMPT = """Analyze this document and extract structured information.

Return a JSON object with these exact keys:
{{
  "title": "inferred document title or topic",
  "summary": "2-3 sentence summary",
  "sections": ["section 1 heading", "section 2 heading", ...],
  "entities": {{
    "people": ["name1", "name2"],
    "organizations": ["org1", "org2"],
    "locations": ["place1"],
    "dates": ["date1"],
    "key_terms": ["term1", "term2", "term3"]
  }},
  "key_takeaways": ["takeaway 1", "takeaway 2", "takeaway 3"],
  "document_type": "article|report|manual|financial|legal|other"
}}

Respond with JSON only, no prose before or after.

Document text (first 1200 chars):
{text}"""


class DocumentProcessor:
    def __init__(self, model=MODEL, cache_file=".doc_processor_cache.json"):
        self.client = Client(host="http://localhost:11434")
        self.model = model
        self.cache_file = Path(cache_file)
        self.cache = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}

    def _save_cache(self):
        self.cache_file.write_text(json.dumps(self.cache, indent=2))

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def extract_text(self, filepath: str) -> str | None:
        p = Path(filepath)
        if not p.exists():
            print(f"  [ERROR] file not found: {filepath}")
            return None

        if p.suffix in (".txt", ".md"):
            return p.read_text(encoding="utf-8", errors="replace")

        if p.suffix == ".pdf":
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(str(p)) as pdf:
                    for page in pdf.pages[:10]:  # max 10 pages
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except ImportError:
                print("  [ERROR] pdfplumber not installed. Run: pip install pdfplumber")
                return None
            except Exception as e:
                print(f"  [ERROR] PDF extraction failed: {e}")
                return None

        print(f"  [SKIP] unsupported extension: {p.suffix}")
        return None

    def _analyze(self, text: str) -> dict:
        key = self._hash(text[:1200])
        if key in self.cache:
            print("  (cached)")
            return self.cache[key]

        prompt = EXTRACT_PROMPT.format(text=text[:1200])
        t0 = time.time()
        r = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        elapsed = time.time() - t0
        content = r["message"]["content"].strip()

        # Strip markdown fences if model wrapped in ```json
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # LLM didn't return valid JSON — wrap raw output
            result = {"raw_analysis": content, "parse_error": True}

        print(f"  [{self.model}] {elapsed:.1f}s")
        self.cache[key] = result
        self._save_cache()
        return result

    def process(self, filepath: str) -> dict | None:
        p = Path(filepath)
        print(f"\nProcessing: {p.name}")

        text = self.extract_text(filepath)
        if text is None:
            return None

        text_len = len(text)
        print(f"  extracted {text_len} chars ({text_len // 250} ~pages)")

        analysis = self._analyze(text)

        return {
            "filename": p.name,
            "filepath": str(p.resolve()),
            "extension": p.suffix,
            "text_length": text_len,
            "analysis": analysis,
        }

    def process_dir(self, dirpath: str) -> list[dict]:
        d = Path(dirpath)
        files = sorted(
            f for f in d.rglob("*")
            if f.suffix in (".txt", ".md", ".pdf") and f.is_file()
        )
        if not files:
            print(f"No supported files found in {dirpath}")
            return []

        print(f"Found {len(files)} file(s) in {dirpath}")
        results = []
        for f in files:
            r = self.process(str(f))
            if r:
                results.append(r)
        return results


def print_result(r: dict):
    a = r.get("analysis", {})
    if a.get("parse_error"):
        print(f"\n  [PARSE ERROR] Raw LLM output:\n{a.get('raw_analysis', '')[:400]}")
        return

    print(f"\n  Title:    {a.get('title', 'n/a')}")
    print(f"  Type:     {a.get('document_type', 'n/a')}")
    print(f"  Summary:  {a.get('summary', 'n/a')[:120]}")

    sections = a.get("sections", [])
    if sections:
        print(f"  Sections: {', '.join(sections[:5])}")

    entities = a.get("entities", {})
    for etype in ("people", "organizations", "key_terms"):
        vals = entities.get(etype, [])
        if vals:
            print(f"  {etype.capitalize()}: {', '.join(vals[:6])}")

    takeaways = a.get("key_takeaways", [])
    if takeaways:
        print("  Takeaways:")
        for t in takeaways[:3]:
            print(f"    - {t}")


DEMO_DOCS = {
    "demo_article.txt": """The Rise of Local AI Inference

In 2024, a shift occurred in how developers think about artificial intelligence deployment.
Rather than sending all inference requests to cloud providers, many teams began running
models locally using tools like Ollama and llama.cpp.

The main driver was privacy. Healthcare companies, legal firms, and financial institutions
found they could not send sensitive documents to external APIs without compliance risk.
Local inference eliminated that concern entirely.

Performance also improved dramatically. The release of quantized models (GGUF format)
allowed 7B and 13B parameter models to run on consumer hardware with acceptable latency.
Phi-4 from Microsoft and GLM-4.7 from Zhipu AI demonstrated that smaller models could
match larger ones on many practical tasks.

The Ollama project, led by a small team in San Francisco, became the standard runtime.
By mid-2025, it had over 40,000 GitHub stars and was the default choice for local LLM
deployment in the Python ecosystem.

Key takeaways: local inference solves privacy, reduces cost, and is now fast enough for
production workloads on modern CPUs and consumer GPUs.
""",
    "demo_financial.txt": """Q2 2025 Earnings Report — TechCorp Inc.

Revenue: $4.2B (+18% YoY)
Net Income: $820M (+12% YoY)
EPS: $2.14 (beat estimate of $1.98)

Segment breakdown:
- Cloud Services: $2.1B (+31%)
- Software Licenses: $1.4B (+5%)
- Hardware: $0.7B (-8%)

CEO John Martinez commented: "The cloud segment continues to outperform expectations,
driven by AI workload demand. We are increasing R&D investment by $200M to accelerate
our inference platform roadmap."

CFO Sarah Chen noted margin pressure from infrastructure costs: gross margin declined
from 68% to 64% YoY due to GPU procurement costs.

Guidance for Q3 2025: revenue $4.4B-$4.6B, EPS $2.20-$2.35.
""",
}


def create_demo_files(tmpdir: Path):
    tmpdir.mkdir(exist_ok=True)
    for name, content in DEMO_DOCS.items():
        (tmpdir / name).write_text(content)
    print(f"Created demo files in {tmpdir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", help="File to process")
    p.add_argument("--dir", help="Process all supported files in directory")
    p.add_argument("--out", help="Save JSON results to file")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--demo", action="store_true", help="Create sample docs and process them")
    args = p.parse_args()

    proc = DocumentProcessor(model=args.model)

    if args.demo:
        demo_dir = Path("demo_docs")
        create_demo_files(demo_dir)
        results = proc.process_dir(str(demo_dir))
    elif args.dir:
        results = proc.process_dir(args.dir)
    elif args.file:
        r = proc.process(args.file)
        results = [r] if r else []
    else:
        p.print_help()
        return

    for r in results:
        print_result(r)

    if args.out and results:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nSaved {len(results)} result(s) → {args.out}")
    elif results:
        print(f"\nProcessed {len(results)} file(s). Use --out to save JSON.")


if __name__ == "__main__":
    main()
