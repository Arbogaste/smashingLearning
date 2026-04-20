#!/usr/bin/env python3
"""
Project 4: Code Analyzer + Refactor Suggestion
Analyze Python repository, suggest refactors, cache results.
Antirez-style: minimal caching, direct analysis.

Usage:
    python project_4_code_analyzer.py [repo_path]
    
Setup:
    pip install ollama
"""

from pathlib import Path
from ollama import Client
import hashlib
import json
import time

class CodeAnalyzer:
    """Analyze code files, suggest refactors."""
    
    def __init__(self, model="phi4:latest", cache_file=".code_analysis_cache.json"):
        self.model = model
        self.client = Client()
        self.cache_file = Path(cache_file)
        self.cache = {}
        self.findings = []
        
        # Load cache if exists
        if self.cache_file.exists():
            self.cache = json.loads(self.cache_file.read_text())
    
    def _save_cache(self):
        """Persist cache to disk."""
        self.cache_file.write_text(json.dumps(self.cache, indent=2))
    
    def _file_hash(self, code):
        """MD5 hash of code (cache key)."""
        return hashlib.md5(code.encode()).hexdigest()
    
    def analyze_file(self, filepath):
        """Analyze single Python file."""
        code = Path(filepath).read_text()
        code_hash = self._file_hash(code)
        
        # Check cache
        if code_hash in self.cache:
            return self.cache[code_hash], True  # True = from cache
        
        # Not cached, analyze
        # Limit code to first 2KB to fit in context
        code_sample = code[:2000]
        lines = len(code.split('\n'))
        
        prompt = f"""Analyze this Python code for:
1) Functions that are too long (>30 lines)
2) Missing error handling
3) Code style issues
4) Security concerns
5) Performance problems

File size: {lines} lines

Code:
{code_sample}

Provide brief assessment (3-4 key points max)."""
        
        t0 = time.time()
        try:
            r = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                timeout=15
            )
            elapsed = time.time() - t0
            
            analysis = r['message']['content']
            
            # Cache result
            self.cache[code_hash] = {
                "analysis": analysis,
                "lines": lines,
                "timestamp": time.time()
            }
            self._save_cache()
            
            return analysis, False  # False = newly analyzed
        except Exception as e:
            return f"ERROR: {str(e)}", False
    
    def analyze_repo(self, repo_path, max_files=10):
        """Analyze all Python files in repo."""
        repo_dir = Path(repo_path)
        
        if not repo_dir.exists():
            print(f"[ERROR] Path not found: {repo_path}")
            return
        
        py_files = list(repo_dir.glob("**/*.py"))[:max_files]
        
        if not py_files:
            print(f"[WARN] No Python files found in {repo_path}")
            return
        
        print(f"\n[CODE-ANALYZER] Analyzing {len(py_files)} files...")
        print(f"[CODE-ANALYZER] Cache file: {self.cache_file}\n")
        
        results = {}
        cached_count = 0
        
        for i, fpath in enumerate(py_files, 1):
            rel_path = fpath.relative_to(repo_dir)
            print(f"[{i}/{len(py_files)}] {rel_path}...", end=" ", flush=True)
            
            analysis, was_cached = self.analyze_file(str(fpath))
            results[str(rel_path)] = {
                "analysis": analysis,
                "cached": was_cached
            }
            
            if was_cached:
                print("(cache)")
                cached_count += 1
            else:
                print("(new)")
        
        self.findings = results
        
        print(f"\n[INFO] Analysis complete. Cached: {cached_count}/{len(py_files)}")
        return results
    
    def aggregate_findings(self):
        """Extract common patterns from all files."""
        patterns = {
            "long_functions": 0,
            "missing_error_handling": 0,
            "style_issues": 0,
            "security_concerns": 0,
            "performance_issues": 0,
            "other": 0
        }
        
        keywords = {
            "long_functions": ["function", "long", ">30", ">50"],
            "missing_error_handling": ["error", "exception", "try", "except"],
            "style_issues": ["style", "naming", "format", "pep"],
            "security_concerns": ["security", "injection", "eval", "pickle"],
            "performance_issues": ["performance", "slow", "loop", "n²"]
        }
        
        for filepath, data in self.findings.items():
            analysis_lower = data["analysis"].lower()
            
            found = False
            for pattern, keywords_list in keywords.items():
                if any(kw in analysis_lower for kw in keywords_list):
                    patterns[pattern] += 1
                    found = True
                    break
            
            if not found:
                patterns["other"] += 1
        
        return patterns
    
    def generate_report(self, output_file="code_analysis_report.md"):
        """Generate markdown report with findings."""
        aggregate = self.aggregate_findings()
        
        report = f"""# Code Analysis Report

**Analysis Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Files Analyzed**: {len(self.findings)}
**Cache Hit Rate**: {sum(1 for f in self.findings.values() if f['cached'])} / {len(self.findings)}

---

## Common Issues Found

| Issue | Count |
|-------|-------|
| Long Functions | {aggregate['long_functions']} |
| Missing Error Handling | {aggregate['missing_error_handling']} |
| Style Issues | {aggregate['style_issues']} |
| Security Concerns | {aggregate['security_concerns']} |
| Performance Issues | {aggregate['performance_issues']} |

---

## Detailed Findings

"""
        
        for filepath, data in self.findings.items():
            cache_marker = " *(from cache)*" if data["cached"] else ""
            report += f"### {filepath}{cache_marker}\n\n"
            report += f"{data['analysis'][:300]}...\n\n"
        
        report += """---

## Refactoring Recommendations

1. **Reduce Function Complexity**: Break down long functions using helper methods
2. **Add Error Handling**: Wrap external calls and file I/O with try-except
3. **Follow PEP 8**: Use consistent naming conventions (snake_case for functions)
4. **Security**: Avoid eval(), pickle untrusted data, validate input
5. **Performance**: Profile hot paths, reduce nested loops, cache results

---

## Next Steps

1. Pick the file with most issues
2. Refactor one function at a time
3. Add test coverage
4. Re-run analyzer to verify improvements
"""
        
        with open(output_file, "w") as f:
            f.write(report)
        
        print(f"[INFO] Report saved to: {output_file}")
        return report


def demo():
    """Demo: analyze this project's files."""
    
    # Analyze current directory Python files
    analyzer = CodeAnalyzer(model="phi4:latest")
    
    # Analyze this directory (where projects are)
    analyzer.analyze_repo(".", max_files=5)
    
    # Generate report
    analyzer.generate_report()
    
    # Summary
    aggregate = analyzer.aggregate_findings()
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    for issue, count in aggregate.items():
        if count > 0:
            print(f"{issue}: {count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = "."
    
    analyzer = CodeAnalyzer()
    analyzer.analyze_repo(repo_path, max_files=10)
    analyzer.generate_report()
