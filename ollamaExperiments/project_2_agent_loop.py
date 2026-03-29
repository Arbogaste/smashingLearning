#!/usr/bin/env python3
"""
Project 2: Agent Loop PRD-Driven Sequential Execution
Execute tasks sequentially, update progress, long-run autonomy.
Antirez-style: minimal, direct loop.

Usage:
    python project_2_agent_loop.py
    
Setup:
    pip install ollama
    
Files created:
    - prd.json: Project goal and task list
    - progress.json: Execution state (auto-created)
"""

from pathlib import Path
from ollama import Client
import json
import time

class AgentLoop:
    """Execute tasks sequentially per PRD."""
    
    def __init__(self, prd_file="prd.json", progress_file="progress.json", model="phi4:latest"):
        self.model = model
        self.client = Client()
        self.prd_file = Path(prd_file)
        self.progress_file = Path(progress_file)
        
        # Load or create PRD
        if self.prd_file.exists():
            self.prd = json.loads(self.prd_file.read_text())
        else:
            print(f"[WARN] {prd_file} not found. Create it first.")
            self.prd = None
            return
        
        # Load or create progress
        if self.progress_file.exists():
            self.progress = json.loads(self.progress_file.read_text())
        else:
            self.progress = {
                "started": True,
                "prd_goal": self.prd.get("goal"),
                "tasks": [
                    {
                        "id": i,
                        "name": t.get("name"),
                        "desc": t.get("desc"),
                        "status": "pending",
                        "result": None,
                        "started_at": None,
                        "completed_at": None
                    }
                    for i, t in enumerate(self.prd.get("tasks", []))
                ]
            }
            self._save_progress()
    
    def _save_progress(self):
        """Save progress to file."""
        self.progress_file.write_text(json.dumps(self.progress, indent=2))
    
    def _find_next_task(self):
        """Find first pending task."""
        for task in self.progress["tasks"]:
            if task["status"] == "pending":
                return task
        return None
    
    def _execute_task(self, task):
        """Execute single task via LLM."""
        prompt = f"""PRD Goal: {self.prd.get('goal')}

Task: {task['name']}
Description: {task['desc']}

Provide a concise solution or plan (2-3 sentences)."""
        
        try:
            t0 = time.time()
            r = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            elapsed = time.time() - t0
            
            result = r['message']['content']
            return result, elapsed, None
        except Exception as e:
            return None, 0, str(e)
    
    def run(self):
        """Main loop: execute all pending tasks."""
        if not self.prd:
            return
        
        start_time = time.time()
        completed = 0
        failed = 0
        
        print(f"\n[AGENT-LOOP] Goal: {self.prd.get('goal')}")
        print(f"[AGENT-LOOP] Tasks: {len(self.progress['tasks'])}\n")
        
        while True:
            task = self._find_next_task()
            if not task:
                break
            
            print(f"[TASK {task['id']+1}] {task['name']}")
            print(f"         {task['desc'][:60]}...")
            
            task["started_at"] = time.time()
            result, elapsed, error = self._execute_task(task)
            task["completed_at"] = time.time()
            
            if error:
                task["status"] = "error"
                task["result"] = error
                print(f"         ✗ ERROR: {error}")
                failed += 1
            else:
                task["status"] = "completed"
                task["result"] = result[:150]
                print(f"         ✓ [{elapsed:.2f}s]")
                print(f"         {result[:100]}...\n")
                completed += 1
            
            # Save after each task
            self._save_progress()
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"Progress saved to: {self.progress_file}")
        print("="*60 + "\n")


def create_sample_prd():
    """Create sample PRD file."""
    prd = {
        "goal": "Design and document a simple caching layer for a web application",
        "success_metrics": [
            "Cache coherence strategy defined",
            "Eviction policy selected and justified",
            "API design documented",
            "Basic pseudocode provided"
        ],
        "tasks": [
            {
                "name": "Select eviction policy",
                "desc": "Choose between LRU, LFU, or FIFO. Justify based on typical web workload."
            },
            {
                "name": "Design cache API",
                "desc": "Define get(key), set(key, value, ttl), invalidate(key) methods."
            },
            {
                "name": "Document concurrency strategy",
                "desc": "How to handle concurrent access? Thread-safe? Use locks or CAS?"
            },
            {
                "name": "Estimate resource usage",
                "desc": "Given 100K cache entries, 1KB avg size, estimate memory and lookup time."
            }
        ]
    }
    
    Path("prd.json").write_text(json.dumps(prd, indent=2))
    print("[INFO] Created prd.json")
    return prd


def reset_progress():
    """Remove progress file to restart execution."""
    if Path("progress.json").exists():
        Path("progress.json").unlink()
        print("[INFO] Reset progress.json")


def main():
    """Demo: create sample PRD, then run agent loop."""
    
    # Option to reset
    import sys
    if "--reset" in sys.argv:
        reset_progress()
    
    # Create sample PRD if needed
    if not Path("prd.json").exists():
        create_sample_prd()
    
    # Run loop
    loop = AgentLoop()
    loop.run()


if __name__ == "__main__":
    main()
