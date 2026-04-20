#!/usr/bin/env python3
"""
Project 3: Multi-Agent Orchestrator (Parallel Role-Based)
Assign tasks to 3 specialized agents, collect results, generate report.
Antirez-style: minimal, sequential execution, no deadlocks.

Usage:
    python project_3_multi_agent.py
    
Setup:
    pip install ollama
"""

from ollama import Client
import time
import json

class MultiAgentOrchestrator:
    """Orchestrate 3 specialized agents for parallel task execution."""
    
    def __init__(self, model="phi4:latest"):
        self.model = model
        self.client = Client()
        
        self.roles = {
            "Frontend Dev": {
                "desc": "Design UI/UX, layout, responsive design",
                "prompt_template": """You are a Senior Frontend Developer.
Project Goal: {goal}
Your Task: {task}

Provide: 1) UI Architecture, 2) Key components, 3) Tech stack recommendation"""
            },
            
            "Backend Dev": {
                "desc": "Design API endpoints, database schema, business logic",
                "prompt_template": """You are a Senior Backend Developer.
Project Goal: {goal}
Your Task: {task}

Provide: 1) API design (endpoints), 2) Database schema, 3) Key services"""
            },
            
            "QA Engineer": {
                "desc": "Plan test strategy, edge cases, quality assurance",
                "prompt_template": """You are a QA Lead.
Project Goal: {goal}
Your Task: {task}

Provide: 1) Test strategy, 2) Critical test cases, 3) Risk assessment"""
            }
        }
    
    def execute_agent(self, role, goal, task):
        """Execute a single agent (LLM call)."""
        prompt = self.roles[role]["prompt_template"].format(goal=goal, task=task)
        
        t0 = time.time()
        try:
            r = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                timeout=30
            )
            elapsed = time.time() - t0
            result = r['message']['content']
            return result, elapsed, None
        except Exception as e:
            elapsed = time.time() - t0
            return None, elapsed, str(e)
    
    def orchestrate(self, project_goal, common_task=None):
        """Execute all agents sequentially, collect results."""
        if common_task is None:
            common_task = "Design core features and architecture"
        
        results = {}
        total_start = time.time()
        
        print(f"\n[MULTI-AGENT] Project Goal: {project_goal}")
        print(f"[MULTI-AGENT] Common Task: {common_task}")
        print(f"[MULTI-AGENT] Orchestrating {len(self.roles)} agents...\n")
        
        for role, role_info in self.roles.items():
            print(f"[{role}] Executing...")
            result, elapsed, error = self.execute_agent(role, project_goal, common_task)
            
            if error:
                print(f"         ✗ ERROR: {error}")
                results[role] = {
                    "status": "error",
                    "error": error,
                    "output": None,
                    "time": elapsed
                }
            else:
                print(f"         ✓ [{elapsed:.2f}s]")
                results[role] = {
                    "status": "success",
                    "output": result,
                    "time": elapsed
                }
        
        total_time = time.time() - total_start
        
        return results, total_time
    
    def format_report(self, results, project_goal):
        """Format results as markdown report."""
        report = f"""# Multi-Agent Architecture Report

**Project Goal**: {project_goal}

---

"""
        for role, data in results.items():
            report += f"## {role}\n\n"
            if data["status"] == "success":
                report += f"{data['output'][:300]}...\n\n"
                report += f"*Execution time: {data['time']:.2f}s*\n\n"
            else:
                report += f"**ERROR**: {data['error']}\n\n"
            report += "---\n\n"
        
        return report
    
    def save_report(self, report, filename="orchestration_report.md"):
        """Save report to file."""
        with open(filename, "w") as f:
            f.write(report)
        print(f"\n[INFO] Report saved to: {filename}")


def main():
    """Demo: orchestrate 3 agents on a sample project."""
    
    project_goal = "Build a real-time collaborative document editor (like Google Docs)"
    
    orchestrator = MultiAgentOrchestrator(model="phi4:latest")
    results, total_time = orchestrator.orchestrate(project_goal)
    
    # Generate and save report
    report = orchestrator.format_report(results, project_goal)
    orchestrator.save_report(report)
    
    # Summary
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "error")
    
    print("\n" + "="*60)
    print("ORCHESTRATION SUMMARY")
    print("="*60)
    print(f"Total agents: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total execution time: {total_time:.2f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
