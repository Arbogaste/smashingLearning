import time
import json
import platform
import subprocess
from ollama import Client

"""
Advanced Ollama experiments leveraging state-of-art models (Phi-4, GLM-4.7-Flash)
and Ollama capabilities (tool calling, streaming, agent loops, RAG simulation).

Run:
    python ollama_advanced_experiments.py > ollama_advanced_experiments.log 2>&1
"""


def calculate_tool(expression: str) -> str:
    """Tool for mathematical calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def retrieve_docs(query: str) -> list:
    """Simulate RAG retrieval: return relevant doc snippets for a query."""
    docs_db = {
        "cache coherence": [
            "Cache coherence protocols: MESI, MOESI, MSI. MESI most common.",
            "Write-invalidate vs write-update tradeoffs in distributed caches."
        ],
        "distributed systems": [
            "CAP theorem: Consistency, Availability, Partition tolerance.",
            "Eventual consistency model used in most large-scale systems."
        ],
        "python": [
            "Python GIL (Global Interpreter Lock) limits true parallelism.",
            "Use multiprocessing or async/await for high-concurrency workloads."
        ]
    }
    # Simple keyword matching
    for key, snippets in docs_db.items():
        if key in query.lower():
            return snippets
    return ["No relevant docs found."]


class AdvancedLLMExperiments:
    """Advanced experiments for multi-turn, RAG-augmented, agent-loop workflows."""
    
    def __init__(self, models):
        self.client = Client(host='http://localhost:11434')
        self.models = models
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression, e.g. '(10 + 5) * 3'"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    
    def exp_rag_augmented_query(self, model):
        """Experiment: RAG-augmented query (retrieve docs, inject into context)."""
        print(f"\n[RAG-Augmented Query] {model}")
        
        query = "How does cache coherence work in distributed systems with multiple regions?"
        docs = retrieve_docs(query)
        
        # Build context with retrieved documents
        context = f"Retrieved documents:\n" + "\n".join([f"- {doc}" for doc in docs])
        augmented_prompt = f"{context}\n\nUser query: {query}\n\nProvide a concise answer using the context above."
        
        t0 = time.time()
        r = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": augmented_prompt}],
            stream=False
        )
        t1 = time.time() - t0
        
        answer = r['message']['content'][:200]
        print(f"Answer ({t1:.2f}s): {answer}...")
    
    def exp_multi_turn_agent_loop(self, model):
        """Experiment: simple agent loop (PRD → task → action → progress)."""
        print(f"\n[Multi-turn Agent Loop] {model}")
        
        # Simulate PRD and progress
        prd = "Build a simple cache invalidation strategy for a distributed system."
        tasks = ["Define eviction policy", "Describe invalidation protocol", "Pseudocode"]
        progress = {"completed": [], "current": 0}
        
        messages = [
            {"role": "user", "content": f"PRD: {prd}\n\nYour task: {tasks[progress['current']]}"}
        ]
        
        for i in range(min(2, len(tasks))):  # Simulate 2 iterations
            t0 = time.time()
            r = self.client.chat(model=model, messages=messages, stream=False)
            t1 = time.time() - t0
            
            response = r['message']['content']
            print(f"Iteration {i+1} ({t1:.2f}s): {response[:150]}...")
            
            # Simulate progress update
            progress['completed'].append(tasks[i])
            progress['current'] = i + 1
            
            # Append response to messages (multi-turn)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Next task: {tasks[progress['current']] if progress['current'] < len(tasks) else 'Done'}"})
    
    def exp_tool_calling_loop(self, model):
        """Experiment: iterative tool calling (model decides when to invoke tools)."""
        print(f"\n[Tool-Calling Loop] {model}")
        
        messages = [
            {"role": "user", "content": "Calculate: (1000 + 500) * 2 - 300. Show your work step by step."}
        ]
        
        loop_count = 0
        max_iterations = 3
        
        while loop_count < max_iterations:
            t0 = time.time()
            r = self.client.chat(
                model=model,
                messages=messages,
                stream=False,
                tools=self.tools
            )
            t1 = time.time() - t0
            
            # Check for tool calls
            if r['message'].get('tool_calls'):
                print(f"Iteration {loop_count+1} ({t1:.2f}s): Model invoked tools")
                for tool_call in r['message']['tool_calls']:
                    tool_name = tool_call['function']['name']
                    args = tool_call['function']['arguments']
                    result = calculate_tool(args.get('expression', ''))
                    print(f"  → {tool_name}({args.get('expression')}) = {result}")
                    
                    # Add assistant response and tool result to messages
                    messages.append({"role": "assistant", "content": r['message']['content']})
                    messages.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": result
                    })
            else:
                # Model produced final answer
                print(f"Iteration {loop_count+1} ({t1:.2f}s): Final answer")
                print(f"  → {r['message']['content'][:150]}...")
                break
            
            loop_count += 1
    
    def exp_streaming_performance(self, model):
        """Experiment: streaming performance comparison."""
        print(f"\n[Streaming Performance] {model}")
        
        prompt = "Explain the CAP theorem in 3 paragraphs with examples."
        
        t0 = time.time()
        chunks = 0
        total_chars = 0
        
        for chunk in self.client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            content = chunk['message']['content']
            total_chars += len(content)
            chunks += 1
            print(content, end="", flush=True)
        
        t1 = time.time() - t0
        chars_per_sec = total_chars / t1 if t1 > 0 else 0
        print(f"\n({t1:.2f}s, {total_chars} chars, {chunks} chunks, {chars_per_sec:.0f} chars/sec)")
    
    def exp_model_comparison(self):
        """Experiment: side-by-side comparison of models on same prompt."""
        print(f"\n[Model Comparison: Phi-4 vs GLM-4.7-Flash]")
        
        prompt = "Design a simple load balancer for 3 backend servers. List key responsibilities."
        
        for model in self.models:
            print(f"\n--- {model} ---")
            t0 = time.time()
            r = self.client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            t1 = time.time() - t0
            
            answer = r['message']['content'][:180]
            print(f"({t1:.2f}s): {answer}...")


def print_system_info():
    """Print system information summary."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    
    # RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.total / (1024**3):.1f} GB (Available: {mem.available / (1024**3):.1f} GB)")
    except ImportError:
        try:
            result = subprocess.run(['free', '-h'], capture_output=True, text=True, timeout=2)
            lines = result.stdout.split('\n')
            if len(lines) > 1:
                mem_line = lines[1].split()
                print(f"RAM: {mem_line[1]} (Available: {mem_line[6]})")
        except:
            print("RAM: Unable to detect")
    
    # GPU info
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            gpus = result.stdout.strip().split('\n')
            for i, gpu in enumerate(gpus, 1):
                print(f"GPU {i}: {gpu}")
        else:
            print("GPU: No NVIDIA GPU detected")
    except FileNotFoundError:
        print("GPU: nvidia-smi not found (no NVIDIA GPU or drivers)")
    except Exception as e:
        print(f"GPU: Unable to detect ({e})")
    
    print("="*60)


def main():
    start_time = time.time()
    models = ["phi4:latest", "glm-4.7-flash:latest"]
    
    exp = AdvancedLLMExperiments(models)
    results = {"success": 0, "failed": 0}
    
    # Run experiments for each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        
        try:
            exp.exp_rag_augmented_query(model)
            exp.exp_multi_turn_agent_loop(model)
            exp.exp_tool_calling_loop(model)
            exp.exp_streaming_performance(model)
            results["success"] += 1
        except Exception as e:
            print(f"ERROR: {e}")
            results["failed"] += 1
    
    # Cross-model comparison
    print(f"\n{'='*60}")
    exp.exp_model_comparison()
    
    # Print final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Models tested: {len(models)}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} min)")
    
    print_system_info()


if __name__ == "__main__":
    print("[Advanced Ollama Experiments]")
    print(f"Models: phi4:latest, glm-4.7-flash:latest")
    print(f"Experiments: RAG-augmented, agent loop, tool calling, streaming, comparison")
    main()
