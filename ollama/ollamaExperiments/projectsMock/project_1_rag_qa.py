#!/usr/bin/env python3
"""
Project 1: Minimal RAG Document QA
Load documents, search semantically, answer questions with LLM context.
Antirez-style: minimal, direct, no bloat.

Usage:
    python project_1_rag_qa.py
    
Setup:
    pip install txtai ollama
"""

from pathlib import Path
from ollama import Client
from txtai.embeddings import Embeddings
import time

class RAGQA:
    """Simple RAG QA system."""
    
    def __init__(self, doc_folder="./docs", model="phi4:latest"):
        self.model = model
        self.client = Client()
        self.embeddings = Embeddings({
            "content": True,
            "path": "sentence-transformers/all-MiniLM-L6-v2"
        })
        self.docs = []
        
        # Load documents
        doc_dir = Path(doc_folder)
        if doc_dir.exists():
            for txt_file in doc_dir.glob("*.txt"):
                content = txt_file.read_text()
                doc_id = len(self.docs)
                self.docs.append({
                    "id": doc_id,
                    "text": content,
                    "source": txt_file.name
                })
            
            if self.docs:
                print(f"[INFO] Loaded {len(self.docs)} documents")
                self.embeddings.index(self.docs)
                print(f"[INFO] Indexed in embeddings")
            else:
                print(f"[WARN] No .txt files in {doc_folder}")
    
    def search(self, query, limit=2):
        """Search documents by query."""
        if not self.docs:
            return []
        return self.embeddings.search(query, limit=limit)
    
    def answer(self, query):
        """Answer question using retrieved documents."""
        results = self.search(query)
        
        if not results:
            return "No documents to search."
        
        # Build context from top results
        context = "\n\n".join([f"[{r['source']}]\n{r['text'][:500]}" 
                               for r in results])
        
        # LLM answer with context
        prompt = f"""Given this context, answer the question concisely:

Context:
{context}

Question: {query}

Answer:"""
        
        t0 = time.time()
        r = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        elapsed = time.time() - t0
        
        answer = r['message']['content']
        return answer, elapsed
    
    def interactive(self):
        """Interactive Q&A loop."""
        print("\n[RAG-QA] Ready. Type 'quit' to exit.\n")
        
        while True:
            try:
                query = input("Q: ").strip()
                if query.lower() == "quit":
                    break
                if not query:
                    continue
                
                answer, elapsed = self.answer(query)
                print(f"A: {answer}\n[{elapsed:.2f}s]\n")
            except KeyboardInterrupt:
                break


def main():
    """Demo: create sample docs, then interactive QA."""
    
    # Create docs folder with sample content
    doc_dir = Path("./docs")
    doc_dir.mkdir(exist_ok=True)
    
    samples = {
        "python.txt": """Python is a high-level, interpreted programming language.
It emphasizes code readability and simplicity. Python supports multiple programming paradigms.
Key features: dynamic typing, automatic memory management, comprehensive standard library.
Python is widely used in data science, web development, and automation.""",
        
        "distributed_systems.txt": """Distributed systems span multiple computers or processes.
Key challenges: fault tolerance, consistency, scalability, network latency.
CAP theorem: Consistency, Availability, Partition tolerance (choose 2 of 3).
Common patterns: replication, sharding, consensus algorithms (Raft, Paxos).""",
        
        "caching.txt": """Caching improves performance by storing frequently accessed data.
Cache levels: L1/L2 (CPU), main memory, disk, network (CDN).
Eviction policies: LRU (Least Recently Used), LFU (Least Frequently Used), FIFO.
Write policies: write-through, write-back, write-behind.""",
    }
    
    for fname, content in samples.items():
        (doc_dir / fname).write_text(content)
    
    print("[INFO] Created sample documents in ./docs/")
    
    # Init and run
    qa = RAGQA(doc_folder="./docs")
    
    # Demo queries
    demo_queries = [
        "What is Python used for?",
        "What is the CAP theorem?",
        "Explain caching mechanisms.",
    ]
    
    print("\n[DEMO] Running sample queries...\n")
    for q in demo_queries:
        print(f"Q: {q}")
        answer, elapsed = qa.answer(q)
        print(f"A: {answer[:150]}...\n[{elapsed:.2f}s]\n")
    
    # Interactive mode
    print("\n[INFO] Starting interactive mode...")
    qa.interactive()


if __name__ == "__main__":
    main()
