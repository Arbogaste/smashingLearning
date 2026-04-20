"""
Ollama LLM text generation test suite.

Tests Qwen3.5-Uncensored model with different thinking modes and API methods.
Generates 4 different outputs and saves them.
Optional: TTS integration (not invoked in basic tests).

Setup:
  ollama pull 
  ollama serve

"""
import json
import os
import time
from typing import Optional

import requests
from ollama import Client

# Hardcoded config
MODEL = "nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive:4b"
OLLAMA_HOST = "http://localhost:11434"
OUT = "outputs/ollama"

# Test prompt (keep it simple, no mandatory inputs)
PROMPT = "What is 2+2? Answer in one word."


def log_output(name: str, content_dict: dict) -> None:
    """Log output to console only (no file)."""
    print(f"\n📋 {name}:")
    print(json.dumps(content_dict, indent=2))


def test_generate_default() -> Optional[str]:
    """Test 1: client.generate() - default settings (think=False)."""
    print(f"\n{'='*60}")
    print("[TEST 1] client.generate() - default (think=False)")
    print(f"{'='*60}")
    
    try:
        client = Client(host=OLLAMA_HOST)
        t0 = time.time()
        response = client.generate(model=MODEL, prompt=PROMPT, stream=False)
        elapsed = time.time() - t0
        
        text = response.get("response", "").strip()
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0
        
        print(f"Generated in {elapsed:.2f}s:")
        print(f"  Tokens: {eval_count} (prompt: {prompt_eval_count})")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s")
        print(f"Response: {text}")
        
        # Log output
        output = {
            "test": "generate_default",
            "think": False,
            "time_sec": elapsed,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tokens_per_sec": tokens_per_sec,
            "prompt": PROMPT,
            "response": text,
        }
        log_output("test_1_generate_default", output)
        return text
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def test_generate_think_true() -> Optional[str]:
    """Test 2: client.generate() - with think=True."""
    print(f"\n{'='*60}")
    print("[TEST 2] client.generate() - with think=True")
    print(f"{'='*60}")
    
    try:
        client = Client(host=OLLAMA_HOST)
        t0 = time.time()
        response = client.generate(
            model=MODEL,
            prompt=PROMPT,
            stream=False,
            options={"think": True},
        )
        elapsed = time.time() - t0
        
        text = response.get("response", "").strip()
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0
        
        print(f"Generated in {elapsed:.2f}s:")
        print(f"  Tokens: {eval_count} (prompt: {prompt_eval_count})")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s")
        print(f"Response: {text}")
        
        # Log output
        output = {
            "test": "generate_think_true",
            "think": True,
            "time_sec": elapsed,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tokens_per_sec": tokens_per_sec,
            "prompt": PROMPT,
            "response": text,
        }
        log_output("test_2_generate_think_true", output)
        return text
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def test_chat_default() -> Optional[str]:
    """Test 3: client.chat() - default settings."""
    print(f"\n{'='*60}")
    print("[TEST 3] client.chat() - default settings")
    print(f"{'='*60}")
    
    try:
        client = Client(host=OLLAMA_HOST)
        t0 = time.time()
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            stream=False,
        )
        elapsed = time.time() - t0
        
        text = response["message"]["content"].strip()
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0
        
        print(f"Generated in {elapsed:.2f}s:")
        print(f"  Tokens: {eval_count} (prompt: {prompt_eval_count})")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s")
        print(f"Response: {text}")
        
        # Log output
        output = {
            "test": "chat_default",
            "think": False,
            "time_sec": elapsed,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tokens_per_sec": tokens_per_sec,
            "prompt": PROMPT,
            "response": text,
        }
        log_output("test_3_chat_default", output)
        return text
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

# NOT WORKING for model "nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive:4b" (returns 404)
"""
def test_chat_think_true() -> Optional[str]:
    # Test 4: client.chat() - with think=True.
    print(f"\n{'='*60}")
    print("[TEST 4] client.chat() - with think=True")
    print(f"{'='*60}")
    
    try:
        client = Client(host=OLLAMA_HOST)
        t0 = time.time()
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            stream=False,
            options={"think": True},
        )
        elapsed = time.time() - t0
        
        text = response["message"]["content"].strip()
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0
        
        print(f"Generated in {elapsed:.2f}s:")
        print(f"  Tokens: {eval_count} (prompt: {prompt_eval_count})")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s")
        print(f"Response: {text}")
        
        # Log output
        output = {
            "test": "chat_think_true",
            "think": True,
            "time_sec": elapsed,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tokens_per_sec": tokens_per_sec,
            "prompt": PROMPT,
            "response": text,
        }
        log_output("test_4_chat_think_true", output)
        return text
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return None
"""

def test_requests_generate() -> Optional[str]:
    """Test 5: requests.post() to /api/generate endpoint."""
    print(f"\n{'='*60}")
    print("[TEST 5] requests.post() - /api/generate")
    print(f"{'='*60}")
    
    try:
        t0 = time.time()
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": MODEL, "prompt": PROMPT, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        elapsed = time.time() - t0
        
        data = response.json()
        text = data.get("response", "").strip()
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0
        
        print(f"Generated in {elapsed:.2f}s:")
        print(f"  Tokens: {eval_count} (prompt: {prompt_eval_count})")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s")
        print(f"Response: {text}")
        
        # Log output
        output = {
            "test": "requests_generate",
            "endpoint": "/api/generate",
            "time_sec": elapsed,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tokens_per_sec": tokens_per_sec,
            "prompt": PROMPT,
            "response": text,
        }
        log_output("test_5_requests_generate", output)
        return text
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

# NOT WORKING for model "nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive:4b" (returns 404)
"""def test_requests_completion() -> Optional[str]:
    #Test 6: requests.post() to /api/completion endpoint.
    print(f"\n{'='*60}")
    print("[TEST 6] requests.post() - /api/completion")
    print(f"{'='*60}")
    
    try:
        t0 = time.time()
        response = requests.post(
            f"{OLLAMA_HOST}/api/completion",
            json={"model": MODEL, "prompt": PROMPT, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        elapsed = time.time() - t0
        
        data = response.json()
        text = data.get("completion", "").strip()
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0
        
        print(f"Generated in {elapsed:.2f}s:")
        print(f"  Tokens: {eval_count} (prompt: {prompt_eval_count})")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s")
        print(f"Response: {text}")
        
        # Log output
        output = {
            "test": "requests_completion",
            "endpoint": "/api/completion",
            "time_sec": elapsed,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tokens_per_sec": tokens_per_sec,
            "prompt": PROMPT,
            "response": text,
        }
        log_output("test_6_requests_completion", output)
        return text
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return None
"""
# NOT WORKING for model "nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive:4b" (returns 404)
"""def test_requests_chat() -> Optional[str]:
    #Test 7: requests.post() to /api/chat endpoint.
    print(f"\n{'='*60}")
    print("[TEST 7] requests.post() - /api/chat")
    print(f"{'='*60}")
    
    try:
        t0 = time.time()
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        elapsed = time.time() - t0
        
        data = response.json()
        text = data["message"]["content"].strip()
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0
        
        print(f"Generated in {elapsed:.2f}s:")
        print(f"  Tokens: {eval_count} (prompt: {prompt_eval_count})")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s")
        print(f"Response: {text}")
        
        # Log output
        output = {
            "test": "requests_chat",
            "endpoint": "/api/chat",
            "time_sec": elapsed,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tokens_per_sec": tokens_per_sec,
            "prompt": PROMPT,
            "response": text,
        }
        log_output("test_7_requests_chat", output)
        return text
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return None
"""

def main():
    print(f"\n{'='*60}")
    print(f"Ollama LLM Text Generation Test")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"API: {OLLAMA_HOST}")
    print(f"Output dir: {OUT}")
    print(f"Prompt: {PROMPT}")
    
    results = {
        #ONLY 
       }
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for i, (name, result) in enumerate(results.items(), 1):
        status = "✓ OK" if result else "✗ FAILED"
        print(f"{i}. {name:<30} {status}")
    

if __name__ == "__main__":
    main()
