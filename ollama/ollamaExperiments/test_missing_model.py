import ollama
try:
    print("Testing non-existent model...")
    resp = ollama.chat(model="non-existent-model", messages=[{"role": "user", "content": "hi"}])
    print(resp)
except Exception as e:
    print(f"Caught error: {e}")
