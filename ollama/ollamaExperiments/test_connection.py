import ollama

def test_no_gpu(name):
    print(f"\n--- Testing {name} with num_gpu=0 (CPU ONLY) ---")
    try:
        resp = ollama.generate(
            model=name, 
            prompt='Say hi',
            options={'num_gpu': 0}
        )
        print(f"   Success: {resp['response'][:60]}...")
    except Exception as e:
        print(f"   Fail: {e}")

models = [
    "frankarenakc/hermes-3-uncensored:latest",
    "guzesqdro/zyx-ai:latest"
]

for m in models:
    test_no_gpu(m)
