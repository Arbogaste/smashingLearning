import asyncio
import os
import argparse
from audio_interface import TTSFactory

# Corpus samples
LONG_EN = "In the heart of the digital transformation, artificial intelligence stands as the most defining technology of our era. From the way we automate complex logistical chains to the subtle art of personalizing the user experience, machine learning is no longer a distant promise but a foundational reality of modern enterprise."

TEST_PLAN = [
    {
        "name": "male_en_qwen",
        "engine": "qwen",
        "voice": "Ryan",
        "text": LONG_EN,
        "kwargs": {"language": "English", "model_size": "1.7B"}
    },
    {
        "name": "female_en_qwen",
        "engine": "qwen",
        "voice": "Emma",
        "text": LONG_EN,
        "kwargs": {"language": "English", "model_size": "1.7B"}
    },
    {
        "name": "authoritative_qwen",
        "engine": "qwen",
        "voice": "Ryan",
        "text": "Breaking news. The markets are reacting sharply to today's Federal Reserve announcement.",
        "kwargs": {
            "language": "English", 
            "model_size": "1.7B", 
            "instruct": "speak slowly and deliberately, authoritative news anchor tone, no emotion"
        }
    },
    {
        "name": "calm_qwen",
        "engine": "qwen",
        "voice": "Emma",
        "text": "Breathe in slowly. Feel the air filling your lungs. Let go of all tension.",
        "kwargs": {
            "language": "English",
            "model_size": "1.7B",
            "instruct": "speak very slowly and gently, calm meditative voice, warm and soothing"
        }
    }
]

async def run_tests(target_engine=None):
    print("🚀 Starting Unified Audio Test Suite")
    out_dir = "outputs/tests"
    os.makedirs(out_dir, exist_ok=True)
    
    for test in TEST_PLAN:
        if target_engine and test["engine"] != target_engine:
            continue
            
        print(f"\n[TEST] {test['name']} ({test['engine']})")
        try:
            engine = TTSFactory.get_engine(test["engine"], **test["kwargs"])
            output_path = os.path.join(out_dir, f"{test['name']}.wav")
            
            result = await engine.generate(test["text"], test["voice"], output_path, **test["kwargs"])
            
            print(f"  ✓ Success: {result['path']}")
            if "rtf" in result:
                print(f"  RTF: {result['rtf']:.2f}x | Duration: {result['duration']:.1f}s")
            else:
                print(f"  Elapsed: {result['elapsed']:.2f}s")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", help="Run only tests for this engine")
    args = parser.parse_args()
    
    asyncio.run(run_tests(args.engine))
