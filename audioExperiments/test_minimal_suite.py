import asyncio
import os
from audio_interface import TTSFactory

async def run_minimal_test():
    print("🎯 Starting Minimal Audio Test Suite")
    out_dir = "outputs/minimal_test"
    os.makedirs(out_dir, exist_ok=True)
    
    # Engines to test
    # Note: openrouter requires an API key in .env, qwen is local, vibevoice is local
    test_cases = [
        {
            "engine": "qwen",
            "voice": "Ryan",
            "text": "Hello world, testing Qwen voice.",
            "kwargs": {"model_size": "0.6B"}
        },
        {
            "engine": "openrouter",
            "voice": "openai/tts-1-hd:alloy", # Example for openrouter
            "text": "Hello world, testing OpenRouter tts.",
            "kwargs": {}
        }
    ]
    
    for tc in test_cases:
        engine_name = tc["engine"]
        print(f"\n--- Testing Engine: {engine_name} ---")
        
        try:
            # For qwen, ensure we use the model we have
            engine = TTSFactory.get_engine(engine_name, **tc["kwargs"])
            output_path = os.path.join(out_dir, f"test_{engine_name}.wav")
            
            print(f"Generating audio for '{tc['text']}' using voice '{tc['voice']}'...")
            result = await engine.generate(tc["text"], tc["voice"], output_path)
            
            print(f"✅ Success! Saved to: {result['path']}")
            print(f"Stats: {result.get('elapsed', 0):.2f}s elapsed")
            
        except Exception as e:
            print(f"❌ Failed {engine_name}: {e}")

if __name__ == "__main__":
    asyncio.run(run_minimal_test())
