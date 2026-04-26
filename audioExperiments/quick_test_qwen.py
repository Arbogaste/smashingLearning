import asyncio
import os
from audio_interface import TTSFactory

async def quick_test():
    print("Testing Qwen3-TTS via Unified Interface...")
    
    # Configuration
    # We'll use 0.6B by default as it's lighter
    engine = TTSFactory.get_engine("qwen", model_size="0.6B")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    text = "Ciao, sono un'intelligenza artificiale cazzuta."
    voice = "vivian"
    output_path = f"outputs/test_qwen_{timestamp}.wav"
    
    print(f"Generating audio for: '{text}' using voice '{voice}'...")
    print(f"Target file: {output_path}")
    try:
        result = await engine.generate(text, voice, output_path, language="Italian")
        
        print("\n✅ Test Success!")
        print(f"  • Engine: {result['engine']}")
        print(f"  • File: {result['path']}")
        print(f"  • Elapsed: {result['elapsed']:.2f}s")
        print(f"  • Duration: {result['duration']:.1f}s")
        print(f"  • RTF: {result['rtf']:.2f}x")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Test Failed: {e}")
        traceback.print_exc()
        if "FileNotFoundError" in str(e):
            print("  Hint: Make sure the Qwen models are downloaded in Qwen3-TTS/models/")

if __name__ == "__main__":
    asyncio.run(quick_test())
