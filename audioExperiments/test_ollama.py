"""
Ollama LLM text generation test suite.

Tests Qwen3.5-Uncensored model with different thinking modes and API methods.
Generates 4 different outputs and saves them.
Optional: TTS integration (not invoked in basic tests).

Setup:
  ollama pull nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive:4b
  ollama serve

Run from audioExperiments/:
  python test_ollama.py
"""
import json
import os
import sys
import time
from typing import Optional

import requests

try:
    import soundfile as sf
    import torch

    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

try:
    import edge_tts
    from pydub import AudioSegment
    from pydub.playback import play

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# Hardcoded config
MODEL = "nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive:4b"
OLLAMA_HOST = "http://localhost:11434"
OUT = "outputs/ollama"

# Test prompt (keep it simple, no mandatory inputs)
PROMPT = "write the first 4 sentences of the Red Book by Mao."

# Setup Qwen3-TTS if available (optional, not invoked by default)
if QWEN_AVAILABLE:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
    try:
        from qwen_tts import Qwen3TTSModel
        QWEN_AVAILABLE = True
    except ImportError:
        QWEN_AVAILABLE = False


def log_output(name: str, content_dict: dict) -> None:
    """Log output to console only (no file)."""
    print(f"\n📋 {name}:")
    print(json.dumps(content_dict, indent=2))


def check_ollama_health() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        print("[CHECK] Verifying Ollama is running...", flush=True)
        sys.stdout.flush()
        import httpx
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"[ERROR] Ollama not reachable at {OLLAMA_HOST}: {e}", flush=True)
        sys.stdout.flush()
        return False


def test_chat_default() -> Optional[str]:
    """Test 3: ollama Client - streaming enabled (official pattern from docs)."""
    print(f"\n{'='*60}")
    print(f"[TEST 3] Client.chat() with streaming", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"{'='*60}", flush=True)
    
    try:
        from ollama import Client
        
        print("[DEBUG] Creating Client...", flush=True)
        sys.stdout.flush()
        client = Client(host=OLLAMA_HOST)
        
        t0 = time.time()
        full_text = ""
        print("\nStreaming response:\n", flush=True)
        sys.stdout.flush()
        
        # EXACT pattern from docs - client.chat with stream=True returns ChatResponse objects
        # Access attributes with .message.content, NOT dictionary keys
        for part in client.chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': PROMPT}],
            stream=True
        ):
            # part is ChatResponse object - use attribute access
            content = part.message.content
            if content:
                full_text += content
                print(content, end='', flush=True)
                sys.stdout.flush()
        
        print("\n", flush=True)
        elapsed = time.time() - t0
        sys.stdout.flush()
        
        print(f"\n[STATS] Generated in {elapsed:.2f}s:", flush=True)
        print(f"  • Text length: {len(full_text)} chars", flush=True)
        sys.stdout.flush()
        
        # Log output
        output = {
            "test": "chat_streaming_client",
            "method": "Client.chat()",
            "time_sec": elapsed,
            "response_length": len(full_text),
        }
        log_output("test_3_chat_client", output)
        return full_text if full_text else None
        
    except ImportError as e:
        print(f"[ERROR TEST 3] Import error: {e}", flush=True)
        sys.stdout.flush()
        return None
    except Exception as e:
        print(f"[ERROR TEST 3] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None


def test_requests_generate() -> Optional[str]:
    """Test 5: requests.post() to /api/chat endpoint with streaming (fallback)."""
    print(f"\n{'='*60}")
    print("[TEST 5] requests.post() - /api/chat - streaming enabled", flush=True)
    print(f"{'='*60}", flush=True)
    
    try:
        t0 = time.time()
        full_text = ""
        last_chunk = None
        
        print("[DEBUG] Sending POST request...", flush=True)
        sys.stdout.flush()
        
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "stream": True,
            },
            timeout=120,
            stream=True,
        )
        
        print(f"[DEBUG] Response status: {response.status_code}", flush=True)
        sys.stdout.flush()
        
        response.raise_for_status()
        print("[DEBUG] Status OK, starting to iterate lines...", flush=True)
        sys.stdout.flush()
        
        print("\nStreaming response:\n", flush=True)
        sys.stdout.flush()
        
        line_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            line_count += 1
            print(f"[DEBUG] Line {line_count} received", flush=True)
            
            try:
                last_chunk = json.loads(line)
                chunk_text = last_chunk.get("message", {}).get("content", "")
                if chunk_text:
                    full_text += chunk_text
                    print(chunk_text, end="", flush=True)
                    sys.stdout.flush()
            except json.JSONDecodeError as je:
                print(f"[DEBUG] JSON decode error on line {line_count}: {je}", flush=True)
                sys.stdout.flush()
        
        print(f"\n[DEBUG] Finished iteration, received {line_count} lines", flush=True)
        print("\n", flush=True)  # newline after streaming
        elapsed = time.time() - t0
        sys.stdout.flush()
        
        print(f"\n[STATS] Generated in {elapsed:.2f}s:", flush=True)
        print(f"  • Lines received: {line_count}", flush=True)
        print(f"  • Text length: {len(full_text)} chars", flush=True)
        sys.stdout.flush()
        
        # Log output
        output = {
            "test": "chat_requests_fallback",
            "endpoint": "/api/chat",
            "method": "requests.post()",
            "time_sec": elapsed,
            "lines_received": line_count,
            "response_length": len(full_text),
        }
        log_output("test_5_chat_requests_fallback", output)
        return full_text if full_text else None
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR TEST 5] Request error: {type(e).__name__}: {e}", flush=True)
        sys.stdout.flush()
        return None
    except Exception as e:
        print(f"[ERROR TEST 5] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None


# ============================================================================
# Optional TTS Functions (not invoked by default, kept for future use)
# ============================================================================

def generate_with_qwen(text: str, device: str = "cpu", speaker: str = "Ryan", language: str = "English") -> Optional[tuple]:
    """Generate audio from text using Qwen3-TTS (optional, not default)."""
    if not QWEN_AVAILABLE:
        print("[ERROR] Qwen3-TTS not available. Install: pip install soundfile torch")
        return None

    model_path = os.path.join(os.path.dirname(__file__), "Qwen3-TTS/models/0.6B-CustomVoice")
    if not os.path.isdir(model_path):
        print(f"[ERROR] Qwen3-TTS model not found at {model_path}")
        return None

    try:
        print(f"[Qwen] Loading model on {device}...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if device == "cuda" else "sdpa",
        )

        print(f"[Qwen] Generating audio with speaker={speaker}...")
        t0 = time.time()
        wavs, sr = tts.generate_custom_voice(text, speaker=speaker, language=language)
        elapsed = time.time() - t0
        duration = len(wavs[0]) / sr
        rtf = elapsed / duration if duration > 0 else 0

        print(f"[Qwen] Generated {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f}x)")
        return wavs, sr

    except Exception as e:
        print(f"[ERROR] Qwen3-TTS error: {e}")
        return None


def generate_with_edge_tts(text: str, voice: str = "en-US-GuyNeural") -> bool:
    """Generate audio stream from text using EdgeTTS (plays directly, no file)."""
    if not EDGE_TTS_AVAILABLE:
        print("[ERROR] edge-tts or pydub not available. Install: pip install edge-tts pydub")
        return False

    import asyncio
    import io

    async def stream_audio():
        try:
            print(f"[EdgeTTS] Generating and streaming audio with voice={voice}...")
            communicate = edge_tts.Communicate(text=text, voice=voice)
            
            # Collect audio chunks in memory
            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                audio_data.write(chunk["audio"])
            
            audio_data.seek(0)
            
            # Load and play audio
            print("[EdgeTTS] Playing audio...")
            audio = AudioSegment.from_file(audio_data, format="mp3")
            play(audio)
            print("[EdgeTTS] ✓ Audio playback complete")
            return True
            
        except Exception as e:
            print(f"[ERROR] EdgeTTS streaming error: {e}")
            return False

    return asyncio.run(stream_audio())


def save_qwen_audio(wavs, sr: int, name: str) -> str:
    """Save Qwen-generated audio as WAV (optional, not default)."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.wav")
    sf.write(path, wavs[0], sr)
    print(f"[Output] saved → {path}")
    return path


def main():
    print(f"\n{'='*60}")
    print(f"Ollama LLM + TTS Pipeline")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"API: {OLLAMA_HOST}")
    print(f"Prompt: {PROMPT}")
    print(f"{'='*60}")
    
    # Check Ollama is running
    if not check_ollama_health():
        print("\n❌ Ollama not running or unreachable!")
        print(f"   Start with: ollama serve")
        print(f"   Pull model: ollama pull {MODEL}")
        return
    
    print(f"✅ Ollama is running\n")
    
    # Step 1: Generate text with streaming
    print(f"{'='*60}")
    print(f"STEP 1: Generate Text (with fallback)")
    print(f"{'='*60}")
    
    # Skip test 3, go directly to test 5 (requests API)
    print(f"\n⏭️  Skipping test 3, trying test 5 (requests API)...")
    generated_text = test_requests_generate()  # Method 5
    
    if not generated_text:
        print(f"\n❌ Test 5 failed!")
        return
    
    print(f"\n✅ Text generated successfully!")
    
    # Step 2: NO AUDIO - just return for now
    print(f"\n{'='*60}")
    print(f"Test complete - no audio for now")
    print(f"{'='*60}")
    print(f"Generated text ({len(generated_text)} chars):")
    print(f"{generated_text}\n")


if __name__ == "__main__":
    main()
