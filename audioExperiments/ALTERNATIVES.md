# TTS Alternatives — CPU-only Linux (Quadro M1200, cc5.0)

> Goal: fast + quality narration for YouTube videos. GPU unusable (PyTorch cc≥7.5 required, GPU is cc5.0).

---

## Rankings (speed priority)

| # | Repo | Model | RTF (CPU est.) | Quality | Voice cloning | Status |
|---|------|-------|---------------|---------|---------------|--------|
| 1 | **kokoro** | Kokoro-82M (StyleTTS2) | ~0.3-0.5x | MEDIUM-HIGH | No (presets) | ✅ test first |
| 2 | **chatterbox** | Chatterbox-Turbo 350M | ~1.0-2.5x | HIGH | Yes | ✅ test second |
| 3 | **OmniVoice** | OmniVoice diffusion | ~2.0-5.0x | HIGHEST | Yes | ⚠ slow, test last |
| 4 | **voicebox** | Multi-engine server | varies | varies | Yes | ❌ skip (complex build) |
| 5 | **Colab-notebooks** | cloud notebooks | N/A | N/A | N/A | ❌ skip (Colab-only) |

---

## Detail

### 1. kokoro — BEST BET
- **82M params**, single forward pass, no diffusion
- ONNX variant available for even faster CPU
- 50+ preset voices, Italian supported (`lang_code='i'`)
- `pip install kokoro misaki[en]` + `apt install espeak-ng`
- RTF 0.3-0.5x = ~3-5s for 10s audio on CPU

### 2. chatterbox (Turbo)
- **350M params**, single-step distilled diffusion
- Zero-shot voice cloning from audio reference
- Paralinguistic tags: `[laugh]`, `[cough]`, `[chuckle]`
- English only in Turbo variant
- `pip install chatterbox-tts`

### 3. OmniVoice
- Best multilingual (600+ languages), voice design via instruct
- Similar API to Qwen3-TTS VoiceDesign (gender, age, pitch attributes)
- Default 32 diffusion steps → reduce to 16 for 2x speed
- `pip install omnivoice`

### 4. voicebox — SKIP
- Tauri (Rust) desktop app + FastAPI backend, wraps all engines above
- Requires Bun + Rust build chain, no Linux prebuilt binary
- Interesting long-term but overkill for testing

### 5. Colab-notebooks — SKIP
- Google Colab wrappers, not local inference

---

## Hypothesis

Kokoro will cover 80% of use cases (English narration, multiple female presets, very fast).
Chatterbox-Turbo as fallback when voice cloning needed.
OmniVoice only if Italian voice design quality matters and we can tolerate 20-50s generation.
