# audioExperiments

Test bench for audio engines under evaluation for [vid-production](https://github.com/arbogaste/vid-production).

**Goal**: pick the best local TTS engine for automated video narration. Secondary goal: evaluate ASR engines for round-trip quality tests and voice-clone workflows.

---

## Projects

| Dir | Type | Status | Runtime | Voice clone | Italian |
|-----|------|--------|---------|-------------|---------|
| `Qwen3-TTS/` | **TTS** | cloned | Python + PyTorch | yes (3s ref) | yes |
| `VibeVoice/` | **TTS** | to clone | Python + PyTorch | preset `.pt` files | yes (`it-Spk0/1`) |
| `edge-tts` | **TTS** | pip only | Python async (cloud) | no | yes |
| `voxtral.c/` | ASR | cloned | C, MPS/BLAS | n/a | no |
| `qwen-asr/` | ASR | cloned | C, BLAS | n/a | yes |

**Note on ASR tools**: `voxtral.c` and `qwen-asr` are speech-to-text, not TTS. Their role here is the round-trip quality test — generate audio with a TTS engine, transcribe it back, compare against the original script to measure intelligibility (WER). They are also useful for future subtitle generation.

---

## Test Scripts

All scripts run from `audioExperiments/` and write output to `outputs/<engine>/`.

```
audioExperiments/
  test_qwen_tts.py      # Qwen3-TTS: CustomVoice, VoiceDesign, VoiceClone
  test_vibevoice.py     # VibeVoice: preset voices, RTF stress test
  test_edgetts.py       # EdgeTTS: voice candidates, rate control, Italian
  outputs/
    qwen/
    vibevoice/
    edgetts/
    ref_audio/          # place your reference WAVs here for voice clone tests
```

---

## TTS Engines

### 1. Qwen3-TTS

Official Alibaba Qwen Python package. Local inference, PyTorch. Three model variants:

| Model | Size | Mode | Notes |
|-------|------|------|-------|
| `0.6B-CustomVoice` | ~1.2GB | 9 preset speakers | fastest, no `instruct` control |
| `1.7B-CustomVoice` | ~3.4GB | 9 preset speakers | + `instruct` style control |
| `1.7B-VoiceDesign` | ~3.4GB | voice from text prompt | design a character voice |
| `0.6B-Base` | ~1.2GB | voice clone | 3s reference audio, x-vector or ICL mode |
| `1.7B-Base` | ~3.4GB | voice clone | same, higher quality |

**Preset speakers (CustomVoice)**

| Speaker | Native language | Voice description |
|---------|----------------|-------------------|
| Ryan | English | Dynamic male, strong rhythmic drive |
| Aiden | English | Sunny American male, clear midrange |
| Vivian | Chinese | Bright, slightly edgy young female |
| Serena | Chinese | Warm, gentle young female |
| Uncle_Fu | Chinese | Low, mellow male |
| Dylan | Chinese (Beijing) | Clear, natural male |
| Eric | Chinese (Sichuan) | Slightly husky, lively male |
| Ono_Anna | Japanese | Light, nimble female |
| Sohee | Korean | Warm, rich emotion female |

**Setup**

```bash
cd Qwen3-TTS
pip install -e .
pip install flash-attn --no-build-isolation  # optional, GPU only

# Download models (only download what you need to test)
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/tokenizer
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/0.6B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir models/1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/0.6B-Base
```

**Run tests**

```bash
# CustomVoice 0.6B (fastest, start here)
python test_qwen_tts.py --model 0.6B --suite custom

# CustomVoice 1.7B with instruct control
python test_qwen_tts.py --model 1.7B --suite custom

# VoiceDesign: generate a persona from a text description
python test_qwen_tts.py --suite design

# VoiceClone: put reference WAVs in outputs/ref_audio/ first
python test_qwen_tts.py --suite clone

# CPU only
python test_qwen_tts.py --device cpu --model 0.6B --suite custom
```

**What the tests cover**

- T1: Short English narration (finance, ~30 words) → measures quality and RTF
- T2: Long English narration (~100 words) → measures consistency and RTF at scale
- T3: Italian narration → multilingual capability
- T4–T5: `instruct` style control (1.7B only) — news anchor tone, mindfulness tone
- T6: Batch generation (3 sections in one call) → measures throughput
- Clone: x-vector mode (no transcript) and ICL mode (with transcript)

**API reference**

```python
from qwen_tts import Qwen3TTSModel
import torch, soundfile as sf

tts = Qwen3TTSModel.from_pretrained(
    "Qwen3-TTS/models/0.6B-CustomVoice",
    device_map="cuda",           # or "cpu"
    torch_dtype=torch.bfloat16,  # float32 on CPU
    attn_implementation="flash_attention_2",  # or "sdpa" on CPU
)

# List available speakers and languages
print(tts.model.get_supported_speakers())
print(tts.model.get_supported_languages())

# Generate — returns (List[np.ndarray], int sample_rate)
wavs, sr = tts.generate_custom_voice(
    text="Your narration text here.",
    speaker="Ryan",
    language="English",
    instruct="speak slowly, news anchor tone",  # 1.7B only
)
sf.write("output.wav", wavs[0], sr)

# Batch — one call, multiple sections
wavs, sr = tts.generate_custom_voice(
    text=["Section one.", "Section two.", "Section three."],
    speaker="Ryan",          # broadcast to all, or list per section
    language="English",
)
for i, w in enumerate(wavs):
    sf.write(f"section_{i}.wav", w, sr)

# VoiceDesign — generate a persona from description
tts_design = Qwen3TTSModel.from_pretrained("Qwen3-TTS/models/1.7B-VoiceDesign", ...)
wavs, sr = tts_design.generate_voice_design(
    text="Text to speak.",
    language="English",
    instruct="Deep, confident male. Seasoned financial journalist tone.",
)

# Voice clone — Base model required
tts_base = Qwen3TTSModel.from_pretrained("Qwen3-TTS/models/0.6B-Base", ...)
wavs, sr = tts_base.generate_voice_clone(
    text="Text to speak.",
    language="English",
    ref_audio="reference.wav",   # 3–10s of clean speech
    x_vector_only_mode=True,     # no transcript needed; False = ICL mode (better quality)
    # ref_text="transcript of reference audio",  # required if x_vector_only_mode=False
)
```

---

### 2. VibeVoice

Microsoft TTS. Local inference, PyTorch. Model: `VibeVoice-Realtime-0.5B`. Works via preset voice latent files (`.pt`). No voice clone — presets only, but 25 voices available including Italian.

**Available preset voices**

| File | Gender | Language |
|------|--------|----------|
| `en-Carter_man.pt` | male | English |
| `en-Davis_man.pt` | male | English |
| `en-Emma_woman.pt` | female | English |
| `en-Frank_man.pt` | male | English |
| `en-Grace_woman.pt` | female | English |
| `en-Mike_man.pt` | male | English |
| `it-Spk0_woman.pt` | female | Italian |
| `it-Spk1_man.pt` | male | Italian |
| `de-Spk0_man.pt` | male | German |
| `fr-Spk0_man.pt` | male | French |
| `jp-Spk0_man.pt` | male | Japanese |
| ... | | |

**Setup**

```bash
# Clone into audioExperiments/VibeVoice
git clone https://github.com/microsoft/VibeVoice VibeVoice
cd VibeVoice
pip install -e .
# Model downloads automatically from HuggingFace on first run
```

If you already have VibeVoice cloned elsewhere:

```bash
VIBEVOICE_ROOT=/path/to/VibeVoice python test_vibevoice.py
```

**Run tests**

```bash
# All English voices
python test_vibevoice.py

# Single voice
python test_vibevoice.py --voice en-Emma_woman

# Italian voices
# (always included — script runs it-* voices automatically)

# CPU
python test_vibevoice.py --device cpu
```

**What the tests cover**

- All English preset voices: short narration, long narration, mindfulness text
- Italian voices: Italian narration
- Stress test: 10 back-to-back segments with same voice → average RTF, consistency

**API reference**

```python
import copy, torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

processor = VibeVoiceStreamingProcessor.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    torch_dtype=torch.bfloat16,       # float32 on CPU/MPS
    attn_implementation="flash_attention_2",  # sdpa on CPU/MPS
    device_map="cuda",
)
model.eval()
model.set_ddpm_inference_steps(num_steps=5)  # quality vs speed tradeoff

# Load voice preset
prefilled = torch.load("VibeVoice/demo/voices/streaming_model/en-Emma_woman.pt",
                       map_location="cuda", weights_only=False)

# Prepare input
inputs = processor.process_input_with_cached_prompt(
    text="Your narration text here.",
    cached_prompt=prefilled,
    padding=True, return_tensors="pt", return_attention_mask=True,
)
for k, v in inputs.items():
    if torch.is_tensor(v): inputs[k] = v.to("cuda")

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=None,
    cfg_scale=1.5,          # higher = more faithful to voice preset
    tokenizer=processor.tokenizer,
    generation_config={"do_sample": False},
    all_prefilled_outputs=copy.deepcopy(prefilled),
)

# Save (processor handles sample rate internally, output is 24kHz WAV)
processor.save_audio(outputs.speech_outputs[0], output_path="output.wav")
```

---

### 3. EdgeTTS

Microsoft Azure TTS via `edge-tts`. Cloud-based, no GPU, no model download. Production-grade quality baseline. No voice cloning.

**Setup**

```bash
pip install edge-tts
```

**Run tests**

```bash
# All candidate voices
python test_edgetts.py

# Single voice
python test_edgetts.py --voice en-US-ChristopherNeural

# List all available voices
python test_edgetts.py --list-voices
```

**What the tests cover**

- T1: All candidate EN voices on short narration → pick best 2–3 for vid-production
- T2: Long narration with default voice
- T3: Rate control (`-15%`, `+0%`, `+15%`) — slower for news anchor, faster for presenter
- T4: Mindfulness narration with female voices at reduced rate
- T5: Italian voices
- T6: Stress test — 10 consecutive segments → measures latency consistency and reliability

**Candidate voices for vid-production channels**

| Channel | Voice | Rate |
|---------|-------|------|
| TheRoarWire (news) | `en-US-ChristopherNeural` | `-10%` |
| PeakyStockRadar (finance) | `en-US-GuyNeural` | `+0%` |
| MomentOfMindfulness | `en-US-AriaNeural` | `-20%` |
| FunLabChannel | `en-US-EricNeural` | `+10%` |

**API reference**

```python
import asyncio
import edge_tts

async def generate(text, voice, path, rate="+0%"):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(path)  # saves as MP3

asyncio.run(generate(
    text="The S&P 500 closed up 1.2 percent today.",
    voice="en-US-GuyNeural",
    path="output.mp3",
    rate="-10%",
))

# List voices
async def list_voices():
    return await edge_tts.list_voices()
voices = asyncio.run(list_voices())
```

---

## ASR Engines (for round-trip quality tests)

### voxtral.c

Speech-to-text. Voxtral Mini 4B. Pure C, MPS on Apple Silicon, BLAS on Linux.

```bash
cd voxtral.c
make blas        # Linux with OpenBLAS
# make mps       # Apple Silicon
./download_model.sh

# Transcribe TTS output
ffmpeg -i output.mp3 -f s16le -ar 16000 -ac 1 output.wav 2>/dev/null
./voxtral -d voxtral-model -i output.wav
```

### qwen-asr

Speech-to-text. Qwen3-ASR 0.6B/1.7B. Pure C, BLAS. Linux-first.

```bash
cd qwen-asr
make blas
./download_model.sh  # choose 0.6B (fast) or 1.7B (accurate)

./qwen_asr -d qwen3-asr-0.6b -i output.wav
# Italian (auto-detected, or force: --language Italian)
```

---

## Round-trip Quality Test

After generating audio with each TTS engine, transcribe with an ASR engine and compute WER.

```bash
pip install jiwer

# Example: EdgeTTS → voxtral round-trip
python test_edgetts.py --voice en-US-GuyNeural
ffmpeg -i outputs/edgetts/edge_en-US-GuyNeural_short.mp3 \
       -f s16le -ar 16000 -ac 1 outputs/tmp_short.wav 2>/dev/null
./voxtral.c/voxtral -d voxtral.c/voxtral-model -i outputs/tmp_short.wav > outputs/transcript.txt

python3 - <<'EOF'
from jiwer import wer
ref = "Welcome to today's market recap. The S and P 500 closed up 1.2 percent, driven by strong earnings in the technology and energy sectors."
hyp = open("outputs/transcript.txt").read().strip()
print(f"WER: {wer(ref, hyp)*100:.1f}%")
EOF
```

---

## Decision Matrix

Fill in after running the tests:

| Engine | Size | Natural? | RTF CPU | RTF GPU | Italian | Clone | WER | Notes |
|--------|------|----------|---------|---------|---------|-------|-----|-------|
| EdgeTTS | cloud | | n/a | n/a | yes | no | | baseline |
| Qwen3-TTS 0.6B | ~1.2GB | | | | yes | no | | |
| Qwen3-TTS 1.7B | ~3.4GB | | | | yes | instruct | | |
| VibeVoice 0.5B | ~1GB | | | | yes (preset) | no | | |

**Target for vid-production**: naturalness ≥ EdgeTTS, WER < 5%, RTF < 2× on CPU (2-min segment).
