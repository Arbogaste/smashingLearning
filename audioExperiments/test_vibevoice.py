"""
VibeVoice test suite for vid-production narration use case.

Model: microsoft/VibeVoice-Realtime-0.5B (auto-downloaded from HuggingFace on first run)
Voice presets: .pt files from VibeVoice/demo/voices/streaming_model/

Assumes VibeVoice is cloned at VibeVoice/ inside audioExperiments/.
If already cloned elsewhere, set VIBEVOICE_ROOT env var:
  VIBEVOICE_ROOT=/path/to/VibeVoice python test_vibevoice.py

Run from audioExperiments/:
  python test_vibevoice.py [--device cpu|cuda|mps] [--cfg_scale 1.5]
"""
import argparse
import copy
import os
import sys
import time

import torch

# Support VIBEVOICE_ROOT override for pre-existing clones
VIBEVOICE_ROOT = os.environ.get(
    "VIBEVOICE_ROOT",
    os.path.join(os.path.dirname(__file__), "VibeVoice"),
)

if not os.path.isdir(VIBEVOICE_ROOT):
    print(f"[ERROR] VibeVoice not found at: {VIBEVOICE_ROOT}")
    print("  Clone it: git clone https://github.com/microsoft/VibeVoice VibeVoice")
    print("  Or set:   VIBEVOICE_ROOT=/path/to/existing/clone python test_vibevoice.py")
    sys.exit(1)

sys.path.insert(0, VIBEVOICE_ROOT)
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

VOICES_DIR = os.path.join(VIBEVOICE_ROOT, "demo", "voices", "streaming_model")
MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
OUT = "outputs/vibevoice"

# Same narration corpus as the other test files for direct comparison
NARRATION_EN_SHORT = (
    "Welcome to today's market recap. The S&P 500 closed up 1.2 percent, "
    "driven by strong earnings in the technology and energy sectors."
)
NARRATION_EN_LONG = (
    "Breaking news from Wall Street. The Federal Reserve held interest rates steady "
    "at its May meeting, citing persistent inflation in services and a resilient labor "
    "market. Fed chair Jerome Powell signaled that the central bank is in no rush to "
    "cut rates, pushing back on market expectations for three cuts this year. "
    "The ten-year Treasury yield climbed six basis points to four-point-six percent. "
    "Meanwhile, Bitcoin crossed the ninety-five thousand dollar mark for the first time "
    "this month, as spot ETF inflows reached a two-week high. Gold retreated slightly "
    "after three consecutive sessions of gains, settling at twenty-three hundred and "
    "forty dollars per ounce."
)
NARRATION_IT = (
    "Benvenuti al riepilogo finanziario di oggi. La Banca Centrale Europea ha mantenuto "
    "i tassi invariati, segnalando un approccio cauto per la seconda metà dell'anno. "
    "I mercati europei hanno chiuso in rialzo, con il FTSE MIB in crescita dello zero "
    "virgola otto percento."
)
NARRATION_MINDFULNESS = (
    "Take a slow, deep breath. Let your shoulders drop away from your ears. "
    "In this moment, there is nothing you need to do, nowhere you need to be. "
    "You are exactly where you are supposed to be."
)


def list_voices():
    if not os.path.isdir(VOICES_DIR):
        return {}
    return {
        os.path.splitext(f)[0]: os.path.join(VOICES_DIR, f)
        for f in sorted(os.listdir(VOICES_DIR))
        if f.endswith(".pt")
    }


def load_voice(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def load_model(device):
    dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
    attn = "sdpa" if device in ("cpu", "mps") else "flash_attention_2"

    print(f"\nLoading {MODEL_ID} on {device} ({dtype}, {attn})...")
    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_ID)

    try:
        if device == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_ID, torch_dtype=dtype, attn_implementation=attn, device_map=None,
            )
            model.to("mps")
        else:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_ID, torch_dtype=dtype, attn_implementation=attn, device_map=device,
            )
    except Exception as e:
        if "flash_attention_2" in str(e) or "flash_attention_2" in attn:
            print(f"  flash_attention_2 unavailable ({e}), falling back to sdpa")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_ID, torch_dtype=dtype, attn_implementation="sdpa", device_map=device,
            )
        else:
            raise

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)
    return processor, model


def generate(processor, model, text, prefilled, device, cfg_scale):
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=prefilled,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        all_prefilled_outputs=copy.deepcopy(prefilled),
    )
    return outputs.speech_outputs[0] if outputs.speech_outputs else None


def save(processor, audio, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.wav")
    processor.save_audio(audio, output_path=path)
    print(f"  saved → {path}")
    return path


def rtf(audio, elapsed, sample_rate=24000):
    n = audio.shape[-1] if len(audio.shape) > 0 else len(audio)
    duration = n / sample_rate
    return elapsed / duration, duration


def run_test(processor, model, prefilled, device, cfg_scale, text, label):
    print(f"\n[{label}] {len(text.split())} words")
    t0 = time.time()
    audio = generate(processor, model, text, prefilled, device, cfg_scale)
    elapsed = time.time() - t0
    if audio is None:
        print("  ERROR: no audio generated")
        return
    r, dur = rtf(audio, elapsed)
    print(f"  duration={dur:.1f}s  elapsed={elapsed:.1f}s  RTF={r:.2f}x")
    return audio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--voice", help="Run only this voice (e.g. en-Emma_woman)")
    args = p.parse_args()

    voices = list_voices()
    if not voices:
        print(f"[ERROR] No voice .pt files found in {VOICES_DIR}")
        sys.exit(1)

    print(f"Found {len(voices)} voices: {', '.join(voices)}")

    processor, model = load_model(args.device)

    # --- English voices ---
    en_voices = {k: v for k, v in voices.items() if k.startswith("en-")}
    target_en = {args.voice: voices[args.voice]} if args.voice and args.voice in voices else en_voices

    for name, path in target_en.items():
        prefilled = load_voice(path, args.device)

        audio = run_test(processor, model, prefilled, args.device, args.cfg_scale,
                         NARRATION_EN_SHORT, f"{name} / short narration")
        if audio is not None:
            save(processor, audio, f"vibe_{name}_short")

        audio = run_test(processor, model, prefilled, args.device, args.cfg_scale,
                         NARRATION_EN_LONG, f"{name} / long narration")
        if audio is not None:
            save(processor, audio, f"vibe_{name}_long")

        audio = run_test(processor, model, prefilled, args.device, args.cfg_scale,
                         NARRATION_MINDFULNESS, f"{name} / mindfulness")
        if audio is not None:
            save(processor, audio, f"vibe_{name}_mindfulness")

    # --- Italian voices ---
    it_voices = {k: v for k, v in voices.items() if k.startswith("it-")}
    for name, path in it_voices.items():
        prefilled = load_voice(path, args.device)
        audio = run_test(processor, model, prefilled, args.device, args.cfg_scale,
                         NARRATION_IT, f"{name} / Italian narration")
        if audio is not None:
            save(processor, audio, f"vibe_{name}_italian")

    # --- RTF stress test: back-to-back 10 segments (simulates pipeline batch) ---
    print(f"\n[Stress] 10 consecutive segments with same voice (en-Emma_woman or first available)")
    stress_voice = "en-emma_woman" if "en-emma_woman" in voices else list(en_voices)[0]
    prefilled = load_voice(voices[stress_voice], args.device)
    segments = [NARRATION_EN_SHORT] * 10

    times = []
    for i, seg in enumerate(segments):
        t0 = time.time()
        audio = generate(processor, model, seg, prefilled, args.device, args.cfg_scale)
        elapsed = time.time() - t0
        if audio is not None:
            r, dur = rtf(audio, elapsed)
            times.append((elapsed, dur, r))

    if times:
        avg_rtf = sum(r for _, _, r in times) / len(times)
        total_gen = sum(e for e, _, _ in times)
        total_audio = sum(d for _, d, _ in times)
        print(f"  10 segments: total_audio={total_audio:.1f}s  total_gen={total_gen:.1f}s  avg_RTF={avg_rtf:.2f}x")

    print(f"\nDone. Listen to {OUT}/ and fill in the decision matrix in README.md")
    print("Available voices for vid-production:")
    for name in voices:
        print(f"  {name}")


if __name__ == "__main__":
    main()
