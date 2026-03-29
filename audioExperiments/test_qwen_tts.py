"""
Qwen3-TTS test suite for vid-production narration use case.

Models (download before running):
  huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir Qwen3-TTS/models/tokenizer
  huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir Qwen3-TTS/models/0.6B-CustomVoice
  huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir Qwen3-TTS/models/1.7B-CustomVoice
  huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir Qwen3-TTS/models/1.7B-VoiceDesign
  huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir Qwen3-TTS/models/0.6B-Base

Run from audioExperiments/:
  python test_qwen_tts.py [--model 0.6B|1.7B] [--device cpu|cuda] [--suite custom|design|clone|all]
"""
import argparse
import os
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
from qwen_tts import Qwen3TTSModel

MODELS = {
    "0.6B-CustomVoice": "Qwen3-TTS/models/0.6B-CustomVoice",
    "1.7B-CustomVoice": "Qwen3-TTS/models/1.7B-CustomVoice",
    "1.7B-VoiceDesign": "Qwen3-TTS/models/1.7B-VoiceDesign",
    "0.6B-Base":        "Qwen3-TTS/models/0.6B-Base",
}

OUT = "outputs/qwen"

# Narration samples representative of vid-production script sections
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
    "forty dollars per ounce. Analysts expect volatility to remain elevated heading "
    "into the Q3 earnings season, with big banks reporting next week."
)
NARRATION_IT = (
    "Benvenuti al riepilogo finanziario di oggi. La Banca Centrale Europea ha mantenuto "
    "i tassi invariati, segnalando un approccio cauto per la seconda metà dell'anno. "
    "I mercati europei hanno chiuso in rialzo, con il FTSE MIB in crescita dello zero "
    "virgola otto percento."
)
# Mindfulness channel sample
NARRATION_MINDFULNESS = (
    "Take a slow, deep breath. Let your shoulders drop away from your ears. "
    "In this moment, there is nothing you need to do, nowhere you need to be. "
    "You are exactly where you are supposed to be."
)


def save(wavs, sr, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.wav")
    sf.write(path, wavs[0], sr)
    print(f"  saved → {path}")
    return path


def load_model(key, device, dtype):
    path = MODELS[key]
    if not os.path.isdir(path):
        print(f"[SKIP] model not found: {path}")
        return None
    print(f"\nLoading {key} on {device}...")
    # Qwen3-TTS docs show dtype= but from_pretrained() forwards **kwargs to
    # AutoModel.from_pretrained() which uses torch_dtype= (HuggingFace standard).
    # Both may work depending on HF version; torch_dtype= is the authoritative param.
    try:
        tts = Qwen3TTSModel.from_pretrained(
            path,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if device == "cuda" else "sdpa",
        )
    except TypeError:
        # Fallback: some HF versions accept dtype= as alias
        tts = Qwen3TTSModel.from_pretrained(
            path,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
    return tts


def rtf(wav, sr, elapsed):
    duration = len(wav) / sr
    return elapsed / duration


def suite_custom(device, dtype, model_size):
    key = f"{model_size}-CustomVoice"
    tts = load_model(key, device, dtype)
    if tts is None:
        return

    supports_instruct = model_size == "1.7B"

    # T1: English narration, preset male voice (Ryan)
    print(f"\n[T1] English narration, speaker=Ryan")
    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(NARRATION_EN_SHORT, speaker="Ryan", language="English")
    elapsed = time.time() - t0
    print(f"  RTF={rtf(wavs[0], sr, elapsed):.2f}x  ({elapsed:.1f}s for {len(wavs[0])/sr:.1f}s audio)")
    save(wavs, sr, f"qwen_{model_size}_ryan_short")

    # T2: Long narration — simulates a full script section
    print(f"\n[T2] Long narration (~{len(NARRATION_EN_LONG.split())} words), speaker=Aiden")
    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(NARRATION_EN_LONG, speaker="Aiden", language="English")
    elapsed = time.time() - t0
    print(f"  RTF={rtf(wavs[0], sr, elapsed):.2f}x  ({elapsed:.1f}s for {len(wavs[0])/sr:.1f}s audio)")
    save(wavs, sr, f"qwen_{model_size}_aiden_long")

    # T3: Italian (for multilingual channel support)
    print(f"\n[T3] Italian narration")
    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(NARRATION_IT, speaker="Ryan", language="Italian")
    elapsed = time.time() - t0
    print(f"  RTF={rtf(wavs[0], sr, elapsed):.2f}x")
    save(wavs, sr, f"qwen_{model_size}_italian")

    # T4: Instruct control (1.7B only — 0.6B silently ignores instruct)
    if supports_instruct:
        print(f"\n[T4] Instruct: slow, authoritative news anchor tone")
        t0 = time.time()
        wavs, sr = tts.generate_custom_voice(
            NARRATION_EN_SHORT,
            speaker="Ryan",
            language="English",
            instruct="speak slowly and deliberately, authoritative news anchor tone, no emotion",
        )
        elapsed = time.time() - t0
        print(f"  RTF={rtf(wavs[0], sr, elapsed):.2f}x")
        save(wavs, sr, f"qwen_{model_size}_instruct_anchor")

        print(f"\n[T5] Instruct: calm, meditative voice")
        wavs, sr = tts.generate_custom_voice(
            NARRATION_MINDFULNESS,
            speaker="Ryan",
            language="English",
            instruct="speak very slowly and gently, calm meditative voice, warm and soothing",
        )
        save(wavs, sr, f"qwen_{model_size}_instruct_mindfulness")

    # T6: Batch — simulate pipeline generating multiple sections at once
    print(f"\n[T6] Batch: 3 sections in one call")
    sections = [NARRATION_EN_SHORT, NARRATION_EN_LONG[:200], NARRATION_MINDFULNESS]
    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(
        text=sections,
        speaker=["Ryan", "Aiden", "Ryan"],
        language=["English", "English", "English"],
    )
    elapsed = time.time() - t0
    total_dur = sum(len(w) for w in wavs) / sr
    print(f"  {len(wavs)} wavs, total={total_dur:.1f}s, elapsed={elapsed:.1f}s, RTF={elapsed/total_dur:.2f}x")
    for i, w in enumerate(wavs):
        save([w], sr, f"qwen_{model_size}_batch_section{i}")


def suite_design(device, dtype):
    tts = load_model("1.7B-VoiceDesign", device, dtype)
    if tts is None:
        return

    # T1: design a voice matching a specific channel persona
    print(f"\n[VoiceDesign T1] Investigative finance host")
    wavs, sr = tts.generate_voice_design(
        text=NARRATION_EN_SHORT,
        language="English",
        instruct=(
            "Deep, confident male voice. Measured pace, slight gravitas. "
            "Sounds like a seasoned financial journalist on cable news."
        ),
    )
    save(wavs, sr, "qwen_design_finance_host")

    # T2: Calm narrator for mindfulness content
    print(f"\n[VoiceDesign T2] Mindfulness narrator")
    wavs, sr = tts.generate_voice_design(
        text=NARRATION_MINDFULNESS,
        language="English",
        instruct="Soft, warm female voice. Very slow, gentle breathing rhythm. Meditative and calming.",
    )
    save(wavs, sr, "qwen_design_mindfulness")

    # T3: Lively, enthusiastic voice for entertainment channel
    print(f"\n[VoiceDesign T3] Enthusiastic YouTube presenter")
    wavs, sr = tts.generate_voice_design(
        text="Welcome back to Fun Lab Channel! Today we're going to do something absolutely insane.",
        language="English",
        instruct="Energetic, enthusiastic young male voice. Upbeat pacing, YouTube presenter style.",
    )
    save(wavs, sr, "qwen_design_funlab")


def suite_clone(device, dtype):
    """
    Voice clone test using the Base model.
    Provide your own reference audio files (3–10 seconds of clean speech).
    Place them in outputs/ref_audio/ before running.
    """
    tts = load_model("0.6B-Base", device, dtype)
    if tts is None:
        return

    ref_dir = "outputs/ref_audio"
    refs = [f for f in os.listdir(ref_dir) if f.endswith(".wav")] if os.path.isdir(ref_dir) else []

    if not refs:
        print(f"\n[Clone] No reference audio found in {ref_dir}/")
        print("  Place 3–10s WAV files there to test voice cloning.")
        print("  Tip: record yourself saying ~2 sentences and save as ref_voice.wav")
        return

    for ref_name in refs:
        ref_path = os.path.join(ref_dir, ref_name)
        stem = os.path.splitext(ref_name)[0]
        print(f"\n[Clone] reference: {ref_name}")

        # ICL mode: ref_text required — transcribe your reference audio first
        # For a quick test without ref_text, use x_vector_only_mode=True
        t0 = time.time()
        wavs, sr = tts.generate_voice_clone(
            text=NARRATION_EN_SHORT,
            language="English",
            ref_audio=ref_path,
            x_vector_only_mode=True,  # no ref_text needed
        )
        elapsed = time.time() - t0
        print(f"  RTF={rtf(wavs[0], sr, elapsed):.2f}x (x-vector only mode)")
        save(wavs, sr, f"qwen_clone_{stem}_xvec")

        # ICL mode for better quality — requires knowing what the ref audio says
        # Uncomment and set ref_text to the transcript of your reference file:
        # wavs, sr = tts.generate_voice_clone(
        #     text=NARRATION_EN_SHORT,
        #     language="English",
        #     ref_audio=ref_path,
        #     ref_text="<transcript of your reference audio here>",
        #     x_vector_only_mode=False,
        # )
        # save(wavs, sr, f"qwen_clone_{stem}_icl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["0.6B", "1.7B", "both"], default="0.6B")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--suite", choices=["custom", "design", "clone", "all"], default="custom")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    print(f"Device: {args.device} | dtype: {dtype}")

    sizes = ["0.6B", "1.7B"] if args.model == "both" else [args.model]

    if args.suite in ("custom", "all"):
        for s in sizes:
            suite_custom(args.device, dtype, s)

    if args.suite in ("design", "all"):
        suite_design(args.device, dtype)

    if args.suite in ("clone", "all"):
        suite_clone(args.device, dtype)

    print("\nDone. Listen to outputs/qwen/ and fill in the decision matrix in README.md")


if __name__ == "__main__":
    main()
