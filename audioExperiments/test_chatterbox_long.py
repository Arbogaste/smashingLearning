#!/usr/bin/env python3
"""
Chatterbox-Turbo long-form generation via text chunking.
Splits text into ~3-sentence chunks, generates each, concatenates.

Run:
  python3 test_chatterbox_long.py
  python3 test_chatterbox_long.py --ref reference.wav
"""

import argparse
import re
import sys
import time
from pathlib import Path

import torch

OUT_DIR = Path("outputs/chatterbox_long")

LONG_TEXT = (
    "The history of artificial intelligence is one of the most fascinating stories in modern science. "
    "It begins in the mid-twentieth century, when a small group of visionary researchers dared to ask whether machines could think. "
    "Alan Turing proposed his famous imitation game in nineteen fifty, laying the philosophical groundwork for decades of inquiry. "
    "Just a few years later, in nineteen fifty-six, John McCarthy coined the term artificial intelligence at a summer conference at Dartmouth College. "
    "Those early pioneers believed that every aspect of human intelligence could, in principle, be described with enough precision for a machine to simulate it. "
    "The optimism was extraordinary. Researchers predicted that a machine as intelligent as a human being would exist within a generation. "
    "Reality, however, proved far more stubborn. Progress stalled repeatedly, and funding dried up during periods that historians now call AI winters. "
    "Yet each winter was followed by a spring. The invention of expert systems in the nineteen seventies and eighties brought renewed excitement. "
    "These programs encoded human expertise as explicit rules, and they achieved remarkable results in narrow domains like medical diagnosis and chess. "
    "Then came the connectionist revolution. Artificial neural networks, loosely inspired by the structure of the human brain, began to outperform rule-based systems on certain tasks. "
    "The turning point arrived in two thousand twelve, when a deep neural network crushed the competition in a major image recognition contest, reducing the error rate by nearly half. "
    "Suddenly every major technology company in the world was racing to hire machine learning researchers and pour resources into the field. "
    "Language models followed. Systems learned to predict the next word in a sentence, and from that humble objective emerged an astonishing range of capabilities. "
    "Today, large language models can write poetry, explain scientific concepts, translate between languages, and hold nuanced conversations on almost any topic. "
    "We are living through one of the most consequential technological transitions in human history, and it is only just beginning. "
    "The question is no longer whether artificial intelligence will transform society, but how quickly and in what directions. "
    "Some researchers believe we are approaching a moment of general artificial intelligence, a system that can match human performance across virtually any cognitive task. "
    "Others are more cautious, arguing that current approaches have fundamental limitations that will require entirely new paradigms to overcome. "
    "What is clear is that the decisions made by governments, companies, and individuals over the coming years will shape the trajectory of this technology for generations. "
    "We must think carefully about how to develop artificial intelligence in ways that are safe, equitable, and aligned with human values. "
    "That is not a technical challenge alone. It is a philosophical, political, and deeply human one."
)


def split_into_chunks(text: str, max_words: int = 40) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_words = 0
    for sent in sentences:
        words = len(sent.split())
        if current_words + words > max_words and current:
            chunks.append(' '.join(current))
            current = [sent]
            current_words = words
        else:
            current.append(sent)
            current_words += words
    if current:
        chunks.append(' '.join(current))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=None, help="Reference audio for voice cloning")
    parser.add_argument("--max-words", type=int, default=40, help="Max words per chunk")
    args = parser.parse_args()

    try:
        import torchaudio
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError as e:
        print(f"✗ Missing dep: {e}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = split_into_chunks(LONG_TEXT, max_words=args.max_words)
    print(f"Text split into {len(chunks)} chunks (max {args.max_words} words each)")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] {len(c.split())} words: {c[:60]}...")

    print(f"\nLoading model...")
    t_load = time.time()
    try:
        model = ChatterboxTurboTTS.from_pretrained(device="cpu")
    except Exception as e:
        print(f"✗ FAIL loading model: {e}")
        return 1
    print(f"✓ model loaded in {time.time()-t_load:.1f}s\n")

    if args.ref and Path(args.ref).exists():
        print(f"Preparing voice from {args.ref}...")
        model.prepare_conditionals(args.ref)

    wav_chunks = []
    total_duration = 0.0
    t_gen_start = time.time()

    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] generating: {chunk[:60]}...")
        t0 = time.time()
        try:
            wav = model.generate(chunk)
        except Exception as e:
            print(f"  ✗ FAIL chunk {i+1}: {e}")
            import traceback; traceback.print_exc()
            continue
        elapsed = time.time() - t0
        sr = 24000
        dur = wav.shape[-1] / sr
        total_duration += dur
        print(f"  ✓ duration={dur:.1f}s elapsed={elapsed:.1f}s RTF={elapsed/dur:.2f}x")
        wav_chunks.append(wav.squeeze(0).cpu())

    if not wav_chunks:
        print("✗ No chunks generated")
        return 1

    # Concatenate all chunks
    full_wav = torch.cat(wav_chunks, dim=-1).unsqueeze(0)
    total_elapsed = time.time() - t_gen_start

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = OUT_DIR / f"{ts}_long_{len(wav_chunks)}chunks.wav"
    torchaudio.save(str(out_path), full_wav, 24000)

    print(f"\n{'='*50}")
    print(f"✅ {len(wav_chunks)}/{len(chunks)} chunks OK")
    print(f"   output:   {out_path}")
    print(f"   duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"   elapsed:  {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"   avg RTF:  {total_elapsed/total_duration:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
