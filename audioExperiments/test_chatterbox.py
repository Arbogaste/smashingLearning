#!/usr/bin/env python3
"""
Test Chatterbox-Turbo CPU inference.

Prerequisites:
  pip install chatterbox-tts torch torchaudio

Run:
  python3 test_chatterbox.py
  python3 test_chatterbox.py --ref reference.wav   # test voice cloning
"""

import argparse
import sys
import time
from pathlib import Path

OUT_DIR = Path("outputs/chatterbox")
TEST_TEXT = (
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
    "We are living through one of the most consequential technological transitions in human history, and it is only just beginning."
)

TEST_TEXT_EXPRESSIVE = (
    "Welcome, everyone, to this exploration of artificial intelligence. [laugh] I know, I know - AI is everywhere these days, and you might be wondering what all the fuss is about. "
    "Well, let me tell you, the story of how we got here is absolutely wild. [laugh] "
    "It starts back in the nineteen fifties, when a brilliant mathematician named Alan Turing asked a deceptively simple question: can machines think? "
    "Now, at the time, computers filled entire rooms and could barely multiply numbers quickly. [laugh] And here is this guy asking if they might one day be conscious! "
    "Fast forward to today, and we have systems that can write novels, compose music, and diagnose diseases - sometimes better than human doctors. "
    "The pace of progress has been genuinely breathtaking. Ten years ago, getting a computer to recognize a cat in a photograph was considered a major achievement. "
    "Now these systems can describe the emotional content of a painting, argue philosophy, and explain quantum mechanics to a ten-year-old. [laugh] "
    "Of course, this raises enormous questions. What does it mean for human work? For creativity? For our sense of identity? "
    "I do not have all the answers - nobody does. But I can promise you this: paying attention to what is happening in artificial intelligence right now "
    "is one of the most important things any thoughtful person can do. The decisions we make in the next few years will shape the world for generations to come."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=None, help="Reference audio for voice cloning")
    args = parser.parse_args()

    try:
        import torchaudio
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError as e:
        print(f"✗ Missing dep: {e}")
        print("  pip install chatterbox-tts torch torchaudio")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tests = [
        ("plain", TEST_TEXT, args.ref),
        ("expressive", TEST_TEXT_EXPRESSIVE, args.ref),
    ]

    results = []
    for name, text, ref in tests:
        print(f"[{name}] loading fresh model instance...")
        t_load = time.time()
        try:
            m = ChatterboxTurboTTS.from_pretrained(device="cpu")
        except Exception as e:
            print(f"  ✗ FAIL loading model: {e}")
            results.append((name, None, None))
            continue
        print(f"  ✓ model loaded in {time.time()-t_load:.1f}s")

        print(f"[{name}] generating...")
        t0 = time.time()
        try:
            kwargs = {}
            if ref and Path(ref).exists():
                kwargs["audio_prompt_path"] = ref
            wav = m.generate(text, **kwargs)
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            import traceback; traceback.print_exc()
            results.append((name, None, None))
            continue

        elapsed = time.time() - t0
        sr = 24000
        duration = wav.shape[-1] / sr
        rtf = elapsed / duration if duration > 0 else 0

        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = OUT_DIR / f"{ts}_{name}.wav"
        torchaudio.save(str(out_path), wav.squeeze(0).unsqueeze(0).cpu(), sr)
        print(
            f"  ✓ {out_path.name} | duration={duration:.1f}s | elapsed={elapsed:.1f}s | RTF={rtf:.2f}x"
        )
        results.append((name, duration, elapsed))

    print(f"\n{'=' * 50}")
    success = [r for r in results if r[1] is not None]
    print(f"✅ {len(success)}/{len(tests)} tests passed")
    if success:
        avg_rtf = sum(r[2] / r[1] for r in success) / len(success)
        print(f"   avg RTF: {avg_rtf:.2f}x")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
