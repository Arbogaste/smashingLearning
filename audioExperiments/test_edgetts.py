"""
EdgeTTS test suite for vid-production narration use case.

No model download needed. Requires internet connection.
pip install edge-tts

Run from audioExperiments/:
  python test_edgetts.py [--voice en-US-GuyNeural] [--list-voices]
"""
import argparse
import asyncio
import os
import time

import edge_tts

OUT = "outputs/edgetts"

# Same corpus as the other test files for direct comparison
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

# Candidate voices for vid-production channels
# TheRoarWire / PeakyStockRadar: authoritative male
# MomentOfMindfulness: calm, warm voice
# FunLabChannel: energetic
EN_VOICES = [
    "en-US-GuyNeural",       # male, neutral — good for finance
    "en-US-ChristopherNeural",  # male, mature
    "en-US-EricNeural",      # male, conversational
    "en-US-JennyNeural",     # female, news style
    "en-US-AriaNeural",      # female, warm
]
IT_VOICES = [
    "it-IT-DiegoNeural",     # male
    "it-IT-ElsaNeural",      # female
    "it-IT-IsabellaNeural",  # female, warm
]


async def speak(text, voice, path, rate="+0%"):
    os.makedirs(OUT, exist_ok=True)
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(path)


async def run_timed(text, voice, name, rate="+0%"):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.mp3")
    t0 = time.time()
    await speak(text, voice, path, rate)
    elapsed = time.time() - t0
    size = os.path.getsize(path) / 1024
    print(f"  [{name}] elapsed={elapsed:.1f}s  size={size:.0f}KB → {path}")
    return path


async def list_voices_cmd(filter_lang=None):
    voices = await edge_tts.list_voices()
    for v in voices:
        if filter_lang and not v["Locale"].startswith(filter_lang):
            continue
        print(f"  {v['ShortName']:40s}  {v['Gender']:7s}  {v['Locale']}")


async def main_async(args):
    if args.list_voices:
        print("English voices:")
        await list_voices_cmd("en-US")
        print("\nItalian voices:")
        await list_voices_cmd("it-IT")
        return

    voices_en = [args.voice] if args.voice else EN_VOICES
    voices_it = IT_VOICES

    # T1: English narration — all candidate EN voices, short text
    print("\n[T1] English narration (short) — all candidate voices")
    for voice in voices_en:
        name = f"edge_{voice.replace('-', '_')}_short"
        await run_timed(NARRATION_EN_SHORT, voice, name)

    # T2: Long narration — default voice
    print("\n[T2] Long narration (~100 words)")
    voice = voices_en[0]
    await run_timed(NARRATION_EN_LONG, voice, f"edge_{voice.replace('-', '_')}_long")

    # T3: Rate control — slower for authority, faster for energetic
    print("\n[T3] Rate control on short narration")
    for rate, label in [("-15%", "slow"), ("+0%", "normal"), ("+15%", "fast")]:
        voice = voices_en[0]
        await run_timed(NARRATION_EN_SHORT, voice, f"edge_{voice.replace('-', '_')}_{label}", rate=rate)

    # T4: Mindfulness — calm female voice
    print("\n[T4] Mindfulness narration")
    await run_timed(NARRATION_MINDFULNESS, "en-US-AriaNeural",
                    "edge_en-US-AriaNeural_mindfulness", rate="-20%")
    await run_timed(NARRATION_MINDFULNESS, "en-US-JennyNeural",
                    "edge_en-US-JennyNeural_mindfulness", rate="-10%")

    # T5: Italian
    print("\n[T5] Italian narration")
    for voice in voices_it:
        await run_timed(NARRATION_IT, voice, f"edge_{voice.replace('-', '_')}_italian")

    # T6: Stress test — 10 consecutive segments (measures reliability and latency)
    print("\n[T6] Stress: 10 consecutive segments, same voice")
    voice = voices_en[0]
    times = []
    for i in range(10):
        t0 = time.time()
        path = os.path.join(OUT, f"edge_stress_{i:02d}.mp3")
        await speak(NARRATION_EN_SHORT, voice, path)
        elapsed = time.time() - t0
        times.append(elapsed)
    avg = sum(times) / len(times)
    print(f"  avg per segment: {avg:.2f}s  min: {min(times):.2f}s  max: {max(times):.2f}s")
    print(f"  Note: EdgeTTS is cloud-bound — latency varies with network conditions")

    print(f"\nDone. Listen to {OUT}/ and fill in the decision matrix in README.md")
    print("\nBest voices summary:")
    print("  Finance/news:    en-US-ChristopherNeural, en-US-GuyNeural")
    print("  Mindfulness:     en-US-AriaNeural (-20% rate)")
    print("  Italian:         it-IT-DiegoNeural, it-IT-IsabellaNeural")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--voice", help="Test a single voice (e.g. en-US-GuyNeural)")
    p.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
