import sys, os, time
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, "Qwen3-TTS"))

from qwen_tts import Qwen3TTSModel
import soundfile as sf

DEVICE = "cpu"
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT = "outputs/sing_italian.wav"

def sing_italian():
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE)

    text = (
        "Tra feedback e crepe, alziamo i ferri della notte. "
        "Chitarre come coltelli, distorsione che scava la verità. "
        "Siamo scorie e scintille, operai del rumore che non chiedono perdono. "
        "Versi taglienti, cori ossessivi — spingi il volume, lascia tremare il cielo."
    )

    instruct = (
        "Sing in Italian: female, singing gritty and energetic; punchy rhythm, sing song, "
        "and strong, explosive choruses."
    )

    melody_hint = (
        "Melody hint (syllable->note): 'Tra(fa)edback(so) e(cre)-pe' -> G4 A4 G4 E4 | 'al(zi)amo i fer(ri)' -> C5 C5 B4 A4"
    )

    t0 = time.time()
    try:
        wavs, sr = tts.generate_voice_design(
            text=text,
            language="Italian",
            instruct=instruct + " " + melody_hint,
            temperature=0.8,
            singing=True,
        )
        print("Used singing=True mode")
    except TypeError:
        wavs, sr = tts.generate_voice_design(
            text=text,
            language="Italian",
            instruct=instruct + " " + melody_hint,
            temperature=0.8,
        )
        print("singing=True not supported, used fallback")
    except Exception as e:
        print(f"singing mode failed - {e}, trying fallback...")
        wavs, sr = tts.generate_voice_design(
            text=text,
            language="Italian",
            instruct=instruct + " " + melody_hint,
            temperature=0.8,
        )

    os.makedirs("outputs", exist_ok=True)
    sf.write(OUTPUT, wavs[0], sr)
    elapsed = time.time() - t0
    duration = len(wavs[0]) / sr
    print(f"done in {elapsed:.1f}s → {OUTPUT} ({duration:.1f}s audio, RTF {elapsed/duration:.2f}x)")


if __name__ == "__main__":
    sing_italian()
