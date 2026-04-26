import os
import sys
import time
import abc
import asyncio
import json
from typing import Optional, Dict, Any, List

# Common imports for audio writing
import soundfile as sf
import torch

class BaseTTSEngine(abc.ABC):
    """Abstract base class for all TTS engines."""
    
    @abc.abstractmethod
    async def generate(self, text: str, voice: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        Generate audio from text.
        Returns a dict with performance metrics and output path.
        """
        pass

    @abc.abstractmethod
    def list_voices(self) -> List[str]:
        """Return a list of available voices/presets."""
        pass

class QwenTTSEngine(BaseTTSEngine):
    def __init__(self, model_size="0.6B", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.model_size = model_size
        self.model = None
        self.models_config = {
            "0.6B": "Qwen3-TTS/models/0.6B-CustomVoice",
            "1.7B": "Qwen3-TTS/models/1.7B-CustomVoice",
        }
        
    def _ensure_model(self):
        if self.model:
            return
        
        sys.path.insert(0, os.path.join(os.getcwd(), "Qwen3-TTS"))
        from qwen_tts import Qwen3TTSModel
        
        path = self.models_config.get(self.model_size)
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"Qwen model not found at {path}")
            
        self.model = Qwen3TTSModel.from_pretrained(
            path,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2" if self.device == "cuda" else "sdpa",
        )

    async def generate(self, text: str, voice: str, output_path: str, **kwargs) -> Dict[str, Any]:
        self._ensure_model()
        
        language = kwargs.get("language", "English").lower().capitalize() # Normalizing case
        voice = voice.lower() # Qwen 0.6B config uses lowercase keys (vivian, ryan, etc.)
        instruct = kwargs.get("instruct", None)
        
        t0 = time.time()
        # Recommended generation parameters from official examples
        common_gen_kwargs = {
            "max_new_tokens": 2048,
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
        }
        
        # Qwen's generate_custom_voice is synchronous, but we can wrap it
        try:
            if instruct and self.model_size == "1.7B":
                wavs, sr = self.model.generate_custom_voice(
                    text, 
                    speaker=voice, 
                    language=language, 
                    instruct=instruct,
                    **common_gen_kwargs
                )
            else:
                wavs, sr = self.model.generate_custom_voice(
                    text, 
                    speaker=voice, 
                    language=language,
                    **common_gen_kwargs
                )
        except Exception as e:
             # Fallback to English if language is not supported or causing issues
             print(f"Generation failed with language {language}, retrying with English: {e}")
             wavs, sr = self.model.generate_custom_voice(
                text, 
                speaker=voice, 
                language="English",
                **common_gen_kwargs
             )
        
        elapsed = time.time() - t0
        duration = len(wavs[0]) / sr
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, wavs[0], sr)
        
        return {
            "engine": "qwen3-tts",
            "voice": voice,
            "elapsed": elapsed,
            "duration": duration,
            "rtf": elapsed / duration,
            "path": output_path
        }

    def list_voices(self) -> List[str]:
        return ["Ryan", "Aiden", "Emma", "Bella"] # Common presets

class VibeVoiceEngine(BaseTTSEngine):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.voices = {}
        
    def _ensure_model(self):
        if self.model:
            return
            
        v_root = os.path.join(os.getcwd(), "VibeVoice")
        if not os.path.isdir(v_root):
            raise FileNotFoundError(f"VibeVoice not found at {v_root}")
            
        sys.path.insert(0, v_root)
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        
        model_id = "microsoft/VibeVoice-Realtime-0.5B"
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id, 
            torch_dtype=torch.float32 if self.device in ("cpu", "mps") else torch.bfloat16,
            device_map=self.device
        )
        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)
        
        voices_dir = os.path.join(v_root, "demo", "voices", "streaming_model")
        if os.path.isdir(voices_dir):
            self.voices = {os.path.splitext(f)[0]: os.path.join(voices_dir, f) for f in os.listdir(voices_dir) if f.endswith(".pt")}

    async def generate(self, text: str, voice: str, output_path: str, **kwargs) -> Dict[str, Any]:
        self._ensure_model()
        voice_path = self.voices.get(voice)
        if not voice_path:
            raise ValueError(f"Voice {voice} not found in VibeVoice presets")
            
        prefilled = torch.load(voice_path, map_location=self.device, weights_only=False)
        cfg_scale = kwargs.get("cfg_scale", 1.5)
        
        t0 = time.time()
        inputs = self.processor.process_input_with_cached_prompt(
            text=text, cached_prompt=prefilled, padding=True, return_tensors="pt"
        )
        for k, v in inputs.items():
            if torch.is_tensor(v): inputs[k] = v.to(self.device)
            
        outputs = self.model.generate(
            **inputs, cfg_scale=cfg_scale, tokenizer=self.processor.tokenizer, 
            all_prefilled_outputs=prefilled
        )
        audio = outputs.speech_outputs[0]
        elapsed = time.time() - t0
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.processor.save_audio(audio, output_path=output_path)
        
        duration = len(audio) / 24000 # Typical VibeVoice SR
        return {
            "engine": "vibevoice",
            "voice": voice,
            "elapsed": elapsed,
            "duration": duration,
            "rtf": elapsed / duration,
            "path": output_path
        }

    def list_voices(self) -> List[str]:
        self._ensure_model()
        return list(self.voices.keys())

class OpenRouterTTSEngine(BaseTTSEngine):
    """
    Placeholder for OpenRouter/OpenAI-compatible cloud TTS.
    Assumes an API that takes a model and returns audio bytes.
    """
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url or "https://openrouter.ai/api/v1/audio/speech"
        self.default_model = model or "openai/tts-1"

    async def generate(self, text: str, voice: str, output_path: str, **kwargs) -> Dict[str, Any]:
        import httpx
        model = kwargs.get("model", self.default_model)
        
        t0 = time.time()
        # Note: OpenRouter doesn't have a standard TTS endpoint yet, 
        # but if we use an OpenAI-compatible provider:
        url = self.base_url
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "speed": kwargs.get("speed", 1.0)
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(resp.content)
        
        elapsed = time.time() - t0
        return {
            "engine": "openrouter",
            "voice": voice,
            "model": model,
            "elapsed": elapsed,
            "path": output_path
        }

    def list_voices(self) -> List[str]:
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

class TTSFactory:
    @staticmethod
    def get_engine(engine_name: str, **kwargs) -> BaseTTSEngine:
        if engine_name == "qwen":
            return QwenTTSEngine(model_size=kwargs.get("model_size", "0.6B"), device=kwargs.get("device"))
        elif engine_name == "vibevoice":
            return VibeVoiceEngine(device=kwargs.get("device"))
        elif engine_name in ("openrouter", "openai"):
            return OpenRouterTTSEngine(
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url"),
                model=kwargs.get("model"),
            )
        else:
            raise ValueError(f"Unknown engine: {engine_name}. Available: qwen, vibevoice, openrouter")

    @staticmethod
    def from_config(config: Dict[str, Any]) -> BaseTTSEngine:
        """Build engine from config dict or env vars.

        Config keys:
          engine     : "qwen" | "vibevoice" | "openrouter"   (or TTS_ENGINE env var)
          model_size : "0.6B" | "1.7B"                       (qwen only)
          model      : openai/tts-1 | openai/tts-1-hd        (openrouter only)
          base_url   : override API base URL                  (openrouter only)
          api_key    : OpenRouter/OpenAI key                  (or OPENROUTER_API_KEY env)
          device     : "cpu" | "cuda" | "mps"                (local engines only)

        Example — local qwen:
            TTSFactory.from_config({"engine": "qwen", "model_size": "0.6B"})

        Example — OpenRouter cloud:
            TTSFactory.from_config({"engine": "openrouter", "model": "openai/tts-1"})

        Example — from env only (TTS_ENGINE=openrouter OPENROUTER_API_KEY=...):
            TTSFactory.from_config({})
        """
        engine = config.get("engine") or os.environ.get("TTS_ENGINE", "qwen")
        return TTSFactory.get_engine(
            engine,
            model_size=config.get("model_size", os.environ.get("TTS_MODEL_SIZE", "0.6B")),
            model=config.get("model", os.environ.get("TTS_MODEL")),
            base_url=config.get("base_url", os.environ.get("TTS_BASE_URL")),
            api_key=config.get("api_key", os.environ.get("OPENROUTER_API_KEY")),
            device=config.get("device", os.environ.get("TTS_DEVICE")),
        )
