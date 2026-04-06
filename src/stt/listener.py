"""
stt/listener.py
Speech-to-Text dùng vinai/PhoWhisper-small.
  - Chạy GPU (CUDA) nếu có, fallback CPU
  - Tự phát hiện im lặng để dừng ghi âm
"""
import io
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from transformers import pipeline as hf_pipeline

from config import (
    STT_MODEL, STT_LANGUAGE,
    STT_SILENCE_THRESHOLD, STT_SILENCE_DURATION, STT_SAMPLE_RATE,
)

# ── Singleton model ────────────────────────────
_pipe = None
_pipe_lock = threading.Lock()


def get_model():
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype  = torch.float16 if device == "cuda" else torch.float32
                print(f"[STT] Loading {STT_MODEL} on {device.upper()}...")
                _pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=STT_MODEL,
                    device=device,
                    dtype=dtype,  # torch_dtype deprecated
                )
                print("[STT] Model ready.")
    return _pipe


# ── Ghi âm có VAD ─────────────────────────────
def record_audio(
    sample_rate: int         = STT_SAMPLE_RATE,
    silence_threshold: float = STT_SILENCE_THRESHOLD,
    silence_duration: float  = STT_SILENCE_DURATION,
    max_duration: float      = 30.0,
):
    print("[STT] 🎤 Đang nghe... (nói xong -> tự dừng)")
    frames = []
    silent_start   = None
    speech_started = False

    def callback(indata, frame_count, time_info, status):
        nonlocal silent_start, speech_started
        frames.append(indata.copy())
        volume = float(np.abs(indata).mean())
        if volume > silence_threshold:
            speech_started = True
            silent_start   = None
        elif speech_started:
            if silent_start is None:
                silent_start = time.time()

    with sd.InputStream(
        samplerate=sample_rate, channels=1,
        dtype="float32", blocksize=1024, callback=callback,
    ):
        start = time.time()
        while True:
            time.sleep(0.05)
            if speech_started and silent_start is not None:
                if time.time() - silent_start >= silence_duration:
                    break
            if time.time() - start >= max_duration:
                break

    if not frames or not speech_started:
        return None
    return np.concatenate(frames, axis=0).flatten()


# ── Transcribe ────────────────────────────────
def transcribe(audio, language: str = STT_LANGUAGE) -> str:
    pipe   = get_model()
    result = pipe({"array": audio, "sampling_rate": STT_SAMPLE_RATE})
    return result["text"].strip()


# ── One-shot helper ───────────────────────────
def listen_once(language: str = STT_LANGUAGE) -> str:
    audio = record_audio()
    if audio is None:
        return ""
    text = transcribe(audio, language=language)
    if text:
        print(f"[STT] ✅ {text}")
    return text