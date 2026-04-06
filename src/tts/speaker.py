"""
tts/speaker.py
Gemini TTS API → phát audio trực tiếp qua speaker (pygame)

Model: gemini-2.5-flash-preview-tts
Output: PCM 24kHz signed 16-bit mono → phát qua pygame
"""
import io
import wave
import threading
import pygame
from google.genai import types as genai_types

from config import GEMINI_API_KEY, GEMINI_TTS_MODEL, GEMINI_TTS_VOICE

# Init pygame mixer: 24kHz, 16-bit, mono (khớp với Gemini TTS output)
pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)

# Gemini client
_client = None

def _get_client():
    global _client
    if _client is None:
        import google.genai as _genai
        _client = _genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _pcm_to_wav_bytes(pcm_data: bytes, rate: int = 24000) -> bytes:
    """Wrap raw PCM bytes thành WAV in-memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit = 2 bytes
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    buf.seek(0)
    return buf.read()


def text_to_speech(
    text: str,
    voice: str = GEMINI_TTS_VOICE,
    blocking: bool = True,
    style_prompt: str = "",
) -> None:
    """
    Chuyển text → Gemini TTS → phát qua speaker.

    Args:
        text:         Văn bản cần đọc
        voice:        Tên voice Gemini (mặc định từ config)
        blocking:     True = chờ xong mới return | False = phát nền
        style_prompt: Hướng dẫn phong cách, vd "Đọc nhẹ nhàng, chậm rãi"
                      Để trống = dùng text thẳng.
    """
    if not text or not text.strip():
        return

    def _play():
        try:
            client = _get_client()

            # Ghép style prompt nếu có
            content = f"{style_prompt}\n\n{text}" if style_prompt else text

            response = client.models.generate_content(
                model=GEMINI_TTS_MODEL,
                contents=content,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )

            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            wav_bytes = _pcm_to_wav_bytes(pcm_data)

            # Phát qua pygame
            sound = pygame.mixer.Sound(io.BytesIO(wav_bytes))
            channel = sound.play()

            if blocking:
                # Chờ đến khi phát xong
                while channel.get_busy():
                    pygame.time.Clock().tick(20)

        except Exception as e:
            print(f"[TTS Error] {e}")

    if blocking:
        _play()
    else:
        t = threading.Thread(target=_play, daemon=True)
        t.start()


def stop_speech():
    """Dừng tất cả audio đang phát."""
    pygame.mixer.stop()