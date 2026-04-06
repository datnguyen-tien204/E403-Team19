import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL  = "models/text-embedding-004"   # chỉ dùng cho embedding nếu cần

# Groq LLM
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL          = "qwen/qwen3-32b"              # hoặc: llama-3.3-70b-versatile, mixtral-8x7b-32768

WEAVIATE_URL        = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY    = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_INDEX_NAME = os.getenv("WEAVIATE_INDEX_NAME", "KnowledgeBase")

# Web search
# ⚠️  Đặt SEARXNG_URL trong .env nếu SearXNG không chạy ở localhost:8888
SEARXNG_URL         = os.getenv("SEARXNG_URL", "http://localhost:8888")
SERPER_API_KEY      = os.getenv("SERPER_API_KEY", "")
WEB_SEARCH_TOP_K    = int(os.getenv("WEB_SEARCH_TOP_K", "5"))
SEARCH_MODE_DEFAULT = os.getenv("SEARCH_MODE_DEFAULT", "hybrid").lower()  # hybrid | on | off

TTS_LANG = "vi"

# Gemini TTS
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_VOICE = "Aoede"  # Aoede=Breezy | Sulafat=Warm | Leda=Youthful | Puck=Upbeat
# xem đủ 30 voices trong doc. Tiếng Việt tự detect, không cần set lang.

# STT - faster-whisper (chạy CPU)
STT_MODEL = "vinai/PhoWhisper-small"  # fine-tuned tiếng Việt, chạy GPU
STT_LANGUAGE = "vi"  # None = auto-detect
STT_SAMPLE_RATE = 16000
STT_SILENCE_THRESHOLD = 0.01  # ngưỡng phát hiện im lặng (0.0 – 1.0)
STT_SILENCE_DURATION = 1.5  # giây im lặng để dừng ghi âm

# FastAPI
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))