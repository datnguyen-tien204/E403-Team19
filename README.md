# AI Agent – LangGraph + Gemini + Weaviate (Windows)

## Stack
| Layer | Tech |
|-------|------|
| LLM | Google Gemini 2.0 Flash |
| Agent | LangGraph (ReAct pattern) |
| RAG | Weaviate v4 + Gemini Embeddings |
| TTS | gTTS → pygame (phát trực tiếp speaker) |
| System control | pycaw (volume) + subprocess (open apps) |

---

## Setup

### 1. Cài Python packages
```bash
pip install -r requirements.txt
pip install psutil   # (optional) để xem battery/CPU/RAM
```

### 2. Tạo file .env
```bash
copy .env.example .env
# Điền GEMINI_API_KEY vào .env
```

### 3. Chạy Weaviate local (Docker)
```bash
docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
```
Hoặc dùng [Weaviate Cloud](https://console.weaviate.cloud) (free tier) – điền URL + API key vào .env

### 4. Nạp tài liệu vào RAG (tùy chọn)
```bash
# Từ file .txt
python ingest.py --files docs/faq.txt

# Từ thư mục
python ingest.py --dir ./docs

# Test nhanh với text thẳng
python ingest.py --text "Công ty ABC thành lập năm 2010, trụ sở tại Hà Nội."
```

### 5. Chạy chatbot
```bash
python main.py
```

---

## Cách dùng

```
Bạn: tăng âm lượng lên 20%
Agent: Đã tăng âm lượng từ 40% lên 60%

Bạn: mở notepad
Agent: Đã mở 'notepad'

Bạn: pin còn bao nhiêu?
Agent: battery: 78% (đang sạc)

Bạn: tìm thông tin về chính sách đổi trả
Agent: [tìm trong Weaviate RAG và trả lời]

Bạn: exit
```

---

## Thêm app vào danh sách mở
Sửa dict `APP_MAP` trong `tools/all_tools.py`:
```python
APP_MAP = {
    "chrome": "chrome",
    "my_app": r"C:\path\to\myapp.exe",   # thêm ở đây
    ...
}
```

## Thêm tài liệu vào RAG
Thêm file `.txt` bất cứ lúc nào rồi chạy lại `ingest.py`.

## Đổi ngôn ngữ TTS
Sửa `TTS_LANG` trong `config.py`:
- `"vi"` → tiếng Việt
- `"en"` → tiếng Anh
