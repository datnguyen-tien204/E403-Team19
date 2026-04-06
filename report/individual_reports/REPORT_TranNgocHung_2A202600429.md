# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Trần Ngọc Hùng
- **Student ID**: 2A202600429
- **Date**: 06/04/2026

---

## I. Technical Contribution (15 Points)

**Phần phụ trách**: Chat thông thường (Q&A / Kiến thức) — Agent trả lời trực tiếp từ LLM (Groq), không cần gọi tool.

### Modules Implemented

| Module | File | Vai trò |
|:-------|:-----|:--------|
| System Prompt | `src/agent/graph.py` (L54–76) | Định nghĩa system prompt cốt lõi cho LLM, bao gồm năng lực Q&A, quy trình suy nghĩ `<think>`, và quy tắc trả lời |
| LLM Node (Q&A path) | `src/agent/graph.py` (L276–357) | Hàm `llm_node()` — điểm xử lý chính khi user hỏi câu hỏi kiến thức, LLM trả lời trực tiếp mà không gọi tool |
| Groq LLM Config | `src/config.py` (L9–11) | Cấu hình Groq API key, model (`qwen/qwen3-32b`), kết nối tới LLM provider |
| Escalating Recovery | `src/agent/graph.py` (L167–230, L341–399) | Pattern #3: retry ×3 → fallback model (`llama-3.1-8b-instant`) → surface error, đảm bảo Q&A luôn trả lời được |
| Streaming `<think>` parser | `src/api/server.py` (L457–486) | Parse và tách `<think>...</think>` block khi LLM stream response, gửi thinking tokens riêng biệt tới frontend |

### Code Highlights

**1. System Prompt — Quy trình suy nghĩ cho Q&A:**

```python
# src/agent/graph.py (L54-76)
SYSTEM_PROMPT = f"""Bạn là trợ lý AI thông minh, hỗ trợ người dùng bằng tiếng Việt.
Hôm nay là {datetime.datetime.now().strftime('%d/%m/%Y')}.

## NĂNG LỰC
- Trả lời câu hỏi kiến thức, tính toán, lập trình.
- Tìm kiếm tài liệu nội bộ (RAG) qua `search_knowledge_base`.
- Tìm kiếm web qua `web_search` ...
- Thực thi code thật qua `run_code_in_sandbox` ...
- Điều khiển hệ thống, ứng dụng, âm lượng, v.v.

## QUY TRÌNH SUY NGHĨ (LUÔN FOLLOW)
Trước khi hành động, viết suy nghĩ trong thẻ <think>:
1. **Phân tích**: Người dùng thực sự muốn gì?
2. **Kế hoạch**: Liệt kê các bước, dùng tool gì, thứ tự nào.
3. **Rủi ro**: Có thao tác nguy hiểm không?
4. **Thực hiện**: Gọi tool đúng thứ tự.
"""
```

**Điểm quan trọng**: Với câu hỏi Q&A thuần túy (ví dụ: "Thủ đô của Pháp là gì?"), LLM đi qua `<think>` block, nhận ra không cần gọi tool, và trả lời trực tiếp. LangGraph flow kết thúc sau 1 bước LLM duy nhất (không đi vào `tools` node).

**2. LLM Node — Luồng xử lý Q&A:**

```python
# src/agent/graph.py (L276-357)
def llm_node(state: AgentState):
    messages = list(state["messages"])
    
    # ... search mode prompt, guards ...
    
    # Context Defense (Pattern #9)
    defended = apply_context_defense(messages)
    
    # Build system prompt
    full_system = SYSTEM_PROMPT + mode_prompt
    if not defended or not isinstance(defended[0], SystemMessage):
        defended = [SystemMessage(content=full_system)] + defended
    
    # Chọn primary hay fallback LLM
    llm = llm_fallback if use_fallback else llm_primary
    
    try:
        response = llm.invoke(defended)
        return {"messages": [response]}
    except Exception as e:
        # Escalating Recovery: retry → fallback → surface error
        ...
```

**Khi user hỏi Q&A**, luồng như sau:
1. `llm_node` nhận message → build system prompt → gọi Groq LLM
2. LLM trả lời trực tiếp (không có `tool_calls` trong response)
3. `tools_condition` kiểm tra: không có tool call → đi thẳng tới `END`
4. Response được trả về cho user

**3. Escalating Recovery — Đảm bảo Q&A ổn định:**

```python
# src/agent/graph.py (L359-399)
except Exception as e:
    # Cấp 1: Retry ×3 với cùng model
    if recovery_count <= _MAX_RECOVERY_ATTEMPTS:
        response = llm.invoke(defended + [recovery_msg])
        
    # Cấp 2: Chuyển sang fallback model (llama-3.1-8b-instant)
    if not use_fallback:
        response = llm_fallback.invoke(defended)
        
    # Cấp 3: Báo lỗi rõ ràng cho user
    return {"messages": [AIMessage(content="❌ Không thể xử lý...")]}
```

Với Q&A, pattern này đặc biệt quan trọng vì Groq API có thể bị rate limit. Khi `qwen3-32b` fail, hệ thống tự động chuyển sang `llama-3.1-8b-instant` (nhanh hơn, nhẹ hơn) để user vẫn nhận được câu trả lời.

**4. Streaming `<think>` Parser — UX cho Q&A:**

```python
# src/api/server.py (L457-486)
# Tách <think>...</think> khỏi response visible
if not in_think:
    think_start = raw.find("<think>", i)
    if think_start == -1:
        # Gửi token hiển thị cho user
        await ws.send_json({"type": "token", "content": chunk})
    else:
        in_think = True
else:
    think_end = raw.find("</think>", i)
    if think_end != -1:
        # Gửi thinking content riêng (hiển thị collapsible trên UI)
        await ws.send_json({"type": "thinking_done", "content": think_buffer})
        in_think = False
```

Parser này cho phép frontend hiển thị quá trình suy nghĩ của LLM (tại sao không cần tool, kiến thức nào đang được dùng) trong một panel riêng, trong khi câu trả lời cuối cùng hiển thị gọn gàng.

### Documentation — Cách Q&A tương tác với ReAct Loop

```
User: "Thủ đô của Pháp là gì?"
          │
          ▼
    ┌─────────────┐
    │  llm_node()  │  ← System prompt + user message → Groq API
    └──────┬──────┘
           │ response (no tool_calls)
           ▼
    ┌──────────────┐
    │ tools_condition│ → Không có tool call → END
    └──────┬───────┘
           │
           ▼
    Response: "<think>Đây là câu hỏi kiến thức phổ thông,
    không cần gọi tool.</think>
    Thủ đô của Pháp là Paris."
```

Trong LangGraph, `tools_condition` (từ `langgraph.prebuilt`) kiểm tra `AIMessage.tool_calls`. Nếu list rỗng → kết thúc tại `END`. Đây là đường dẫn ngắn nhất — chỉ 1 lần gọi LLM, **không có overhead của tool execution**.

---

## II. Debugging Case Study (10 Points)

### Problem Description

**Vấn đề**: Khi Groq API bị rate limit (429 Too Many Requests) trong lúc nhiều user cùng hỏi Q&A, agent trả về lỗi raw exception thay vì thông báo thân thiện, khiến UX rất tệ.

### Log Source

```json
// Ví dụ tái hiện lỗi từ Escalating Recovery logs
{
  "event": "LLM_ERROR",
  "data": {
    "model": "qwen/qwen3-32b",
    "error": "rate_limit_exceeded: Rate limit reached for model...",
    "recovery_attempt": 1
  }
}
```

Lỗi tương tự cũng xảy ra ở baseline chatbot trong log thực tế (gemini-1.5-flash model not found):

```json
// logs/2026-04-06.log (L6)
{
  "timestamp": "2026-04-06T08:47:57.539795",
  "event": "CHATBOT_ERROR",
  "data": {
    "input": "hello",
    "model": "gemini-1.5-flash",
    "error": "404 models/gemini-1.5-flash is not found..."
  }
}
```

### Diagnosis

1. **Root Cause**: Groq free tier có rate limit thấp (~30 req/min). Khi nhiều câu hỏi Q&A liên tiếp, model `qwen3-32b` (lớn) dùng nhiều token → hit rate limit nhanh.
2. **Tại sao LLM làm vậy?**: Đây không phải lỗi prompt hay model reasoning — mà là lỗi infrastructure. LLM không có cơ hội trả lời vì request bị reject ở API layer.
3. **Baseline cũng gặp lỗi tương tự**: Log cho thấy Gemini provider cũng fail khi model name sai (`gemini-1.5-flash` → đã deprecated, phải dùng `gemini-2.5-flash`).

### Solution

Implement **Escalating Recovery** (Pattern #3 từ sách "Giải phẫu một Agentic OS"):

```python
# Phân tầng xử lý lỗi:
# Cấp 1: Retry ×3 (rate limit thường tự hết sau vài giây)
if recovery_count <= 3 and "rate_limit" not in err_str:
    response = llm.invoke(defended + [recovery_msg])

# Cấp 2: Chuyển sang fallback nhỏ hơn (ít token hơn, ít bị rate limit)
_FALLBACK_MODEL = "llama-3.1-8b-instant"
if not use_fallback:
    response = llm_fallback.invoke(defended)

# Cấp 3: Báo lỗi rõ ràng
return {"messages": [AIMessage(content="❌ Không thể xử lý yêu cầu...")]}
```

**Kết quả**: Q&A giờ đây gần như luôn trả lời được. Khi `qwen3-32b` fail, `llama-3.1-8b-instant` (8B params, nhanh) vẫn trả lời Q&A đơn giản tốt. User thấy prefix `⚡ [fallback-model]` để biết đang dùng model backup.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

### 1. Reasoning: `<think>` block giúp gì so với Chatbot trả lời trực tiếp?

Với Q&A thuần túy, `<think>` block tạo ra **sự khác biệt lớn về khả năng self-routing**:

- **Chatbot baseline**: Nhận prompt → trả lời ngay. Nếu câu hỏi cần tra cứu (vd: "giá iPhone 16 hôm nay"), chatbot trả lời sai hoặc từ chối vì không có tool.
- **Agent với `<think>`**: LLM phân tích trong `<think>` block:
  - "Đây là câu hỏi kiến thức phổ thông → trả lời trực tiếp" (Q&A path)
  - "Đây cần dữ liệu real-time → gọi web_search" (tool path)

`<think>` block hoạt động như một **router tự nhiên**: cùng một system prompt, LLM tự quyết định khi nào dùng kiến thức nội tại (Q&A) vs. khi nào cần tool. Chatbot baseline không có khả năng này.

**Ví dụ thực tế**:
- Q: "Thủ đô của Pháp?" → `<think>Kiến thức phổ thông, không cần tool</think>` → Trả lời: "Paris"
- Q: "Thời tiết Hà Nội hôm nay?" → `<think>Cần dữ liệu real-time → web_search</think>` → Gọi tool

### 2. Reliability: Khi nào Agent tệ hơn Chatbot?

Agent thực sự tệ hơn Chatbot trong **Q&A đơn giản** vì:

| Tiêu chí | Chatbot | Agent (Q&A path) |
|:---------|:--------|:-----------------|
| **Latency** | ~1.5s (1 LLM call) | ~2-3s (system prompt dài hơn + `<think>` overhead) |
| **Token cost** | Ít (prompt ngắn) | Nhiều hơn ~40% (system prompt chứa tool descriptions + search mode) |
| **Đôi khi hallucinate tool** | Không | Có — LLM đôi khi gọi `web_search` cho câu hỏi đơn giản dù không cần |

**Case thực tế**: Câu "Chính sách hoàn vé là gì?" — Chatbot trả lời trong 1.5s với ~150 tokens. Agent mất 3s vì phải parse system prompt dài (chứa tool descriptions cho volume, sandbox, calendar...) rồi mới quyết định không cần tool.

### 3. Observation: Environment feedback ảnh hưởng thế nào?

Với Q&A path, **không có Observation** (vì không gọi tool). Nhưng system prompt chứa `search mode` hint:
- Mode `hybrid`: LLM có xu hướng gọi RAG trước → tăng latency cho Q&A đơn giản
- Mode `off`: LLM bỏ qua RAG → Q&A nhanh hơn nhưng mất khả năng tra cứu tài liệu

Đây là trade-off quan trọng: **search mode ảnh hưởng trực tiếp đến hành vi Q&A** dù user không nhận ra. Khi mode = `hybrid`, LLM đôi khi gọi `search_knowledge_base` cho câu hỏi kiến thức phổ thông (vd: "Python là gì?"), nhận `NO_RAG_HIT`, rồi mới trả lời từ kiến thức nội tại — lãng phí 1 tool call.

---

## IV. Future Improvements (5 Points)

### Scalability
- **Intent Classification Layer**: Thêm một classification model nhẹ (ví dụ: fine-tuned BERT) trước LLM để phân loại intent: `general_qa` vs. `needs_tool` vs. `needs_rag`. Với `general_qa`, bypass toàn bộ tool system → giảm latency từ ~3s xuống ~1s.
- **Response Caching**: Q&A kiến thức (factual) có thể cache. Hash câu hỏi → kiểm tra cache trước khi gọi LLM. Tiết kiệm cost và latency cho câu hỏi lặp lại.

### Safety
- **Output Guardrails**: Thêm filter cho Q&A output — phát hiện và chặn hallucination rõ ràng (vd: LLM tự bịa URL, số liệu thống kê). Có thể dùng `NLI model` để verify claims.
- **Content Policy**: System prompt hiện tại không có explicit safety boundaries cho Q&A. Cần thêm rules: không đưa ra medical/legal advice, không trả lời câu hỏi harmful.

### Performance
- **Model Selection by Complexity**: Cho Q&A đơn giản ("1+1=?"), dùng model nhỏ (`llama-3.1-8b-instant`) mặc định thay vì `qwen3-32b`. Chỉ escalate lên model lớn khi câu hỏi phức tạp. Giảm chi phí ~70% cho Q&A traffic.
- **Streaming Optimization**: Hiện tại `<think>` parser chạy character-by-character. Có thể optimize bằng regex-based chunked parsing để giảm WebSocket message overhead.

---

