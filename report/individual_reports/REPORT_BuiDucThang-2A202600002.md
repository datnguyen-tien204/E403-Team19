# REPORT_BUI_DUC_THANG.md

# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Bui Duc Thang
- **Student ID**: 2A202600002   
- **Date**: 4/6/2026

---

## I. Technical Contribution (15 Points)

- **Modules Implementated**:  
  - `src/tools/web_search.py` (web search tool với SearxNG + DuckDuckGo fallback)  
  - `src/agent/graph.py` (ReAct agent bằng LangGraph)  
  - `main.py` (khởi tạo agent + toolset)  

---

- **Code Highlights**:  

```python
@tool
def web_search(query: str, k: int = WEB_SEARCH_TOP_K) -> str:
    """
    Tìm kiếm web và trả về nội dung đầy đủ của các trang liên quan.
    Dùng SearxNG (localhost:8888), tự động fallback DuckDuckGo nếu lỗi.

    Để kết quả tốt nhất:
      - Truyền từ khoá ngắn gọn, súc tích (tiếng Anh hoặc tiếng Việt đều được).
      - Tách nhiều chủ đề thành nhiều lần gọi web_search riêng.
      - Với câu hỏi thực tế/mới nhất, thêm năm hiện tại vào query.

    Args:
        query: Câu truy vấn tìm kiếm
        k:     Số kết quả tối đa (mặc định từ config, thường là 5)

    Returns:
        Chuỗi text chứa nội dung các trang, nguồn (domain), URL.
        Prefix "[SearxNG]" hoặc "[DuckDuckGo]" cho biết nguồn dữ liệu.
    """
    query = (query or "").strip()
    if not query:
        return "❌ Không có truy vấn tìm kiếm."

    source_label = "SearxNG"
    try:
        results = _searxng_search(query, k)
        if not results:
            raise ValueError("SearxNG trả về 0 kết quả")
    except Exception as searxng_err:
        source_label = "DuckDuckGo"
        try:
            results = _ddg_search(query, k)
        except Exception as ddg_err:
            return (
                f"❌ Tìm kiếm thất bại.\n"
                f"  SearxNG: {searxng_err}\n"
                f"  DuckDuckGo: {ddg_err}"
            )

    if not results:
        return f"[{source_label}] Không tìm thấy kết quả nào cho: {query}"

    # Sắp xếp: kết quả có full content lên trước
    results.sort(key=lambda r: r.get("content_len", 0), reverse=True)

    parts = [f"🔍 [{source_label}] Kết quả cho: «{query}»\n{'─'*60}"]
    for i, r in enumerate(results, 1):
        parts.append(_format_result(i, r))

    return "\n\n".join(parts)
#web_search.py trong tool folder
```


---

- **Documentation**:  

Hệ thống sử dụng **ReAct Agent (LangGraph)** với tool `web_search`.

### 1. Web Search Tool

- Sử dụng SearxNG (localhost:8888)  
- Fallback sang DuckDuckGo nếu lỗi  
- Fetch nội dung URL song song (ThreadPoolExecutor)  
- Lọc:
  - bỏ nội dung < 200 ký tự  
  - loại trùng URL  
- Trả về: title, URL, domain, content, snippet  

---

### 2. ReAct Flow

**Thought → Action (web_search) → Observation → Final Answer**

Agent:
- phân tích query  
- gọi tool khi cần  
- tổng hợp kết quả  

---

### 3. Advanced Mechanisms

- Loop guard (`_MAX_SAME_TOOL_CALLS`)  
- Retry + fallback  
- Context truncation  

---

## II. Debugging Case Study (10 Points)

- **Problem Description**:  
Agent gọi web_search lặp lại nhiều lần với cùng query.

- **Log Source**:  

```
[2026-04-06 16:10:21] [STEP 1]
  Thought     : I need more information about the query to answer accurately.
  Action      : web_search
  Action Input: query="Tập Cận Bình", engine="SearxNG"
  Observation : Retrieved general profile information from Wikipedia and major news outlets.

[2026-04-06 16:10:24] [STEP 2]
  Thought     : The information might not be sufficient; I should search again to confirm.
  Action      : web_search
  Action Input: query="Tập Cận Bình", engine="SearxNG"
  Observation : The same sources were returned with no additional details.

[2026-04-06 16:10:27] [STEP 3]
  Thought     : I still think more context is needed, but I do not have a clear stopping rule.
  Action      : web_search
  Action Input: query="Tập Cận Bình", engine="SearxNG"
  Observation : Repeated results detected; no new information found.
```

- **Diagnosis**:  
- Thiếu điều kiện dừng  
- Model không biết khi nào đủ thông tin  

- **Solution**:  
- Thêm loop guard  
- Giới hạn số bước  
- Improve prompt  

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

*Reflect on the reasoning capability difference.*

1. **Reasoning**:  

ReAct agent sử dụng `Thought` như một bước suy luận trung gian trước khi hành động. Điều này giúp quá trình reasoning trở nên **explicit** thay vì implicit như chatbot truyền thống.

Cụ thể:
- Chatbot:  
  - Nhận input → sinh output trực tiếp (one-shot)  
  - Không thể hiện quá trình suy nghĩ  
  - Dễ bị hallucination khi thiếu thông tin  

- ReAct Agent:  
  - Phân tích câu hỏi (`Thought`)  
  - Quyết định hành động (`Action`)  
  - Thu thập thông tin (`Observation`)  
  - Tổng hợp kết quả  

→ Điều này tạo thành một pipeline:
**Reason → Act → Observe → Refine**

Trong bài toán web search:
- Agent không “đoán” câu trả lời  
- Mà **chủ động truy xuất thông tin từ web rồi mới trả lời**  

→ Kết quả:
- Giảm hallucination  
- Tăng độ chính xác factual  
- Có khả năng xử lý câu hỏi open-domain tốt hơn  

---

2. **Reliability**:  

Mặc dù mạnh hơn về reasoning, ReAct agent không phải lúc nào cũng đáng tin cậy hơn chatbot.

Các trường hợp agent hoạt động kém:

- **Query đơn giản** (ví dụ: “2 + 2 = ?”):  
  → Agent vẫn có thể gọi `web_search` không cần thiết  
  → Tăng latency và chi phí  

- **Dữ liệu web nhiễu**:  
  → Kết quả search có thể không liên quan  
  → Agent tổng hợp sai hoặc thiếu chính xác  

- **Loop behavior**:  
  → Agent có thể gọi tool nhiều lần nếu không có termination condition  
  → Gây lãng phí tài nguyên  

- **Tool dependency**:  
  → Nếu web_search fail hoặc trả về dữ liệu kém  
  → Toàn bộ pipeline bị ảnh hưởng  

→ So sánh:

| Tiêu chí | Chatbot | ReAct Agent |
|---------|--------|------------|
| Tốc độ | Nhanh | Chậm hơn |
| Độ ổn định | Cao (simple task) | Thấp hơn nếu tool lỗi |
| Độ chính xác factual | Thấp hơn | Cao hơn (nếu tool tốt) |

→ Kết luận:  
ReAct mạnh hơn trong **complex / real-world queries**, nhưng kém hiệu quả với **simple tasks**.

---

3. **Observation**:  

Observation là yếu tố quan trọng nhất giúp ReAct agent vượt trội hơn chatbot.

Trong hệ thống này:
- Observation chính là kết quả từ `web_search`
- Bao gồm:
  - nội dung web  
  - snippet  
  - thông tin thực tế  

Vai trò của Observation:

- **Grounding**:  
  → Giúp agent dựa vào dữ liệu thật thay vì suy đoán  

- **Feedback loop**:  
  → Agent sử dụng Observation để quyết định bước tiếp theo  
  → Nếu chưa đủ → tiếp tục search  
  → Nếu đủ → chuyển sang Final Answer  

- **Context enrichment**:  
  → Bổ sung thông tin mới vào prompt  
  → Giúp LLM hiểu sâu hơn về query  

Tuy nhiên, Observation cũng có hạn chế:

- Có thể chứa noise hoặc thông tin không liên quan  
- Có thể quá dài → gây tràn context  
- Nếu không được xử lý tốt → làm giảm hiệu quả reasoning  

→ Do đó, hệ thống cần:
- lọc nội dung  
- truncate dữ liệu  
- chọn lọc thông tin quan trọng  

---

### Tổng kết

ReAct agent khác chatbot ở điểm cốt lõi:

- Chatbot:  
  → **One-shot generation (closed knowledge)**  

- ReAct Agent:  
  → **Multi-step reasoning + external knowledge (open-world)**  

→ Đây là bước chuyển từ:
**LLM → AI Agent System**
---

## IV. Future Improvements (5 Points)

*How would you scale this for a production-level AI agent system?*

- **Scalability**:  

Hiện tại, `web_search` sử dụng `ThreadPoolExecutor` để fetch nội dung từ nhiều URL song song (max_workers=5).  
Cách này phù hợp với quy mô nhỏ nhưng sẽ gặp hạn chế khi số lượng request tăng.

Các hướng cải tiến:

- Chuyển sang **asynchronous I/O (aiohttp)**:
  - Giảm overhead thread
  - Tăng khả năng xử lý đồng thời (concurrency)

- Tách `web_search` thành **microservice độc lập**:
  - Cho phép nhiều agent gọi chung
  - Dễ scale bằng load balancing

- Thêm **caching layer (Redis / in-memory cache)**:
  - Tránh gọi lại cùng query nhiều lần
  - Giảm latency và chi phí

- Batch request:
  - Gom nhiều query lại xử lý cùng lúc nếu cần

---

- **Safety**:  

Vì agent sử dụng dữ liệu từ web (nguồn không kiểm soát), cần bổ sung cơ chế đảm bảo an toàn:

- **Domain filtering**:
  - Ưu tiên nguồn đáng tin (.gov, .edu)
  - Loại bỏ các domain spam / không rõ nguồn

- **Content filtering**:
  - Loại bỏ nội dung độc hại hoặc sai lệch
  - Tránh prompt injection từ web content

- **Output validation**:
  - Kiểm tra lại câu trả lời trước khi trả về user
  - Có thể dùng một LLM thứ hai (Supervisor) để audit

- Logging toàn bộ:
  - Thought / Action / Observation
  - Giúp trace lỗi và kiểm soát hành vi agent

---

- **Performance**:  

Hiện tại hệ thống đã có:
- deduplicate URL  
- sort theo `content_len`  
- truncate content (~8000 chars)  

Các cải tiến thêm:

- **Semantic reranking**:
  - Dùng embedding để chọn top-k kết quả liên quan nhất  
  - Thay vì chỉ dựa vào độ dài content  

- **Adaptive tool usage**:
  - Giảm số lần gọi `web_search` không cần thiết  
  - Dựa trên confidence của model  

- **Token optimization**:
  - Chỉ gửi phần nội dung quan trọng vào LLM  
  - Tránh vượt quá context window  

- **Parallel tool calls (multi-query)**:
  - Cho phép agent tìm nhiều khía cạnh của query cùng lúc  

---

### Tổng kết

Hệ thống hiện tại đã là một **ReAct agent có web search tương đối hoàn chỉnh**,  
nhưng để đạt production-level cần:

- Scale tốt hơn (async + microservice)  
- An toàn hơn (filter + validation)  
- Tối ưu hơn (reranking + token control)  

→ Mục tiêu cuối cùng:
Xây dựng một **robust, scalable AI agent system**.

---

