# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Dương Trịnh Hoài An
- **Student ID**: 2A202600050
- **Date**: 2026/4/6

---

## I. Technical Contribution (15 Points)

*Tôi phụ trách module `Hybrid Search (RAG + Web)`. Sau khi đối chiếu lại với `flow4_hybrid_search.xml` trong thư mục `drawio-workflows`, tôi xác định rõ rằng hybrid mode không chỉ là cơ chế “RAG fail thì mới ra web”, mà là một pipeline 2 bước: ưu tiên RAG trước, sau đó đánh giá kết quả RAG để quyết định trả lời luôn hay gọi thêm web nhằm bổ sung, kiểm chứng hoặc cập nhật thông tin.*

- **Modules Implemented**:
  - `src/agent/graph.py`: xây dựng `mode_hint = hybrid` để LLM hiểu quy trình “RAG trước, đánh giá sau, rồi mới quyết định có gọi `web_search` hay không”.
  - `src/tools/all_tools.py`: xây dựng `search_knowledge_base(query)` để lấy dữ liệu nội bộ từ Weaviate.
  - `src/rag/reranker.py`: dùng reranker `BAAI/bge-reranker-v2-m3` để sắp xếp lại passage sau vector search.
  - `src/tools/web_search.py`: cung cấp nhánh bổ sung dữ liệu web bằng SearXNG, có fallback DuckDuckGo khi cần.
- **Code Highlights**:

```python
docs = vector_store.similarity_search(query, k=8)
if not docs:
    return "NO_RAG_HIT: Không tìm thấy thông tin liên quan trong knowledge base."

ranked = rerank(query, passages, top_k=4)
```

```python
"hybrid": (
    "Kết hợp RAG và web. Ưu tiên RAG trước, bổ sung web nếu cần.\n"
    "Nếu search_knowledge_base trả về NO_RAG_HIT -> gọi thêm web_search."
)
```

- **Documentation**:
  - Bước 1: người dùng gửi câu hỏi với `search_mode='hybrid'`.
  - Bước 2: LLM Turn 1 phân tích yêu cầu và luôn gọi `search_knowledge_base` trước.
  - Bước 3: RAG pipeline chạy theo flow: Weaviate `k=8` rồi rerank còn `top_k=4`.
  - Bước 4: nếu RAG trả về `NO_RAG_HIT`, agent buộc phải gọi `web_search`.
  - Bước 5: nếu RAG có kết quả, agent vẫn phải có một lượt đánh giá tiếp theo để xem thông tin đã đủ chưa. Nếu chưa đủ, thiếu dữ liệu mới, hoặc cần kiểm chứng, agent tiếp tục gọi web.
  - Bước 6: kết quả RAG và Web được đưa qua `Context Defense` để tránh phình context.
  - Bước 7: LLM tổng hợp câu trả lời cuối, ưu tiên knowledge base nội bộ; web chỉ đóng vai trò bổ sung, verify, hoặc update. Phần trả lời cần tách được nguồn `[KB]` và `[Web]`.

---

## II. Debugging Case Study (10 Points)

*Case debug quan trọng nhất của tôi là phát hiện mô tả hybrid search ban đầu đang lệch so với flow thực tế: tôi từng hiểu hybrid gần như một nhánh fallback-only.*

- **Problem Description**: Trong mô tả cũ, tôi viết rằng hybrid search chỉ gọi `web_search` khi `search_knowledge_base` trả về `NO_RAG_HIT`. Tuy nhiên `flow4_hybrid_search.xml` cho thấy còn một nhánh khác: ngay cả khi RAG đã hit, agent vẫn cần một bước đánh giá lại để quyết định xem kết quả đó đã đủ chưa. Nếu dữ liệu nội bộ còn thiếu hoặc đã cũ, agent vẫn phải gọi web.
- **Log Source**:

```text
[tool] search_knowledge_base("câu hỏi hybrid")
-> trả về các chunk nội bộ liên quan

[llm] đánh giá lại kết quả RAG
-> thông tin chưa đủ / cần dữ liệu mới hơn

[tool] web_search("truy vấn bổ sung")
-> lấy thêm nguồn web để verify/update
```

- **Diagnosis**: Điểm cốt lõi của hybrid flow là “RAG-first but not RAG-only”. Quyết định không chỉ dựa vào việc có hit hay không hit, mà còn dựa vào độ đầy đủ của kết quả RAG. Nếu mô tả chỉ dừng ở `NO_RAG_HIT -> web_search`, thì sẽ bỏ mất nửa sau của sơ đồ là bước đánh giá sau khi RAG hit.
- **Solution**:
  - Sửa lại phần mô tả kỹ thuật để phản ánh đủ hai nhánh: `NO_RAG_HIT` và `RAG HIT nhưng chưa đủ`.
  - Nhấn mạnh vai trò của LLM Turn 2a: đánh giá RAG result trước khi quyết định có cần web hay không.
  - Bổ sung `Context Defense` sau khi ghép cả RAG và Web result.
  - Làm rõ câu trả lời cuối phải ưu tiên nguồn nội bộ nhưng vẫn tách được `[KB]` và `[Web]`, đồng thời nêu rõ khi hai nguồn có khác biệt.
- **Kết quả sau khi sửa**: Báo cáo hiện phản ánh đúng flow hybrid trong sơ đồ. Agent không chỉ “fallback sang web”, mà thực sự có một bước reasoning trung gian để quyết định có cần bổ sung web hay không.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: Điểm mạnh nhất của ReAct trong flow hybrid là agent có hai lần suy nghĩ. Lần đầu quyết định “RAG trước”, lần hai đánh giá “RAG đã đủ chưa”. Chatbot trực tiếp thường chỉ trả lời ngay nên không có bước tự đánh giá trung gian này.
2. **Reliability**: Agent có thể kém chatbot ở các câu hỏi rất ngắn hoặc rất đơn giản vì chi phí điều phối cao hơn. Ngoài ra, nếu RAG hit nhưng passage chưa thật sự đúng ý, agent có thể bị kéo sang một hướng chưa tối ưu trước khi kịp bổ sung web.
3. **Observation**: Trong hybrid flow, observation không chỉ là tín hiệu “có dữ liệu hay không”. Nó còn là cơ sở để agent đánh giá độ đầy đủ của RAG, quyết định có gọi web, và xử lý trường hợp nguồn nội bộ với nguồn web không hoàn toàn trùng nhau.

---

## IV. Future Improvements (5 Points)

*Nếu tiếp tục phát triển module này theo đúng hướng của flow 4, tôi sẽ ưu tiên ba cải tiến sau:*

- **Scalability**: Thêm một scoring layer rõ ràng sau RAG để quyết định tự động khi nào “RAG đủ”, khi nào “RAG cần bổ sung web”, thay vì phụ thuộc hoàn toàn vào diễn giải tự nhiên của LLM.
- **Safety**: Bổ sung cơ chế phát hiện mâu thuẫn giữa `[KB]` và `[Web]`, ưu tiên nguồn nội bộ nhưng vẫn cảnh báo nếu dữ liệu web mới hơn hoặc khác biệt đáng kể.
- **Performance**: Tối ưu phần ghép context RAG + Web bằng cache, threshold relevance và nén citation để giảm độ dài context trước khi vào bước synthesis cuối.

---

> [!NOTE]
> Report usecase khớp với `flow4_hybrid_search.xml` trong thư mục `report/drawio-workflows`.
