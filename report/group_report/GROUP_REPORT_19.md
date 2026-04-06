# Group Report: Lab 3 - Production-Grade Agentic System

- **Team Name**: Team-19
- **Team Members**: 
    + Nguyễn Tiến Đạt - 
    + Dương Trịnh Hoài An - 2A202600050
    + Trần Ngọc Hùng - 2A202600429
    + Bùi Đức Thắng - 2A202600002
- **Deployment Date**: 2026-04-06

---

## 1. Executive Summary

*Hệ thống của nhóm là một "production-grade agentic assistant" dùng LangGraph + Groq + Weaviate + SearXNG + DuckDuckGo Search để xử lý 6 luồng tác vụ chính được mô hình hóa trong thư mục `report/drawio-workflows`: chat thường, web search, RAG, hybrid search, code sandbox và system control.*

- **Success Rate**: `6/6` luồng nghiệp vụ chính đã có workflow riêng và có ánh xạ sang implementation trong codebase. Đây là functional coverage theo thiết kế, không phải benchmark accuracy trên tập test lớn.
- **Key Outcome**: So với chatbot baseline chỉ mạnh ở trả lời trực tiếp, agent của nhóm mở rộng được thêm 5 nhóm nhiệm vụ cần tool hoặc môi trường thực thi thật, đặc biệt mạnh ở các truy vấn nhiều bước, truy vấn thời sự, truy vấn tài liệu nội bộ và tác vụ thực thi hành động.

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation
Toàn bộ 6 flow đều đi qua cùng một lõi ReAct:

- Người dùng gửi yêu cầu vào lớp FastAPI/WebSocket, kèm `search_mode` nếu cần.
- `llm_node()` trong `src/agent/graph.py` nhận message, inject system prompt và mode hint tương ứng: `off`, `on`, hoặc `hybrid`.
- LLM suy nghĩ trong khối `<think>`, sau đó hoặc trả lời trực tiếp, hoặc sinh `tool_calls`.
- `tools_condition()` quyết định rẽ sang `ToolNode` hay kết thúc luôn.
- Sau khi tool trả về `ToolMessage`, agent quan sát kết quả rồi quyết định gọi tiếp tool hay tổng hợp câu trả lời cuối.
- Vòng lặp được bảo vệ bởi các guardrails: tối đa `12` tool rounds, tối đa `4` lần lặp cùng một tool, dừng khi sandbox bị chặn liên tiếp, và có context defense để cắt bớt tool output quá dài.

Ánh xạ 6 flow vào kiến trúc:

- **Flow 1 - Chat Q&A**: `search_mode=off`, không gọi tool, LLM trả lời trực tiếp.
- **Flow 2 - Web Search**: LLM gọi `web_search`, nhận kết quả từ SearXNG hoặc DuckDuckGo fallback, rồi tổng hợp có citation.
- **Flow 3 - RAG**: LLM gọi `search_knowledge_base`, lấy top-k từ Weaviate rồi rerank trước khi trả lời.
- **Flow 4 - Hybrid**: ưu tiên RAG trước, nếu `NO_RAG_HIT` hoặc thiếu context thì gọi thêm `web_search`.
- **Flow 5 - Code Sandbox**: LLM viết code, gọi `run_code_in_sandbox`, đọc output thật rồi phản hồi hoặc debug.
- **Flow 6 - System Control**: LLM phân loại intent và gọi đúng tool hệ thống như volume, mở ứng dụng, lấy thông tin máy.

### 2.2 Tool Definitions (Inventory)
| Tool Name | Input Format | Use Case |
| :--- | :--- | :--- |
| `web_search` | `query: str, k?: int` | Tìm kiếm web thời gian thực bằng SearXNG, fallback DuckDuckGo nếu lỗi. |
| `search_knowledge_base` | `query: str` | Tìm kiếm tài liệu nội bộ trong Weaviate và rerank kết quả RAG. |
| `run_code_in_sandbox` | `language: str, code: str, timeout?: int` | Chạy Python/PowerShell/Bash thật trong sandbox và trả về stdout/stderr. |
| `inspect_permission` | `language: str, code: str, working_dir?: str` | Kiểm tra trước code có bị sandbox chặn hay không. |
| `control_volume` | `action: str, value?: int` | Tăng/giảm/set/mute/unmute âm lượng trên Windows. |
| `open_application` | `app_name: str` | Mở ứng dụng, path hoặc URL từ agent. |
| `get_system_info` | `info_type?: str` | Lấy battery, CPU, RAM, disk, datetime hoặc all. |
| `MCP / extended system tools` | `tool-specific` | Mở rộng sang Gmail, Calendar và các integration ngoài lõi chính. |

### 2.3 LLM Providers Used
- **Primary**: Groq qua `ChatGroq`, model mặc định hiện tại trong repo là `qwen/qwen3-32b`.
- **Secondary (Backup)**: `llama-3.1-8b-instant`, dùng cho cơ chế `Escalating Recovery` khi model chính lỗi hoặc đụng vấn đề context/output.

---

## 3. Telemetry & Performance Dashboard

*Repo hiện đã có hạ tầng telemetry thời gian thực ở mức per-request, nhưng chưa có file benchmark tổng hợp chính thức cho cả test suite. Vì vậy phần này tách rõ giữa “đã instrument” và “đã đo batch”.*

- **Average Latency (P50)**: Chưa có benchmark batch được commit kèm báo cáo. Tuy nhiên server đã stream `latency_ms` về UI trong quá trình chat.
- **Max Latency (P99)**: Chưa có thống kê P99 chính thức. Theo kiến trúc, các flow nặng nhất là `hybrid search` và `code sandbox` vì có nhiều bước tool hơn Q&A trực tiếp.
- **Average Tokens per Task**: Đã có đếm `prompt_tokens`, `completion_tokens`, `total_tokens` trong `src/api/server.py`, nhưng chưa có file tổng hợp trung bình theo test suite.
- **Total Cost of Test Suite**: Chưa có cost collector tự động trong repo, nên chưa xuất được chi phí cuối cùng cho toàn bộ bài test.

Quan sát vận hành từ 6 flow:

- `Flow 1` có độ trễ thấp nhất vì không gọi tool.
- `Flow 2` và `Flow 4` phụ thuộc web fetch nên độ trễ biến động theo mạng và số nguồn cần tổng hợp.
- `Flow 3` phụ thuộc tốc độ Weaviate và reranker.
- `Flow 5` có độ trễ cao nhất khi code cần chạy lại hoặc chạm timeout.
- `Flow 6` thường nhanh, nhưng bị giới hạn theo nền tảng Windows và hành động hệ thống thực tế.

---

## 4. Root Cause Analysis (RCA) - Failure Traces

*Case tiêu biểu nhất nằm ở luồng `Hybrid Search`, vì đây là nơi kết hợp cả reasoning, RAG và web tool.*

### Case Study: Hybrid search bị gãy khi RAG miss và web layer không sẵn sàng
- **Input**: "Tóm tắt tài liệu nội bộ về dự án X và bổ sung thông tin cập nhật mới nhất liên quan."
- **Observation**: Agent gọi `search_knowledge_base(query)` trước. Khi RAG trả về `NO_RAG_HIT`, agent cần chuyển sang `web_search`, nhưng nhánh web có thể thất bại nếu SearXNG local không chạy hoặc timeout.
- **Root Cause**: Thiết kế hybrid đúng về mặt logic, nhưng độ bền vận hành phụ thuộc vào availability của service web search. Nếu chỉ dựa vào SearXNG mà không có health check và fallback, toàn bộ flow 4 sẽ bị suy giảm dù planner của agent ra quyết định đúng.
- **Mitigation in code**: Bổ sung kiểm tra `_check_searxng_health()`, retry/backoff cho SearXNG, fallback sang DuckDuckGo, đồng thời giữ tín hiệu `NO_RAG_HIT` để agent biết khi nào phải rẽ sang web.
- **Takeaway**: Với agentic system, failure không chỉ đến từ LLM hallucination mà còn đến từ độ bền của tool ecosystem. Reliability của orchestration phụ thuộc trực tiếp vào khả năng degrade gracefully của từng tool.

---

## 5. Ablation Studies & Experiments

### Experiment 1: Prompt v1 vs Prompt v2
- **Diff**: Thêm `search_mode` hint rõ ràng trong system prompt:
  - `off`: chỉ dùng web hoặc trả lời trực tiếp, không gọi RAG.
  - `on`: chỉ dùng RAG, không gọi web; nếu `NO_RAG_HIT` thì báo người dùng.
  - `hybrid`: ưu tiên RAG, nếu thiếu hoặc `NO_RAG_HIT` thì gọi thêm web.
- **Result**: Hành vi chọn tool nhất quán hơn giữa 3 flow tìm kiếm. Agent giảm tình trạng gọi sai nhánh, đặc biệt ở RAG-only và Hybrid.

### Experiment 2 (Bonus): Chatbot vs Agent
| Case | Chatbot Result | Agent Result | Winner |
| :--- | :--- | :--- | :--- |
| Flow 1 - Q&A thông thường | Trả lời đúng với câu hỏi kiến thức ổn định | Trả lời đúng, nhưng không vượt trội rõ rệt | Draw |
| Flow 2 - Web search | Dễ trả lời theo kiến thức cũ hoặc không có nguồn | Gọi `web_search`, tổng hợp kết quả mới và có citation | **Agent** |
| Flow 3 - RAG nội bộ | Không có quyền truy cập knowledge base thực | Gọi `search_knowledge_base`, đọc tài liệu nội bộ và trích dẫn nguồn | **Agent** |
| Flow 4 - Hybrid Search | Không kết hợp được dữ liệu nội bộ với dữ liệu web mới | Ưu tiên KB nội bộ, bổ sung web khi thiếu và ghi rõ hai loại nguồn | **Agent** |
| Flow 5 - Code Sandbox | Chỉ sinh code mẫu, không chạy thật | Chạy code thật, đọc output, debug lại nếu cần | **Agent** |
| Flow 6 - System Control | Chỉ mô tả thao tác, không thực hiện | Gọi tool hệ thống để thay đổi trạng thái máy hoặc đọc system info | **Agent** |

---

## 6. Production Readiness Review

*Xét theo 6 flow hiện có, hệ thống đã vượt mức demo đơn giản nhưng vẫn cần thêm một số lớp để đạt mức production hoàn chỉnh.*

- **Security**: Sandbox đã có nhiều lớp chặn lệnh nguy hiểm, path nhạy cảm, shell injection và destructive patterns; system tools có platform guard cho Windows; web search đã có trích dẫn nguồn. Tuy nhiên vẫn nên bổ sung allowlist domain, structured validation cho tool args và audit log tập trung.
- **Guardrails**: Agent hiện đã có `max_tool_rounds=12`, `same_tool_calls=4`, context defense ở ngưỡng `8,000` ký tự cho tool result và `90,000` ký tự cho toàn context, cùng với fallback model khi model chính gặp lỗi. Đây là nền guardrail tốt để tránh loop và phình context.
- **Scaling**: Bước tiếp theo nên là tạo bộ evaluation chuẩn cho cả 6 flow, lưu telemetry vào database thay vì chỉ stream về UI, tách query router cho `off/on/hybrid`, và chuyển các tác vụ lâu như web aggregation hoặc sandbox execution sang hàng đợi bất đồng bộ nếu tải tăng.

---

> [!NOTE]
> File này đã được điền nội dung dựa trên 6 workflow trong `report/drawio-workflows`.
