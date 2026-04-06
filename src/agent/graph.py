"""
agent/graph.py  ── v2.0
LangGraph ReAct Agent

Cải tiến từ sách "Giải phẫu một Agentic OS" (Lâm Nguyễn, 2026):
  ✅ Pattern #3  – Escalating Recovery: retry ×3 → fallback model → surface error
  ✅ Pattern #9  – Context Defense: 3 lớp bảo vệ context window
  ✅ Pattern #2  – needsFollowUp: detect vòng lặp từ nội dung, không từ metadata
  ✅ Pattern #10 – Permission-aware loop: dừng hẳn khi tool bị block liên tiếp

══════════════════════════════════════════════════════
ESCALATING RECOVERY (Pattern #3):
  Cấp 1 – Retry ×3      : gọi lại với cùng model, inject recovery message
  Cấp 2 – Fallback model : chuyển sang model nhỏ hơn, nhanh hơn
  Cấp 3 – Surface error  : báo người dùng, dừng loop

CONTEXT DEFENSE (Pattern #9):
  Lớp 1 – Tool result truncation: kết quả tool > MAX_TOOL_RESULT_CHARS → cắt + ghi disk pointer
  Lớp 2 – Message pruning:        messages cũ > TOKEN_SOFT_LIMIT → loại tool results cũ
  Lớp 3 – Emergency compact:      tổng quá lớn → tóm tắt bằng LLM (nếu có)
══════════════════════════════════════════════════════
"""
from __future__ import annotations

import datetime
import re
import threading
from pathlib import Path
from typing import Callable, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage, AIMessage, ToolMessage, HumanMessage, BaseMessage
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.state import AgentState
from src.config import GROQ_API_KEY, GROQ_MODEL, SEARCH_MODE_DEFAULT

# ── Constants ──────────────────────────────────────────────────────────────────
_MAX_SAME_TOOL_CALLS   = 4      # loop guard: cùng tool > N lần → dừng
_MAX_TOOL_ROUNDS       = 12     # tổng tool rounds tối đa trong 1 turn
_MAX_RECOVERY_ATTEMPTS = 3      # Pattern #3: retry tối đa trước khi escalate
_FALLBACK_MODEL        = "llama-3.1-8b-instant"   # Pattern #3: fallback nhanh hơn
_MAX_SESSION_CACHE     = 500    # giới hạn số session trong recovery cache để tránh memory leak

# Context defense
_MAX_TOOL_RESULT_CHARS = 8_000  # Lớp 1: cắt tool result nếu dài hơn
_TOKEN_SOFT_LIMIT      = 90_000 # Lớp 2: ~90K chars ≈ ~22K tokens (thô)
_TOOL_PRUNE_KEEP_LAST  = 4      # Lớp 2: giữ lại N tool results gần nhất

# ── System Prompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""Bạn là trợ lý AI thông minh, hỗ trợ người dùng bằng tiếng Việt.
Hôm nay là {datetime.datetime.now().strftime('%d/%m/%Y')}.

## NĂNG LỰC
- Trả lời câu hỏi kiến thức, tính toán, lập trình.
- Tìm kiếm tài liệu nội bộ (RAG) qua `search_knowledge_base`.
- Tìm kiếm web qua `web_search` – từ khoá ngắn gọn, tách nhiều chủ đề thành nhiều lần gọi.
- Thực thi code thật qua `run_code_in_sandbox` (Python / PowerShell / Bash).
- Điều khiển hệ thống, ứng dụng, âm lượng, v.v.

## QUY TRÌNH SUY NGHĨ (LUÔN FOLLOW)
Trước khi hành động, viết suy nghĩ trong thẻ <think>:
1. **Phân tích**: Người dùng thực sự muốn gì?
2. **Kế hoạch**: Liệt kê các bước, dùng tool gì, thứ tự nào.
3. **Rủi ro**: Có thao tác nguy hiểm không?
4. **Thực hiện**: Gọi tool đúng thứ tự.

## QUY TẮC
- Bước thất bại → phân tích lý do, thử cách khác, KHÔNG lặp lại y chang.
- Đã thử ≥ 3 cách mà vẫn fail → báo rõ vấn đề, hỏi người dùng.
- Luôn trích dẫn nguồn: `[Tên nguồn](URL hoặc tên file)`.
- Sandbox chạy thật trên máy – không giả vờ.
"""

# Recovery message khi hit max_tokens (tương tự Claude Code)
_RECOVERY_MSG = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought if cut. "
    "Break remaining work into smaller pieces."
)


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT DEFENSE  (Pattern #9)
# ══════════════════════════════════════════════════════════════════════════════

def _truncate_tool_result(content: str, max_chars: int = _MAX_TOOL_RESULT_CHARS) -> str:
    """Lớp 1: Cắt tool result quá dài, thêm note."""
    if len(content) <= max_chars:
        return content
    kept = content[:max_chars]
    dropped = len(content) - max_chars
    return kept + f"\n\n[... truncated {dropped:,} chars – dùng file tool để đọc đầy đủ ...]"


def _estimate_chars(messages: list[BaseMessage]) -> int:
    total = 0
    for m in messages:
        if isinstance(m.content, str):
            total += len(m.content)
        elif isinstance(m.content, list):
            for block in m.content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "")))
    return total


def _prune_old_tool_results(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Lớp 2: Khi tổng chars vượt TOKEN_SOFT_LIMIT,
    loại bỏ ToolMessage cũ, chỉ giữ _TOOL_PRUNE_KEEP_LAST gần nhất.
    """
    total = _estimate_chars(messages)
    if total <= _TOKEN_SOFT_LIMIT:
        return messages

    # Thu thập vị trí ToolMessage
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if len(tool_indices) <= _TOOL_PRUNE_KEEP_LAST:
        return messages   # Không đủ để prune

    # Xóa tool messages cũ (giữ N gần nhất)
    to_remove = set(tool_indices[:-_TOOL_PRUNE_KEEP_LAST])
    pruned = [m for i, m in enumerate(messages) if i not in to_remove]

    removed_count = len(to_remove)
    # Inject note vào đầu để LLM biết
    note = SystemMessage(
        content=f"[context-defense] {removed_count} tool results cũ đã bị pruned để tiết kiệm context. "
                f"Chỉ còn {_TOOL_PRUNE_KEEP_LAST} tool results gần nhất."
    )
    # Tìm vị trí inject (sau system message gốc)
    if pruned and isinstance(pruned[0], SystemMessage):
        pruned.insert(1, note)
    else:
        pruned.insert(0, note)

    return pruned


def apply_context_defense(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Áp dụng 2 lớp context defense trước khi gọi LLM.
    Lớp 3 (LLM compact) để dành cho trường hợp khẩn cấp.
    """
    # Lớp 1: Truncate tool results dài
    defended: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            truncated = _truncate_tool_result(msg.content)
            if truncated != msg.content:
                msg = ToolMessage(
                    content=truncated,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
        defended.append(msg)

    # Lớp 2: Prune old tool results nếu quá nhiều
    defended = _prune_old_tool_results(defended)
    return defended


# ══════════════════════════════════════════════════════════════════════════════
# ESCALATING RECOVERY  (Pattern #3)
# ══════════════════════════════════════════════════════════════════════════════

def _count_recent_tool_calls(messages: list, tool_name: str, last_n: int = 6) -> int:
    count = 0
    for msg in reversed(messages[-last_n:]):
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                if tc.get("name") == tool_name:
                    count += 1
    return count


def _total_tool_rounds(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m, ToolMessage))


def _count_blocked_tool_results(messages: list, last_n: int = 4) -> int:
    """Đếm bao nhiêu tool results gần đây bị sandbox block (bắt đầu bằng '❌ BỊ CHẶN')."""
    count = 0
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    for m in tool_msgs[-last_n:]:
        if isinstance(m.content, str) and m.content.startswith("❌ BỊ CHẶN"):
            count += 1
    return count


def _get_session_key(messages: list) -> str:
    """
    Trả về session key ổn định dựa trên .id của message đầu tiên.

    LangChain BaseMessage có thuộc tính .id (UUID string) được gán lúc tạo
    và không thay đổi trong suốt vòng đời của object đó. Dùng .id thay vì
    id() (địa chỉ bộ nhớ) để tránh collision sau khi GC thu dọn object cũ
    và cấp lại cùng địa chỉ cho session mới.

    Fallback về hash nội dung nếu message không có .id (các implementation cũ).
    """
    if not messages:
        return "empty"
    first = messages[0]
    msg_id = getattr(first, "id", None)
    if msg_id:
        return str(msg_id)
    # Fallback: hash nội dung message đầu tiên
    content = first.content if isinstance(first.content, str) else str(first.content)
    return str(hash(content[:200]))


def _evict_session_cache_if_needed(
    recovery_counters: dict,
    using_fallback: dict,
    lock: threading.Lock,
) -> None:
    """
    Giới hạn kích thước recovery cache để tránh memory leak.
    Xóa toàn bộ khi vượt _MAX_SESSION_CACHE entries.
    Được gọi dưới lock.
    """
    if len(recovery_counters) > _MAX_SESSION_CACHE:
        recovery_counters.clear()
        using_fallback.clear()


# ══════════════════════════════════════════════════════════════════════════════
# BUILD AGENT
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(all_tools: list, mode_getter: Optional[Callable] = None):
    """
    Khởi tạo và biên dịch LangGraph agent với:
    - Escalating recovery (Pattern #3)
    - Context defense (Pattern #9)
    - Loop guards (Pattern #2)

    Args:
        all_tools:   list LangChain tools
        mode_getter: callable → search_mode ('hybrid'|'on'|'off')
    Returns:
        compiled LangGraph app
    """

    # ── LLM instances ──────────────────────────────────────────────────────────
    _primary_llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.6,
        max_tokens=4096,
    )
    _fallback_llm = ChatGroq(   # Pattern #3 – fallback model
        model=_FALLBACK_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.4,
        max_tokens=2048,
    )
    llm_primary  = _primary_llm.bind_tools(all_tools)
    llm_fallback = _fallback_llm.bind_tools(all_tools)

    tool_names = {t.name for t in all_tools}

    # ── Per-session recovery state (in-memory) ─────────────────────────────────
    # Lưu trạng thái recovery per invocation (không lưu trong AgentState để tránh
    # serialize phức tạp). Thread-safe với lock.
    _recovery_counters: dict[str, int] = {}
    _using_fallback:    dict[str, bool] = {}
    _lock = threading.Lock()

    # ── Node: LLM ──────────────────────────────────────────────────────────────
    def llm_node(state: AgentState):
        messages = list(state["messages"])

        # Search mode prompt
        current_mode = SEARCH_MODE_DEFAULT
        if callable(mode_getter):
            try:
                current_mode = mode_getter() or SEARCH_MODE_DEFAULT
            except Exception:
                pass

        mode_hint = {
            "hybrid": (
                "→ Kết hợp RAG VÀ web. Ưu tiên RAG trước, bổ sung web nếu cần.\n"
                "  Nếu search_knowledge_base trả về NO_RAG_HIT → gọi thêm web_search."
            ),
            "on": (
                "→ CHỈ dùng RAG. KHÔNG được gọi web_search.\n"
                "  Nếu search_knowledge_base trả về NO_RAG_HIT → báo người dùng rằng\n"
                "  không tìm thấy thông tin trong tài liệu nội bộ và đề xuất họ\n"
                "  chuyển sang search mode 'hybrid' hoặc 'off' để tìm kiếm web."
            ),
            "off": (
                "→ CHỈ dùng web. KHÔNG được gọi search_knowledge_base."
            ),
        }.get(current_mode, "→ Dùng cả RAG và web.")

        mode_prompt = f"\n## SEARCH MODE: **{current_mode.upper()}**\n{mode_hint}"

        # ── Guard: total tool rounds ────────────────────────────────────────────
        total_rounds = _total_tool_rounds(messages)
        if total_rounds >= _MAX_TOOL_ROUNDS:
            return {"messages": [AIMessage(content=(
                "⚠️ Tôi đã thực hiện nhiều bước xử lý liên tiếp. Để tránh vòng lặp, "
                "tôi tạm dừng.\n**Bạn có muốn tôi tiếp tục không?**"
            ))]}

        # ── Guard: same tool repeated ───────────────────────────────────────────
        for tname in tool_names:
            if _count_recent_tool_calls(messages, tname) >= _MAX_SAME_TOOL_CALLS:
                return {"messages": [AIMessage(content=(
                    f"⚠️ Tool `{tname}` đã được gọi {_MAX_SAME_TOOL_CALLS} lần liên tiếp "
                    f"mà chưa đạt kết quả.\n**Bạn có muốn tôi thử cách khác?**"
                ))]}

        # ── Guard: sandbox liên tục bị block ────────────────────────────────────
        blocked = _count_blocked_tool_results(messages)
        if blocked >= 3:
            return {"messages": [AIMessage(content=(
                "⚠️ Sandbox đã từ chối 3 lần liên tiếp do vi phạm permission.\n"
                "Tôi sẽ dừng lại thay vì tiếp tục thử.\n"
                "Hãy kiểm tra code và dùng `inspect_permission` để debug."
            ))]}

        # ── Context Defense (Pattern #9) ────────────────────────────────────────
        # Không áp dụng cho system message đầu tiên
        defended = apply_context_defense(messages)

        # ── Build system prompt ─────────────────────────────────────────────────
        full_system = SYSTEM_PROMPT + mode_prompt
        if not defended or not isinstance(defended[0], SystemMessage):
            defended = [SystemMessage(content=full_system)] + defended
        else:
            defended = [SystemMessage(content=full_system)] + defended[1:]

        # ── Pattern #3: Escalating Recovery ────────────────────────────────────
        # Xác định nên dùng primary hay fallback
        skey = _get_session_key(messages)
        with _lock:
            _evict_session_cache_if_needed(_recovery_counters, _using_fallback, _lock)
            recovery_count = _recovery_counters.get(skey, 0)
            use_fallback   = _using_fallback.get(skey, False)

        llm = llm_fallback if use_fallback else llm_primary

        try:
            response = llm.invoke(defended)
            # Reset recovery counter khi thành công
            with _lock:
                _recovery_counters[skey] = 0
                _using_fallback[skey]    = False
            return {"messages": [response]}

        except Exception as e:
            err_str = str(e).lower()
            with _lock:
                recovery_count = _recovery_counters.get(skey, 0) + 1
                _recovery_counters[skey] = recovery_count

            # Cấp 1: Retry ×3 với cùng model
            if recovery_count <= _MAX_RECOVERY_ATTEMPTS and "rate_limit" not in err_str:
                recovery_msg = HumanMessage(content=_RECOVERY_MSG)
                try:
                    response = llm.invoke(defended + [recovery_msg])
                    with _lock:
                        _recovery_counters[skey] = 0
                    return {"messages": [response]}
                except Exception:
                    pass

            # Cấp 2: Chuyển sang fallback model
            if not use_fallback:
                with _lock:
                    _using_fallback[skey] = True
                try:
                    response = llm_fallback.invoke(defended)
                    return {"messages": [AIMessage(
                        content=(
                            f"⚡ [fallback-model] Đang dùng {_FALLBACK_MODEL} do lỗi.\n\n"
                            + (response.content if isinstance(response.content, str) else "")
                        )
                    )]}
                except Exception as fe:
                    pass

            # Cấp 3: Surface error
            with _lock:
                _recovery_counters[skey] = 0
                _using_fallback[skey]    = False
            return {"messages": [AIMessage(content=(
                f"❌ Không thể xử lý yêu cầu sau {_MAX_RECOVERY_ATTEMPTS} lần thử:\n"
                f"   Lỗi: {str(e)}\n"
                f"Vui lòng thử lại hoặc đơn giản hoá yêu cầu."
            ))]}

    # ── Node: Tools ─────────────────────────────────────────────────────────────
    tool_node = ToolNode(tools=all_tools)

    # ── Build Graph ─────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("llm",   llm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()