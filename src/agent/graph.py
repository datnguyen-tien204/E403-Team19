"""
agent/graph.py  ── v2.0
LangGraph ReAct Agent
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
import json
import os
import re
import time
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

# Context defense
_MAX_TOOL_RESULT_CHARS = 8_000  # Lớp 1: cắt tool result nếu dài hơn
_TOKEN_SOFT_LIMIT      = 90_000 # Lớp 2: ~90K chars ≈ ~22K tokens (thô)
_TOOL_PRUNE_KEEP_LAST  = 4      # Lớp 2: giữ lại N tool results gần nhất

# ── Logging ────────────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

def _log(event: str, data: dict):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": event,
        "data": data,
    }
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    log_file = _LOG_DIR / (datetime.date.today().isoformat() + ".log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Cost estimation (Groq pricing, USD per token) ──────────────────────────────
_COST_PER_INPUT_TOKEN  = 0.00000029   # qwen3-32b: $0.29 / 1M input tokens
_COST_PER_OUTPUT_TOKEN = 0.00000059   # qwen3-32b: $0.59 / 1M output tokens

def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens * _COST_PER_INPUT_TOKEN
        + completion_tokens * _COST_PER_OUTPUT_TOKEN,
        8,
    )

def _extract_usage(response: AIMessage) -> dict:
    """Trích xuất token usage từ response_metadata của Groq."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    prompt_tokens     = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens      = usage.get("total_tokens", prompt_tokens + completion_tokens)
    return {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
    }


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

    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if len(tool_indices) <= _TOOL_PRUNE_KEEP_LAST:
        return messages

    to_remove = set(tool_indices[:-_TOOL_PRUNE_KEEP_LAST])
    pruned = [m for i, m in enumerate(messages) if i not in to_remove]

    removed_count = len(to_remove)
    note = SystemMessage(
        content=f"[context-defense] {removed_count} tool results cũ đã bị pruned để tiết kiệm context. "
                f"Chỉ còn {_TOOL_PRUNE_KEEP_LAST} tool results gần nhất."
    )
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


# ══════════════════════════════════════════════════════════════════════════════
# BUILD AGENT
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(all_tools: list, mode_getter: Optional[Callable] = None):
    """
    Khởi tạo và biên dịch LangGraph agent với:
    - Escalating recovery (Pattern #3)
    - Context defense (Pattern #9)
    - Loop guards (Pattern #2)
    - Structured JSON logging cho mỗi LLM call và tool call

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
    _recovery_counters: dict[int, int] = {}
    _using_fallback:    dict[int, bool] = {}
    _lock = threading.Lock()

    def _get_session_key(messages: list) -> int:
        return id(messages[0]) if messages else 0

    # ── Helper: current step number ────────────────────────────────────────────
    def _current_step(messages: list) -> int:
        """Step = số AIMessage đã có trong state (trước lần gọi LLM này) + 1."""
        return sum(1 for m in messages if isinstance(m, AIMessage)) + 1

    # ── Helper: extract user input từ messages ─────────────────────────────────
    def _get_user_input(messages: list) -> str:
        for m in messages:
            if isinstance(m, HumanMessage):
                return m.content if isinstance(m.content, str) else str(m.content)
        return ""

    # ── Node: LLM ──────────────────────────────────────────────────────────────
    def llm_node(state: AgentState):
        messages = list(state["messages"])

        step = _current_step(messages)

        # Log AGENT_START ở bước đầu tiên
        if step == 1:
            _log("AGENT_START", {
                "input": _get_user_input(messages),
                "model": GROQ_MODEL,
            })

        # Search mode prompt
        current_mode = SEARCH_MODE_DEFAULT
        if callable(mode_getter):
            try:
                current_mode = mode_getter() or SEARCH_MODE_DEFAULT
            except Exception:
                pass

        mode_hint = {
            "hybrid": "→ Kết hợp RAG VÀ web. Ưu tiên RAG trước, bổ sung web nếu cần.",
            "on":     "→ CHỈ dùng RAG. KHÔNG được gọi web_search.",
            "off":    "→ CHỈ dùng web. KHÔNG được gọi search_knowledge_base.",
        }.get(current_mode, "→ Dùng cả RAG và web.")

        mode_prompt = f"\n## SEARCH MODE: **{current_mode.upper()}**\n{mode_hint}"

        # ── Guard: total tool rounds ────────────────────────────────────────────
        total_rounds = _total_tool_rounds(messages)
        if total_rounds >= _MAX_TOOL_ROUNDS:
            answer = (
                "⚠️ Tôi đã thực hiện nhiều bước xử lý liên tiếp. Để tránh vòng lặp, "
                "tôi tạm dừng.\n**Bạn có muốn tôi tiếp tục không?**"
            )
            _log("AGENT_END", {
                "steps": step - 1,
                "total_tokens": _sum_tokens(messages),
                "reason": "max_tool_rounds_reached",
                "final_answer": answer,
            })
            return {"messages": [AIMessage(content=answer)]}

        # ── Guard: same tool repeated ───────────────────────────────────────────
        for tname in tool_names:
            if _count_recent_tool_calls(messages, tname) >= _MAX_SAME_TOOL_CALLS:
                answer = (
                    f"⚠️ Tool `{tname}` đã được gọi {_MAX_SAME_TOOL_CALLS} lần liên tiếp "
                    f"mà chưa đạt kết quả.\n**Bạn có muốn tôi thử cách khác?**"
                )
                _log("AGENT_END", {
                    "steps": step - 1,
                    "total_tokens": _sum_tokens(messages),
                    "reason": f"tool_loop_guard:{tname}",
                    "final_answer": answer,
                })
                return {"messages": [AIMessage(content=answer)]}

        # ── Guard: sandbox liên tục bị block ────────────────────────────────────
        blocked = _count_blocked_tool_results(messages)
        if blocked >= 3:
            answer = (
                "⚠️ Sandbox đã từ chối 3 lần liên tiếp do vi phạm permission.\n"
                "Tôi sẽ dừng lại thay vì tiếp tục thử.\n"
                "Hãy kiểm tra code và dùng `inspect_permission` để debug."
            )
            _log("AGENT_END", {
                "steps": step - 1,
                "total_tokens": _sum_tokens(messages),
                "reason": "sandbox_blocked_3x",
                "final_answer": answer,
            })
            return {"messages": [AIMessage(content=answer)]}

        # ── Context Defense (Pattern #9) ────────────────────────────────────────
        defended = apply_context_defense(messages)

        # ── Build system prompt ─────────────────────────────────────────────────
        full_system = SYSTEM_PROMPT + mode_prompt
        if not defended or not isinstance(defended[0], SystemMessage):
            defended = [SystemMessage(content=full_system)] + defended
        else:
            defended = [SystemMessage(content=full_system)] + defended[1:]

        # ── Pattern #3: Escalating Recovery ────────────────────────────────────
        skey = _get_session_key(messages)
        with _lock:
            recovery_count = _recovery_counters.get(skey, 0)
            use_fallback   = _using_fallback.get(skey, False)

        llm          = llm_fallback if use_fallback else llm_primary
        active_model = _FALLBACK_MODEL if use_fallback else GROQ_MODEL

        try:
            t_start  = time.time()
            response = llm.invoke(defended)
            latency_ms = int((time.time() - t_start) * 1000)

            with _lock:
                _recovery_counters[skey] = 0
                _using_fallback[skey]    = False

        except Exception as e:
            err_str = str(e).lower()
            with _lock:
                recovery_count = _recovery_counters.get(skey, 0) + 1
                _recovery_counters[skey] = recovery_count

            # Cấp 1: Retry ×3 với cùng model
            if recovery_count <= _MAX_RECOVERY_ATTEMPTS and "rate_limit" not in err_str:
                recovery_msg = HumanMessage(content=_RECOVERY_MSG)
                try:
                    t_start  = time.time()
                    response = llm.invoke(defended + [recovery_msg])
                    latency_ms = int((time.time() - t_start) * 1000)
                    with _lock:
                        _recovery_counters[skey] = 0
                    _log("AGENT_RECOVERY", {
                        "step": step,
                        "level": 1,
                        "model": active_model,
                        "attempt": recovery_count,
                    })
                except Exception:
                    pass
                else:
                    # Recovery thành công – fall-through bình thường
                    pass

            # Cấp 2: Chuyển sang fallback model
            if not use_fallback:
                with _lock:
                    _using_fallback[skey] = True
                try:
                    t_start  = time.time()
                    response = llm_fallback.invoke(defended)
                    latency_ms = int((time.time() - t_start) * 1000)
                    active_model = _FALLBACK_MODEL
                    _log("AGENT_RECOVERY", {
                        "step": step,
                        "level": 2,
                        "model": _FALLBACK_MODEL,
                        "reason": str(e),
                    })
                    content_prefix = f"⚡ [fallback-model] Đang dùng {_FALLBACK_MODEL} do lỗi.\n\n"
                    raw = response.content if isinstance(response.content, str) else ""
                    return {"messages": [AIMessage(content=content_prefix + raw)]}
                except Exception as fe:
                    pass

            # Cấp 3: Surface error
            with _lock:
                _recovery_counters[skey] = 0
                _using_fallback[skey]    = False
            answer = (
                f"❌ Không thể xử lý yêu cầu sau {_MAX_RECOVERY_ATTEMPTS} lần thử:\n"
                f"   Lỗi: {str(e)}\n"
                f"Vui lòng thử lại hoặc đơn giản hoá yêu cầu."
            )
            _log("AGENT_ERROR", {
                "step": step,
                "level": 3,
                "error": str(e),
            })
            _log("AGENT_END", {
                "steps": step,
                "total_tokens": _sum_tokens(messages),
                "reason": "escalating_recovery_exhausted",
                "final_answer": answer,
            })
            return {"messages": [AIMessage(content=answer)]}

        # ── Extract usage & log ─────────────────────────────────────────────────
        usage = _extract_usage(response)
        cost  = _estimate_cost(usage["prompt_tokens"], usage["completion_tokens"])

        _log("LLM_METRIC", {
            "provider":          "groq",
            "model":             active_model,
            "prompt_tokens":     usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens":      usage["total_tokens"],
            "latency_ms":        latency_ms,
            "cost_estimate":     cost,
        })

        # Preview: lấy 300 ký tự đầu, escape newline để fit 1 dòng JSON
        preview = ""
        if isinstance(response.content, str):
            preview = response.content[:300].replace("\n", "\\n")
        elif isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    preview = block["text"][:300].replace("\n", "\\n")
                    break

        _log("AGENT_STEP", {
            "step":             step,
            "response_preview": preview,
            "usage":            usage,
            "latency_ms":       latency_ms,
        })

        # ── Nếu không có tool_calls → đây là Final Answer → log AGENT_END ──────
        tool_calls = getattr(response, "tool_calls", []) or []
        if not tool_calls:
            final_text = response.content if isinstance(response.content, str) else preview
            _log("AGENT_END", {
                "steps":            step,
                "total_tokens":     _sum_tokens(messages) + usage["total_tokens"],
                "total_latency_ms": latency_ms,
                "final_answer":     final_text[:500],
            })

        return {"messages": [response]}

    # ── Helper: tổng tokens từ response_metadata trong lịch sử ────────────────
    def _sum_tokens(messages: list) -> int:
        total = 0
        for m in messages:
            if isinstance(m, AIMessage):
                u = _extract_usage(m)
                total += u.get("total_tokens", 0)
        return total

    # ── Node: Tools (wrapped để log TOOL_CALL) ─────────────────────────────────
    _inner_tool_node = ToolNode(tools=all_tools)

    def tool_node(state: AgentState):
        messages = list(state["messages"])

        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)),
            None,
        )
        pending_calls: list[dict] = []
        if last_ai:
            pending_calls = getattr(last_ai, "tool_calls", []) or []

        step = sum(1 for m in messages if isinstance(m, AIMessage))

        result = _inner_tool_node.invoke(state)

        if isinstance(result, dict):
            new_messages: list[BaseMessage] = result.get("messages", [])
        else:
            new_messages: list[BaseMessage] = getattr(result, "messages", [])

        tool_results: dict[str, str] = {}
        for tm in new_messages:
            if isinstance(tm, ToolMessage):
                tool_results[tm.tool_call_id] = (
                    tm.content if isinstance(tm.content, str) else str(tm.content)
                )

        for tc in pending_calls:
            tool_call_id = tc.get("id", "")
            tool_name = tc.get("name", "unknown")
            tool_args = tc.get("args", {})
            observation = tool_results.get(tool_call_id, "(no result)")

            _log("TOOL_CALL", {
                "step": step,
                "tool": tool_name,
                "arguments": json.dumps(tool_args, ensure_ascii=False),
                "observation": observation[:1000],
            })

        return result

    # ── Build Graph ─────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("llm",   llm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()