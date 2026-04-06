"""
tools/tool_orchestrator.py  ── v1.0  (file MỚI)

Cải tiến từ sách "Giải phẫu một Agentic OS" (Lâm Nguyễn, 2026):
  ✅ Pattern #4 – Concurrency-safe partitioning
  ✅ Pattern #5 – Parallel execution cho safe tools

══════════════════════════════════════════════════════
VẤN ĐỀ ĐƯỢC GIẢI QUYẾT:
  Khi agent gọi 5 tools cùng lúc (vd: web_search × 3 + search_kb + run_code),
  LangGraph mặc định chạy TUẦN TỰ. Orchestrator này phân loại và chạy SONG SONG
  các tools an toàn (read-only), tuần tự các tools có side-effect.

CÁCH HOẠT ĐỘNG:
  1. partitionToolCalls() gom consecutive safe tools vào cùng batch.
  2. Concurrent batch → ThreadPoolExecutor (max 8 workers).
  3. Exclusive batch   → chạy từng tool một (tuần tự).

VÍ DỤ:
  Input:  [web_search, web_search, search_kb, run_code, web_search]
  Batches: [{web_search × 2 + search_kb, concurrent=True},
            {run_code,                   concurrent=False},
            {web_search,                 concurrent=True}]
══════════════════════════════════════════════════════

CÁCH DÙNG:
  Bước 1: Build toolset như bình thường.
  Bước 2: Gọi build_concurrent_tool_node(tools) thay vì ToolNode(tools).
  Bước 3: Dùng trong graph như ToolNode thông thường.

  Ví dụ:
    from tools.tool_orchestrator import build_concurrent_tool_node, register_safe_tools
    concurrent_node = build_concurrent_tool_node(all_tools)
    graph.add_node("tools", concurrent_node)

  Hoặc chỉ dùng partitioner để biết tool nào safe:
    from tools.tool_orchestrator import is_tool_safe
    print(is_tool_safe("web_search"))   # True
    print(is_tool_safe("run_code_in_sandbox"))  # False
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

# ══════════════════════════════════════════════════════════════════════════════
# SAFE TOOL REGISTRY  (Pattern #4: each tool declares its own safety)
# ══════════════════════════════════════════════════════════════════════════════

# Tools LUÔN safe (read-only, không side-effect):
_ALWAYS_SAFE_TOOLS: set[str] = {
    # RAG & search
    "search_knowledge_base",
    "web_search",
    # System info (read-only)
    "get_system_info",
    "get_network_info",
    # Workspace
    "list_workspace",
    "read_task_log",
    "inspect_permission",
    # Process info
    "manage_process",   # action='list' hoặc 'find' → safe; action='kill' → exclusive
}

# Tools LUÔN exclusive (có side-effect, phải chạy tuần tự):
_ALWAYS_EXCLUSIVE_TOOLS: set[str] = {
    "run_code_in_sandbox",
    "start_background_task",
    "stop_background_task",
    "install_packages",
    "control_volume",
    "control_brightness",
    "take_screenshot",
    "power_control",
    "clipboard_control",
    "open_application",
}

# Tools CONDITIONAL: safe hay không tuỳ thuộc vào input
_CONDITIONAL_CLASSIFIERS: dict[str, Callable[[dict], bool]] = {}


def register_conditional_tool(
    tool_name: str,
    classifier: Callable[[dict], bool],
):
    """
    Đăng ký classifier cho tool conditional.
    classifier(input_dict) → True nếu safe, False nếu exclusive.

    Ví dụ:
        register_conditional_tool(
            "manage_process",
            lambda inp: inp.get("action", "") in ("list", "find")
        )
    """
    _CONDITIONAL_CLASSIFIERS[tool_name] = classifier


# Đăng ký sẵn manage_process: list/find → safe, kill → exclusive
register_conditional_tool(
    "manage_process",
    lambda inp: inp.get("action", "kill") in ("list", "find")
)


def is_tool_safe(tool_name: str, tool_input: dict | None = None) -> bool:
    """
    Kiểm tra xem tool có safe để chạy song song không.
    Safe-by-default = False (conservative, tránh race condition).
    """
    if tool_name in _ALWAYS_SAFE_TOOLS:
        return True
    if tool_name in _ALWAYS_EXCLUSIVE_TOOLS:
        return False
    if tool_name in _CONDITIONAL_CLASSIFIERS and tool_input is not None:
        try:
            return bool(_CONDITIONAL_CLASSIFIERS[tool_name](tool_input))
        except Exception:
            return False  # Default exclusive nếu classifier throw
    return False  # Unknown tool → exclusive (safe-by-default = conservative)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PARTITIONING  (Pattern #4 – partitionToolCalls)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCallBatch:
    is_concurrent: bool
    calls: list[dict]   # list của {id, name, args}


def partition_tool_calls(tool_calls: list[dict]) -> list[ToolCallBatch]:
    """
    Phân chia danh sách tool calls thành batches:
    - Consecutive safe tools → 1 concurrent batch
    - Exclusive tool → 1 serial batch riêng

    Greedy algorithm: gom consecutive safe tools vào cùng batch.

    Input: [{"id": "...", "name": "web_search", "args": {...}}, ...]
    Output: [ToolCallBatch(is_concurrent=True, calls=[...]), ...]
    """
    if not tool_calls:
        return []

    batches: list[ToolCallBatch] = []

    for tc in tool_calls:
        name  = tc.get("name", "")
        args  = tc.get("args", {}) or {}
        safe  = is_tool_safe(name, args)

        if safe and batches and batches[-1].is_concurrent:
            # Gom vào batch concurrent hiện tại
            batches[-1].calls.append(tc)
        else:
            # Tạo batch mới
            batches.append(ToolCallBatch(is_concurrent=safe, calls=[tc]))

    return batches


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

_MAX_CONCURRENT_WORKERS = 8   # max tools chạy song song

def _execute_single_tool(
    tool: BaseTool,
    tool_call: dict,
) -> ToolMessage:
    """Thực thi 1 tool call, trả về ToolMessage."""
    call_id = tool_call.get("id", "unknown")
    args    = tool_call.get("args", {}) or {}
    name    = tool_call.get("name", "")

    try:
        result = tool.invoke(args)
        content = str(result) if not isinstance(result, str) else result
    except Exception as e:
        content = f"❌ Tool '{name}' lỗi: {str(e)}"

    return ToolMessage(content=content, tool_call_id=call_id, name=name)


def execute_batch(
    batch: ToolCallBatch,
    tools_by_name: dict[str, BaseTool],
) -> list[ToolMessage]:
    """
    Thực thi 1 batch:
    - Concurrent batch → ThreadPoolExecutor
    - Serial batch     → loop tuần tự
    Kết quả được sắp xếp theo thứ tự input (giống Claude Code).
    """
    results: list[ToolMessage | None] = [None] * len(batch.calls)

    if batch.is_concurrent and len(batch.calls) > 1:
        with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENT_WORKERS, len(batch.calls))) as exe:
            futures = {
                exe.submit(
                    _execute_single_tool,
                    tools_by_name.get(tc.get("name", ""), _noop_tool()),
                    tc,
                ): i
                for i, tc in enumerate(batch.calls)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    tc = batch.calls[idx]
                    results[idx] = ToolMessage(
                        content=f"❌ Concurrent execution error: {str(e)}",
                        tool_call_id=tc.get("id", "unknown"),
                        name=tc.get("name", ""),
                    )
    else:
        # Serial
        for i, tc in enumerate(batch.calls):
            tool = tools_by_name.get(tc.get("name", ""), _noop_tool())
            results[i] = _execute_single_tool(tool, tc)

    return [r for r in results if r is not None]


class _NoopTool(BaseTool):
    name:        str = "_noop"
    description: str = "Tool not found"

    def _run(self, *args, **kwargs):
        return "❌ Tool không tồn tại."

def _noop_tool() -> _NoopTool:
    return _NoopTool()


# ══════════════════════════════════════════════════════════════════════════════
# CONCURRENT TOOL NODE  (Drop-in replacement cho LangGraph ToolNode)
# ══════════════════════════════════════════════════════════════════════════════

def build_concurrent_tool_node(tools: list[BaseTool]):
    """
    Tạo node function thay thế LangGraph ToolNode.
    Node này tự động chạy song song các tools safe, tuần tự các tools exclusive.

    Drop-in replacement:
        # Thay vì:
        tool_node = ToolNode(tools=all_tools)
        # Dùng:
        tool_node = build_concurrent_tool_node(all_tools)
        graph.add_node("tools", tool_node)

    Returns:
        Callable nhận AgentState, trả về {"messages": [ToolMessage, ...]}
    """
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    def concurrent_tool_node(state: dict) -> dict:
        """Node thực thi tools với concurrency partitioning."""
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        # Lấy tool calls từ message AI cuối cùng
        last_ai = None
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                last_ai = msg
                break

        if last_ai is None:
            return {"messages": []}

        tool_calls = last_ai.tool_calls  # list[dict]

        # Partition
        batches = partition_tool_calls(tool_calls)

        # Stats cho logging
        concurrent_count = sum(len(b.calls) for b in batches if b.is_concurrent)
        serial_count     = sum(len(b.calls) for b in batches if not b.is_concurrent)
        total_batches    = len(batches)

        if concurrent_count > 1:
            # Chỉ log khi thực sự có parallel execution
            print(
                f"[orchestrator] {len(tool_calls)} tools → {total_batches} batches "
                f"({concurrent_count} parallel / {serial_count} serial)"
            )

        # Execute
        all_results: list[ToolMessage] = []
        for batch in batches:
            batch_results = execute_batch(batch, tools_by_name)
            all_results.extend(batch_results)

        return {"messages": all_results}

    return concurrent_tool_node


# ══════════════════════════════════════════════════════════════════════════════
# STATS HELPER  (debug/monitoring)
# ══════════════════════════════════════════════════════════════════════════════

def explain_partitioning(tool_calls: list[dict]) -> str:
    """
    Giải thích cách tool_calls sẽ được phân chia thành batches.
    Dùng để debug hoặc logging.

    Ví dụ:
        explain_partitioning([
            {"name": "web_search",          "args": {}},
            {"name": "search_knowledge_base","args": {}},
            {"name": "run_code_in_sandbox", "args": {}},
            {"name": "web_search",          "args": {}},
        ])
        →
        "Batch 1 [CONCURRENT]: web_search, search_knowledge_base
         Batch 2 [SERIAL]:     run_code_in_sandbox
         Batch 3 [CONCURRENT]: web_search"
    """
    batches = partition_tool_calls(tool_calls)
    lines = []
    for i, b in enumerate(batches, 1):
        tag   = "CONCURRENT" if b.is_concurrent else "SERIAL"
        names = ", ".join(tc.get("name", "?") for tc in b.calls)
        lines.append(f"Batch {i} [{tag}]: {names}")
    return "\n".join(lines) if lines else "No tool calls"