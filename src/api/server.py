"""
api/server.py
FastAPI wrapper cho AI Agent.

Endpoints:
  POST /chat              – gửi text, nhận text reply
  POST /chat/audio        – upload audio file, nhận text reply + TTS base64
  WS   /ws/chat           – WebSocket real-time streaming
  GET  /health            – health check
  POST /rag/ingest        – nạp tài liệu vào Weaviate
  GET  /tools             – liệt kê tools có sẵn

  ─── Chat History ───────────────────────────────────────
  GET  /sessions                 – danh sách session
  GET  /sessions/{id}/messages   – tin nhắn trong session
  DELETE /sessions/{id}          – xóa session
  GET  /mode | POST /mode        – search mode
"""
import io
import base64
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TTS_LANG, API_HOST, API_PORT, SEARCH_MODE_DEFAULT
from rag.weaviate_store import get_weaviate_client, get_vector_store, ingest_text_files
from rag.chat_history_store import (
    ensure_chat_collection, save_turn, get_sessions,
    get_session_messages, delete_session, search_chat_history,
)
from tools.all_tools import build_toolset
from agent.graph import build_agent
from tts.speaker import text_to_speech
from agent.state import SearchModeState


# ── Shared state ───────────────────────────────────────────────────────────────
class AppState:
    agent = None
    chat_history: list = []
    weaviate_client = None
    mode_state = SearchModeState(SEARCH_MODE_DEFAULT)
    # Session tracking
    current_session_id: str = ""
    current_session_name: str = ""
    current_turn_index: int = 0
    history_enabled: bool = False  # True nếu Weaviate khả dụng


state = AppState()


def extract_reply(messages) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            if isinstance(msg.content, str):
                return msg.content
            if isinstance(msg.content, list):
                parts = [p.get("text", "") for p in msg.content if isinstance(p, dict)]
                return " ".join(parts).strip()
    return ""


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Khởi tạo agent...")
    try:
        state.weaviate_client = get_weaviate_client()
        vs = get_vector_store(state.weaviate_client)
        all_tools = build_toolset(vs)
        # Tạo collection lịch sử chat
        if ensure_chat_collection(state.weaviate_client):
            state.history_enabled = True
            print("[API] Chat history storage: OK")
        print("[API] Weaviate OK")
    except Exception as e:
        print(f"[API] Weaviate không khả dụng: {e}. Chạy không có RAG.")
        all_tools = build_toolset(None)

    state.agent = build_agent(all_tools, mode_getter=state.mode_state.get_mode)
    print("[API] Agent sẵn sàng!")
    yield

    if state.weaviate_client:
        try:
            state.weaviate_client.close()
        except Exception:
            pass


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Agent API",
    description="LangGraph + Groq + Weaviate RAG + Sandbox + System Control",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "gui", "web"))
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def root_page():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "UI not found"}


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    tts: bool = False
    clear_history: bool = False
    session_id: Optional[str] = None
    session_name: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    audio_base64: Optional[str] = None

class IngestRequest(BaseModel):
    texts: list[str] = []
    file_paths: list[str] = []

class IngestResponse(BaseModel):
    ingested: int
    message: str

class ModeRequest(BaseModel):
    mode: str


# ── Helpers ────────────────────────────────────────────────────────────────────
async def run_agent(user_input: str) -> str:
    state.chat_history.append(HumanMessage(content=user_input))
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: state.agent.invoke({"messages": state.chat_history})
    )
    reply = extract_reply(result["messages"])
    state.chat_history = list(result["messages"])
    return reply


def make_tts_base64(text: str) -> Optional[str]:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=TTS_LANG, slow=False).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return None


def _save_turn_async(session_id: str, session_name: str, user_msg: str, ai_msg: str, turn_idx: int):
    """Lưu turn vào Weaviate trong background thread."""
    if not state.history_enabled or not state.weaviate_client:
        return
    try:
        save_turn(state.weaviate_client, session_id, session_name, user_msg, ai_msg, turn_idx)
    except Exception as e:
        print(f"[ChatHistory] Lỗi lưu background: {e}")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_ready": state.agent is not None,
        "search_mode": state.mode_state.get_mode(),
        "history_enabled": state.history_enabled,
    }


@app.get("/mode")
async def get_mode():
    return {"mode": state.mode_state.get_mode()}


@app.post("/mode")
async def set_mode(req: ModeRequest):
    state.mode_state.set_mode(req.mode)
    return {"mode": state.mode_state.get_mode()}


@app.get("/tools")
async def list_tools():
    if not state.agent:
        raise HTTPException(503, "Agent chưa sẵn sàng")
    tool_names = []
    try:
        from langgraph.prebuilt import ToolNode
        for node_name, node in state.agent.nodes.items():
            if isinstance(node, ToolNode):
                tool_names = [t.name for t in node.tools_by_name.values()]
    except Exception:
        tool_names = ["Không lấy được danh sách"]
    return {"tools": tool_names}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not state.agent:
        raise HTTPException(503, "Agent chưa sẵn sàng")
    if req.clear_history:
        state.chat_history = []
        state.current_turn_index = 0
    reply = await run_agent(req.message)
    audio_b64 = make_tts_base64(reply) if req.tts else None

    # Auto-save nếu có session
    if req.session_id:
        sid = req.session_id
        sname = req.session_name or req.message[:60]
        import threading
        threading.Thread(
            target=_save_turn_async,
            args=(sid, sname, req.message, reply, state.current_turn_index),
            daemon=True,
        ).start()
        state.current_turn_index += 1

    return ChatResponse(reply=reply, audio_base64=audio_b64)


@app.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(file: UploadFile = File(...), tts: bool = True):
    if not state.agent:
        raise HTTPException(503, "Agent chưa sẵn sàng")
    content = await file.read()
    buf = io.BytesIO(content)
    try:
        audio_np, sr = sf.read(buf, dtype="float32")
    except Exception as e:
        raise HTTPException(400, f"Không đọc được file audio: {e}")

    from config import STT_SAMPLE_RATE
    if sr != STT_SAMPLE_RATE:
        try:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=STT_SAMPLE_RATE)
        except ImportError:
            pass

    from stt.listener import transcribe
    text = transcribe(audio_np)
    if not text:
        raise HTTPException(400, "Không nhận dạng được giọng nói.")

    reply = await run_agent(text)
    audio_b64 = make_tts_base64(reply) if tts else None
    return ChatResponse(reply=f"[Bạn nói: {text}]\n{reply}", audio_base64=audio_b64)


@app.post("/rag/ingest", response_model=IngestResponse)
async def rag_ingest(req: IngestRequest):
    if not state.weaviate_client:
        raise HTTPException(503, "Weaviate không khả dụng")
    from rag.weaviate_store import get_vector_store, ingest_documents
    from langchain_core.documents import Document
    vs = get_vector_store(state.weaviate_client)
    total = 0
    if req.texts:
        docs = [Document(page_content=t, metadata={"source": "api"}) for t in req.texts]
        ingest_documents(vs, docs)
        total += len(docs)
    if req.file_paths:
        total += ingest_text_files(vs, req.file_paths)
    state.mode_state.set_mode("hybrid")
    return IngestResponse(ingested=total, message=f"Đã ingest {total} documents. search_mode=hybrid")


# ── Session History Endpoints ──────────────────────────────────────────────────

@app.get("/sessions")
async def list_sessions():
    """Lấy danh sách tất cả session chat."""
    if not state.history_enabled or not state.weaviate_client:
        return {"sessions": [], "enabled": False}
    sessions = get_sessions(state.weaviate_client)
    return {"sessions": sessions, "enabled": True}


@app.get("/sessions/{session_id}/messages")
async def get_session(session_id: str):
    """Lấy toàn bộ tin nhắn của 1 session."""
    if not state.history_enabled or not state.weaviate_client:
        raise HTTPException(503, "Chat history không khả dụng")
    messages = get_session_messages(state.weaviate_client, session_id)
    return {"session_id": session_id, "messages": messages}


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """Xóa 1 session."""
    if not state.history_enabled or not state.weaviate_client:
        raise HTTPException(503, "Chat history không khả dụng")
    count = delete_session(state.weaviate_client, session_id)
    return {"deleted": count, "session_id": session_id}


@app.get("/sessions/search/{query}")
async def search_history(query: str, top_k: int = 5):
    """Semantic search lịch sử chat."""
    if not state.history_enabled or not state.weaviate_client:
        return {"results": [], "enabled": False}
    results = search_chat_history(state.weaviate_client, query, top_k=top_k)
    return {"results": results}


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    print("[WS] Client kết nối")

    # Per-connection state
    local_session_id: str = ""
    local_session_name: str = ""
    local_turn_index: int = 0

    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action", "chat")

            if action == "clear":
                state.chat_history = []
                local_turn_index = 0
                await ws.send_json({"type": "system", "content": "Đã xóa lịch sử."})
                continue

            if action == "mode":
                state.mode_state.set_mode(data.get("mode", ""))
                await ws.send_json({"type": "mode", "content": state.mode_state.get_mode()})
                continue

            if action == "set_session":
                # Client thông báo session ID + name khi đổi session
                local_session_id = data.get("session_id", "")
                local_session_name = data.get("session_name", "")
                local_turn_index = data.get("turn_index", 0)
                state.chat_history = []  # reset in-memory history khi đổi session
                await ws.send_json({"type": "system", "content": f"Session: {local_session_id[:8]}..."})
                continue

            message = data.get("message", "").strip()
            if not message:
                continue

            # Lấy session info từ message (nếu client gửi kèm)
            msg_session_id = data.get("session_id", local_session_id)
            msg_session_name = data.get("session_name", local_session_name or message[:60])

            if msg_session_id:
                local_session_id = msg_session_id
            if msg_session_name:
                local_session_name = msg_session_name

            await ws.send_json({"type": "thinking", "content": "⏳ Đang suy nghĩ..."})

            from langchain_core.messages import HumanMessage, AIMessage
            import time as _time
            from datetime import datetime, timezone

            def _ts() -> str:
                return datetime.now(timezone.utc).isoformat()

            local_history = list(state.chat_history)
            local_history.append(HumanMessage(content=message))
            await ws.send_json({"type": "start"})

            # ── Hook sandbox output → WebSocket ──────────────────
            loop = asyncio.get_event_loop()

            def sandbox_output_cb(line: str):
                try:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_json({"type": "sandbox_output", "content": line}),
                        loop,
                    )
                except Exception:
                    pass

            from tools.sandbox import set_output_callback
            set_output_callback(sandbox_output_cb)

            # ── Stream agent ──────────────────────────────────────
            full_reply = ""
            think_buffer = ""
            in_think = False
            _agent_start = _time.time()
            _step = 0
            _llm_prompt_tokens = 0
            _llm_completion_tokens = 0

            # Emit AGENT_START
            await ws.send_json({"type": "thinking_log", "event": "AGENT_START", "ts": _ts(),
                                 "data": {"input_preview": message[:120],
                                          "mode": state.mode_state.get_mode()}})

            try:
                for msg_chunk, metadata in state.agent.stream(
                    {"messages": local_history},
                    stream_mode="messages",
                ):
                    node = metadata.get("langgraph_node")

                    if node == "llm":
                        raw = msg_chunk.content if isinstance(msg_chunk.content, str) else ""

                        # Emit LLM_STEP khi nhận được chunk đầu tiên của mỗi bước llm
                        usage = getattr(msg_chunk, "usage_metadata", None) or {}
                        if usage:
                            pt = usage.get("input_tokens", 0)
                            ct = usage.get("output_tokens", 0)
                            if pt or ct:
                                _llm_prompt_tokens += pt
                                _llm_completion_tokens += ct
                                lat = int((_time.time() - _agent_start) * 1000)
                                await ws.send_json({"type": "thinking_log", "event": "LLM_METRIC", "ts": _ts(),
                                                    "data": {"prompt_tokens": _llm_prompt_tokens,
                                                             "completion_tokens": _llm_completion_tokens,
                                                             "total_tokens": _llm_prompt_tokens + _llm_completion_tokens,
                                                             "latency_ms": lat}})

                        if not raw:
                            continue

                        i = 0
                        while i < len(raw):
                            if not in_think:
                                think_start = raw.find("<think>", i)
                                if think_start == -1:
                                    chunk = raw[i:]
                                    if chunk:
                                        full_reply += chunk
                                        await ws.send_json({"type": "token", "content": chunk})
                                    break
                                else:
                                    before = raw[i:think_start]
                                    if before:
                                        full_reply += before
                                        await ws.send_json({"type": "token", "content": before})
                                    in_think = True
                                    i = think_start + len("<think>")
                            else:
                                think_end = raw.find("</think>", i)
                                if think_end == -1:
                                    think_buffer += raw[i:]
                                    await ws.send_json({"type": "thinking_token", "content": raw[i:]})
                                    break
                                else:
                                    think_buffer += raw[i:think_end]
                                    await ws.send_json({"type": "thinking_token", "content": raw[i:think_end]})
                                    await ws.send_json({"type": "thinking_done", "content": think_buffer})
                                    think_buffer = ""
                                    in_think = False
                                    i = think_end + len("</think>")

                    elif node == "tools":
                        _step += 1
                        tool_name = (
                            getattr(msg_chunk, "name", None)
                            or getattr(msg_chunk, "tool", None)
                            or metadata.get("tool_name", "tool")
                        )
                        preview = getattr(msg_chunk, "content", "")
                        preview_str = str(preview)[:300] if preview else ""
                        await ws.send_json({
                            "type": "tool",
                            "tool": tool_name,
                            "content": preview_str,
                        })
                        # Emit TOOL_CALL log
                        await ws.send_json({"type": "thinking_log", "event": "TOOL_CALL", "ts": _ts(),
                                            "data": {"step": _step, "tool": tool_name,
                                                     "result_preview": preview_str[:200]}})

                # Emit AGENT_END
                elapsed_ms = int((_time.time() - _agent_start) * 1000)
                await ws.send_json({"type": "thinking_log", "event": "AGENT_END", "ts": _ts(),
                                    "data": {"steps": _step, "elapsed_ms": elapsed_ms,
                                             "total_tokens": _llm_prompt_tokens + _llm_completion_tokens}})

                # Cập nhật in-memory history
                state.chat_history = local_history + [AIMessage(content=full_reply)]

                # ── Auto-save lịch sử vào Weaviate ──────────────────
                if local_session_id and full_reply:
                    import threading
                    threading.Thread(
                        target=_save_turn_async,
                        args=(local_session_id, local_session_name, message, full_reply, local_turn_index),
                        daemon=True,
                    ).start()
                    local_turn_index += 1

                await ws.send_json({
                    "type": "done",
                    "content": full_reply,
                    "turn_index": local_turn_index,
                })

            except Exception as stream_err:
                await ws.send_json({"type": "error", "content": str(stream_err)})
            finally:
                set_output_callback(None)

    except WebSocketDisconnect:
        print("[WS] Client ngắt kết nối")
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=False)