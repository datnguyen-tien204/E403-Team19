"""
rag/chat_history_store.py
Lưu lịch sử chat vào Weaviate + semantic search (RAG) qua lịch sử.

Features:
  ✅ Tự động tạo collection ChatHistory nếu chưa có
  ✅ Lưu từng turn (user + assistant) kèm embedding
  ✅ Semantic search lịch sử qua near_vector
  ✅ Quản lý sessions: list / get / delete
"""

from __future__ import annotations

import datetime
import threading
from functools import lru_cache
from typing import Optional

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

CHAT_COLLECTION = "ChatHistory"

# ── Embedding model (singleton) ────────────────────────────────────────────────

_embed_lock = threading.Lock()
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        with _embed_lock:
            if _embedder is None:
                from langchain_huggingface import HuggingFaceEmbeddings
                _embedder = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
    return _embedder


def _embed(text: str) -> list[float]:
    try:
        return _get_embedder().embed_query(text)
    except Exception:
        return []


# ── Collection setup ───────────────────────────────────────────────────────────

def ensure_chat_collection(client: weaviate.WeaviateClient) -> bool:
    """
    Tạo collection ChatHistory trong Weaviate nếu chưa tồn tại.
    Trả về True nếu thành công.
    """
    try:
        existing = [c.name for c in client.collections.list_all().values()]
        if CHAT_COLLECTION in existing:
            return True

        client.collections.create(
            CHAT_COLLECTION,
            properties=[
                Property(name="session_id",   data_type=DataType.TEXT),
                Property(name="session_name", data_type=DataType.TEXT),
                Property(name="role",         data_type=DataType.TEXT),   # "user" | "assistant"
                Property(name="content",      data_type=DataType.TEXT),
                Property(name="timestamp",    data_type=DataType.TEXT),
                Property(name="turn_index",   data_type=DataType.INT),
            ],
            # Dùng embedding thủ công (near_vector), không dùng module tự động
            vectorizer_config=Configure.Vectorizer.none(),
        )
        print(f"[ChatHistory] Đã tạo collection '{CHAT_COLLECTION}'")
        return True
    except Exception as e:
        print(f"[ChatHistory] Lỗi tạo collection: {e}")
        return False


# ── Save ───────────────────────────────────────────────────────────────────────

def save_turn(
    client: weaviate.WeaviateClient,
    session_id: str,
    session_name: str,
    user_msg: str,
    ai_msg: str,
    turn_index: int,
) -> bool:
    """
    Lưu 1 turn (user + assistant) vào Weaviate.
    Embed nội dung để cho phép semantic search sau này.
    """
    try:
        col = client.collections.get(CHAT_COLLECTION)
        ts = datetime.datetime.now().isoformat()

        # Embed combined text for richer semantic search
        combined = f"User: {user_msg}\nAssistant: {ai_msg}"
        vector = _embed(combined)

        base = dict(
            session_id=session_id,
            session_name=session_name,
            timestamp=ts,
            turn_index=turn_index,
        )

        insert_kwargs_user = dict(
            properties={**base, "role": "user", "content": user_msg},
        )
        insert_kwargs_ai = dict(
            properties={**base, "role": "assistant", "content": ai_msg},
        )
        if vector:
            insert_kwargs_ai["vector"] = vector  # chỉ embed AI reply để search

        col.data.insert(**insert_kwargs_user)
        col.data.insert(**insert_kwargs_ai)
        return True

    except Exception as e:
        print(f"[ChatHistory] Lỗi lưu turn: {e}")
        return False


# ── Query sessions ─────────────────────────────────────────────────────────────

def get_sessions(client: weaviate.WeaviateClient, limit: int = 100) -> list[dict]:
    """
    Lấy danh sách session, sắp xếp theo last_timestamp giảm dần.
    Trả về: [{session_id, session_name, last_timestamp, message_count}]
    """
    try:
        col = client.collections.get(CHAT_COLLECTION)
        results = col.query.fetch_objects(
            limit=2000,
            return_properties=["session_id", "session_name", "timestamp"],
        )

        sessions: dict[str, dict] = {}
        for obj in results.objects:
            p = obj.properties
            sid = p.get("session_id", "")
            ts  = p.get("timestamp", "")
            if not sid:
                continue
            if sid not in sessions:
                sessions[sid] = {
                    "session_id":   sid,
                    "session_name": p.get("session_name", "Cuộc trò chuyện"),
                    "last_timestamp": ts,
                    "message_count": 0,
                }
            sessions[sid]["message_count"] += 1
            if ts > sessions[sid]["last_timestamp"]:
                sessions[sid]["last_timestamp"] = ts

        return sorted(sessions.values(), key=lambda x: x["last_timestamp"], reverse=True)[:limit]

    except Exception as e:
        print(f"[ChatHistory] Lỗi get_sessions: {e}")
        return []


def get_session_messages(client: weaviate.WeaviateClient, session_id: str) -> list[dict]:
    """
    Lấy toàn bộ tin nhắn của 1 session, sắp xếp theo turn_index.
    Trả về: [{role, content, timestamp, turn_index}]
    """
    try:
        col = client.collections.get(CHAT_COLLECTION)
        results = col.query.fetch_objects(
            filters=Filter.by_property("session_id").equal(session_id),
            limit=2000,
            return_properties=["role", "content", "timestamp", "turn_index"],
        )
        msgs = [obj.properties for obj in results.objects]
        return sorted(msgs, key=lambda x: (x.get("turn_index", 0), x.get("role", "")))

    except Exception as e:
        print(f"[ChatHistory] Lỗi get_session_messages: {e}")
        return []


def delete_session(client: weaviate.WeaviateClient, session_id: str) -> int:
    """Xóa toàn bộ tin nhắn của session. Trả về số bản ghi đã xóa."""
    try:
        col = client.collections.get(CHAT_COLLECTION)
        result = col.data.delete_many(
            where=Filter.by_property("session_id").equal(session_id)
        )
        return result.successful if hasattr(result, "successful") else 0
    except Exception as e:
        print(f"[ChatHistory] Lỗi delete_session: {e}")
        return 0


# ── Semantic search ────────────────────────────────────────────────────────────

def search_chat_history(
    client: weaviate.WeaviateClient,
    query: str,
    top_k: int = 5,
    session_id: Optional[str] = None,
) -> list[dict]:
    """
    Semantic search lịch sử chat qua near_vector.
    Nếu session_id được cung cấp, chỉ search trong session đó.
    Trả về: [{role, content, timestamp, session_name, score}]
    """
    try:
        vector = _embed(query)
        if not vector:
            return []

        col = client.collections.get(CHAT_COLLECTION)

        kwargs: dict = dict(
            near_vector=vector,
            limit=top_k,
            return_properties=["role", "content", "timestamp", "session_id", "session_name"],
            return_metadata=["distance"],
        )

        if session_id:
            kwargs["filters"] = Filter.by_property("session_id").equal(session_id)
        # Chỉ search assistant messages (đã có embedding)
        else:
            kwargs["filters"] = Filter.by_property("role").equal("assistant")

        results = col.query.near_vector(**kwargs)

        out = []
        for obj in results.objects:
            p = obj.properties
            score = 1 - (obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 1)
            out.append({
                "role":         p.get("role", ""),
                "content":      p.get("content", ""),
                "timestamp":    p.get("timestamp", ""),
                "session_id":   p.get("session_id", ""),
                "session_name": p.get("session_name", ""),
                "score":        round(score, 3),
            })
        return out

    except Exception as e:
        print(f"[ChatHistory] Lỗi search: {e}")
        return []