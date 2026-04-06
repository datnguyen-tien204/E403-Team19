"""
agent/profile_builder.py
Phân tích lịch sử chat → xây dựng user profile → lưu Weaviate.

Features:
  ✅ Phân tích chủ đề hay hỏi
  ✅ Detect giờ active cao nhất
  ✅ Phát hiện workflow lặp lại
  ✅ Nhận ra pain point thường gặp
  ✅ Lưu profile vào Weaviate để agent đọc khi cần
  ✅ Tự động cập nhật hàng tuần

Schema profile (lưu Weaviate collection "UserProfile"):
  user_id, top_topics, active_hours, repeated_workflows,
  pain_points, preferred_tools, last_updated
"""
from __future__ import annotations

import datetime
import json
import threading
from collections import Counter
from typing import Optional

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from src.rag.chat_history_store import get_sessions, get_session_messages

PROFILE_COLLECTION = "UserProfile"
DEFAULT_USER_ID    = "default"


# ── Collection setup ───────────────────────────────────────────────────────────

def ensure_profile_collection(client: weaviate.WeaviateClient) -> bool:
    try:
        existing = [c.name for c in client.collections.list_all().values()]
        if PROFILE_COLLECTION in existing:
            return True
        client.collections.create(
            PROFILE_COLLECTION,
            properties=[
                Property(name="user_id",              data_type=DataType.TEXT),
                Property(name="top_topics",           data_type=DataType.TEXT),   # JSON list
                Property(name="active_hours",         data_type=DataType.TEXT),   # JSON list[int]
                Property(name="repeated_workflows",   data_type=DataType.TEXT),   # JSON list
                Property(name="pain_points",          data_type=DataType.TEXT),   # JSON list
                Property(name="preferred_tools",      data_type=DataType.TEXT),   # JSON list
                Property(name="daily_patterns",       data_type=DataType.TEXT),   # JSON {weekday: [topics]}
                Property(name="last_updated",         data_type=DataType.TEXT),
                Property(name="total_sessions_analyzed", data_type=DataType.INT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
        )
        print(f"[Profile] Đã tạo collection '{PROFILE_COLLECTION}'")
        return True
    except Exception as e:
        print(f"[Profile] Lỗi tạo collection: {e}")
        return False


# ── Data extraction từ chat history ───────────────────────────────────────────

def _extract_hours(messages: list[dict]) -> list[int]:
    """Trích xuất giờ từ timestamps."""
    hours = []
    for msg in messages:
        ts = msg.get("timestamp", "")
        try:
            dt = datetime.datetime.fromisoformat(ts)
            hours.append(dt.hour)
        except Exception:
            pass
    return hours


def _extract_weekday_pattern(messages: list[dict]) -> dict[str, list[str]]:
    """Phân tích pattern theo ngày trong tuần."""
    weekday_map: dict[str, list[str]] = {
        "Mon": [], "Tue": [], "Wed": [], "Thu": [],
        "Fri": [], "Sat": [], "Sun": [],
    }
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for msg in messages:
        if msg.get("role") != "user":
            continue
        ts = msg.get("timestamp", "")
        content = msg.get("content", "")[:200]
        try:
            dt = datetime.datetime.fromisoformat(ts)
            day_key = days[dt.weekday()]
            weekday_map[day_key].append(content)
        except Exception:
            pass
    return weekday_map


def _build_analysis_prompt(
    user_messages: list[str],
    hours: list[int],
    total_sessions: int,
) -> str:
    """Tạo prompt phân tích cho LLM."""
    # Lấy sample messages để tránh quá dài
    sample = user_messages[:80]
    hour_counter = Counter(hours)
    top_hours = [h for h, _ in hour_counter.most_common(5)]

    messages_text = "\n".join(f"- {m}" for m in sample)

    return f"""Phân tích lịch sử chat sau và trả về JSON (KHÔNG có markdown, KHÔNG có giải thích):

THỐNG KÊ:
- Tổng sessions: {total_sessions}
- Giờ active nhất: {top_hours}
- Số tin nhắn phân tích: {len(sample)}

TIN NHẮN NGƯỜI DÙNG (sample):
{messages_text}

Trả về JSON với format sau (KHÔNG thêm bất kỳ text nào ngoài JSON):
{{
  "top_topics": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "repeated_workflows": [
    "Mô tả workflow lặp lại 1",
    "Mô tả workflow lặp lại 2"
  ],
  "pain_points": [
    "Pain point 1 hay gặp",
    "Pain point 2 hay gặp"
  ],
  "preferred_tools": ["web_search", "search_knowledge_base"],
  "behavior_summary": "Tóm tắt ngắn về hành vi và thói quen người dùng (2-3 câu)"
}}

Chú ý:
- top_topics: các chủ đề hay được hỏi nhất (tiếng Việt)
- repeated_workflows: quy trình làm việc thấy lặp lại nhiều lần
- pain_points: vấn đề hay gặp phải, hay stuck
- preferred_tools: tools hay được dùng/hỏi đến"""


# ── LLM analysis ──────────────────────────────────────────────────────────────

def _analyze_with_llm(prompt: str) -> Optional[dict]:
    """Dùng Groq LLM để phân tích pattern."""
    try:
        from langchain_groq import ChatGroq
        from config import GROQ_API_KEY, GROQ_MODEL

        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.2,
            max_tokens=1500,
        )
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown nếu có
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[Profile] LLM trả về JSON không hợp lệ: {e}")
        return None
    except Exception as e:
        print(f"[Profile] LLM analysis error: {e}")
        return None


# ── Save / Load profile ────────────────────────────────────────────────────────

def save_profile(client: weaviate.WeaviateClient, profile: dict) -> bool:
    """Lưu hoặc cập nhật profile vào Weaviate."""
    try:
        ensure_profile_collection(client)
        col = client.collections.get(PROFILE_COLLECTION)
        user_id = profile.get("user_id", DEFAULT_USER_ID)

        # Xóa profile cũ nếu có
        col.data.delete_many(
            where=Filter.by_property("user_id").equal(user_id)
        )

        col.data.insert({
            "user_id":                 user_id,
            "top_topics":              json.dumps(profile.get("top_topics", []), ensure_ascii=False),
            "active_hours":            json.dumps(profile.get("active_hours", []), ensure_ascii=False),
            "repeated_workflows":      json.dumps(profile.get("repeated_workflows", []), ensure_ascii=False),
            "pain_points":             json.dumps(profile.get("pain_points", []), ensure_ascii=False),
            "preferred_tools":         json.dumps(profile.get("preferred_tools", []), ensure_ascii=False),
            "daily_patterns":          json.dumps(profile.get("daily_patterns", {}), ensure_ascii=False),
            "last_updated":            datetime.datetime.now().isoformat(),
            "total_sessions_analyzed": profile.get("total_sessions_analyzed", 0),
        })
        return True
    except Exception as e:
        print(f"[Profile] Lỗi lưu profile: {e}")
        return False


def load_profile(
    client: weaviate.WeaviateClient,
    user_id: str = DEFAULT_USER_ID,
) -> Optional[dict]:
    """Đọc profile từ Weaviate."""
    try:
        ensure_profile_collection(client)
        col = client.collections.get(PROFILE_COLLECTION)
        results = col.query.fetch_objects(
            filters=Filter.by_property("user_id").equal(user_id),
            limit=1,
        )
        if not results.objects:
            return None

        p = results.objects[0].properties

        def _safe_json(val):
            try:
                return json.loads(val) if val else []
            except Exception:
                return []

        return {
            "user_id":                 p.get("user_id", user_id),
            "top_topics":              _safe_json(p.get("top_topics")),
            "active_hours":            _safe_json(p.get("active_hours")),
            "repeated_workflows":      _safe_json(p.get("repeated_workflows")),
            "pain_points":             _safe_json(p.get("pain_points")),
            "preferred_tools":         _safe_json(p.get("preferred_tools")),
            "daily_patterns":          _safe_json(p.get("daily_patterns")) or {},
            "last_updated":            p.get("last_updated", ""),
            "total_sessions_analyzed": p.get("total_sessions_analyzed", 0),
        }
    except Exception as e:
        print(f"[Profile] Lỗi đọc profile: {e}")
        return None


# ── Main builder ───────────────────────────────────────────────────────────────

def build_user_profile(
    client: weaviate.WeaviateClient,
    user_id: str = DEFAULT_USER_ID,
    max_sessions: int = 50,
) -> Optional[dict]:
    """
    Phân tích lịch sử chat và xây dựng user profile.

    Args:
        client:       Weaviate client
        user_id:      ID người dùng
        max_sessions: số sessions tối đa để phân tích

    Returns:
        dict profile hoặc None nếu lỗi
    """
    print(f"[Profile] Bắt đầu build profile cho '{user_id}'...")

    # 1. Lấy danh sách sessions
    sessions = get_sessions(client, limit=max_sessions)
    if not sessions:
        print("[Profile] Không có lịch sử chat để phân tích.")
        return None

    print(f"[Profile] Phân tích {len(sessions)} sessions...")

    # 2. Gom tất cả messages
    all_messages: list[dict] = []
    for session in sessions:
        msgs = get_session_messages(client, session["session_id"])
        all_messages.extend(msgs)

    # 3. Tách user messages để phân tích
    user_messages = [
        m.get("content", "")[:300]
        for m in all_messages
        if m.get("role") == "user" and m.get("content")
    ]

    if not user_messages:
        print("[Profile] Không có tin nhắn người dùng để phân tích.")
        return None

    # 4. Thống kê giờ active
    hours = _extract_hours(all_messages)
    hour_counter = Counter(hours)
    top_active_hours = [h for h, _ in hour_counter.most_common(5)]

    # 5. Pattern theo ngày trong tuần
    daily_patterns = _extract_weekday_pattern(all_messages)
    # Giữ tối đa 5 messages mẫu mỗi ngày
    daily_patterns = {
        day: msgs[:5] for day, msgs in daily_patterns.items() if msgs
    }

    # 6. LLM phân tích patterns
    prompt = _build_analysis_prompt(user_messages, hours, len(sessions))
    llm_result = _analyze_with_llm(prompt)

    if not llm_result:
        # Fallback: phân tích đơn giản không dùng LLM
        llm_result = {
            "top_topics": ["general"],
            "repeated_workflows": [],
            "pain_points": [],
            "preferred_tools": [],
            "behavior_summary": f"Đã phân tích {len(sessions)} sessions.",
        }

    # 7. Ghép kết quả
    profile = {
        "user_id":                 user_id,
        "top_topics":              llm_result.get("top_topics", []),
        "active_hours":            top_active_hours,
        "repeated_workflows":      llm_result.get("repeated_workflows", []),
        "pain_points":             llm_result.get("pain_points", []),
        "preferred_tools":         llm_result.get("preferred_tools", []),
        "daily_patterns":          daily_patterns,
        "behavior_summary":        llm_result.get("behavior_summary", ""),
        "total_sessions_analyzed": len(sessions),
    }

    # 8. Lưu vào Weaviate
    if save_profile(client, profile):
        print(f"[Profile] ✅ Profile saved — {len(sessions)} sessions, topics: {profile['top_topics'][:3]}")
    else:
        print("[Profile] ⚠️ Không lưu được profile.")

    return profile


def get_profile_summary(profile: dict) -> str:
    """
    Tạo tóm tắt ngắn từ profile để inject vào system prompt của agent.
    """
    if not profile:
        return ""

    topics   = ", ".join(profile.get("top_topics", [])[:5]) or "chưa rõ"
    hours    = profile.get("active_hours", [])
    hour_str = f"{min(hours)}h–{max(hours)}h" if len(hours) >= 2 else str(hours)
    workflows = profile.get("repeated_workflows", [])
    pains    = profile.get("pain_points", [])

    lines = [
        f"## PROFILE NGƯỜI DÙNG (tự động học từ lịch sử)",
        f"- Hay hỏi về: {topics}",
        f"- Active nhất lúc: {hour_str}",
    ]
    if workflows:
        lines.append(f"- Workflow lặp lại: {'; '.join(workflows[:2])}")
    if pains:
        lines.append(f"- Hay gặp khó: {'; '.join(pains[:2])}")

    summary = profile.get("behavior_summary", "")
    if summary:
        lines.append(f"- Ghi chú: {summary}")

    return "\n".join(lines)
