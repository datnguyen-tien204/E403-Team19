"""
agent/briefing.py
Tạo morning briefing cá nhân hoá dựa trên user profile.

Features:
  ✅ Tóm tắt tin tức theo chủ đề người dùng hay quan tâm
  ✅ Đọc Google Calendar (nếu có MCP) → nhắc lịch hôm nay
  ✅ Đọc Gmail unread (nếu có MCP) → tóm tắt email quan trọng
  ✅ Đọc file Excel/Sheet tracking (nếu có MCP)
  ✅ Gợi ý task dựa trên pattern ngày hôm nay
  ✅ Deliver qua TTS hoặc in ra console
"""
from __future__ import annotations

import datetime
import json
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import GROQ_API_KEY, GROQ_MODEL


# ── Briefing builder ──────────────────────────────────────────────────────────

class MorningBriefing:
    """
    Tạo morning briefing cá nhân hoá.

    Cách dùng:
        briefing = MorningBriefing(agent, weaviate_client)
        text = await briefing.generate()
        briefing.deliver(text)
    """

    def __init__(self, agent, weaviate_client=None):
        self.agent          = agent          # compiled LangGraph agent
        self.weaviate_client = weaviate_client
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.5,
            max_tokens=2000,
        )

    # ── Data collectors ────────────────────────────────────────────────────────

    def _load_profile(self) -> dict:
        """Đọc user profile từ Weaviate."""
        if not self.weaviate_client:
            return {}
        try:
            from src.agent.profile_builder import load_profile
            return load_profile(self.weaviate_client) or {}
        except Exception as e:
            print(f"[Briefing] load_profile error: {e}")
            return {}

    def _get_todays_pattern(self, profile: dict) -> list[str]:
        """Lấy pattern hôm nay là ngày gì → gợi ý liên quan."""
        today = datetime.datetime.now().strftime("%a")  # "Mon", "Tue"...
        daily = profile.get("daily_patterns", {})
        return daily.get(today, [])

    def _search_news(self, topics: list[str]) -> str:
        """Tìm tin tức theo topics từ profile."""
        if not topics:
            return "Không có chủ đề theo dõi."

        # Chỉ lấy top 3 topics để tránh quá nhiều search
        top_topics = topics[:3]
        results = []

        for topic in top_topics:
            try:
                from langchain_core.messages import HumanMessage
                query = f"tin tức mới nhất về {topic} hôm nay"
                response = self.agent.invoke({
                    "messages": [HumanMessage(content=f"Tìm kiếm nhanh: {query}. Tóm tắt trong 3 câu.")]
                })
                last_msg = response["messages"][-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    results.append(f"**{topic.upper()}**: {last_msg.content[:400]}")
            except Exception as e:
                results.append(f"**{topic}**: Không lấy được tin ({e})")

        return "\n\n".join(results) if results else "Không tìm được tin tức."

    def _get_calendar_events(self) -> str:
        """Đọc lịch hôm nay qua MCP (nếu có)."""
        try:
            from langchain_core.messages import HumanMessage
            today = datetime.datetime.now().strftime("%d/%m/%Y")
            response = self.agent.invoke({
                "messages": [HumanMessage(
                    content=f"Dùng mcp_google_calendar tool để lấy các sự kiện ngày {today}. "
                            "Nếu không có tool, trả lời 'Không có calendar tool'."
                )]
            })
            last = response["messages"][-1]
            content = last.content if hasattr(last, "content") else ""

            if "không có" in content.lower() or "no tool" in content.lower():
                return "Chưa kết nối Google Calendar."
            return content[:500]
        except Exception as e:
            return f"Không đọc được calendar: {e}"

    def _get_unread_emails(self) -> str:
        """Tóm tắt email chưa đọc qua MCP (nếu có)."""
        try:
            from langchain_core.messages import HumanMessage
            response = self.agent.invoke({
                "messages": [HumanMessage(
                    content="Dùng mcp_gmail tool để lấy 5 email chưa đọc quan trọng nhất. "
                            "Tóm tắt ngắn gọn từng email. "
                            "Nếu không có tool, trả lời 'Không có gmail tool'."
                )]
            })
            last = response["messages"][-1]
            content = last.content if hasattr(last, "content") else ""
            if "không có" in content.lower():
                return "Chưa kết nối Gmail."
            return content[:600]
        except Exception as e:
            return f"Không đọc được email: {e}"

    def _suggest_tasks(self, profile: dict, todays_pattern: list[str]) -> str:
        """Gợi ý task hôm nay dựa trên profile và pattern."""
        if not profile:
            return "Chưa có đủ dữ liệu để gợi ý task."

        today_name = datetime.datetime.now().strftime("%A")  # "Monday"
        topics     = profile.get("top_topics", [])
        workflows  = profile.get("repeated_workflows", [])
        pains      = profile.get("pain_points", [])

        prompt = f"""Dựa vào thông tin sau, gợi ý 3-5 task cụ thể cho người dùng hôm nay ({today_name}).
Trả lời bằng tiếng Việt, ngắn gọn, dạng bullet list.

Thông tin:
- Chủ đề hay làm: {', '.join(topics[:5])}
- Workflow lặp lại: {'; '.join(workflows[:3])}
- Hay gặp khó: {'; '.join(pains[:3])}
- Hôm nay ({today_name}): {'; '.join(todays_pattern[:3]) if todays_pattern else 'không có pattern đặc biệt'}

Chỉ gợi ý task thực tế, liên quan trực tiếp đến thói quen đã biết. KHÔNG bịa task."""

        try:
            response = self.llm.invoke([
                SystemMessage(content="Bạn là trợ lý gợi ý công việc dựa trên thói quen thực tế."),
                HumanMessage(content=prompt),
            ])
            return response.content.strip()
        except Exception as e:
            return f"Không tạo được gợi ý: {e}"

    # ── Generate briefing ──────────────────────────────────────────────────────

    def generate(self, include_news: bool = True) -> str:
        """
        Tạo toàn bộ morning briefing.

        Args:
            include_news: có tìm tin tức không (tắt để test nhanh)

        Returns:
            str: briefing text hoàn chỉnh
        """
        now    = datetime.datetime.now()
        date_str = now.strftime("%A, %d/%m/%Y %H:%M")

        print(f"[Briefing] Đang tạo briefing lúc {date_str}...")

        # Load profile
        profile         = self._load_profile()
        todays_pattern  = self._get_todays_pattern(profile)
        topics          = profile.get("top_topics", [])

        sections: list[str] = []

        # ── Header ──
        greeting = _get_greeting(now.hour)
        sections.append(f"# 🌅 {greeting}!\n📅 {date_str}")

        # ── Calendar ──
        print("[Briefing] Đọc calendar...")
        calendar = self._get_calendar_events()
        sections.append(f"## 📆 Lịch hôm nay\n{calendar}")

        # ── Email ──
        print("[Briefing] Đọc email...")
        emails = self._get_unread_emails()
        sections.append(f"## 📧 Email chưa đọc\n{emails}")

        # ── Tin tức (tuỳ chọn) ──
        if include_news and topics:
            print(f"[Briefing] Tìm tin tức: {topics[:3]}...")
            news = self._search_news(topics)
            sections.append(f"## 📰 Tin tức theo dõi\n{news}")

        # ── Task gợi ý ──
        print("[Briefing] Tạo gợi ý task...")
        tasks = self._suggest_tasks(profile, todays_pattern)
        sections.append(f"## ✅ Gợi ý task hôm nay\n{tasks}")

        # ── Profile reminder ──
        if profile:
            active = profile.get("active_hours", [])
            if active:
                peak = max(set(active), key=active.count) if active else None
                if peak:
                    sections.append(
                        f"## 💡 Nhắc nhở\nBạn thường productive nhất lúc **{peak}:00**. "
                        f"Sắp xếp task quan trọng vào khung giờ này."
                    )

        briefing_text = "\n\n".join(sections)
        print("[Briefing] ✅ Briefing ready!")
        return briefing_text

    # ── Delivery ──────────────────────────────────────────────────────────────

    def deliver(self, text: str, method: str = "print"):
        """
        Gửi briefing tới người dùng.

        Args:
            text:   nội dung briefing
            method: 'print' | 'tts' | 'notification'
        """
        if method == "print":
            print("\n" + "="*60)
            print(text)
            print("="*60 + "\n")

        elif method == "tts":
            self._deliver_tts(text)

        elif method == "notification":
            self._deliver_notification(text)

    def _deliver_tts(self, text: str):
        """Đọc briefing bằng TTS (pyttsx3)."""
        try:
            import pyttsx3
            # Strip markdown cho TTS
            import re
            clean = re.sub(r"[#*`_\[\]()]", "", text)
            clean = re.sub(r"\n{2,}", ". ", clean)

            engine = pyttsx3.init()
            engine.setProperty("rate", 160)  # tốc độ nói
            engine.say(clean[:2000])         # giới hạn độ dài
            engine.runAndWait()
        except ImportError:
            print("[Briefing] pyttsx3 chưa cài. Dùng: pip install pyttsx3")
            self.deliver(text, method="print")
        except Exception as e:
            print(f"[Briefing] TTS error: {e}")
            self.deliver(text, method="print")

    def _deliver_notification(self, text: str):
        """Gửi Windows notification."""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            # Notification chỉ hiện excerpt
            excerpt = text.split("\n")[2] if len(text.split("\n")) > 2 else text[:100]
            toaster.show_toast(
                "🌅 Morning Briefing",
                excerpt,
                duration=10,
                threaded=True,
            )
        except ImportError:
            print("[Briefing] win10toast chưa cài. Dùng: pip install win10toast")
            self.deliver(text, method="print")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_greeting(hour: int) -> str:
    if hour < 12:
        return "Chào buổi sáng"
    elif hour < 18:
        return "Chào buổi chiều"
    else:
        return "Chào buổi tối"
