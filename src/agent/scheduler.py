"""
agent/scheduler.py
Chạy các job tự động theo lịch:
  - 08:00 hàng ngày   → Morning Briefing
  - 02:00 Chủ nhật    → Rebuild user profile từ lịch sử chat
  - Mỗi 30 phút       → Nhắc task nếu đang active (tuỳ chọn)

Cách dùng (trong main.py):
    from agent.scheduler import AgentScheduler
    scheduler = AgentScheduler(agent, weaviate_client, mode_state)
    scheduler.start()
    # ... app chạy ...
    scheduler.stop()
"""
from __future__ import annotations

import datetime
import threading
from typing import Optional, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


class AgentScheduler:
    """
    Quản lý các scheduled jobs cho agent.

    Args:
        agent:            compiled LangGraph agent
        weaviate_client:  Weaviate client (có thể None nếu Weaviate không chạy)
        mode_state:       SearchModeState object
        briefing_method:  'print' | 'tts' | 'notification'
        briefing_hour:    giờ gửi morning briefing (mặc định 8)
        briefing_minute:  phút gửi briefing (mặc định 0)
        on_briefing_ready: callback(text: str) khi briefing xong — dùng để update GUI
    """

    def __init__(
        self,
        agent,
        weaviate_client=None,
        mode_state=None,
        briefing_method:    str = "print",
        briefing_hour:      int = 8,
        briefing_minute:    int = 0,
        on_briefing_ready:  Optional[Callable[[str], None]] = None,
    ):
        self.agent            = agent
        self.weaviate_client  = weaviate_client
        self.mode_state       = mode_state
        self.briefing_method  = briefing_method
        self.briefing_hour    = briefing_hour
        self.briefing_minute  = briefing_minute
        self.on_briefing_ready = on_briefing_ready

        self._scheduler = BackgroundScheduler(
            job_defaults={"misfire_grace_time": 300}  # 5 phút grace period
        )
        self._running = False

    # ── Jobs ──────────────────────────────────────────────────────────────────

    def _job_morning_briefing(self):
        """Job: tạo và deliver morning briefing."""
        print(f"\n[Scheduler] ⏰ Morning briefing job started — {datetime.datetime.now().strftime('%H:%M')}")
        try:
            from agent.briefing import MorningBriefing
            briefing = MorningBriefing(self.agent, self.weaviate_client)
            text = briefing.generate()

            # Deliver theo method đã config
            briefing.deliver(text, method=self.briefing_method)

            # Gọi callback nếu có (để update GUI)
            if self.on_briefing_ready:
                self.on_briefing_ready(text)

        except Exception as e:
            print(f"[Scheduler] Briefing job lỗi: {e}")

    def _job_rebuild_profile(self):
        """Job: rebuild user profile từ lịch sử chat (chạy hàng tuần)."""
        print(f"\n[Scheduler] 🔄 Profile rebuild job started — {datetime.datetime.now()}")
        if not self.weaviate_client:
            print("[Scheduler] Không có Weaviate, bỏ qua profile rebuild.")
            return
        try:
            from agent.profile_builder import build_user_profile
            profile = build_user_profile(self.weaviate_client)
            if profile:
                topics = profile.get("top_topics", [])[:3]
                print(f"[Scheduler] ✅ Profile rebuilt — topics: {topics}")
            else:
                print("[Scheduler] Profile rebuild không có data.")
        except Exception as e:
            print(f"[Scheduler] Profile rebuild lỗi: {e}")

    def _job_proactive_check(self):
        """
        Job: kiểm tra xem có nên chủ động nhắc gì không.
        Chạy mỗi giờ trong giờ active của người dùng.
        """
        if not self.weaviate_client:
            return
        try:
            from agent.profile_builder import load_profile
            profile = load_profile(self.weaviate_client)
            if not profile:
                return

            current_hour   = datetime.datetime.now().hour
            active_hours   = profile.get("active_hours", [])
            workflows      = profile.get("repeated_workflows", [])

            # Chỉ nhắc khi đang trong giờ active
            if current_hour not in active_hours:
                return

            # Kiểm tra có workflow nào thường làm vào giờ này không
            today = datetime.datetime.now().strftime("%a")
            daily = profile.get("daily_patterns", {})
            todays_msgs = daily.get(today, [])

            if todays_msgs:
                print(
                    f"\n[Scheduler] 💡 Proactive reminder ({current_hour}:00): "
                    f"Bạn thường làm: {todays_msgs[0][:80]}"
                )

        except Exception as e:
            print(f"[Scheduler] Proactive check lỗi: {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Khởi động scheduler với tất cả jobs."""
        if self._running:
            print("[Scheduler] Đã chạy rồi.")
            return

        # Job 1: Morning briefing — mỗi ngày đúng giờ config
        self._scheduler.add_job(
            self._job_morning_briefing,
            trigger=CronTrigger(
                hour=self.briefing_hour,
                minute=self.briefing_minute,
            ),
            id="morning_briefing",
            replace_existing=True,
        )

        # Job 2: Rebuild profile — mỗi Chủ nhật 02:00
        self._scheduler.add_job(
            self._job_rebuild_profile,
            trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="rebuild_profile",
            replace_existing=True,
        )

        # Job 3: Proactive check — mỗi giờ
        self._scheduler.add_job(
            self._job_proactive_check,
            trigger=CronTrigger(minute=0),  # đầu mỗi giờ
            id="proactive_check",
            replace_existing=True,
        )

        self._scheduler.start()
        self._running = True

        print(
            f"[Scheduler] ✅ Started — "
            f"Briefing lúc {self.briefing_hour:02d}:{self.briefing_minute:02d} | "
            f"Profile rebuild: Chủ nhật 02:00"
        )

    def stop(self):
        """Dừng scheduler."""
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            print("[Scheduler] Stopped.")

    def run_briefing_now(self):
        """Chạy briefing ngay lập tức (dùng để test)."""
        print("[Scheduler] Running briefing NOW...")
        thread = threading.Thread(
            target=self._job_morning_briefing,
            daemon=True,
        )
        thread.start()

    def run_profile_rebuild_now(self):
        """Rebuild profile ngay lập tức (dùng để test / lần đầu setup)."""
        print("[Scheduler] Running profile rebuild NOW...")
        thread = threading.Thread(
            target=self._job_rebuild_profile,
            daemon=True,
        )
        thread.start()

    def get_next_jobs(self) -> list[dict]:
        """Lấy danh sách jobs và lần chạy tiếp theo."""
        jobs = []
        for job in self._scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append({
                "id":       job.id,
                "next_run": next_run.strftime("%d/%m/%Y %H:%M") if next_run else "N/A",
            })
        return jobs
