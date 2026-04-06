"""
main.py  ── v2.0
Entry point với:
  ✅ MCP servers (Excel, Calendar, Gmail...)
  ✅ User profile builder
  ✅ Scheduler (morning briefing + weekly profile rebuild)

Chạy:
  python main.py           → GUI (PyQt5)
  python main.py --api     → FastAPI server
  python main.py --briefing-now   → test briefing ngay
  python main.py --rebuild-profile → rebuild profile ngay

Config qua biến môi trường (hoặc .env):
  MCP_SERVERS=excel,google_calendar,gmail
  BRIEFING_HOUR=8
  BRIEFING_METHOD=print   # print | tts | notification
"""
import sys
import argparse
import os


# ── MCP servers muốn bật (đọc từ env hoặc hardcode) ──────────────────────────
_MCP_SERVERS = [s.strip() for s in os.getenv("MCP_SERVERS", "excel").split(",") if s.strip()]
_BRIEFING_HOUR   = int(os.getenv("BRIEFING_HOUR", "8"))
_BRIEFING_MINUTE = int(os.getenv("BRIEFING_MINUTE", "0"))
_BRIEFING_METHOD = os.getenv("BRIEFING_METHOD", "print")   # print | tts | notification


def init_agent():
    """Khởi tạo agent, weaviate, tools, scheduler."""
    from src.rag.weaviate_store import get_weaviate_client, get_vector_store
    from src.tools.all_tools import build_toolset
    from src.agent.graph import build_agent
    from src.agent.state import SearchModeState
    from src.agent.profile_builder import build_user_profile, load_profile
    from src.config import SEARCH_MODE_DEFAULT

    mode_state = SearchModeState(SEARCH_MODE_DEFAULT)

    # ── Weaviate ──────────────────────────────────────────────────────────────
    print("[*] Kết nối Weaviate...")
    weaviate_client = None
    vector_store    = None
    try:
        weaviate_client = get_weaviate_client()
        vector_store    = get_vector_store(weaviate_client)
        print("[✓] Weaviate OK")
    except Exception as e:
        print(f"[!] Weaviate không khả dụng: {e}")

    # ── Tools (bao gồm MCP) ───────────────────────────────────────────────────
    print(f"[*] Khởi tạo tools (MCP: {_MCP_SERVERS})...")
    all_tools = build_toolset(
        vector_store=vector_store,
        mcp_servers=_MCP_SERVERS,
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    print("[*] Khởi tạo agent...")
    agent = build_agent(all_tools, mode_getter=mode_state.get_mode)
    print("[✓] Agent sẵn sàng!")

    # ── User Profile ──────────────────────────────────────────────────────────
    if weaviate_client:
        profile = load_profile(weaviate_client)
        if profile:
            print(f"[✓] Profile loaded — topics: {profile.get('top_topics', [])[:3]}")
        else:
            print("[*] Chưa có profile. Chạy --rebuild-profile để tạo lần đầu.")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = None
    try:
        from src.agent.scheduler import AgentScheduler
        scheduler = AgentScheduler(
            agent=agent,
            weaviate_client=weaviate_client,
            mode_state=mode_state,
            briefing_method=_BRIEFING_METHOD,
            briefing_hour=_BRIEFING_HOUR,
            briefing_minute=_BRIEFING_MINUTE,
        )
        scheduler.start()
    except ImportError:
        print("[!] APScheduler chưa cài. Bỏ qua scheduler. Dùng: pip install apscheduler")
    except Exception as e:
        print(f"[!] Scheduler lỗi: {e}")

    print()
    return agent, weaviate_client, mode_state, scheduler


def run_api():
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("src.api.server:app", host=API_HOST, port=API_PORT, reload=False)


def run_briefing_now(agent, weaviate_client):
    """Chạy briefing ngay để test."""
    from src.agent.briefing import MorningBriefing
    briefing = MorningBriefing(agent, weaviate_client)
    text = briefing.generate()
    briefing.deliver(text, method=_BRIEFING_METHOD)


def run_rebuild_profile(weaviate_client):
    """Rebuild profile ngay."""
    if not weaviate_client:
        print("[!] Weaviate không khả dụng, không thể rebuild profile.")
        return
    from src.agent.profile_builder import build_user_profile
    profile = build_user_profile(weaviate_client)
    if profile:
        print("\n=== PROFILE ===")
        print(f"Topics:    {profile['top_topics']}")
        print(f"Hours:     {profile['active_hours']}")
        print(f"Workflows: {profile['repeated_workflows']}")
        print(f"Pains:     {profile['pain_points']}")
        print(f"Summary:   {profile.get('behavior_summary', '')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent v2")
    parser.add_argument("--api",            action="store_true", help="Chạy FastAPI server")
    parser.add_argument("--briefing-now",   action="store_true", help="Test morning briefing ngay")
    parser.add_argument("--rebuild-profile",action="store_true", help="Rebuild user profile ngay")
    args = parser.parse_args()

    agent, weaviate_client, mode_state, scheduler = init_agent()

    try:
        if args.briefing_now:
            run_briefing_now(agent, weaviate_client)

        elif args.rebuild_profile:
            run_rebuild_profile(weaviate_client)

        elif args.api:
            run_api()

        # else: GUI (PyQt5) — giữ nguyên như cũ

    except KeyboardInterrupt:
        print("\n[*] Đang tắt...")
    finally:
        if scheduler:
            scheduler.stop()
        # Dọn dẹp MCP servers
        try:
            from src.tools.mcp_client import shutdown_mcp_servers
            shutdown_mcp_servers()
        except Exception:
            pass
        print("[✓] Tắt xong.")
