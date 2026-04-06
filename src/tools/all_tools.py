from __future__ import annotations

import os
import sys
import subprocess
import datetime
from langchain_core.tools import tool

from src.rag.reranker import rerank
from src.tools.web_search import web_search


def get_rag_tool(vector_store):
    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Tìm kiếm thông tin trong knowledge base (RAG).
        Dùng khi cần trả lời câu hỏi dựa trên tài liệu đã lưu.
        Nếu trả về "NO_RAG_HIT" nghĩa là không có kết quả → hãy thử web_search.
        """
        try:
            docs = vector_store.similarity_search(query, k=8)
            if not docs:
                return "NO_RAG_HIT: Không tìm thấy thông tin liên quan trong knowledge base."
            passages = [doc.page_content for doc in docs]
            source_map = {doc.page_content: doc.metadata.get("source", "unknown") for doc in docs}
            ranked = rerank(query, passages, top_k=4)
            results = []
            for i, (text, score) in enumerate(ranked, 1):
                source = source_map.get(text, "unknown")
                results.append(f"[{i}] (nguồn: {source}) [score={score:.2f}]\n{text}")
            return "\n\n".join(results)
        except Exception as e:
            return f"Lỗi khi tìm kiếm: {str(e)}"

    return search_knowledge_base


@tool
def control_volume(action: str, value: int = 10) -> str:
    """
    Điều khiển âm lượng laptop trên Windows.
    action: 'set' | 'increase' | 'decrease' | 'mute' | 'unmute' | 'get'
    value: phần trăm (0-100), chỉ dùng cho 'set', 'increase', 'decrease'
    """
    if sys.platform != "win32":
        return "❌ Tool này chỉ hỗ trợ Windows. Hệ điều hành hiện tại không tương thích."
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        vol_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
        current = round(vol_ctrl.GetMasterVolumeLevelScalar() * 100)

        if action == "get":
            return f"Âm lượng hiện tại: {current}%"
        elif action == "set":
            new_val = max(0, min(100, value))
            vol_ctrl.SetMasterVolumeLevelScalar(new_val / 100.0, None)
            return f"Đã đặt âm lượng về {new_val}%"
        elif action == "increase":
            new_val = max(0, min(100, current + value))
            vol_ctrl.SetMasterVolumeLevelScalar(new_val / 100.0, None)
            return f"Đã tăng âm lượng từ {current}% lên {new_val}%"
        elif action == "decrease":
            new_val = max(0, min(100, current - value))
            vol_ctrl.SetMasterVolumeLevelScalar(new_val / 100.0, None)
            return f"Đã giảm âm lượng từ {current}% xuống {new_val}%"
        elif action == "mute":
            vol_ctrl.SetMute(1, None)
            return "Đã tắt tiếng (mute)"
        elif action == "unmute":
            vol_ctrl.SetMute(0, None)
            return f"Đã bật tiếng (unmute), âm lượng: {current}%"
        else:
            return f"Action không hợp lệ: {action}. Dùng: set/increase/decrease/mute/unmute/get"
    except ImportError:
        return "Lỗi: pycaw chưa được cài. Chạy: pip install pycaw comtypes"
    except Exception as e:
        return f"Lỗi điều khiển âm lượng: {str(e)}"


APP_MAP = {
    "chrome": "chrome", "firefox": "firefox", "edge": "msedge",
    "notepad": "notepad", "calculator": "calc", "explorer": "explorer",
    "word": "winword", "excel": "excel", "powerpoint": "powerpnt",
    "cmd": "cmd", "terminal": "wt", "spotify": "spotify",
    "vscode": "code", "task manager": "taskmgr",
    "control panel": "control", "settings": "ms-settings:",
    "paint": "mspaint", "vlc": "vlc",
}

@tool
def open_application(app_name: str) -> str:
    """
    Mở ứng dụng, thư mục, hoặc URL trên Windows.
    app_name: tên ứng dụng (vd: 'chrome', 'notepad') hoặc path hoặc URL
    """
    if sys.platform != "win32":
        return "❌ Tool này chỉ hỗ trợ Windows. Hệ điều hành hiện tại không tương thích."
    try:
        if app_name.startswith("http://") or app_name.startswith("https://"):
            os.startfile(app_name)
            return f"Đã mở URL: {app_name}"
        if os.path.exists(app_name):
            os.startfile(app_name)
            return f"Đã mở: {app_name}"
        key = app_name.lower().strip()
        command = APP_MAP.get(key, key)
        subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Đã mở '{app_name}'"
    except Exception as e:
        return f"Không thể mở '{app_name}': {str(e)}"


@tool
def get_system_info(info_type: str = "all") -> str:
    """
    Lấy thông tin hệ thống: battery, cpu, memory, disk, datetime.
    info_type: 'battery' | 'cpu' | 'memory' | 'disk' | 'datetime' | 'all'
    """
    results = {}
    try:
        import psutil
        if info_type in ("battery", "all"):
            bat = psutil.sensors_battery()
            if bat:
                results["battery"] = f"{bat.percent:.0f}% {'(đang sạc)' if bat.power_plugged else '(không sạc)'}"
        if info_type in ("cpu", "all"):
            results["cpu"] = f"{psutil.cpu_percent(interval=0.5)}%"
        if info_type in ("memory", "all"):
            mem = psutil.virtual_memory()
            results["memory"] = f"{mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB ({mem.percent}%)"
        if info_type in ("disk", "all"):
            if sys.platform == "win32":
                import string
                drives = [f"{d}:\\" for d in string.ascii_uppercase]
                for d in drives:
                    if os.path.exists(d):
                        try:
                            disk = psutil.disk_usage(d)
                            results[f"disk_{d[0]}"] = f"còn {disk.free/1e9:.1f}GB / {disk.total/1e9:.1f}GB"
                        except Exception:
                            pass
            else:
                for part in psutil.disk_partitions(all=False):
                    try:
                        disk = psutil.disk_usage(part.mountpoint)
                        results[f"disk_{part.mountpoint}"] = f"còn {disk.free/1e9:.1f}GB / {disk.total/1e9:.1f}GB"
                    except Exception:
                        pass
    except ImportError:
        results["note"] = "Cài psutil để xem thêm: pip install psutil"
    if info_type in ("datetime", "all"):
        results["datetime"] = datetime.datetime.now().strftime("%H:%M %d/%m/%Y")
    if not results:
        return "Không lấy được thông tin hệ thống."
    return "\n".join(f"{k}: {v}" for k, v in results.items())


@tool
def get_outlook_calendar_local(days_ahead: int = 1) -> str:
    """
    Lấy lịch từ Outlook desktop qua Win32 COM API.
    Yêu cầu: Microsoft Outlook đang chạy trên máy và pywin32 đã được cài.
    days_ahead: số ngày muốn xem kể từ hôm nay (mặc định 1 = chỉ hôm nay)
    """
    if sys.platform != "win32":
        return "❌ Tool này chỉ hỗ trợ Windows với Microsoft Outlook. Hệ điều hành hiện tại không tương thích."
    try:
        import win32com.client

        outlook = win32com.client.Dispatch("Outlook.Application")
        ns = outlook.GetNamespace("MAPI")
        calendar = ns.GetDefaultFolder(9)
        items = calendar.Items
        items.IncludeRecurrences = True
        items.Sort("[Start]")

        now = datetime.datetime.now()
        end = now + datetime.timedelta(days=days_ahead)

        restriction = (
            f"[Start] >= '{now.strftime('%m/%d/%Y %H:%M')}' AND "
            f"[Start] <= '{end.strftime('%m/%d/%Y %H:%M')}'"
        )
        filtered = items.Restrict(restriction)

        if filtered.Count == 0:
            return "Không có sự kiện nào trong lịch Outlook."

        lines = []
        for item in filtered:
            try:
                start_str = item.Start.strftime("%H:%M %d/%m")
                location = item.Location if item.Location else "không có địa điểm"
                lines.append(f"- {start_str}: {item.Subject} ({location})")
            except Exception:
                continue

        return "\n".join(lines) if lines else "Không có sự kiện nào."

    except ImportError:
        return "Lỗi: pywin32 chưa cài. Chạy: pip install pywin32"
    except Exception as e:
        return f"Lỗi Outlook COM Calendar: {str(e)}"


@tool
def get_outlook_emails_local(top: int = 5, unread_only: bool = True) -> str:
    """
    Lấy email từ Outlook desktop qua Win32 COM API.
    Yêu cầu: Microsoft Outlook đang chạy trên máy và pywin32 đã được cài.
    top: số email tối đa muốn lấy (mặc định 5)
    unread_only: True = chỉ lấy email chưa đọc, False = lấy tất cả (mặc định True)
    """
    if sys.platform != "win32":
        return "❌ Tool này chỉ hỗ trợ Windows với Microsoft Outlook. Hệ điều hành hiện tại không tương thích."
    try:
        import win32com.client

        outlook = win32com.client.Dispatch("Outlook.Application")
        ns = outlook.GetNamespace("MAPI")
        inbox = ns.GetDefaultFolder(6)
        items = inbox.Items
        items.Sort("[ReceivedTime]", True)

        results = []
        count = 0
        for item in items:
            if count >= top:
                break
            try:
                if unread_only and item.UnRead is False:
                    continue
                received = item.ReceivedTime.strftime("%d/%m %H:%M")
                sender = item.SenderName
                subject = item.Subject
                preview = (item.Body or "")[:150].replace("\n", " ").strip()
                results.append(
                    f"- [{received}] {subject}\n"
                    f"  Từ: {sender}\n"
                    f"  {preview}..."
                )
                count += 1
            except Exception:
                continue

        return "\n\n".join(results) if results else "Không có email chưa đọc trong Outlook."

    except ImportError:
        return "Lỗi: pywin32 chưa cài. Chạy: pip install pywin32"
    except Exception as e:
        return f"Lỗi Outlook COM Email: {str(e)}"


OUTLOOK_COM_TOOLS = [
    get_outlook_calendar_local,
    get_outlook_emails_local,
]


def get_all_system_tools():
    from src.tools.system_tools import EXTENDED_SYSTEM_TOOLS
    from src.tools.sandbox import SANDBOX_TOOLS
    return (
        [control_volume, open_application, get_system_info]
        + SANDBOX_TOOLS
        + EXTENDED_SYSTEM_TOOLS
        + OUTLOOK_COM_TOOLS
    )


SYSTEM_TOOLS = [control_volume, open_application, get_system_info]


def build_toolset(vector_store=None, mcp_servers: list[str] | None = None):
    tools = []

    if vector_store is not None:
        tools.append(get_rag_tool(vector_store))

    tools.append(web_search)

    tools.extend(get_all_system_tools())

    if mcp_servers:
        try:
            from tools.mcp_client import build_mcp_tools
            mcp_tools = build_mcp_tools(mcp_servers)
            tools.extend(mcp_tools)
            print(f"[Tools] MCP tools loaded: {[t.name for t in mcp_tools]}")
        except Exception as e:
            print(f"[Tools] MCP load lỗi (bỏ qua): {e}")

    print(f"[Tools] Total tools: {len(tools)} — {[t.name for t in tools]}")
    return tools