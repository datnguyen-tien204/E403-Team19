"""
tools/system_tools.py
Tools hệ thống mở rộng cho Windows:
  - Độ sáng màn hình
  - Screenshot
  - Shutdown / Restart / Sleep
  - Danh sách / kill process
  - Clipboard
  - Wifi info
"""
import os
import subprocess
import datetime
from langchain_core.tools import tool


# ──────────────────────────────────────────────
# TOOL: Điều chỉnh độ sáng màn hình
# ──────────────────────────────────────────────
@tool
def control_brightness(action: str, value: int = 10) -> str:
    """
    Điều chỉnh độ sáng màn hình laptop.
    action: 'set' | 'increase' | 'decrease' | 'get'
    value: phần trăm (0-100)
    """
    try:
        import screen_brightness_control as sbc

        current = sbc.get_brightness(display=0)
        if isinstance(current, list):
            current = current[0]

        if action == "get":
            return f"Độ sáng hiện tại: {current}%"
        elif action == "set":
            new_val = max(0, min(100, value))
            sbc.set_brightness(new_val, display=0)
            return f"Đã đặt độ sáng về {new_val}%"
        elif action == "increase":
            new_val = max(0, min(100, current + value))
            sbc.set_brightness(new_val, display=0)
            return f"Đã tăng độ sáng từ {current}% lên {new_val}%"
        elif action == "decrease":
            new_val = max(0, min(100, current - value))
            sbc.set_brightness(new_val, display=0)
            return f"Đã giảm độ sáng từ {current}% xuống {new_val}%"
        else:
            return "action không hợp lệ. Dùng: set/increase/decrease/get"

    except ImportError:
        return "Lỗi: screen-brightness-control chưa cài. Chạy: pip install screen-brightness-control"
    except Exception as e:
        return f"Lỗi điều chỉnh độ sáng: {str(e)}"


# ──────────────────────────────────────────────
# TOOL: Chụp màn hình
# ──────────────────────────────────────────────
@tool
def take_screenshot(save_path: str = "") -> str:
    """
    Chụp màn hình và lưu vào file.
    save_path: đường dẫn file ảnh (mặc định: Desktop/screenshot_YYYYMMDD_HHMMSS.png)
    """
    try:
        from PIL import ImageGrab
        import time

        if not save_path:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            os.makedirs(desktop, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(desktop, f"screenshot_{ts}.png")

        time.sleep(0.3)  # đợi màn hình ổn định
        img = ImageGrab.grab()
        img.save(save_path)
        return f"Đã chụp màn hình: {save_path}"

    except ImportError:
        return "Lỗi: pillow chưa cài. Chạy: pip install pillow"
    except Exception as e:
        return f"Lỗi chụp màn hình: {str(e)}"


# ──────────────────────────────────────────────
# TOOL: Shutdown / Restart / Sleep
# ──────────────────────────────────────────────
@tool
def power_control(action: str, delay_seconds: int = 0) -> str:
    """
    Điều khiển nguồn điện laptop.
    action: 'shutdown' | 'restart' | 'sleep' | 'lock' | 'cancel'
    delay_seconds: số giây chờ trước khi thực hiện (mặc định 0)
    """
    try:
        if action == "shutdown":
            subprocess.run(f"shutdown /s /t {delay_seconds}", shell=True)
            if delay_seconds > 0:
                return f"Sẽ tắt máy sau {delay_seconds} giây."
            return "Đang tắt máy..."

        elif action == "restart":
            subprocess.run(f"shutdown /r /t {delay_seconds}", shell=True)
            return f"Đang khởi động lại... (sau {delay_seconds}s)"

        elif action == "sleep":
            subprocess.run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
            return "Đang chuyển sang chế độ sleep..."

        elif action == "lock":
            subprocess.run("rundll32.exe user32.dll,LockWorkStation", shell=True)
            return "Đã khóa màn hình."

        elif action == "cancel":
            subprocess.run("shutdown /a", shell=True)
            return "Đã hủy lệnh shutdown/restart."

        else:
            return "action không hợp lệ. Dùng: shutdown/restart/sleep/lock/cancel"

    except Exception as e:
        return f"Lỗi: {str(e)}"


# ──────────────────────────────────────────────
# TOOL: Quản lý tiến trình (Process)
# ──────────────────────────────────────────────
@tool
def manage_process(action: str, name_or_pid: str = "") -> str:
    """
    Quản lý tiến trình Windows.
    action: 'list' | 'kill' | 'find'
    name_or_pid: tên process hoặc PID (dùng cho kill/find)
    Ví dụ: action='kill', name_or_pid='notepad.exe'
           action='find', name_or_pid='chrome'
    """
    try:
        import psutil

        if action == "list":
            procs = []
            for p in sorted(psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]), 
                            key=lambda x: x.info["memory_info"].rss if x.info["memory_info"] else 0,
                            reverse=True)[:15]:
                mem_mb = p.info["memory_info"].rss / 1e6 if p.info["memory_info"] else 0
                procs.append(f"  {p.info['pid']:6d}  {p.info['name']:<30s}  {mem_mb:6.1f} MB")
            return "Top 15 process (theo RAM):\n   PID   Name                            RAM\n" + "\n".join(procs)

        elif action == "find":
            query = name_or_pid.lower()
            found = [p for p in psutil.process_iter(["pid", "name"])
                     if query in p.info["name"].lower()]
            if not found:
                return f"Không tìm thấy process '{name_or_pid}'"
            lines = [f"  PID {p.info['pid']}: {p.info['name']}" for p in found]
            return "\n".join(lines)

        elif action == "kill":
            import signal
            killed = []
            for p in psutil.process_iter(["pid", "name"]):
                match = (name_or_pid.isdigit() and p.info["pid"] == int(name_or_pid)) or \
                        (name_or_pid.lower() in p.info["name"].lower())
                if match:
                    p.kill()
                    killed.append(f"{p.info['name']} (PID {p.info['pid']})")
            if killed:
                return f"Đã kết thúc: {', '.join(killed)}"
            return f"Không tìm thấy process '{name_or_pid}'"

        else:
            return "action không hợp lệ. Dùng: list/find/kill"

    except ImportError:
        return "Lỗi: psutil chưa cài. Chạy: pip install psutil"
    except Exception as e:
        return f"Lỗi: {str(e)}"


# ──────────────────────────────────────────────
# TOOL: Clipboard
# ──────────────────────────────────────────────
@tool
def clipboard_control(action: str, text: str = "") -> str:
    """
    Đọc/ghi clipboard Windows.
    action: 'get' | 'set' | 'clear'
    text: nội dung cần copy (dùng cho action='set')
    """
    try:
        import subprocess

        if action == "get":
            result = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"],
                capture_output=True, text=True
            )
            content = result.stdout.strip()
            if not content:
                return "Clipboard đang trống."
            preview = content[:300] + ("..." if len(content) > 300 else "")
            return f"Nội dung clipboard:\n{preview}"

        elif action == "set":
            if not text:
                return "Cần truyền text để copy vào clipboard."
            escaped = text.replace('"', '`"')
            subprocess.run(
                ["powershell", "-command", f'Set-Clipboard -Value "{escaped}"'],
                capture_output=True
            )
            return f"Đã copy vào clipboard: {text[:80]}{'...' if len(text)>80 else ''}"

        elif action == "clear":
            subprocess.run(
                ["powershell", "-command", "Set-Clipboard -Value ''"],
                capture_output=True
            )
            return "Đã xóa clipboard."

        else:
            return "action không hợp lệ. Dùng: get/set/clear"

    except Exception as e:
        return f"Lỗi clipboard: {str(e)}"


# ──────────────────────────────────────────────
# TOOL: Wifi / Network info
# ──────────────────────────────────────────────
@tool
def get_network_info(action: str = "status") -> str:
    """
    Xem thông tin mạng / Wifi.
    action: 'status' | 'list_wifi' | 'ip'
    """
    try:
        if action == "ip":
            result = subprocess.run(
                ["powershell", "-command",
                 "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -ne '127.0.0.1'}).IPAddress"],
                capture_output=True, text=True
            )
            ips = result.stdout.strip()
            return f"Địa chỉ IP: {ips}" if ips else "Không tìm thấy IP"

        elif action == "status":
            result = subprocess.run(
                ["netsh", "wlan", "show", "interfaces"],
                capture_output=True, text=True, encoding="utf-8", errors="ignore"
            )
            lines = result.stdout.strip().split("\n")
            info = {}
            for line in lines:
                if "SSID" in line and "BSSID" not in line:
                    info["SSID"] = line.split(":", 1)[-1].strip()
                elif "Signal" in line or "Cường độ" in line:
                    info["Signal"] = line.split(":", 1)[-1].strip()
                elif "State" in line or "Trạng thái" in line:
                    info["State"] = line.split(":", 1)[-1].strip()
            if not info:
                return "Không có kết nối Wifi đang hoạt động."
            return "\n".join(f"{k}: {v}" for k, v in info.items())

        elif action == "list_wifi":
            result = subprocess.run(
                ["netsh", "wlan", "show", "networks"],
                capture_output=True, text=True, encoding="utf-8", errors="ignore"
            )
            lines = [l.strip() for l in result.stdout.split("\n") 
                     if "SSID" in l and "BSSID" not in l]
            networks = [l.split(":", 1)[-1].strip() for l in lines if ":" in l]
            if not networks:
                return "Không tìm thấy mạng Wifi nào."
            return "Các mạng Wifi xung quanh:\n" + "\n".join(f"  • {n}" for n in networks)

        else:
            return "action không hợp lệ. Dùng: status/list_wifi/ip"

    except Exception as e:
        return f"Lỗi network: {str(e)}"


# ──────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────
EXTENDED_SYSTEM_TOOLS = [
    control_brightness,
    take_screenshot,
    power_control,
    manage_process,
    clipboard_control,
    get_network_info,
]
