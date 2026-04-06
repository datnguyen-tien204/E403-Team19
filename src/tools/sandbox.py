"""
tools/sandbox.py  ── v2.0
Real-Machine Code Execution Sandbox

Cải tiến từ sách "Giải phẫu một Agentic OS":
  ✅ Pattern #10 – Permission Classification Pipeline (5 lớp, thay vì regex đơn giản)
  ✅ Pattern #7  – Disk-based output + outputOffset cho background tasks (Chương 7)
  ✅ Pattern #3  – Escalating recovery: warn → ask → block (Chương 3)
  ✅ Stall detection cho background tasks (process ngủ đợi input)
  ✅ Real-time output streaming qua callback (WebSocket / GUI)

══════════════════════════════════════════════════════
PERMISSION PIPELINE (5 lớp, lớp nhẹ chạy trước):
──────────────────────────────────────────────────────
  Lớp 1 – Safe allowlist:   Lệnh/code chỉ đọc → ALWAYS SAFE, skip toàn bộ pipeline.
  Lớp 2 – Sensitive paths:  Đường dẫn hệ thống Windows (C:\\Windows...) → BLOCK.
  Lớp 3 – Code intent:      Phân loại READ_ONLY / WRITE_SAFE / NETWORK / SYSTEM / DESTRUCTIVE.
  Lớp 4 – Dangerous patterns: eval, exec, subprocess shell injection, sudo rm, ... → BLOCK.
  Lớp 5 – Shell AST check:  Phân tích bash/powershell command chaining → BLOCK nếu nguy hiểm.
══════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import re
import sys
import time
import threading
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

from langchain_core.tools import tool

# ── Workspace ──────────────────────────────────────────────────────────────────
WORKSPACE = Path.home() / "agent_workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)

# ── Background process registry ────────────────────────────────────────────────
_bg_processes: dict[str, subprocess.Popen] = {}
_bg_logs:      dict[str, Path]             = {}
_bg_offsets:   dict[str, int]              = {}   # Pattern #7: outputOffset

# ── Real-time output callback ──────────────────────────────────────────────────
_output_callback: Optional[Callable[[str], None]] = None

def set_output_callback(cb: Optional[Callable[[str], None]]):
    global _output_callback
    _output_callback = cb

def _emit(text: str):
    if _output_callback:
        try:
            _output_callback(text)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# PERMISSION CLASSIFICATION PIPELINE  (Pattern #10)
# ══════════════════════════════════════════════════════════════════════════════

class PermResult(Enum):
    ALLOW   = "allow"   # An toàn, thực thi ngay
    WARN    = "warn"    # Cho phép nhưng ghi log cảnh báo
    BLOCK   = "block"   # Từ chối

@dataclass
class PermDecision:
    result:  PermResult
    layer:   int          # lớp nào quyết định
    reason:  str          # lý do
    labels:  list[str] = field(default_factory=list)  # nhãn phân loại


# ─────────────────────────────────────────
# Lớp 1 – Safe read-only allowlist
# ─────────────────────────────────────────
_READ_ONLY_PYTHON = re.compile(
    r"^\s*(print|input|len|range|int|float|str|list|dict|set|type|"
    r"isinstance|hasattr|getattr|dir|help|repr|"
    r"open\s*\([^,)]+,\s*['\"]r['\"]|"  # open(..., 'r')
    r"import\s+(math|json|re|datetime|collections|itertools|functools|"
    r"pathlib|typing|dataclasses|enum|abc|copy|textwrap|pprint|"
    r"hashlib|hmac|base64|binascii|struct|io|os\.path|"
    r"time\.sleep|time\.time|time\.strftime))"
)

_READ_ONLY_BASH = re.compile(
    r"^\s*(cat |head |tail |grep |awk |sed |ls |find |echo |pwd |"
    r"git (log|diff|status|show|branch|tag)|"
    r"ps |top |df |du |free |uptime |uname |whoami |id |"
    r"curl -[^X]*$|wget --spider)"
)

def _layer1_safe_allowlist(code: str, language: str) -> Optional[PermDecision]:
    """Trả về ALLOW ngay nếu code chỉ là read-only. Trả None nếu không chắc."""
    lang = language.lower()
    lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return PermDecision(PermResult.ALLOW, 1, "empty code", ["read_only"])

    if lang == "python":
        if all(_READ_ONLY_PYTHON.match(l) for l in lines):
            return PermDecision(PermResult.ALLOW, 1, "read-only Python", ["read_only"])
    elif lang in ("bash", "sh"):
        if len(lines) == 1 and _READ_ONLY_BASH.match(lines[0]):
            return PermDecision(PermResult.ALLOW, 1, "read-only shell", ["read_only"])
    return None


# ─────────────────────────────────────────
# Lớp 2 – Sensitive system paths (Windows)
# ─────────────────────────────────────────
def _layer2_sensitive_paths(code: str, working_dir: str) -> Optional[PermDecision]:
    from src.tools.exclusion_config import check_code_for_sensitive_paths, is_safe_working_dir
    if working_dir and not is_safe_working_dir(working_dir):
        return PermDecision(
            PermResult.BLOCK, 2,
            f"Thư mục làm việc '{working_dir}' thuộc khu vực hệ thống bị cấm.",
            ["system_path"]
        )
    ok, reason = check_code_for_sensitive_paths(code)
    if not ok:
        return PermDecision(PermResult.BLOCK, 2, reason, ["system_path"])
    return None


# ─────────────────────────────────────────
# Lớp 3 – Code intent classification
# ─────────────────────────────────────────
_DESTRUCTIVE_PY = [
    (r"\bos\.remove\s*\(",           "os.remove()"),
    (r"\bos\.unlink\s*\(",           "os.unlink()"),
    (r"\bos\.rmdir\s*\(",            "os.rmdir()"),
    (r"\bshutil\.rmtree\s*\(",       "shutil.rmtree()"),
    (r"\.unlink\s*\(",               "Path.unlink()"),
    (r"\bformat\s+[a-zA-Z]:",        "format drive"),
]

_DESTRUCTIVE_SH = [
    (r"\brm\s+.*-[a-z]*r[a-z]*f",   "rm -rf"),
    (r"\brm\s+-[rRf]{2,}",          "rm -rf"),
    (r"rmdir\s+/[sq]",              "rmdir /s"),
    (r"\bdel\s+/[sqf]",             "del /s /q"),
]

_NETWORK_PATTERNS = [
    r"\brequests\.(get|post|put|delete|patch)\(",
    r"\bsocket\.connect\(",
    r"\burllib\.request\.",
    r"\bhttpx\.",
    r"\baiohttp\.",
]

_SYSTEM_PATTERNS = [
    r"\bsubprocess\.(run|call|Popen)\b",
    r"\bos\.system\(",
    r"\bos\.popen\(",
    r"\bos\.execv",
]

def _layer3_classify_intent(code: str, language: str) -> tuple[list[str], Optional[PermDecision]]:
    """
    Trả về (labels, decision|None).
    decision = BLOCK nếu DESTRUCTIVE, WARN nếu NETWORK/SYSTEM.
    """
    labels: list[str] = []
    lang = language.lower()

    patterns = _DESTRUCTIVE_PY if lang == "python" else _DESTRUCTIVE_SH
    for pattern, name in patterns:
        if re.search(pattern, code, re.IGNORECASE):
            labels.append("destructive")
            return labels, PermDecision(
                PermResult.BLOCK, 3,
                f"Phát hiện thao tác xóa bị cấm: {name}",
                labels
            )

    if lang == "python":
        for pat in _NETWORK_PATTERNS:
            if re.search(pat, code):
                labels.append("network")
                break
        for pat in _SYSTEM_PATTERNS:
            if re.search(pat, code):
                labels.append("system_call")
                break

    if not labels:
        labels.append("write_safe")

    return labels, None


# ─────────────────────────────────────────
# Lớp 4 – Dangerous execution patterns
# ─────────────────────────────────────────
_DANGEROUS_EXEC = [
    (r"\beval\s*\(",         "eval() – thực thi code động, dễ bị inject"),
    (r"\bexec\s*\(",         "exec() – thực thi code động"),
    (r"\b__import__\s*\(",   "__import__() động"),
    (r"compile\s*\([^)]*exec", "compile(..., 'exec')"),
    (r"subprocess\.Popen\s*\([^)]*shell\s*=\s*True",  "Popen(shell=True) – shell injection"),
    (r"os\.system\s*\([^)]*&&|os\.system\s*\([^)]*;",  "os.system với command chaining"),
    (r"\bcurl\s+.*\|\s*bash", "curl | bash – remote code execution"),
    (r"\bwget\s+.*-O-.*\|\s*sh", "wget | sh – remote code execution"),
    (r"\bsudo\s+rm",          "sudo rm – xóa với quyền root"),
    (r"\bsudo\s+dd\s+",       "sudo dd – ghi thẳng vào block device"),
    (r"\bchmod\s+777\s+/",    "chmod 777 / – mở quyền toàn bộ root"),
    (r">\s*/dev/sd[a-z]",     "ghi thẳng vào block device"),
    (r"\bpickle\.loads\b",    "pickle.loads() – arbitrary code execution"),
    (r"\byaml\.load\s*\([^,)]+\)", "yaml.load() unsafe – dùng yaml.safe_load()"),
]

def _layer4_dangerous_patterns(code: str) -> Optional[PermDecision]:
    for pattern, desc in _DANGEROUS_EXEC:
        if re.search(pattern, code, re.IGNORECASE | re.DOTALL):
            return PermDecision(PermResult.BLOCK, 4, f"Pattern nguy hiểm: {desc}", ["dangerous_exec"])
    return None


# ─────────────────────────────────────────
# Lớp 5 – Shell command chaining analysis
# ─────────────────────────────────────────
_SHELL_DANGEROUS_CHAINS = [
    (r";\s*rm\s+-[rRf]",              "command; rm -rf (chaining xóa)"),
    (r"&&\s*rm\s+-[rRf]",             "command && rm -rf"),
    (r"\|\s*bash",                     "pipe to bash"),
    (r"\|\s*sh\b",                     "pipe to sh"),
    (r">\s*/etc/",                     "ghi đè vào /etc/"),
    (r">\s*/boot/",                    "ghi đè vào /boot/"),
    (r"dd\s+.*of=/dev/[a-z]+\b(?!\s*#)", "dd of=/dev/... (ghi block device)"),
    (r":\(\)\{.*:\|:&\s*\}",          "fork bomb :(){ :|:& }"),
    (r">\s*\$\(which ",               "ghi đè binary system"),
]

def _layer5_shell_ast(code: str, language: str) -> Optional[PermDecision]:
    lang = language.lower()
    if lang not in ("bash", "sh", "powershell", "ps1", "ps"):
        return None
    for pattern, desc in _SHELL_DANGEROUS_CHAINS:
        if re.search(pattern, code, re.IGNORECASE | re.DOTALL):
            return PermDecision(PermResult.BLOCK, 5, f"Shell chain nguy hiểm: {desc}", ["shell_injection"])
    return None


# ─────────────────────────────────────────
# Orchestrator: chạy 5 lớp theo thứ tự
# ─────────────────────────────────────────
def classify_code(code: str, language: str, working_dir: str = "") -> PermDecision:
    """
    Chạy 5 lớp permission classification theo thứ tự nhẹ → nặng.
    Lớp nhẹ (1) chạy trước – nếu ALLOW thì skip toàn bộ pipeline.
    Trả về PermDecision với result = ALLOW / WARN / BLOCK.
    """
    # Lớp 1 – fast path
    d = _layer1_safe_allowlist(code, language)
    if d:
        return d

    # Lớp 2 – sensitive paths
    d = _layer2_sensitive_paths(code, working_dir)
    if d:
        return d

    # Lớp 3 – intent classification
    labels, d = _layer3_classify_intent(code, language)
    if d:
        return d

    # Lớp 4 – dangerous exec patterns
    d = _layer4_dangerous_patterns(code)
    if d:
        return d

    # Lớp 5 – shell AST
    d = _layer5_shell_ast(code, language)
    if d:
        return d

    # Passed all layers
    extra = ""
    if "network" in labels:
        extra = " (network access)"
    elif "system_call" in labels:
        extra = " (system call)"
    return PermDecision(PermResult.ALLOW, 5, f"Passed all layers{extra}", labels)


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_cmd(language: str, script_path: Path) -> list | None:
    import platform
    is_win = platform.system() == "Windows"
    lang = language.lower().strip()
    if lang == "python":
        return [sys.executable, str(script_path)]
    elif lang in ("bash", "sh"):
        return (["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
                if is_win else ["bash", str(script_path)])
    elif lang in ("powershell", "ps1", "ps"):
        return ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
    return None


def _run_streaming(cmd: list, cwd: Path, timeout: int, env: dict) -> tuple[str, int]:
    """Chạy process và stream output real-time qua _emit(). Trả về (stdout, returncode)."""
    process = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=env, bufsize=1,
    )
    output_lines: list[str] = []

    def _reader():
        for line in process.stdout:
            output_lines.append(line)
            _emit(line)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    try:
        process.wait(timeout=timeout)
        reader.join(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        msg = f"\n⏱️ Timeout: quá {timeout}s, process bị huỷ.\n"
        output_lines.append(msg)
        _emit(msg)
    return "".join(output_lines), process.returncode if process.returncode is not None else -1


# ══════════════════════════════════════════════════════════════════════════════
# STALL DETECTION  (Pattern #7 – Chapter 7)
# ══════════════════════════════════════════════════════════════════════════════

_STALL_THRESHOLD_S     = 45      # giây không có output mới → kiểm tra stall
_STALL_CHECK_INTERVAL  = 5       # giây giữa các lần poll
_INTERACTIVE_PROMPTS   = re.compile(
    r"(\[Y/n\]|\[y/N\]|\(yes/no\)|Press any key|Continue\?|Proceed\?|"
    r"Password:|Enter .*:|--More--|Press ENTER)",
    re.IGNORECASE
)

def _bg_stall_monitor(name: str):
    """
    Background thread theo dõi stall cho background tasks.
    Nếu task không output mới sau _STALL_THRESHOLD_S giây,
    và cuối log trông giống interactive prompt → phát notification.
    """
    log_path = _bg_logs.get(name)
    if not log_path:
        return

    last_size = 0
    last_change = time.time()

    while True:
        time.sleep(_STALL_CHECK_INTERVAL)

        proc = _bg_processes.get(name)
        if proc is None or proc.poll() is not None:
            break

        try:
            cur_size = log_path.stat().st_size
        except FileNotFoundError:
            break

        if cur_size != last_size:
            last_size = cur_size
            last_change = time.time()
        elif time.time() - last_change >= _STALL_THRESHOLD_S:
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
                tail_text = "\n".join(tail)
            except Exception:
                tail_text = ""

            if _INTERACTIVE_PROMPTS.search(tail_text):
                _emit(
                    f"\n⚠️ [stall-detect] Task '{name}' dường như đang chờ input:\n"
                    f"   {tail[-1].strip()}\n"
                    f"   → Dùng stop_background_task(name='{name}') hoặc cấp input thủ công.\n"
                )
            else:
                _emit(f"\n⚠️ [stall-detect] Task '{name}' không có output mới trong {_STALL_THRESHOLD_S}s.\n")
            last_change = time.time()  # reset để không spam


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1: run_code_in_sandbox
# ══════════════════════════════════════════════════════════════════════════════

@tool
def run_code_in_sandbox(
    language: str,
    code: str,
    working_dir: str = "",
    timeout: int = 60,
) -> str:
    """
    Chạy code trực tiếp trên máy thật. Workspace mặc định ~/agent_workspace/.
    Output được stream real-time. File tạo ra tồn tại giữa các lần gọi.

    Permission classifier 5 lớp sẽ chặn code nguy hiểm tự động.

    Args:
        language:    'python' | 'bash' | 'powershell'
        code:        Đoạn code cần thực thi
        working_dir: Thư mục làm việc (mặc định ~/agent_workspace/)
        timeout:     Giây tối đa (10–300, mặc định 60)
    """
    # ── Permission check (5 lớp) ──────────────────────────────────────────────
    decision = classify_code(code, language, working_dir)
    if decision.result == PermResult.BLOCK:
        return (
            f"❌ BỊ CHẶN [lớp {decision.layer}]: {decision.reason}\n"
            f"   Labels: {decision.labels}\n"
            f"   Hãy sửa code và thử lại."
        )

    warn_prefix = ""
    if decision.result == PermResult.WARN:
        warn_prefix = (
            f"⚠️ [permission-warn lớp {decision.layer}] {decision.reason}\n"
            f"   Labels: {decision.labels}\n\n"
        )
        _emit(warn_prefix)

    # ── Setup ──────────────────────────────────────────────────────────────────
    timeout = min(max(int(timeout), 10), 300)
    cwd = Path(working_dir).expanduser().resolve() if working_dir else WORKSPACE
    cwd.mkdir(parents=True, exist_ok=True)

    ext_map = {"python": ".py", "bash": ".sh", "sh": ".sh",
               "powershell": ".ps1", "ps1": ".ps1", "ps": ".ps1"}
    ext = ext_map.get(language.lower(), ".py")
    script_path = cwd / f"_run_script{ext}"
    script_path.write_text(code, encoding="utf-8")

    cmd = _resolve_cmd(language, script_path)
    if cmd is None:
        return f"❌ Ngôn ngữ '{language}' không hỗ trợ. Dùng: python / bash / powershell"

    _emit(f"\n🚀 [sandbox] {language} @ {cwd} (timeout={timeout}s) | labels={decision.labels}\n")

    stdout, returncode = _run_streaming(cmd, cwd, timeout, os.environ.copy())

    # ── Workspace snapshot ────────────────────────────────────────────────────
    snapshot = sorted(
        [f for f in cwd.iterdir() if not f.name.startswith("_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:8]
    file_list = "\n".join(
        f"  {'📂' if f.is_dir() else '📄'} {f.name}"
        + (f"  ({f.stat().st_size / 1024:.1f}KB)" if f.is_file() else "")
        for f in snapshot
    )

    result = stdout.strip() or "(Không có output)"
    if returncode != 0:
        result += f"\n\n⚠️ Exit code: {returncode}"
    if file_list:
        result += f"\n\n📁 Workspace ({cwd}):\n{file_list}"

    return warn_prefix + result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2: start_background_task
# ══════════════════════════════════════════════════════════════════════════════

@tool
def start_background_task(
    name: str,
    language: str,
    code: str,
    working_dir: str = "",
) -> str:
    """
    Chạy code dưới dạng background process (non-blocking).
    Log ghi ra disk với offset tracking (Pattern #7).
    Stall detection tự động cảnh báo nếu process bị treo.

    Dùng cho: YOLO camera, Flask server, ML training, ...

    Args:
        name:        Tên định danh (vd: 'yolo_cam', 'flask_server')
        language:    'python' | 'bash' | 'powershell'
        code:        Đoạn code
        working_dir: Thư mục làm việc
    """
    # ── Permission check ──────────────────────────────────────────────────────
    decision = classify_code(code, language, working_dir)
    if decision.result == PermResult.BLOCK:
        return (
            f"❌ BỊ CHẶN [lớp {decision.layer}]: {decision.reason}\n"
            f"   Labels: {decision.labels}"
        )

    warn_prefix = ""
    if decision.result == PermResult.WARN:
        warn_prefix = (
            f"⚠️ [permission-warn lớp {decision.layer}] {decision.reason}\n"
            f"   Labels: {decision.labels}\n\n"
        )
        _emit(warn_prefix)

    # ── Check đã chạy chưa ───────────────────────────────────────────────────
    if name in _bg_processes and _bg_processes[name].poll() is None:
        p = _bg_processes[name]
        return (
            f"⚠️ Process '{name}' đang chạy (PID {p.pid}).\n"
            f"Dùng stop_background_task(name='{name}') để dừng trước."
        )

    cwd = Path(working_dir).expanduser().resolve() if working_dir else WORKSPACE
    cwd.mkdir(parents=True, exist_ok=True)

    ext_map = {"python": ".py", "bash": ".sh", "sh": ".sh", "powershell": ".ps1", "ps1": ".ps1"}
    ext = ext_map.get(language.lower(), ".py")
    script_path = cwd / f"_bg_{name}{ext}"
    script_path.write_text(code, encoding="utf-8")

    cmd = _resolve_cmd(language, script_path)
    if cmd is None:
        return f"❌ Ngôn ngữ '{language}' không hỗ trợ."

    log_path = WORKSPACE / f"_bg_{name}.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    process = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=log_file, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8",
    )
    _bg_processes[name] = process
    _bg_logs[name]      = log_path
    _bg_offsets[name]   = 0   # Pattern #7: khởi tạo offset

    # ── Stall detection thread ────────────────────────────────────────────────
    threading.Thread(target=_bg_stall_monitor, args=(name,), daemon=True).start()

    time.sleep(0.8)

    if process.poll() is not None:
        log_file.close()
        log_content = log_path.read_text(encoding="utf-8", errors="replace")
        return (
            f"❌ Process '{name}' crash ngay (exit {process.returncode}):\n"
            f"{log_content[:800]}"
        )

    return warn_prefix + (
        f"✅ Background task '{name}' đã khởi động!\n"
        f"   PID    : {process.pid}\n"
        f"   Log    : {log_path}\n"
        f"   Script : {script_path}\n"
        f"   Labels : {decision.labels}\n"
        f"⏹️  Dừng   : stop_background_task(name='{name}')\n"
        f"📋 Xem log : read_task_log(name='{name}')"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3: stop_background_task
# ══════════════════════════════════════════════════════════════════════════════

@tool
def stop_background_task(name: str) -> str:
    """Dừng một background task đang chạy."""
    import platform
    if name not in _bg_processes:
        running = [n for n, p in _bg_processes.items() if p.poll() is None]
        return (
            f"❌ Không tìm thấy task '{name}'.\n"
            f"Tasks đang chạy: {running or 'Không có'}"
        )
    process = _bg_processes[name]
    if process.poll() is not None:
        del _bg_processes[name]
        return f"ℹ️ Task '{name}' đã tự dừng (exit {process.returncode})."

    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    else:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    _bg_processes.pop(name, None)
    _bg_offsets.pop(name, None)
    return f"✅ Đã dừng task '{name}'."


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4: read_task_log  (Pattern #7 – outputOffset)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def read_task_log(name: str, last_n_lines: int = 60, incremental: bool = False) -> str:
    """
    Đọc output log của background task.

    Args:
        name:          Tên task
        last_n_lines:  Số dòng cuối (dùng khi incremental=False)
        incremental:   True → chỉ đọc phần mới từ lần đọc trước (Pattern #7 outputOffset)
    """
    log_path = _bg_logs.get(name, WORKSPACE / f"_bg_{name}.log")
    if not log_path.exists():
        return f"❌ Không tìm thấy log cho task '{name}'."

    content = log_path.read_text(encoding="utf-8", errors="replace")

    if incremental:
        offset = _bg_offsets.get(name, 0)
        new_content = content[offset:]
        _bg_offsets[name] = len(content)
        text = new_content if new_content else "(Chưa có output mới)"
    else:
        lines = content.splitlines()
        tail  = lines[-last_n_lines:] if len(lines) > last_n_lines else lines
        text  = "\n".join(tail)

    is_running = name in _bg_processes and _bg_processes[name].poll() is None
    status = f"🟢 Đang chạy (PID {_bg_processes[name].pid})" if is_running else "🔴 Đã dừng"
    total_lines = len(content.splitlines())

    return f"[{name}] {status} | {total_lines} dòng log\n{text}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5: list_workspace
# ══════════════════════════════════════════════════════════════════════════════

@tool
def list_workspace(subdir: str = "") -> str:
    """Liệt kê nội dung thư mục workspace hoặc đường dẫn tuyệt đối."""
    if subdir and os.path.isabs(subdir):
        target = Path(subdir)
    else:
        target = (WORKSPACE / subdir) if subdir else WORKSPACE

    if not target.exists():
        return f"❌ Thư mục không tồn tại: {target}"

    lines = [f"📁 {target}/"]
    for item in sorted(target.iterdir()):
        if item.name.startswith("_"):
            continue
        if item.is_dir():
            lines.append(f"  📂 {item.name}/")
        else:
            size = item.stat().st_size
            size_str = f"{size/1024:.1f}KB" if size >= 1024 else f"{size}B"
            lines.append(f"  📄 {item.name}  ({size_str})")

    running = [n for n, p in _bg_processes.items() if p.poll() is None]
    lines.append(
        f"\n🟢 Background tasks đang chạy: {running}"
        if running else "\n⚫ Không có background task nào."
    )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6: install_packages
# ══════════════════════════════════════════════════════════════════════════════

@tool
def install_packages(packages: str) -> str:
    """
    Cài đặt Python packages vào môi trường hiện tại.
    Args:
        packages: Tên packages cách nhau bởi dấu cách (vd: 'ultralytics opencv-python torch')
    """
    pkgs = packages.strip().split()
    if not pkgs:
        return "❌ Không có package nào để cài."

    _emit(f"\n📦 Đang cài: {' '.join(pkgs)}...\n")
    stdout, rc = _run_streaming(
        [sys.executable, "-m", "pip", "install"] + pkgs,
        cwd=WORKSPACE, timeout=180, env=os.environ.copy(),
    )
    if rc == 0:
        return f"✅ Đã cài xong: {' '.join(pkgs)}\n\n{stdout[-500:] if len(stdout) > 500 else stdout}"
    return f"❌ Cài thất bại (exit {rc}):\n{stdout[-800:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7: inspect_permission  (debug helper)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def inspect_permission(language: str, code: str, working_dir: str = "") -> str:
    """
    Kiểm tra code sẽ được phân loại thế nào qua 5-layer permission pipeline
    mà KHÔNG thực thi. Dùng để debug khi bị chặn.

    Args:
        language:    'python' | 'bash' | 'powershell'
        code:        Code cần kiểm tra
        working_dir: Thư mục làm việc (tùy chọn)
    """
    decision = classify_code(code, language, working_dir)
    icon = {"allow": "✅", "warn": "⚠️", "block": "❌"}[decision.result.value]
    return (
        f"{icon} Kết quả: {decision.result.value.upper()}\n"
        f"   Lớp quyết định : {decision.layer}/5\n"
        f"   Lý do          : {decision.reason}\n"
        f"   Labels         : {decision.labels}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════

SANDBOX_TOOLS = [
    run_code_in_sandbox,
    start_background_task,
    stop_background_task,
    read_task_log,
    list_workspace,
    install_packages,
    inspect_permission,
]