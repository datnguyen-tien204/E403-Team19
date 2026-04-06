"""
exclusion_config.py
Cấu hình các thư mục và tệp nhạy cảm (như thư mục hệ thống Windows, thư mục gốc C:...)
để loại trừ khỏi các thao tác nguy hiểm của Sandbox, nhằm đảm bảo an toàn cho máy chủ/máy trạm.
"""

import os

# Danh sách các thư mục hệ thống, nhạy cảm cần được bảo vệ
# Thao tác đọc thì vẫn an toàn, nhưng chặn các hành vi tạo, sửa, xóa, chạy mã độc trong này.
SENSITIVE_DIRS = [
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
    r"C:\Users\Default",
    r"C:\$Recycle.Bin",
    r"C:\System Volume Information",
    r"C:\PerfLogs",
    r"C:\Recovery",
    r"C:\Documents and Settings",
    r"C:\Boot",
]

SENSITIVE_FILES = [
    r"C:\pagefile.sys",
    r"C:\swapfile.sys",
    r"C:\hiberfil.sys",
    r"C:\DumpStack.log.tmp",
    r"C:\bootmgr",
]

def is_safe_working_dir(path: str) -> bool:
    """Kiểm tra xem thư mục có nằm trong danh sách cấm hoặc là gốc hệ điều hành không."""
    try:
        abs_path = os.path.abspath(path).lower()
        if abs_path == r"c:\"" or abs_path == "c:\\":
            return False

        for s_dir in SENSITIVE_DIRS:
            if abs_path.startswith(os.path.abspath(s_dir).lower()):
                return False

        for s_file in SENSITIVE_FILES:
            if abs_path == os.path.abspath(s_file).lower():
                return False

        return True
    except Exception:
        return False

def check_code_for_sensitive_paths(code: str) -> tuple[bool, str]:
    """Kiểm tra nhanh nếu trong source code chứa chuỗi đường dẫn nhạy cảm."""
    code_lower = code.lower().replace('/', '\\')

    # Check các dirs
    for d in SENSITIVE_DIRS:
        if d.lower() in code_lower:
            return False, f"Code chứa đường dẫn nhạy cảm bị cấm: {d}"

    # Check các files
    for f in SENSITIVE_FILES:
        if f.lower() in code_lower:
            return False, f"Code chứa tệp nhạy cảm bị cấm: {f}"

    # Check một số pattern nguy hiểm chung nhắm vào Windows
    if 'c:\\windows\\system32' in code_lower or 'c:\\windows' in code_lower:
        return False, "Tránh thao tác trực tiếp vào C:\\Windows"

    return True, ""

