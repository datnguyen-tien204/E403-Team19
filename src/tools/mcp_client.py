from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model

GOOGLE_CREDENTIALS_FILE = r"E:\Working\VinUni-FinalProj\AgentAI-v2\AgentAI\credential\google\gmail\credentials.json"
GOOGLE_TOKEN_CALENDAR   = r"E:\Working\VinUni-FinalProj\AgentAI-v2\AgentAI\credential\google\gmail\token_calendar.json"
GOOGLE_TOKEN_GMAIL      = r"E:\Working\VinUni-FinalProj\AgentAI-v2\AgentAI\credential\google\gmail\token_gmail.json"
GOOGLE_TOKEN_SHEETS     = r"E:\Working\VinUni-FinalProj\AgentAI-v2\AgentAI\credential\google\gmail\token_sheets.json"

MCP_SERVER_CONFIGS: dict[str, dict] = {
    "excel": {
        "package": "excel-mcp-server",
        "command": [sys.executable, "-m", "excel_mcp_server"],
        "description": "Đọc/ghi file Excel và CSV",
        "env": {},
    },
    "google_sheets": {
        "package": "mcp-google-sheets",
        "command": [sys.executable, "-m", "mcp_google_sheets"],
        "description": "Đọc/ghi Google Sheets",
        "env": {
            "GOOGLE_CREDENTIALS_FILE": GOOGLE_CREDENTIALS_FILE,
            "GOOGLE_TOKEN_FILE":       GOOGLE_TOKEN_SHEETS,
        },
    },
    "google_calendar": {
        "package": "mcp-google-calendar",
        "command": [sys.executable, "-m", "mcp_google_calendar"],
        "description": "Đọc/tạo sự kiện Google Calendar",
        "env": {
            "GOOGLE_CREDENTIALS_FILE": GOOGLE_CREDENTIALS_FILE,
            "GOOGLE_TOKEN_FILE":       GOOGLE_TOKEN_CALENDAR,
        },
    },
    "gmail": {
        "package": "mcp-gmail",
        "command": [sys.executable, "-m", "mcp_gmail"],
        "description": "Đọc/gửi Gmail",
        "env": {
            "GOOGLE_CREDENTIALS_FILE": GOOGLE_CREDENTIALS_FILE,
            "GOOGLE_TOKEN_FILE":       GOOGLE_TOKEN_GMAIL,
        },
    },
    "notion": {
        "package": "notion-mcp-server",
        "command": [sys.executable, "-m", "notion_mcp_server"],
        "description": "Đọc/ghi Notion pages và databases",
        "env": {},
    },
    "github": {
        "package": "mcp-github",
        "command": [sys.executable, "-m", "mcp_github"],
        "description": "GitHub repos, issues, PRs",
        "env": {},
    },
}


class MCPSession:
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.config = MCP_SERVER_CONFIGS.get(server_name, {})
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()

    def start(self) -> bool:
        try:
            import os
            env = {**os.environ, **self.config.get("env", {})}
            self._process = subprocess.Popen(
                self.config["command"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
            )
            self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ai-agent", "version": "1.0"},
            })
            print(f"[MCP] {self.server_name} started ✓")
            return True
        except Exception as e:
            print(f"[MCP] Lỗi khởi động {self.server_name}: {e}")
            return False

    def stop(self):
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                self._process.kill()

    def _send_request(self, method: str, params: dict) -> dict:
        with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
            try:
                line = json.dumps(request) + "\n"
                self._process.stdin.write(line)
                self._process.stdin.flush()
                response_line = self._process.stdout.readline()
                if response_line:
                    return json.loads(response_line)
                return {"error": "empty response"}
            except Exception as e:
                return {"error": str(e)}

    def list_tools(self) -> list[dict]:
        try:
            response = self._send_request("tools/list", {})
            return response.get("result", {}).get("tools", [])
        except Exception as e:
            print(f"[MCP] list_tools error: {e}")
            return []

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        try:
            response = self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })
            if "error" in response:
                return f"❌ MCP error: {response['error']}"
            content = response.get("result", {}).get("content", [])
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(texts) if texts else str(content)
        except Exception as e:
            return f"❌ MCP call error: {str(e)}"


def _mcp_schema_to_pydantic(schema: dict, tool_name: str) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        python_type = type_map.get(prop_type, str)

        if prop_name not in required:
            from typing import Optional as Opt
            python_type = Optional[python_type]
            fields[prop_name] = (python_type, None)
        else:
            fields[prop_name] = (python_type, ...)

    if not fields:
        fields["query"] = (str, "")

    return create_model(f"{tool_name}_input", **fields)


def _wrap_tool(session: MCPSession, mcp_tool: dict) -> StructuredTool:
    tool_name    = mcp_tool.get("name", "unknown")
    description  = mcp_tool.get("description", "")
    input_schema = mcp_tool.get("inputSchema", {})

    lc_tool_name = f"mcp_{session.server_name}_{tool_name}"

    ArgsModel = _mcp_schema_to_pydantic(input_schema, lc_tool_name)

    def tool_func(**kwargs) -> str:
        return session.call_tool(tool_name, kwargs)

    return StructuredTool.from_function(
        func=tool_func,
        name=lc_tool_name,
        description=f"[{session.server_name.upper()}] {description}",
        args_schema=ArgsModel,
    )


_active_sessions: dict[str, MCPSession] = {}


def build_mcp_tools(server_names: list[str]) -> list[StructuredTool]:
    all_tools = []

    for name in server_names:
        if name not in MCP_SERVER_CONFIGS:
            print(f"[MCP] Server '{name}' không có trong config. Bỏ qua.")
            continue

        if name in _active_sessions:
            session = _active_sessions[name]
        else:
            session = MCPSession(name)
            if not session.start():
                print(f"[MCP] Không thể khởi động {name}. Bỏ qua.")
                print(f"[MCP] Gợi ý: pip install {MCP_SERVER_CONFIGS[name]['package']}")
                continue
            _active_sessions[name] = session

        mcp_tools = session.list_tools()
        if not mcp_tools:
            print(f"[MCP] {name}: không có tool nào.")
            continue

        wrapped = [_wrap_tool(session, t) for t in mcp_tools]
        all_tools.extend(wrapped)
        print(f"[MCP] {name}: loaded {len(wrapped)} tools → {[t.name for t in wrapped]}")

    return all_tools


def shutdown_mcp_servers():
    for name, session in _active_sessions.items():
        session.stop()
        print(f"[MCP] {name} stopped.")
    _active_sessions.clear()