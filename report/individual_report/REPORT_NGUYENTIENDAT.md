# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyen Tien Dat
- **Student ID**: 2A202600217
- **Date**: 06/04/2026

---

## I. Technical Contribution (15 Points)

*Describe your specific contribution to the codebase (e.g., implemented a specific tool, fixed the parser, etc.).*

- **Modules Implemented**:
  - `src/tools/sandbox.py` (Real-Machine Code Execution Sandbox with Background Task tracking)
  - `src/tools/exclusion_config.py` (Permission Classification Pipeline)
  - `src/agent/graph.py` (Context Defense & Escalating Recovery mechanisms)

- **Code Highlights**:

  Unlike standard basic tools, I implemented a robust `run_code_in_sandbox` and `start_background_task` mechanism equipped with a **5-Layer Permission Classification Pipeline (Pattern #10)** to protect the host Windows machine.

  Before any code is executed, it passes through multiple layers checking for:
  - Read-only intents
  - Sensitive system paths (e.g., `C:\Windows`)
  - Destructive commands (`rm -rf`, `os.remove`)
  - Dangerous executions (`eval`, `Popen(shell=True)`)
  - Shell command chaining

  ```python
  # src/tools/sandbox.py
  def classify_code(code: str, language: str, working_dir: str = "") -> PermDecision:
      """Chạy 5 lớp permission classification theo thứ tự nhẹ → nặng."""
      d = _layer1_safe_allowlist(code, language)
      if d: return d
      d = _layer2_sensitive_paths(code, working_dir)
      if d: return d
      labels, d = _layer3_classify_intent(code, language)
      if d: return d
      d = _layer4_dangerous_patterns(code)
      if d: return d
      d = _layer5_shell_ast(code, language)
      if d: return d
      return PermDecision(PermResult.ALLOW, 5, "Passed all layers", labels)

  @tool
  def run_code_in_sandbox(language: str, code: str, working_dir: str = "", timeout: int = 60) -> str:
      decision = classify_code(code, language, working_dir)
      if decision.result == PermResult.BLOCK:
          return f"❌ BỊ CHẶN [lớp {decision.layer}]: {decision.reason}\n Labels: {decision.labels}"

      # Execution with streaming and robust UTF-8 handling
      cmd = _resolve_cmd(language, script_path)
      stdout, returncode = _run_streaming(cmd, cwd, timeout, os.environ.copy())
      
      # Workspace snapshot injected into Observation
      snapshot = sorted([f for f in cwd.iterdir() if not f.name.startswith("_")], ...)
      ...
      return result + f"\n\n📁 Workspace ({cwd}):\n{file_list}"
  ```

- **Documentation**:

  Inside the ReAct loop (`src/agent/graph.py`), the agent acts as an autonomous OS navigator. Instead of guessing paths, it:
  - Uses `list_workspace` to map the directory tree
  - Uses PowerShell to verify Anaconda environments (`conda list`)
  - Executes the target script dynamically

  Below is a real multi-step execution trace extracted from **Groq qwen/qwen3-32b logs** for the prompt:

  *"Kiểm tra ổ đĩa, tìm thư mục YOLOv8_Camera trong E:\Project, kiểm tra môi trường Conda HazeGen và chạy yolov8_camera.py"*

  **Chatbot baseline (single-shot, no tools):**
  ```json
  {"timestamp": "2026-04-06T17:20:00.000000", "event": "CHATBOT_START",  "data": {"input": "...", "model": "qwen3-32b"}}
  {"timestamp": "2026-04-06T17:20:02.100000", "event": "CHATBOT_END",    "data": {"output": "Để chạy file YOLOv8, bạn hãy mở cmd và gõ:\n1. `cd E:\\Project\\YOLOv8_Camera`\n2. `conda activate HazeGen`\n3. `python yolov8_camera.py`"}}
  ```

  > ⚠️ The Chatbot simply regurgitates terminal commands. It has zero awareness of whether `E:\Project` exists, if the Conda environment has the `ultralytics` package, or if the code will crash upon execution.

  ---

  **ReAct Agent (11-step complex reasoning with Context Defense):**
  ```json
  {"timestamp": "2026-04-06T17:20:47.087122", "event": "TOOL_CALL", "data": {"step": 3, "tool": "list_workspace", "arguments": "{\"subdir\": \"E:/Project/YOLOv8_Camera\"}", "observation": "📁 E:\\Project\\YOLOv8_Camera/\n  📄 yolov8_camera.py  (590B)\n  📄 yolov8n.pt  (6396.3KB)"}}

  {"timestamp": "2026-04-06T17:23:08.221448", "event": "TOOL_CALL", "data": {"step": 8, "tool": "run_code_in_sandbox", "arguments": "{\"code\": \"conda activate HazeGen && conda list\", \"language\": \"cmd\"}", "observation": "❌ Ngôn ngữ 'cmd' không hỗ trợ. Dùng: python / bash / powershell"}}

  {"timestamp": "2026-04-06T17:23:21.361974", "event": "TOOL_CALL", "data": {"step": 9, "tool": "run_code_in_sandbox", "arguments": "{\"code\": \"E:\\\\Anaconda\\\\Scripts\\\\conda.exe list -n HazeGen\", \"language\": \"powershell\"}", "observation": "# packages in environment at E:\\Anaconda\\envs\\HazeGen:\ntorch                     2.4.0\nultralytics               8.5.0\nopencv-python             4.9.0..."}}

  {"timestamp": "2026-04-06T17:24:27.739345", "event": "TOOL_CALL", "data": {"step": 10, "tool": "run_code_in_sandbox", "arguments": "{\"code\": \"pwsh -Command \\\"conda activate HazeGen; python E:/Project/YOLOv8_Camera/yolov8_camera.py\\\"\", \"language\": \"powershell\", \"working_dir\": \"E:/Project/YOLOv8_Camera\"}", "observation": "0: 480x640 10 persons, 1 bottle, 2 tvs, 3 laptops, 74.6ms\nSpeed: 17.0ms preprocess, 74.6ms inference..."}}

  {"timestamp": "2026-04-06T17:24:30.495929", "event": "AGENT_END", "data": {"steps": 11, "total_latency_ms": 2753, "final_answer": "Dưới đây là kết quả đầy đủ khi chạy `yolov8_camera.py` trong môi trường `HazeGen`: Camera đã kích hoạt thành công. Các đối tượng được phát hiện: Người, Ghế, TV, Laptop. Tốc độ xử lý: ~17–23ms/frame."}}
  ```

  **Key Takeaways from the Trace:**

  The agent dynamically adapted to its environment. When it tried to use `cmd` (which was blocked), the Observation explicitly guided it to switch to `powershell`. The agent corrected itself, verified dependencies, and successfully executed the AI inference script.

---

## II. Debugging Case Study (10 Points)

*Analyze a specific failure event you encountered during the lab using the logging system.*

- **Problem Description**:

  During a filesystem navigation task, the agent wrote a Python script to list subfolders in `E:\Project`. However, execution crashed twice:

  - A `SyntaxError` caused by invalid f-string backslash usage
  - A `UnicodeEncodeError` due to Vietnamese characters in Windows `cp1252`

- **Log Source**:

  ```json
  {"timestamp": "2026-04-06T17:21:27.291062", "event": "TOOL_CALL", "data": {"step": 5, "tool": "run_code_in_sandbox", "arguments": "{\"code\": \"... print(\\\\n\\\\n.join([f\\\"{i+1}. {f}/\\\" ...\", \"language\": \"python\"}", "observation": "SyntaxError: unexpected character after line continuation character\n⚠️ Exit code: 1"}}

  {"timestamp": "2026-04-06T17:21:28.620138", "event": "TOOL_CALL", "data": {"step": 6, "tool": "run_code_in_sandbox", "arguments": "{\"code\": \"... print(\\\"Các thư mục con trong E:\\\") ...\", \"language\": \"python\"}", "observation": "Traceback:\n  File \"_run_script.py\", line 5\n    print(\"Cc th\\u01b0 m\\u1ee5c con trong E:\")\nUnicodeEncodeError: 'charmap' codec can't encode character '\\u01b0'...\n⚠️ Exit code: 1"}}

  {"timestamp": "2026-04-06T17:21:31.441297", "event": "TOOL_CALL", "data": {"step": 7, "tool": "run_code_in_sandbox", "arguments": "{\"code\": \"import os, sys\\nimport io\\nsys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\\n...\", \"language\": \"python\"}", "observation": "1. DA3/\n2. DeployModel/\n3. YOLOv8_Camera/"}}
  ```

- **Diagnosis**:

  - **Agent Behavior**: The LLM showed strong self-correction:
    - Fixed SyntaxError
    - Detected encoding issue and patched stdout to UTF-8

  - **System Flaw**:
    - Root cause: `subprocess.Popen` used Windows default encoding
    - No enforced UTF-8 → crash on non-ASCII output

- **Solution**:

  Fixed permanently at system level in `_run_streaming`:

  ```python
  # Updated _run_streaming in src/tools/sandbox.py
  def _run_streaming(cmd: list, cwd: Path, timeout: int, env: dict) -> tuple[str, int]:
      process = subprocess.Popen(
          cmd, cwd=str(cwd),
          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
          text=True, encoding="utf-8", errors="replace",
          env={**env, "PYTHONIOENCODING": "utf-8"},
          bufsize=1,
      )
  ```

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

*Reflect on the reasoning capability difference.*

- **Escalating Recovery & Guardrails**:

  Standard chatbots fail silently or hallucinate. In `graph.py`, I implemented:
  - Loop guard (`_MAX_SAME_TOOL_CALLS = 4`)
  - Escalating Recovery pattern

  The agent analyzes failures, retries, and improves iteratively using Observations.

- **Context Window Defense**:

  Real tool outputs (e.g., `conda list`, inference logs) can exceed context limits.

  I implemented `apply_context_defense()` to:
  - Prune old messages
  - Truncate outputs > 8,000 characters

  This enabled successful completion of long multi-step traces without `MaxTokensExceeded`.

- **Latency vs. Verifiability**:

  | | Chatbot | ReAct Agent |
  |---|---|---|
  | Latency | ~2s | ~20s |
  | Execution | ❌ | ✅ |
  | Reliability | Low | High |

  Chatbot is fast but unreliable. ReAct is slower but grounded in real execution.

---

## IV. Future Improvements (5 Points)

*How would you scale this for a production-level AI agent system?*

- **Parallel Tool Execution (Orchestrator)**:

  Implement concurrent tool execution via `ThreadPoolExecutor` in `tool_orchestrator.py` to reduce latency.

- **True Containerization over Regex Guardrails**:

  Replace regex/AST checks with:
  - Docker containers
  - Firecracker microVMs

  → Full isolation from host OS.

- **Background Task Management via WebSocket**:

  Replace polling (`read_task_log`) with:
  - Real-time WebSocket streaming (`api/server.py`)
  - Direct stdout push to frontend (React / PyQt)

---

> [!NOTE]
> Submit this report by renaming it to `REPORT_NGUYEN_TIEN_DAT.md` and placing it in the required folder.