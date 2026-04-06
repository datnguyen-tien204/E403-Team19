# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyen Tien Dat
- **Student ID**: 2A202600217
- **Date**: 06/04/2026

---

## I. Technical Contribution (15 Points)

*Describe your specific contribution to the codebase (e.g., implemented a specific tool, fixed the parser, etc.).*

- **Modules Implemented**:
  - `src/tools/code_sandbox.py`
  - `src/tools/tool_registry.py`

- **Code Highlights**:

  I implemented the `run_code_in_sandbox` tool, which allows the ReAct agent to execute Python / Bash / PowerShell code snippets inside an isolated subprocess and return the real `stdout` as an `Observation` back into the loop.

  ```python
  # src/tools/code_sandbox.py
  import subprocess, tempfile, os, sys

  TIMEOUT_SECONDS = 10

  def run_code_in_sandbox(code: str, language: str = "python") -> str:
      """
      Execute a code snippet in an isolated subprocess and return stdout/stderr.
      Supported languages: python, bash, powershell.
      """
      if not code or code.strip().lower() in ("", "none"):
          return "Error: No code provided."

      suffix_map = {"python": ".py", "bash": ".sh", "powershell": ".ps1"}
      cmd_map = {
          "python":     [sys.executable],
          "bash":       ["bash"],
          "powershell": ["powershell", "-ExecutionPolicy", "Bypass", "-File"],
      }

      if language not in suffix_map:
          return f"Error: Unsupported language '{language}'."

      with tempfile.NamedTemporaryFile(
          mode="w", suffix=suffix_map[language], delete=False, encoding="utf-8"
      ) as f:
          f.write(code)
          tmp_path = f.name

      try:
          proc = subprocess.run(
              cmd_map[language] + [tmp_path],
              capture_output=True, text=True, timeout=TIMEOUT_SECONDS
          )
          if proc.returncode != 0:
              return f"[ERROR] Exit {proc.returncode}\n{proc.stderr.strip()}"
          return proc.stdout.strip() or "(No output)"
      except subprocess.TimeoutExpired:
          return f"Error: Execution timed out after {TIMEOUT_SECONDS}s."
      finally:
          os.unlink(tmp_path)
  ```

  The tool is then registered into `tool_registry.py`:

  ```python
  # src/tools/tool_registry.py
  from src.tools.code_sandbox import run_code_in_sandbox

  TOOL_REGISTRY = {
      "run_code_in_sandbox": {
          "func": run_code_in_sandbox,
          "description": (
              "Execute a code snippet and return the real stdout output. "
              "Supports Python, Bash, and PowerShell. "
              "Input format: '<language>|||<code>'. "
              "Example: 'python|||print(2 + 2)'"
          ),
      },
  }
  ```

- **Documentation**:

  Inside the ReAct loop (`src/agent/react_runner.py`), once the parser extracts `Action: run_code_in_sandbox` and its `Action Input`, the runner splits the input on `|||` to separate `language` from `code`, then calls the function above. The return value is injected back into the conversation context as `Observation: <stdout>` and passed into the next LLM call. If `exit_code != 0`, the `[ERROR]` prefix in the Observation naturally triggers the LLM to diagnose and rewrite the code in the next `Thought` step.

  Real execution trace for the prompt *"find all prime numbers from 1 to 100 in Python"*:

  ```
  Thought      : I should run actual code to get an accurate result rather than guessing.
  Action       : run_code_in_sandbox
  Action Input : python|||primes = [n for n in range(2, 101)
                 if all(n % i != 0 for i in range(2, n))]
                 print(primes)
  Observation  : [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                  43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
  Final Answer : There are 25 prime numbers between 1 and 100: 2, 3, 5, ...
  ```

---

## II. Debugging Case Study (10 Points)

*Analyze a specific failure event you encountered during the lab using the logging system.*

- **Problem Description**: The agent repeatedly called `run_code_in_sandbox` with code containing a `pass` placeholder, causing the output to always be `[]`. The LLM treated this as a valid result and returned an incorrect `Final Answer` without raising any error.

- **Log Source** (snippet from `logs/2026-04-05.log`):

  ```
  [2026-04-05 20:43:11] [STEP 1]
    Thought     : I need to write a helper function to identify prime numbers.
    Action      : run_code_in_sandbox
    Action Input: python|||def is_prime(n):
                      pass  # TODO: implement
                  print([n for n in range(1, 101) if is_prime(n)])
    Observation : []

  [2026-04-05 20:43:13] [STEP 2]
    Thought     : The code executed successfully and returned the list.
    Final Answer: The prime numbers from 1 to 100 are: []
  ```

- **Diagnosis**: The root cause was a **system prompt gap** — the tool description did not explicitly require that the submitted code be complete and immediately runnable. LLMs frequently generate skeleton code with `pass` / `TODO` bodies because this pattern is abundant in training data. More critically, the tool had no way to distinguish between a legitimately empty output and an empty output caused by broken logic, so it returned `[]` silently, giving the LLM no signal to retry.

- **Solution**: Two fixes were applied in parallel:

  1. **System prompt update** — Added a hard constraint to the tool description:
     ```
     IMPORTANT: Code passed to this tool must be complete and immediately runnable.
     Never use `pass`, `...`, or `# TODO` as placeholders.
     If the output is an empty collection or None, re-examine your logic before concluding.
     ```

  2. **Output guard inside the tool** — Detect suspiciously trivial output and warn the LLM:
     ```python
     result = proc.stdout.strip()
     if result in ("[]", "{}", "", "None"):
         return (
             f"Warning: Output is empty or trivial ('{result}'). "
             "The code may be logically incomplete. Please rewrite and retry."
         )
     return result
     ```

  After both fixes, the `Warning` string in the Observation correctly prompted the LLM to rewrite the function with a real implementation on the next step instead of accepting a wrong answer.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

*Reflect on the reasoning capability difference.*

1. **Reasoning**: The `Thought` block acts as an explicit decision gate — forcing the model to assess *whether* a tool call is necessary before acting. For trivial questions like *"What is 2 + 2?"*, the agent correctly skipped the sandbox and answered directly. For complex computations, however, `Thought` surfaced the model's uncertainty (*"I might hallucinate this result — I should run actual code"*), leading to a grounded and verifiable answer. A plain Chatbot has no equivalent step; it responds immediately and can be confidently wrong.

2. **Reliability**: The agent performed **worse** than the Chatbot in several observable cases:
   - **Simple conceptual questions** (e.g., *"What is Python used for?"*): The agent introduced ~1.8s of unnecessary overhead spawning a subprocess that was never needed, while the Chatbot answered instantly and correctly.
   - **Ambiguous language requests** (e.g., *"write code to compute factorial"* without specifying a language): The `_infer_language()` heuristic guessed incorrectly in 2 out of 10 test runs, causing interpreter errors that the Chatbot simply avoided by generating plain text.
   - **Missing interpreter environments**: On machines without `bash` installed, the tool raised `FileNotFoundError`, whereas the Chatbot still produced a usable response.

3. **Observation**: The `Observation` injected after each tool call acted as a real-time quality control loop. In one test case, the first code submission produced a `ZeroDivisionError`. The Observation fed the full `stderr` back to the LLM, which read the traceback in its next `Thought`, identified the division-by-zero condition, patched the guard clause, and re-ran successfully — all without any user intervention. This self-correcting behavior is entirely absent in a single-shot Chatbot.

---

## IV. Future Improvements (5 Points)

*How would you scale this for a production-level AI agent system?*

- **Scalability**: Replace the synchronous subprocess call with **async execution** (`asyncio.create_subprocess_exec`) so that multiple agent sessions can run code concurrently without blocking each other. For a multi-user deployment, each sandbox job should be dispatched to an isolated **task queue** (Celery + Redis), allowing sandbox workers to scale horizontally and preventing a single heavy computation from degrading the entire system.

- **Safety**: Run every sandbox execution inside a **dedicated Docker container** with strict resource limits (`--memory=64m --cpus=0.5 --network=none --read-only`) to block access to the network, host filesystem, and system calls. Additionally, introduce a lightweight **Supervisor LLM** that inspects `Action Input` before execution and rejects code patterns flagged as dangerous (e.g., `os.system`, `subprocess.call`, `open("/etc/passwd")`).

- **Performance**: Cache sandbox results by the **SHA-256 hash of the code string** to avoid redundant re-execution of identical snippets within the same session. For systems with many available tools, adopt a **vector database** (Chroma / Qdrant) for dynamic tool retrieval — injecting only the top-3 most semantically relevant tool descriptions into the prompt per step rather than listing all tools, significantly reducing token cost and attention dilution.

---

> [!NOTE]
> Submit this report by renaming it to `REPORT_[YOUR_NAME].md` and placing it in this folder.