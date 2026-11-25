---
name: dashboard
description: >-
  Fast 1-turn launcher for the Cardiac Fat Research Dashboard on http://127.0.0.1:8050/.
  Use whenever the user invokes /dashboard or asks to bring up, start, launch, open, or show the dashboard or cases.
---

# Dashboard Quick Launch (`/dashboard`)

Follow this zero-investigation, 1-turn runbook whenever the user asks to launch or show the dashboard:

1. **Skip codebase exploration and the `browser` subagent** unless the user explicitly asks to automate clicks/interactions inside Chrome or troubleshoot UI rendering.
2. **Check if already running**: Call `manage_task(Action="list")` first. If a dashboard server daemon on `:8050` is already `RUNNING`, inspect its log (`Active Cohort: ...`) and reply immediately with `http://127.0.0.1:8050/`.
3. **Launch once cleanly (if not running)**: Omit `--no-browser` so `launch_dashboard` automatically opens `http://127.0.0.1:8050/` in the OS default browser via `webbrowser.open()`. Omit `--cohort-dir` unless the user specifies one—`get_default_cohort_dir()` automatically resolves `CARDIAC_COHORT_DIR` (environment or gitignored `.env`) or falls back to `example_source_dir/`. Run as a background daemon (`IsDaemon: true`, `WaitMsBeforeAsync: 5000`):

   - **Windows:**

     ```powershell
     .venv\Scripts\python.exe -m cardiac_fat_fl.dashboard --port 8050
     ```

   - **Linux / macOS:**

     ```bash
     .venv/bin/python -m cardiac_fat_fl.dashboard --port 8050
     ```

4. **Reply concisely**: Read the `Active Cohort: <path> (<N> cases)` line from the daemon startup output and share `http://127.0.0.1:8050/`, the active cohort, 1-click **`View OHIF`** (`/viewer/<pid>/`), and the header **Switch** option.
