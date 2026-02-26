#!/usr/bin/env python3
"""Lightweight HTTP server providing a browser UI for Spiky."""

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRACKING_DIR = DATA_DIR / "tracking"
LOGS_DIR = DATA_DIR / "logs"
UI_FILE = BASE_DIR / "ui.html"
SCHEDULES_FILE = DATA_DIR / "schedules.json"

SCHEDULABLE_COMMANDS = ("triage", "implement", "revise", "push", "retry")

STATUS_ORDER = {
    "awaiting_approval": 0, "approved": 1, "implemented": 2, "pushed": 3,
    "failed": 4, "triage_failed": 5, "easy": 6, "medium": 7,
    "needs_clarification": 8, "complex_skip": 9, "asked_clarification": 10,
}

# Job tracking (in-memory for current session)
_jobs: dict[str, dict] = {}
_job_procs: dict[str, subprocess.Popen] = {}
_jobs_lock = threading.Lock()

# Schedule tracking
_last_run: dict[str, float] = {}
_scheduler_stop = threading.Event()


def _load_schedules() -> dict:
    if SCHEDULES_FILE.exists():
        try:
            return json.loads(SCHEDULES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_schedules(schedules: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCHEDULES_FILE.write_text(json.dumps(schedules, indent=2) + "\n")


def _is_command_running(command: str) -> bool:
    with _jobs_lock:
        for job in _jobs.values():
            if job["command"] == command and job["finished_at"] is None:
                return True
    return False


def _scheduler_loop() -> None:
    """Background thread that fires scheduled jobs at their configured intervals."""
    while not _scheduler_stop.is_set():
        try:
            schedules = _load_schedules()
            now = time.time()
            for command, sched in schedules.items():
                interval = sched.get("interval_seconds")
                if not interval or interval <= 0:
                    continue
                last = _last_run.get(command, 0)
                if now - last < interval:
                    continue
                if _is_command_running(command):
                    continue
                args = list(sched.get("args", []))
                if command == "revise" and "--max-revisions" not in args:
                    args.extend(["--max-revisions", "3"])
                _last_run[command] = now
                _run_job(command, args)
        except Exception:
            pass
        _scheduler_stop.wait(30)


def _discover_running_jobs(exclude_children_of: set[int] | None = None) -> list[dict]:
    """Find scout_agent/cursor-agent processes via ps not already tracked by this server."""
    exclude_children_of = exclude_children_of or set()
    try:
        out = subprocess.check_output(
            ["ps", "axo", "pid,ppid,lstart,command"], text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    server_pid = os.getpid()
    found = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except (ValueError, IndexError):
            continue
        if pid == server_pid:
            continue
        if pid in exclude_children_of or ppid in exclude_children_of:
            continue

        if "uv run scout_agent.py" in line and "server.py" not in line:
            cmd_start = line.find("scout_agent.py")
            cmd_str = line[cmd_start + len("scout_agent.py"):].strip()
            cmd_parts = cmd_str.split() if cmd_str else []
            command = cmd_parts[0] if cmd_parts else "unknown"
            args = cmd_parts[1:] if len(cmd_parts) > 1 else []
            found.append({
                "id": f"ps-{pid}",
                "command": command,
                "args": args,
                "pid": pid,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "output": "(started outside server — output not captured)",
            })
    return found


def _load_tracking(issue_id: str) -> dict | None:
    p = TRACKING_DIR / f"{issue_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_tracking(issue_id: str, data: dict) -> None:
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    (TRACKING_DIR / f"{issue_id}.json").write_text(json.dumps(data, indent=2) + "\n")


def _load_all_tracking() -> list[dict]:
    if not TRACKING_DIR.exists():
        return []
    items = []
    for p in TRACKING_DIR.glob("*.json"):
        items.append(json.loads(p.read_text()))
    items.sort(key=lambda t: t.get("triaged_at", ""), reverse=True)
    items.sort(key=lambda t: STATUS_ORDER.get(t.get("status", ""), 99))
    return items


def _run_job(command: str, args: list[str] | None = None) -> str:
    """Launch ./run.sh <command> [args...] as a background subprocess. Returns job ID."""
    job_id = uuid.uuid4().hex[:8]
    cmd = ["./run.sh", command] + (args or [])

    def _worker():
        with _jobs_lock:
            _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR), text=True, start_new_session=True,
            )
            with _jobs_lock:
                _jobs[job_id]["pid"] = proc.pid
                _job_procs[job_id] = proc
            output_lines = []
            for line in proc.stdout:
                output_lines.append(line)
                with _jobs_lock:
                    _jobs[job_id]["output"] = "".join(output_lines)
            proc.wait()
            with _jobs_lock:
                if _jobs[job_id]["finished_at"] is None:
                    _jobs[job_id]["exit_code"] = proc.returncode
                    _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _job_procs.pop(job_id, None)
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]["exit_code"] = -1
                _jobs[job_id]["output"] = _jobs[job_id].get("output", "") + f"\nError: {e}"
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "command": command,
            "args": args or [],
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "output": "",
        }

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return job_id


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default request logging

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            content = UI_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/issues":
            self._json_response(_load_all_tracking())
            return

        if path.startswith("/api/issues/") and path.count("/") == 3:
            issue_id = path.split("/")[3]
            tracking = _load_tracking(issue_id)
            if tracking:
                self._json_response(tracking)
            else:
                self._json_response({"error": "not found"}, 404)
            return

        if path == "/api/jobs":
            with _jobs_lock:
                jobs = list(_jobs.values())
            known_pids = {j.get("pid") for j in jobs if j.get("pid")}
            if known_pids:
                discovered = _discover_running_jobs(exclude_children_of=known_pids)
            else:
                discovered = _discover_running_jobs()
            for d in discovered:
                jobs.append(d)
            self._json_response(jobs)
            return

        if path == "/api/schedules":
            schedules = _load_schedules()
            now = time.time()
            result = {}
            for cmd in SCHEDULABLE_COMMANDS:
                sched = schedules.get(cmd)
                if sched and sched.get("interval_seconds"):
                    last = _last_run.get(cmd, 0)
                    next_in = max(0, sched["interval_seconds"] - (now - last)) if last else 0
                    result[cmd] = {**sched, "next_run_in_seconds": round(next_in)}
                else:
                    result[cmd] = None
            self._json_response(result)
            return

        if path.startswith("/api/logs/"):
            rel = path[len("/api/logs/"):]
            target = (LOGS_DIR / rel).resolve()
            if not str(target).startswith(str(LOGS_DIR.resolve())):
                self._json_response({"error": "invalid path"}, 400)
                return
            if target.is_dir():
                files = sorted(f.name for f in target.iterdir() if f.suffix == ".log")
                self._json_response(files)
                return
            if target.exists() and target.suffix == ".log":
                content = target.read_text()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(content.encode())))
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self._json_response({"error": "not found"}, 404)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = self.path.split("?")[0]

        if path.startswith("/api/issues/") and path.endswith("/approve"):
            parts = path.split("/")
            issue_id = parts[3]
            tracking = _load_tracking(issue_id)
            if not tracking:
                self._json_response({"error": "not found"}, 404)
                return
            if tracking["status"] != "awaiting_approval":
                self._json_response({"error": f"status is '{tracking['status']}', expected 'awaiting_approval'"}, 400)
                return
            tracking["status"] = "approved"
            tracking["approved_at"] = datetime.now(timezone.utc).isoformat()
            _save_tracking(issue_id, tracking)
            self._json_response({"ok": True, "issue_id": issue_id})
            return

        if path.startswith("/api/jobs/") and path.endswith("/kill"):
            job_id = path.split("/")[3]
            with _jobs_lock:
                job = _jobs.get(job_id)
            # Handle discovered (ps-*) jobs by extracting pid directly
            if not job and job_id.startswith("ps-"):
                pid = int(job_id.removeprefix("ps-"))
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                self._json_response({"ok": True})
                return
            if not job:
                self._json_response({"error": "not found"}, 404)
                return
            if job.get("finished_at") is not None:
                self._json_response({"ok": True, "already_finished": True})
                return
            pid = job.get("pid")
            with _jobs_lock:
                proc = _job_procs.get(job_id)
            killed = False
            if pid:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    killed = True
                except (ProcessLookupError, PermissionError):
                    pass
            if not killed and proc:
                try:
                    proc.kill()
                except OSError:
                    pass
            with _jobs_lock:
                _jobs[job_id]["exit_code"] = -9
                _jobs[job_id]["output"] = _jobs[job_id].get("output", "") + "\n[killed by user]"
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _job_procs.pop(job_id, None)
            self._json_response({"ok": True})
            return

        if path.startswith("/api/jobs/"):
            command = path.split("/")[3]
            if command not in ("triage", "implement", "revise", "push", "retry", "ask_clarification"):
                self._json_response({"error": f"unknown command: {command}"}, 400)
                return
            body = self._read_body()
            args = json.loads(body).get("args", []) if body else []
            job_id = _run_job(command, args)
            self._json_response({"ok": True, "job_id": job_id})
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_PUT(self):
        path = self.path.split("?")[0]

        if path.startswith("/api/schedules/") and path.count("/") == 3:
            command = path.split("/")[3]
            if command not in SCHEDULABLE_COMMANDS:
                self._json_response({"error": f"unknown command: {command}"}, 400)
                return
            body = json.loads(self._read_body())
            schedules = _load_schedules()
            interval = body.get("interval_seconds")
            if interval and interval > 0:
                schedules[command] = {
                    "interval_seconds": int(interval),
                    "args": body.get("args", []),
                }
            else:
                schedules.pop(command, None)
                _last_run.pop(command, None)
            _save_schedules(schedules)
            self._json_response({"ok": True})
            return

        if path.startswith("/api/issues/") and path.endswith("/plan"):
            parts = path.split("/")
            issue_id = parts[3]
            tracking = _load_tracking(issue_id)
            if not tracking:
                self._json_response({"error": "not found"}, 404)
                return
            body = json.loads(self._read_body())
            tracking["plan"] = body.get("plan", tracking.get("plan"))
            tracking["plan_summary"] = body.get("plan_summary", tracking.get("plan_summary"))
            _save_tracking(issue_id, tracking)
            self._json_response({"ok": True, "issue_id": issue_id})
            return

        self.send_error(HTTPStatus.NOT_FOUND)


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def main():
    port = 8111
    server = ReusableHTTPServer(("127.0.0.1", port), Handler)
    scheduler = threading.Thread(target=_scheduler_loop, daemon=True)
    scheduler.start()
    print(f"Spiky running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        _scheduler_stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()
