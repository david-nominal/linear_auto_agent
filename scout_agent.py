#!/usr/bin/env python3
"""Spiky: fetch open issues, triage, plan, and implement via cursor agent CLI."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

LINEAR_API_URL = "https://api.linear.app/graphql"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRACKING_DIR = DATA_DIR / "tracking"
WORKTREES_DIR = DATA_DIR / "worktrees"
LOGS_DIR = DATA_DIR / "logs"
WORKSPACE_DIR = DATA_DIR / "workspace"

REPOS = {
    "backend": {"env_var": "BACKEND_REPO", "clone_url": "git@github.com:nominal-io/scout.git", "description": "Java + Python backend"},
    "frontend": {"env_var": "FRONTEND_REPO", "clone_url": "git@github.com:nominal-io/galaxy.git", "description": "TypeScript frontend + tRPC backend, nominal-mcp, and multiplayer servers"},
}

TRIAGE_TIMEOUT = 60
PLAN_TIMEOUT = 120
IMPL_TIMEOUT = 600
DEFAULT_WORKERS = 3


_start_time = time.monotonic()
_log_lock = threading.Lock()


def _ts() -> str:
    elapsed = time.monotonic() - _start_time
    m, s = divmod(int(elapsed), 60)
    return f"{m:02d}:{s:02d}"


def log(msg: str, *, issue: str = "", file=None) -> None:
    prefix = f"[{_ts()}]"
    if issue:
        prefix += f" [{issue}]"
    with _log_lock:
        print(f"{prefix} {msg}", file=file, flush=True)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Image extraction & downloading
# ---------------------------------------------------------------------------

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

_IMAGE_HOSTS = {"uploads.linear.app", "user-images.githubusercontent.com",
                "github.com", "private-user-images.githubusercontent.com"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

MAX_IMAGES_PER_ISSUE = 10


def _is_image_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if any(h in host for h in _IMAGE_HOSTS):
        return True
    path_lower = parsed.path.lower()
    return any(path_lower.endswith(ext) for ext in _IMAGE_EXTS)


def extract_and_download_images(
    texts: list[tuple[str, str]],
    dest_dir: str | Path,
    issue_id: str = "",
) -> list[dict]:
    """Extract markdown image URLs from texts and download them.

    Args:
        texts: list of (markdown_text, source_label) pairs,
               e.g. [("![img](url)", "issue description"), ...]
        dest_dir: directory to save images into (a .scout-images/ subdir is created)
        issue_id: for logging

    Returns:
        list of dicts with keys: local_path, source, url
    """
    images_dir = Path(dest_dir) / ".scout-images"
    seen_urls: set[str] = set()
    to_download: list[tuple[str, str]] = []  # (url, source_label)

    for text, source in texts:
        if not text:
            continue
        for _alt, url in _IMAGE_RE.findall(text):
            url = url.strip()
            if url in seen_urls or not _is_image_url(url):
                continue
            seen_urls.add(url)
            to_download.append((url, source))
            if len(to_download) >= MAX_IMAGES_PER_ISSUE:
                break
        if len(to_download) >= MAX_IMAGES_PER_ISSUE:
            break

    if not to_download:
        return []

    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict] = []

    for url, source in to_download:
        try:
            headers = {}
            parsed_host = urlparse(url).hostname or ""
            if parsed_host == "uploads.linear.app":
                headers["Authorization"] = linear_api_key()
            elif parsed_host.endswith("githubusercontent.com"):
                gh_token = os.environ.get("GH_TOKEN", "")
                if gh_token:
                    headers["Authorization"] = f"token {gh_token}"
            resp = requests.get(url, timeout=15, stream=True, headers=headers)
            resp.raise_for_status()
            content = resp.content
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
            content_type = resp.headers.get("content-type", "")
            ext = ".png"
            if "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            elif "gif" in content_type:
                ext = ".gif"
            elif "webp" in content_type:
                ext = ".webp"
            else:
                parsed_path = urlparse(url).path.lower()
                for e in _IMAGE_EXTS:
                    if parsed_path.endswith(e):
                        ext = e
                        break

            filename = f"{url_hash}{ext}"
            local_path = images_dir / filename
            local_path.write_bytes(content)
            downloaded.append({"local_path": str(local_path), "source": source, "url": url})
            log(f"Downloaded image {filename} ({len(content)} bytes, from {source})", issue=issue_id)
        except Exception as exc:
            log(f"Failed to download image from {source}: {exc}", issue=issue_id)

    if downloaded:
        log(f"Downloaded {len(downloaded)} image(s) total", issue=issue_id)

    return downloaded


def format_images_for_prompt(downloaded: list[dict]) -> str:
    """Build prompt section listing downloaded images. Returns empty string if none."""
    if not downloaded:
        return ""
    lines = [
        "\n\n## Attached images\n",
        "The following images were attached to the issue or review comments. "
        "Read them to understand visual context:\n",
    ]
    for img in downloaded:
        lines.append(f"- {img['local_path']} (from {img['source']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Repo configuration
# ---------------------------------------------------------------------------

def get_repo_path(repo_key: str) -> str:
    cfg = REPOS[repo_key]
    path = os.environ.get(cfg["env_var"], "")
    if not path:
        print(f"ERROR: {cfg['env_var']} not set in .env", file=sys.stderr)
        sys.exit(1)
    return path


def ensure_workspace() -> Path:
    """Ensure data/workspace/ exists, clone missing repos, and write root AGENTS.md."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    for repo_key, cfg in REPOS.items():
        repo_path = Path(os.environ.get(cfg["env_var"], ""))
        if not repo_path or not str(repo_path).strip():
            continue
        if not repo_path.is_dir():
            log(f"Cloning {repo_key} into {repo_path}...")
            result = subprocess.run(
                ["git", "clone", cfg["clone_url"], str(repo_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log(f"  Clone failed for {repo_key}: {result.stderr.strip()}", file=sys.stderr)

    agents_md = WORKSPACE_DIR / "AGENTS.md"
    agents_md.write_text(
        "# Workspace\n\n"
        "This workspace contains two repos:\n\n"
        "- **scout/** — Java + Python backend (see `scout/AGENTS.md` for conventions)\n"
        "- **galaxy/** — TypeScript frontend, tRPC backend, nominal-mcp, and multiplayer servers (see `galaxy/AGENTS.md` for conventions)\n\n"
        "Read the repo-specific AGENTS.md files before making decisions about code structure, "
        "patterns, or implementation approach.\n"
    )
    return WORKSPACE_DIR


def _fetch_repos() -> None:
    """Fetch latest default branch for all configured repos."""
    for repo_key in REPOS:
        repo_path = get_repo_path(repo_key)
        default_branch = _get_default_branch(repo_path)
        log(f"Fetching latest {default_branch} for {repo_key}...")
        result = subprocess.run(
            ["git", "-C", repo_path, "pull", "--ff-only", "origin", default_branch],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log(f"  Warning: pull failed for {repo_key}: {result.stderr.strip()}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Linear API
# ---------------------------------------------------------------------------

def linear_api_key() -> str:
    key = os.environ.get("LINEAR_API_KEY", "")
    if not key:
        print("ERROR: LINEAR_API_KEY not set. Get one from https://linear.app/settings/account/security", file=sys.stderr)
        sys.exit(1)
    return key


def fetch_issues() -> list[dict]:
    """Fetch all open issues that are not started, completed, canceled, or duplicates."""
    query = """
    query($cursor: String) {
        issues(
            first: 50
            after: $cursor
            filter: {
                state: {
                    type: { nin: ["started", "completed", "cancelled"] }
                }
                hasDuplicateRelations: { eq: false }
            }
            orderBy: updatedAt
        ) {
            pageInfo { hasNextPage endCursor }
            nodes {
                identifier
                title
                description
                url
                priority
                team { key name }
                state { name type }
                labels { nodes { name } }
            }
        }
    }
    """
    headers = {"Authorization": linear_api_key(), "Content-Type": "application/json"}
    all_issues = []
    cursor = None

    while True:
        resp = requests.post(
            LINEAR_API_URL,
            json={"query": query, "variables": {"cursor": cursor}},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"ERROR: Linear API returned {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
            sys.exit(1)
        data = resp.json()

        if "errors" in data:
            print(f"ERROR: Linear API returned errors: {data['errors']}", file=sys.stderr)
            sys.exit(1)

        issues_data = data["data"]["issues"]
        for node in issues_data["nodes"]:
            all_issues.append({
                "id": node["identifier"],
                "title": node["title"],
                "description": node.get("description") or "",
                "url": node["url"],
                "priority": node.get("priority"),
                "team": node["team"]["key"] if node.get("team") else None,
                "state": node["state"]["name"],
                "state_type": node["state"]["type"],
                "labels": [l["name"] for l in node.get("labels", {}).get("nodes", [])],
            })

        if not issues_data["pageInfo"]["hasNextPage"]:
            break
        cursor = issues_data["pageInfo"]["endCursor"]

    return all_issues


def fetch_issue_comments(issue_id: str) -> dict:
    """Fetch comments and attachments for a single Linear issue by identifier (e.g. 'ENG-123')."""
    match = re.match(r"^([A-Za-z]+)-(\d+)$", issue_id)
    if not match:
        log(f"Cannot parse identifier for comments lookup", issue=issue_id, file=sys.stderr)
        return {"comments": [], "attachments": []}
    team_key = match.group(1).upper()
    number = int(match.group(2))

    query = """
    query($teamKey: String!, $number: Float!) {
        issues(
            filter: { team: { key: { eq: $teamKey } }, number: { eq: $number } }
            first: 1
        ) {
            nodes {
                comments(first: 50) {
                    nodes {
                        body
                        user { name }
                        createdAt
                    }
                }
                attachments(first: 20) {
                    nodes { url title subtitle }
                }
            }
        }
    }
    """
    headers = {"Authorization": linear_api_key(), "Content-Type": "application/json"}
    resp = requests.post(
        LINEAR_API_URL,
        json={"query": query, "variables": {"teamKey": team_key, "number": number}},
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 200:
        log(f"Failed to fetch comments (HTTP {resp.status_code})", issue=issue_id, file=sys.stderr)
        return {"comments": [], "attachments": []}

    data = resp.json()
    nodes = data.get("data", {}).get("issues", {}).get("nodes", [])
    issue_data = nodes[0] if nodes else None
    if not issue_data:
        log(f"No issue data in comments response", issue=issue_id, file=sys.stderr)
        return {"comments": [], "attachments": []}

    comments = []
    for node in issue_data.get("comments", {}).get("nodes", []):
        comments.append({
            "author": node.get("user", {}).get("name", "Unknown"),
            "created_at": node.get("createdAt", ""),
            "body": node.get("body", ""),
        })

    attachments = []
    for node in issue_data.get("attachments", {}).get("nodes", []):
        attachments.append({
            "title": node.get("title") or node.get("subtitle") or "",
            "url": node.get("url", ""),
        })

    return {"comments": comments, "attachments": attachments}


def _format_linear_comments(data: dict) -> str:
    """Format Linear issue comments and attachments into readable text for prompts."""
    parts = []

    comments = data.get("comments", [])
    if comments:
        for c in comments:
            author = c.get("author", "Unknown")
            date = c.get("created_at", "")[:10]
            body = c.get("body", "").strip()
            if body:
                parts.append(f"**{author}** ({date}):\n{body}")
    else:
        parts.append("(no comments)")

    attachments = data.get("attachments", [])
    if attachments:
        parts.append("\n### Attachments")
        for a in attachments:
            title = a.get("title", "Untitled")
            url = a.get("url", "")
            parts.append(f"- [{title}]({url})" if url else f"- {title}")

    return "\n\n".join(parts)


def post_linear_comment(issue_id: str, body: str) -> bool:
    """Post a comment on a Linear issue. Returns True on success."""
    match = re.match(r"^([A-Za-z]+)-(\d+)$", issue_id)
    if not match:
        log(f"Cannot parse identifier for comment post", issue=issue_id, file=sys.stderr)
        return False
    team_key = match.group(1).upper()
    number = int(match.group(2))

    headers = {"Authorization": linear_api_key(), "Content-Type": "application/json"}

    # Resolve identifier to internal UUID
    id_query = """
    query($teamKey: String!, $number: Float!) {
        issues(filter: { team: { key: { eq: $teamKey } }, number: { eq: $number } }, first: 1) {
            nodes { id }
        }
    }
    """
    resp = requests.post(
        LINEAR_API_URL,
        json={"query": id_query, "variables": {"teamKey": team_key, "number": number}},
        headers=headers, timeout=30,
    )
    if resp.status_code != 200:
        log(f"Failed to resolve issue UUID (HTTP {resp.status_code})", issue=issue_id, file=sys.stderr)
        return False
    nodes = resp.json().get("data", {}).get("issues", {}).get("nodes", [])
    if not nodes:
        log(f"Issue not found in Linear", issue=issue_id, file=sys.stderr)
        return False
    internal_id = nodes[0]["id"]

    # Post comment
    mutation = """
    mutation($issueId: String!, $body: String!) {
        commentCreate(input: { issueId: $issueId, body: $body }) {
            success
        }
    }
    """
    resp = requests.post(
        LINEAR_API_URL,
        json={"query": mutation, "variables": {"issueId": internal_id, "body": body}},
        headers=headers, timeout=30,
    )
    if resp.status_code != 200:
        log(f"Failed to post comment (HTTP {resp.status_code})", issue=issue_id, file=sys.stderr)
        return False
    success = resp.json().get("data", {}).get("commentCreate", {}).get("success", False)
    if not success:
        log(f"commentCreate returned success=false", issue=issue_id, file=sys.stderr)
    return success


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def tracking_path(issue_id: str) -> Path:
    return TRACKING_DIR / f"{issue_id}.json"


def load_tracking(issue_id: str) -> dict | None:
    p = tracking_path(issue_id)
    if p.exists():
        return json.loads(p.read_text())
    return None


def save_tracking(issue_id: str, data: dict) -> None:
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    tracking_path(issue_id).write_text(json.dumps(data, indent=2) + "\n")


def load_all_tracking() -> dict[str, dict]:
    if not TRACKING_DIR.exists():
        return {}
    result = {}
    for p in TRACKING_DIR.glob("*.json"):
        data = json.loads(p.read_text())
        result[data["issue_id"]] = data
    return result


# ---------------------------------------------------------------------------
# Cursor Agent helpers
# ---------------------------------------------------------------------------

def run_cursor_agent(prompt: str, *, mode: str | None = None, workspace: str | None = None,
                     yolo: bool = False, sandbox: bool = False,
                     log_name: str | None = None, timeout: int = IMPL_TIMEOUT,
                     model: str | None = None) -> tuple[int, str]:
    """Run cursor agent CLI and return (exit_code, output).

    Security:
    - --print mode: MCPs require interactive approval, which is impossible headlessly,
      so all MCP servers are effectively blocked (unless --approve-mcps is passed, which we never do).
    - --mode ask/plan: read-only, no file edits, no shell commands.
    - --sandbox enabled: OS-level (macOS Seatbelt) restriction — writes only to workspace + /tmp,
      no network access. Used for implementation agents to hard-scope writes to the worktree.
    """
    cmd = ["cursor", "agent", "--print", "--trust"]

    api_key = os.environ.get("CURSOR_API_KEY", "")
    if api_key:
        cmd += ["--api-key", api_key]

    if mode:
        cmd += ["--mode", mode]
    if model:
        cmd += ["--model", model]
    if workspace:
        cmd += ["--workspace", workspace]
    if yolo:
        cmd.append("--yolo")
    if sandbox:
        cmd += ["--sandbox", "enabled"]

    cmd.append(prompt)

    log_path = None
    if log_name:
        log_path = LOGS_DIR / f"{log_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

    issue_hint = ""
    if log_name:
        issue_hint = log_name.split("/")[0]

    log(f"Agent starting ({mode or 'default'} mode, sandbox={'on' if sandbox else 'off'}, timeout={timeout}s)",
        issue=issue_hint)
    if log_path:
        log(f"Log: {log_path}", issue=issue_hint)

    max_security_retries = 2
    t0 = time.monotonic()

    for attempt in range(max_security_retries + 1):
        output_chunks: list[str] = []
        log_file = open(log_path, "w") if log_path else None
        timed_out = False
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                output_chunks.append(line)
                if log_file:
                    log_file.write(line)
                    log_file.flush()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
        finally:
            if log_file:
                log_file.close()

        exit_code = proc.returncode or 0
        output = "".join(output_chunks)

        is_keychain_error = exit_code != 0 and (
            "Security" in output or "Password not found" in output
        )
        if is_keychain_error and attempt < max_security_retries:
            delay = 2 * (attempt + 1)
            log(f"Keychain error on attempt {attempt + 1}, retrying in {delay}s...", issue=issue_hint)
            time.sleep(delay)
            continue
        break

    elapsed = time.monotonic() - t0
    status_str = "TIMEOUT" if timed_out else ("OK" if exit_code == 0 else f"FAILED(exit={exit_code})")
    log(f"Agent finished: {status_str} in {_fmt_duration(elapsed)}", issue=issue_hint)

    return exit_code, output


def parse_json_from_output(output: str) -> dict | None:
    """Extract a JSON block from agent output (between ```json and ``` or raw JSON)."""
    match = re.search(r"```json\s*\n(.*?)```", output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[^{}]*\"decision\"[^{}]*\}", output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

TRIAGE_PROMPT = """\
You are triaging a Linear issue for an automated agent system.

## Issue: {id} — {title}

{description}

## Comments

{comments}

## Context

The workspace contains two repos:
- scout/ — Java + Python backend (Gradle, Conjure APIs, Postgres, ClickHouse, Temporal)
- galaxy/ — TypeScript frontend (React), tRPC backend, nominal-mcp, and multiplayer servers

## Instructions

1. Is this task well-defined enough to implement without asking clarifying questions?
2. If yes, classify the complexity:
   - "easy": isolated, single-repo change, < ~1 hour of focused work, no migrations, no cross-service breaking changes
   - "medium": well-defined but cross-repo or slightly larger (1-3 hours), still automatable with clear steps
   - "complex_skip": needs design decisions, migrations, unclear scope, or multi-day effort
3. Which repo(s) need changes?
4. What type of change is this? "feat" for new features, "fix" for bug fixes, "chore" for refactoring/maintenance.

You MUST output ONLY a JSON block (no other text) in this exact format:

```json
{{"decision": "<needs_clarification|complex_skip|medium|easy>", "repos": ["backend", "frontend"], "pr_type": "<feat|fix|chore>", "reasoning": "<1-2 sentence explanation>"}}
```

"repos" should only include repos that need changes. Use "backend" for scout, "frontend" for galaxy.
"""


TRIAGE_MAX_RETRIES = int(os.environ.get("TRIAGE_MAX_RETRIES", "2"))


def triage_issue(issue: dict, workspace: str) -> dict:
    """Triage a single issue. Returns the parsed decision dict."""
    prompt = TRIAGE_PROMPT.format(**issue)
    model = os.environ.get("TRIAGE_MODEL") or None

    for attempt in range(TRIAGE_MAX_RETRIES + 1):
        exit_code, output = run_cursor_agent(prompt, mode="ask", workspace=workspace,
                                             log_name=f"{issue['id']}/triage",
                                             timeout=TRIAGE_TIMEOUT, model=model)

        if exit_code != 0:
            tail = output.strip()[-500:] if output.strip() else "(no output)"
            log(f"Agent failed (exit={exit_code}). Last output:\n    {tail}", issue=issue["id"], file=sys.stderr)

        parsed = parse_json_from_output(output)
        if parsed and "decision" in parsed:
            if "repos" not in parsed:
                parsed["repos"] = ["backend"]
            if "pr_type" not in parsed:
                parsed["pr_type"] = "feat"
            return parsed

        if attempt < TRIAGE_MAX_RETRIES:
            delay = 3 * (attempt + 1)
            log(f"Triage attempt {attempt + 1} failed, retrying in {delay}s...", issue=issue["id"])
            time.sleep(delay)

    return {
        "decision": "triage_failed",
        "repos": [],
        "pr_type": "feat",
        "reasoning": f"Failed to parse agent output after {TRIAGE_MAX_RETRIES + 1} attempts (exit={exit_code})",
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

PLAN_PROMPT = """\
You are creating an implementation plan for a Linear issue that has been triaged as automatable.

## Issue: {id} — {title}

{description}

## Triage reasoning
{reasoning}

## Affected repos: {repos_str}

## Context

The workspace contains two repos:
- scout/ — Java + Python backend (Gradle, Conjure APIs, Postgres, ClickHouse, Temporal). Conventions in scout/AGENTS.md.
- galaxy/ — TypeScript frontend (React), tRPC backend, nominal-mcp, and multiplayer servers

## Instructions

Start your response with a "## Summary" section containing 2-3 sentences describing the change \
for a PR reviewer. Then provide the detailed implementation plan.

Create a concrete implementation plan. For each affected repo, provide:
1. Which files need to be created or modified (specific paths)
2. What the changes are (be specific)
3. What tests need to be added or updated
4. Any risks or edge cases

If both repos are affected, clearly separate the plan into "## Backend (scout)" and "## Frontend (galaxy)" sections.
Note: if both are affected, the backend will be implemented first and its APIs linked locally \
into the frontend worktree, so the frontend agent can compile against the new backend types.

Output your plan as a clear, numbered list per repo. Be specific about file paths.
"""

UI_PLAN_ADDENDUM = """

## Additional instructions for frontend / UI changes

Before writing the plan, explore 2-3 existing pages or components in galaxy/ that are \
most similar to the feature being implemented. Use them as visual and structural references.

Your frontend plan MUST include a "## UI Design" section covering:

1. **Design precedent** — Name a specific existing page or feature in galaxy/ that the new UI \
should visually match (e.g., "model this after the Runs table page"). The implementation agent \
will use this as a reference.
2. **Component reuse** — List which existing shared components to use (buttons, modals, tables, \
form inputs, layout containers, etc.). Prefer reusing existing components over creating new ones.
3. **Layout structure** — Describe the visual hierarchy and layout (e.g., "sidebar with filter \
panel on left, scrollable data table on right, action bar pinned to top").
4. **Styling approach** — Specify which CSS patterns, theme variables, spacing tokens, or \
design tokens to follow, based on what you observe in the codebase. Reference specific files \
if helpful.

The goal: the implementation agent should be able to build a visually consistent UI without \
guessing at layout or style. Be concrete — "use the same card layout as WorkspaceSettings" \
is better than "use a card layout".
"""


def _extract_plan_summary(plan_text: str) -> str | None:
    """Extract the ## Summary section from a plan."""
    match = re.search(r"## Summary\s*\n(.*?)(?=\n## |\Z)", plan_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def plan_issue(issue: dict, reasoning: str, repos: list[str], workspace: str) -> tuple[str, str | None]:
    """Generate an implementation plan. Returns (plan_text, plan_summary)."""
    repos_str = ", ".join(repos)
    prompt = PLAN_PROMPT.format(**issue, reasoning=reasoning, repos_str=repos_str)
    if "frontend" in repos:
        prompt += UI_PLAN_ADDENDUM
    model = os.environ.get("PLAN_MODEL") or None
    exit_code, output = run_cursor_agent(prompt, mode="plan", workspace=workspace,
                                         log_name=f"{issue['id']}/plan",
                                         timeout=PLAN_TIMEOUT, model=model)

    if exit_code != 0:
        text = f"[Plan generation failed with exit code {exit_code}]\n\n{output[-2000:]}"
        return text, None

    plan_text = output.strip()
    plan_summary = _extract_plan_summary(plan_text)
    return plan_text, plan_summary


# ---------------------------------------------------------------------------
# FE/BE linking
# ---------------------------------------------------------------------------

def link_be_to_fe(be_worktree: str, fe_worktree: str) -> bool:
    """Build scout TS APIs in BE worktree and link them into the FE worktree.

    Runs conjure TS codegen + proto zip generation in scout, then uses galaxy's
    pnpm scout:link to point at the local build.
    """
    t0 = time.monotonic()

    log("  Building conjure TypeScript...")
    result = subprocess.run(
        ["./gradlew", ":scout-service-api:compileConjureTypeScript"],
        capture_output=True, text=True, cwd=be_worktree, timeout=300,
    )
    if result.returncode != 0:
        log(f"  compileConjureTypeScript failed: {result.stderr[:500]}")
        return False
    log(f"  compileConjureTypeScript done ({_fmt_duration(time.monotonic() - t0)})")

    t1 = time.monotonic()
    log("  Building proto zip package...")
    result = subprocess.run(
        ["./gradlew", ":scout-service-api:generateProtoZipNpmPackage"],
        capture_output=True, text=True, cwd=be_worktree, timeout=300,
    )
    if result.returncode != 0:
        log(f"  generateProtoZipNpmPackage failed (non-fatal): {result.stderr[:500]}")
    else:
        log(f"  generateProtoZipNpmPackage done ({_fmt_duration(time.monotonic() - t1)})")

    t2 = time.monotonic()
    log("  Installing frontend dependencies...")
    result = subprocess.run(
        ["pnpm", "install", "--frozen-lockfile"],
        capture_output=True, text=True, cwd=fe_worktree, timeout=300,
    )
    if result.returncode != 0:
        log(f"  pnpm install failed: {result.stderr[:500]}")
        return False
    log(f"  pnpm install done ({_fmt_duration(time.monotonic() - t2)})")

    t3 = time.monotonic()
    log("  Linking scout APIs in frontend worktree...")
    try:
        result = subprocess.run(
            ["pnpm", "scout:link", be_worktree],
            capture_output=True, text=True, cwd=fe_worktree, timeout=300,
        )
    except subprocess.TimeoutExpired:
        log("  scout:link timed out")
        return False
    if result.returncode != 0:
        log(f"  scout:link failed: {result.stderr[:500]}")
        return False
    log(f"  scout:link done ({_fmt_duration(time.monotonic() - t3)})")

    t4 = time.monotonic()
    log("  Running proto codegen...")
    result = subprocess.run(
        ["pnpm", "generate:proto"],
        capture_output=True, text=True, cwd=fe_worktree, timeout=120,
    )
    if result.returncode != 0:
        log(f"  generate:proto failed (non-fatal): {result.stderr[:500]}")
    else:
        log(f"  generate:proto done ({_fmt_duration(time.monotonic() - t4)})")

    log(f"  FE/BE linking complete ({_fmt_duration(time.monotonic() - t0)})")
    return True


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

IMPLEMENT_PROMPT_BACKEND = """\
You are implementing a Linear issue on a git worktree of the Java + Python backend repo (scout).

## Issue: {id} — {title}

{description}

## Implementation plan (pre-approved)

{plan}

## Instructions

- Follow the repo conventions in AGENTS.md
- Run compilation and tests for affected modules after making changes
- Run spotlessApply for affected modules
- Do NOT push to remote. Do NOT create a PR.
- Commit your changes with a conventional commit message referencing the issue.
"""

IMPLEMENT_PROMPT_FRONTEND = """\
You are implementing a Linear issue on a git worktree of the galaxy repo (TypeScript frontend, tRPC backend, nominal-mcp, and multiplayer servers).

## Issue: {id} — {title}

{description}

## Implementation plan (pre-approved)

{plan}

{be_note}

## Instructions

- Follow the repo conventions and coding standards
- Run linting and type checking after making changes
- Do NOT push to remote. Do NOT create a PR.
- Commit your changes with a conventional commit message referencing the issue.
"""

BE_DEPENDENCY_NOTE = """\
NOTE: This issue also has backend changes in a separate PR. The backend API changes
may not be published yet. Compilation failures from missing backend dependencies and API changes are
expected and acceptable — the frontend PR will be mergeable after the backend merges."""

BE_LINKED_NOTE = """\
NOTE: This issue also has backend changes. The backend APIs have been built and linked locally
into this worktree via scout:link, so you can compile against the new backend types.
Compilation should work — if you see missing types, check that the import paths are correct."""


def implement_issue_for_repo(issue_id: str, tracking: dict, repo_key: str,
                             be_worktree_path: str | None = None) -> dict:
    """Create a worktree in the given repo and run the implementation agent. Returns impl record."""
    repo_path = get_repo_path(repo_key)
    branch_name = f"agent/{issue_id}"
    worktree_name = f"{issue_id}-{repo_key}"
    worktree_path = str((WORKTREES_DIR / worktree_name).resolve())

    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)

    default_branch = _get_default_branch(repo_path)
    remote_ref = f"origin/{default_branch}"

    log(f"Fetching latest {remote_ref}...", issue=issue_id)
    subprocess.run(["git", "-C", repo_path, "fetch", "origin", default_branch],
                   capture_output=True, text=True)

    wt_cmd = ["git", "-C", repo_path, "worktree", "add", worktree_path, "-b", branch_name, remote_ref]
    wt_result = subprocess.run(wt_cmd, capture_output=True, text=True)
    if wt_result.returncode != 0:
        if "already exists" in wt_result.stderr:
            log(f"Worktree/branch {branch_name} already exists, reusing", issue=issue_id)
        else:
            return {
                "repo": repo_key,
                "error": f"Failed to create worktree: {wt_result.stderr}",
                "exit_code": -1,
                "started_at": now_iso(),
                "completed_at": now_iso(),
            }

    be_linked = False
    repos_affected = tracking.get("repos", [])
    if repo_key == "frontend" and be_worktree_path and "backend" in repos_affected:
        log("Linking local scout APIs from backend worktree...", issue=issue_id)
        be_linked = link_be_to_fe(be_worktree_path, worktree_path)
        if not be_linked:
            log("WARNING: Failed to link scout APIs, FE may have compilation issues", issue=issue_id)

    if be_linked:
        be_note = BE_LINKED_NOTE
    elif repo_key == "frontend" and "backend" in repos_affected:
        be_note = BE_DEPENDENCY_NOTE
    else:
        be_note = ""

    image_texts: list[tuple[str, str]] = [(tracking.get("description", ""), "issue description")]
    impl_images = extract_and_download_images(image_texts, worktree_path, issue_id=issue_id)
    images_section = format_images_for_prompt(impl_images)

    if repo_key == "backend":
        prompt = IMPLEMENT_PROMPT_BACKEND.format(
            id=issue_id, title=tracking["title"],
            description=tracking.get("description", ""),
            plan=tracking.get("plan", "No plan available"),
        )
    else:
        prompt = IMPLEMENT_PROMPT_FRONTEND.format(
            id=issue_id, title=tracking["title"],
            description=tracking.get("description", ""),
            plan=tracking.get("plan", "No plan available"),
            be_note=be_note,
        )

    if images_section:
        prompt += images_section

    model = os.environ.get("IMPL_MODEL") or None
    started_at = now_iso()
    exit_code, output = run_cursor_agent(prompt, workspace=worktree_path, yolo=True, sandbox=True,
                                         log_name=f"{issue_id}/{repo_key}", timeout=IMPL_TIMEOUT,
                                         model=model)
    completed_at = now_iso()

    log_file = LOGS_DIR / f"{issue_id}/{repo_key}.log"

    return {
        "repo": repo_key,
        "worktree_branch": branch_name,
        "worktree_path": worktree_path,
        "started_at": started_at,
        "completed_at": completed_at,
        "exit_code": exit_code,
        "output_log": str(log_file),
    }


def _get_default_branch(repo: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo, "symbolic-ref", "refs/remotes/origin/HEAD"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip().replace("refs/remotes/origin/", "")
    return "main"


# ---------------------------------------------------------------------------
# PR review / revision
# ---------------------------------------------------------------------------

REVISE_PROMPT = """\
You are addressing PR review feedback on a git worktree.

## Issue: {id} — {title}

{description}

## Triage

{triage}

## Implementation plan

{plan}

## Prior implementation log (tail)

{prior_log}

## Unresolved PR feedback

{comments_formatted}

## Instructions

- Address each review comment listed above.
- Keep the original issue description, triage assessment, and implementation plan in mind — \
ensure your changes stay consistent with the overall intent.
- If a comment requests a code change, make the change.
- If a comment is a question, answer it with a code change or clarifying code comment.
- Run compilation and tests after making changes.
- Commit your changes with message "fix: address review feedback for {id}"
- Do NOT push to remote. Do NOT create a PR.
"""

REVIEW_PROMPT = """\
You are reviewing an implementation for a Linear issue on a git worktree.

## Issue: {id} — {title}

{description}

## Implementation plan that was followed

{plan}

## Instructions

1. Run `git diff origin/{default_branch}...HEAD` to see all changes made.
2. Read through the diff and the surrounding code carefully, then evaluate:

   a) **Does the implementation address the issue?** Check that the requirements from the issue \
description are actually fulfilled by the changes, not just partially or superficially.

   b) **Code duplication / missed reuse**: Are there places where an existing API, class, utility, \
or component should have been modified or extended instead of adding new code? Look for patterns \
that already exist in the codebase that the new code is duplicating.

   c) **Simplification opportunities**: Now that the new code exists, could existing code elsewhere \
be updated or simplified to use it? For example, old callers that could use a new abstraction, or \
redundant logic that can now be consolidated.

   d) **Code quality** Is the code in line with the conventions in the AGENTS.md file and the surrounding code?
   If it's an FE change is the UX design in line with the existing design and user-friendly?

3. Output your verdict:
   - If everything looks good, output exactly: LGTM
   - If there are issues, output a numbered list of **specific, actionable corrections**. \
Each item must reference a concrete file and describe exactly what should change. \
Do NOT list vague suggestions — only real problems worth fixing.
"""

REVIEW_REVISE_PROMPT = """\
You are addressing self-review feedback on a git worktree.

## Issue: {id} — {title}

{description}

## Implementation plan

{plan}

## Review feedback to address

{feedback}

## Instructions

- Address each item in the review feedback above.
- Keep the original issue description and implementation plan in mind — \
ensure your changes stay consistent with the overall intent.
- Run compilation and tests after making changes.
- Commit your changes with message "fix: address review feedback for {id}"
- Do NOT push to remote. Do NOT create a PR.
"""

DEFAULT_MAX_REVIEW_REVISIONS = 5


def resolve_pr_threads(thread_ids: list[str], issue_id: str = "") -> int:
    """Resolve review threads by ID via GitHub GraphQL. Returns count of resolved threads."""
    resolved = 0
    for tid in thread_ids:
        if not tid:
            continue
        mutation = """
        mutation($threadId: ID!) {
          resolveReviewThread(input: {threadId: $threadId}) {
            thread { isResolved }
          }
        }
        """
        result = subprocess.run(
            ["gh", "api", "graphql",
             "-f", f"query={mutation}",
             "-F", f"threadId={tid}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            resolved += 1
        else:
            log(f"Failed to resolve thread {tid}: {result.stderr[:200]}", issue=issue_id)
    return resolved


def react_to_pr_comments(pr_url: str, database_ids: list[int], issue_id: str = "") -> int:
    """Add :eyes: reaction to PR issue comments. Returns count of reacted comments."""
    parsed = _parse_pr_url(pr_url)
    if not parsed:
        return 0
    owner, repo, _number = parsed
    reacted = 0
    for cid in database_ids:
        if not cid:
            continue
        result = subprocess.run(
            ["gh", "api",
             f"repos/{owner}/{repo}/issues/comments/{cid}/reactions",
             "-f", "content=eyes"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            reacted += 1
        else:
            log(f"Failed to react to comment {cid}: {result.stderr[:200]}", issue=issue_id)
    return reacted


def fetch_pr_state(pr_url: str) -> str | None:
    """Return 'merged', 'closed', or 'open' for a GitHub PR. None on error."""
    parsed = _parse_pr_url(pr_url)
    if not parsed:
        return None
    owner, repo, number = parsed
    result = subprocess.run(
        ["gh", "pr", "view", number, "--repo", f"{owner}/{repo}", "--json", "state", "-q", ".state"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    state = result.stdout.strip().upper()
    if state == "MERGED":
        return "merged"
    if state == "CLOSED":
        return "closed"
    return "open"


def _parse_pr_url(pr_url: str) -> tuple[str, str, str] | None:
    """Extract (owner, repo, number) from a GitHub PR URL."""
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def fetch_pr_comments(pr_url: str) -> list[dict]:
    """Fetch unresolved review comments from a GitHub PR using GraphQL."""
    parsed = _parse_pr_url(pr_url)
    if not parsed:
        log(f"Could not parse PR URL: {pr_url}")
        return []

    owner, repo, number = parsed

    query = """
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $number) {
          reviewThreads(first: 100) {
            nodes {
              id
              isResolved
              comments(first: 10) {
                nodes {
                  body
                  path
                  line
                  author { login }
                }
              }
            }
          }
          comments(first: 100) {
            nodes {
              id
              databaseId
              body
              author { login }
              reactionGroups {
                content
                users(first: 1) { totalCount }
              }
            }
          }
        }
      }
    }
    """

    result = subprocess.run(
        ["gh", "api", "graphql",
         "-f", f"query={query}",
         "-F", f"owner={owner}",
         "-F", f"repo={repo}",
         "-F", f"number={number}"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        log(f"Failed to fetch PR comments: {result.stderr[:500]}")
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        log("Failed to parse PR comments response")
        return []

    pr_data = data.get("data", {}).get("repository", {}).get("pullRequest")
    if not pr_data:
        log("PR not found in GraphQL response")
        return []

    threads = pr_data.get("reviewThreads", {}).get("nodes", [])

    comments = []
    for thread in threads:
        if thread.get("isResolved"):
            continue
        thread_comments = thread.get("comments", {}).get("nodes", [])
        if not thread_comments:
            continue

        first = thread_comments[0]
        body_parts = [first.get("body", "")]
        for reply in thread_comments[1:]:
            author = reply.get("author", {}).get("login", "unknown")
            body_parts.append(f"\n> Reply from {author}: {reply.get('body', '')}")

        comments.append({
            "type": "review_thread",
            "thread_id": thread.get("id"),
            "path": first.get("path"),
            "line": first.get("line"),
            "author": first.get("author", {}).get("login", "unknown"),
            "body": "\n".join(body_parts),
        })

    # PR-level issue comments (conversation tab)
    issue_comments = pr_data.get("comments", {}).get("nodes", [])
    for ic in issue_comments:
        author = ic.get("author", {}).get("login", "unknown")
        if author.endswith("[bot]"):
            continue
        # Skip if already reacted with EYES
        has_eyes = False
        for rg in ic.get("reactionGroups", []):
            if rg.get("content") == "EYES" and rg.get("users", {}).get("totalCount", 0) > 0:
                has_eyes = True
                break
        if has_eyes:
            continue
        body = ic.get("body", "").strip()
        if not body:
            continue
        comments.append({
            "type": "issue_comment",
            "database_id": ic.get("databaseId"),
            "author": author,
            "body": body,
        })

    return comments


def _format_comments_for_prompt(comments: list[dict]) -> str:
    """Format PR comments into a readable string for the revision prompt."""
    if not comments:
        return "(no comments)"

    parts = []
    for i, c in enumerate(comments, 1):
        ctype = c.get("type", "review_thread")
        label = "[PR comment]" if ctype == "issue_comment" else "[Code review]"
        location = ""
        if c.get("path"):
            location = f" in `{c['path']}`"
            if c.get("line"):
                location += f" (line {c['line']})"
        author = c.get("author", "unknown")
        parts.append(f"{i}. {label} **{author}**{location}:\n   {c['body']}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _triage_and_plan_issue(issue: dict, workspace: str) -> dict:
    """Triage a single issue and generate a plan if easy/medium. Returns tracking data."""
    iid = issue["id"]
    log(f"Triaging: {issue['title']}", issue=iid)

    comment_data = fetch_issue_comments(iid)
    issue["comments"] = _format_linear_comments(comment_data)

    image_texts: list[tuple[str, str]] = [(issue.get("description", ""), "issue description")]
    for c in comment_data.get("comments", []):
        image_texts.append((c.get("body", ""), f"comment by {c.get('author', 'unknown')}"))
    downloaded = extract_and_download_images(image_texts, workspace, issue_id=iid)
    images_section = format_images_for_prompt(downloaded)
    if images_section:
        issue["description"] = issue.get("description", "") + images_section

    decision = triage_issue(issue, workspace)
    repos = decision.get("repos", [])
    repos_str = ", ".join(repos) if repos else "none"
    log(f"Decision: {decision['decision']} | repos: {repos_str} — {decision['reasoning']}", issue=iid)

    tracking_data = {
        "issue_id": issue["id"],
        "title": issue["title"],
        "description": issue.get("description", ""),
        "url": issue["url"],
        "team": issue.get("team"),
        "labels": issue.get("labels", []),
        "triaged_at": now_iso(),
        "decision": decision["decision"],
        "repos": repos,
        "pr_type": decision.get("pr_type", "feat"),
        "reasoning": decision["reasoning"],
        "status": decision["decision"],
        "plan": None,
        "plan_summary": None,
        "implementations": {},
    }

    if decision["decision"] in ("easy", "medium"):
        log("Generating implementation plan...", issue=iid)
        plan, plan_summary = plan_issue(issue, decision["reasoning"], repos, workspace)
        tracking_data["plan"] = plan
        tracking_data["plan_summary"] = plan_summary
        tracking_data["status"] = "awaiting_approval"
        log("Plan saved — awaiting approval", issue=iid)

    return tracking_data


def cmd_triage(args: argparse.Namespace) -> None:
    workspace = str(ensure_workspace())
    _fetch_repos()

    log("Fetching issues from Linear...")
    issues = fetch_issues()
    log(f"Found {len(issues)} open issues")

    tracked = load_all_tracking()
    untracked = [i for i in issues if i["id"] not in tracked]
    log(f"{len(untracked)} untracked issues")

    # Re-triage asked_clarification issues that received a new comment
    issues_by_id = {i["id"]: i for i in issues}
    clarification_retriage: list[dict] = []
    for t in tracked.values():
        if t["status"] != "asked_clarification":
            continue
        asked_at = t.get("clarification_asked_at", "")
        if not asked_at:
            continue
        issue = issues_by_id.get(t["issue_id"])
        if not issue:
            continue
        comment_data = fetch_issue_comments(t["issue_id"])
        has_new = any(
            c.get("created_at", "") > asked_at
            for c in comment_data.get("comments", [])
        )
        if has_new:
            log(f"New comment found after clarification — re-triaging", issue=t["issue_id"])
            clarification_retriage.append(issue)

    to_triage = untracked + clarification_retriage
    if args.limit:
        to_triage = to_triage[:args.limit]
        log(f"Limited to {args.limit} issues")

    if not to_triage:
        log("No new issues to triage.")
        return

    total = len(to_triage)
    workers = getattr(args, "workers", DEFAULT_WORKERS)
    log(f"Triaging {total} issue(s) with {workers} worker(s)...")
    t0 = time.monotonic()
    done = 0
    counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_triage_and_plan_issue, issue, workspace): issue
                   for issue in to_triage}
        for future in as_completed(futures):
            issue = futures[future]
            try:
                tracking_data = future.result()
                save_tracking(issue["id"], tracking_data)
                decision = tracking_data.get("decision", "unknown")
                counts[decision] = counts.get(decision, 0) + 1
            except Exception as exc:
                log(f"Triage failed with exception: {exc}", issue=issue["id"], file=sys.stderr)
                counts["error"] = counts.get("error", 0) + 1
            done += 1
            log(f"Progress: {done}/{total} triaged")

    elapsed = time.monotonic() - t0
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    log(f"Triage complete in {_fmt_duration(elapsed)}: {summary}")
    log("Run 'status' to review.")


def cmd_status(_args: argparse.Namespace) -> None:
    tracked = load_all_tracking()
    if not tracked:
        log("No tracked issues yet. Run 'triage' first.")
        return

    status_order = {
        "awaiting_approval": 0, "approved": 1, "implemented": 2, "pushed": 3,
        "failed": 4, "triage_failed": 5, "easy": 6, "medium": 7,
        "needs_clarification": 8, "complex_skip": 9, "asked_clarification": 10,
    }
    sorted_items = sorted(tracked.values(), key=lambda t: status_order.get(t["status"], 99))

    id_w = max(len(t["issue_id"]) for t in sorted_items)
    st_w = max(len(t["status"]) for t in sorted_items)
    cx_w = max((len(t.get("decision") or "-") for t in sorted_items), default=4)
    cx_w = max(cx_w, len("Complexity"))

    print(f"{'Issue':<{id_w}}  {'Status':<{st_w}}  {'Complexity':<{cx_w}}  {'Repos':<12}  Summary")
    print(f"{'-' * id_w}  {'-' * st_w}  {'-' * cx_w}  {'-' * 12}  {'-' * 50}")

    for t in sorted_items:
        repos = ",".join(t.get("repos", []))[:12] or "-"
        complexity = t.get("decision") or "-"
        summary = ""
        if t["status"] == "awaiting_approval":
            summary = (t.get("plan_summary") or t.get("plan", "") or "").replace("\n", " ")
        elif t.get("reasoning"):
            summary = t["reasoning"]
        impls = t.get("implementations", {})
        if impls:
            branches = [v.get("worktree_branch", "") for v in impls.values() if isinstance(v, dict)]
            if branches:
                summary = f"[{', '.join(branches)}] {summary}"
        print(f"{t['issue_id']:<{id_w}}  {t['status']:<{st_w}}  {complexity:<{cx_w}}  {repos:<12}  {summary}")


def cmd_approve(args: argparse.Namespace) -> None:
    for issue_id in args.issue_ids:
        tracking = load_tracking(issue_id)
        if not tracking:
            log("Not found in tracking", issue=issue_id)
            continue
        if tracking["status"] != "awaiting_approval":
            log(f"Status is '{tracking['status']}', expected 'awaiting_approval'", issue=issue_id)
            continue
        tracking["status"] = "approved"
        tracking["approved_at"] = now_iso()
        save_tracking(issue_id, tracking)
        log("Approved", issue=issue_id)


def _self_review_repo(tracking: dict, repo_key: str, rev_num: int = 0) -> str | None:
    """Run a read-only self-review on a repo's implementation.

    Returns feedback string if corrections are needed, None if LGTM.
    """
    issue_id = tracking["issue_id"]
    impl = tracking.get("implementations", {}).get(repo_key, {})
    worktree_path = impl.get("worktree_path")
    if not worktree_path or not Path(worktree_path).is_dir():
        return None

    default_branch = _get_default_branch(worktree_path)

    prompt = REVIEW_PROMPT.format(
        id=issue_id, title=tracking["title"],
        description=tracking.get("description", ""),
        plan=tracking.get("plan", "No plan available"),
        default_branch=default_branch,
    )

    image_texts: list[tuple[str, str]] = [(tracking.get("description", ""), "issue description")]
    review_images = extract_and_download_images(image_texts, worktree_path, issue_id=issue_id)
    images_section = format_images_for_prompt(review_images)
    if images_section:
        prompt += images_section

    model = os.environ.get("IMPL_MODEL") or None
    log_name = f"{issue_id}/{repo_key}-review-{rev_num}"
    log(f"{repo_key}: running self-review...", issue=issue_id)
    exit_code, output = run_cursor_agent(
        prompt, mode="ask", workspace=worktree_path,
        log_name=log_name, timeout=IMPL_TIMEOUT, model=model,
    )

    if exit_code != 0:
        log(f"{repo_key}: review agent failed (exit={exit_code}), skipping", issue=issue_id)
        return None

    if re.search(r"\bLGTM\b", output):
        log(f"{repo_key}: review passed (LGTM)", issue=issue_id)
        return None

    feedback = output.strip()
    if not feedback:
        log(f"{repo_key}: review output empty (no LGTM, no feedback)", issue=issue_id)
        return None
    log(f"{repo_key}: review found issues", issue=issue_id)
    return feedback


def _review_revise_loop(tracking: dict, repo_key: str, max_revisions: int) -> int:
    """Run review→revise loop for a repo. Returns number of revisions made."""
    issue_id = tracking["issue_id"]
    impl = tracking.get("implementations", {}).get(repo_key, {})
    worktree_path = impl.get("worktree_path")
    if not worktree_path:
        return 0

    rev_count = 0
    while rev_count < max_revisions:
        feedback = _self_review_repo(tracking, repo_key, rev_num=rev_count + 1)
        if not feedback:
            break

        rev_count += 1
        log(f"{repo_key}: self-review revision {rev_count}/{max_revisions}...", issue=issue_id)

        prompt = REVIEW_REVISE_PROMPT.format(
            id=issue_id, title=tracking["title"],
            description=tracking.get("description", ""),
            plan=tracking.get("plan", "No plan available"),
            feedback=feedback,
        )

        image_texts: list[tuple[str, str]] = [(tracking.get("description", ""), "issue description")]
        revise_images = extract_and_download_images(image_texts, worktree_path, issue_id=issue_id)
        images_section = format_images_for_prompt(revise_images)
        if images_section:
            prompt += images_section

        model = os.environ.get("IMPL_MODEL") or None
        log_name = f"{issue_id}/{repo_key}-review-revise-{rev_count}"
        exit_code, _output = run_cursor_agent(
            prompt, workspace=worktree_path, yolo=True, sandbox=True,
            log_name=log_name, timeout=IMPL_TIMEOUT, model=model,
        )

        if exit_code != 0:
            log(f"{repo_key}: review-revise failed (exit={exit_code}), stopping loop", issue=issue_id)
            break

    tracking["revision_count"] = tracking.get("revision_count", 0) + rev_count
    return rev_count


def _implement_issue(tracking: dict, max_revisions: int = DEFAULT_MAX_REVIEW_REVISIONS) -> dict:
    """Implement all repos for a single issue. Returns updated tracking dict."""
    issue_id = tracking["issue_id"]
    repos = tracking.get("repos", ["backend"])
    log(f"Implementing: {tracking['title']} (repos: {', '.join(repos)})", issue=issue_id)

    t0 = time.monotonic()
    implementations = tracking.get("implementations", {})
    all_ok = True
    be_worktree_path = None

    for repo_key in repos:
        if repo_key in implementations and implementations[repo_key].get("exit_code") == 0:
            log(f"{repo_key}: already implemented, skipping", issue=issue_id)
            if repo_key == "backend":
                be_worktree_path = implementations[repo_key].get("worktree_path")
            continue

        log(f"{repo_key}: creating worktree and running agent...", issue=issue_id)
        impl = implement_issue_for_repo(issue_id, tracking, repo_key,
                                        be_worktree_path=be_worktree_path)
        implementations[repo_key] = impl

        if impl.get("exit_code", -1) != 0:
            all_ok = False
            log(f"{repo_key}: FAILED (exit={impl.get('exit_code')})", issue=issue_id)
        else:
            log(f"{repo_key}: done", issue=issue_id)
            if repo_key == "backend":
                be_worktree_path = impl.get("worktree_path")

            _review_revise_loop(tracking, repo_key, max_revisions)

        if impl.get("output_log"):
            log(f"  Log: {impl['output_log']}", issue=issue_id)

    elapsed = time.monotonic() - t0
    status = "implemented" if all_ok else "failed"
    tracking["implementations"] = implementations
    tracking["status"] = status
    log(f"Implementation {status} in {_fmt_duration(elapsed)}", issue=issue_id)
    return tracking


def cmd_implement(args: argparse.Namespace) -> None:
    tracked = load_all_tracking()
    issue_ids = getattr(args, "issue_ids", None)
    if issue_ids:
        approved = [t for t in tracked.values() if t["issue_id"] in issue_ids and t["status"] == "approved"]
    else:
        approved = [t for t in tracked.values() if t["status"] == "approved"]

    if not approved:
        log("No approved issues to implement. Run 'approve <issue-id>' first.")
        return

    total = len(approved)
    workers = getattr(args, "workers", DEFAULT_WORKERS)
    log(f"Implementing {total} issue(s) with {workers} worker(s)...")
    t0 = time.monotonic()
    done = 0
    succeeded = 0

    max_revisions = getattr(args, "max_revisions", DEFAULT_MAX_REVIEW_REVISIONS)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_implement_issue, t, max_revisions): t for t in approved}
        for future in as_completed(futures):
            orig = futures[future]
            try:
                tracking = future.result()
                save_tracking(tracking["issue_id"], tracking)
                if tracking["status"] == "implemented":
                    succeeded += 1
            except Exception as exc:
                log(f"Implementation failed with exception: {exc}",
                    issue=orig["issue_id"], file=sys.stderr)
            done += 1
            log(f"Progress: {done}/{total} done ({succeeded} succeeded)")

    elapsed = time.monotonic() - t0
    log(f"Implementation complete in {_fmt_duration(elapsed)}: {succeeded}/{total} succeeded")

    if succeeded:
        push_args = argparse.Namespace(all=True, issue_ids=[], link_issue=False)
        cmd_push(push_args)


def cmd_retry(args: argparse.Namespace) -> None:
    """Retry failed triage or implementation for specific issues (or all failed)."""
    tracked = load_all_tracking()
    workspace = str(ensure_workspace())
    _fetch_repos()

    if args.issue_ids:
        issue_ids = args.issue_ids
    else:
        issue_ids = [
            t["issue_id"] for t in tracked.values()
            if t["status"] in ("triage_failed", "failed")
        ]

    if not issue_ids:
        log("No failed issues to retry.")
        return

    triage_targets = []
    impl_targets = []
    for iid in issue_ids:
        t = tracked.get(iid)
        if not t:
            log(f"Not found in tracking", issue=iid)
            continue
        if t["status"] == "triage_failed":
            triage_targets.append(t)
        elif t["status"] == "failed":
            impl_targets.append(t)
        else:
            log(f"Status is '{t['status']}', nothing to retry", issue=iid)

    if triage_targets:
        log(f"Re-triaging {len(triage_targets)} issue(s)...")
        issues_by_id = {i["id"]: i for i in fetch_issues()}
        workers = getattr(args, "workers", DEFAULT_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for t in triage_targets:
                issue = issues_by_id.get(t["issue_id"])
                if not issue:
                    issue = {"id": t["issue_id"], "title": t["title"],
                             "description": t.get("description", ""), "url": t["url"]}
                futures[pool.submit(_triage_and_plan_issue, issue, workspace)] = t
            for future in as_completed(futures):
                orig = futures[future]
                try:
                    new_data = future.result()
                    save_tracking(orig["issue_id"], new_data)
                    log(f"Re-triage result: {new_data['decision']}", issue=orig["issue_id"])
                except Exception as exc:
                    log(f"Re-triage failed: {exc}", issue=orig["issue_id"], file=sys.stderr)

    if impl_targets:
        log(f"Re-implementing {len(impl_targets)} issue(s)...")
        for t in impl_targets:
            t["status"] = "approved"
            save_tracking(t["issue_id"], t)
        workers = getattr(args, "workers", DEFAULT_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_implement_issue, t): t for t in impl_targets}
            for future in as_completed(futures):
                orig = futures[future]
                try:
                    updated = future.result()
                    save_tracking(updated["issue_id"], updated)
                    log(f"Re-implementation: {updated['status']}", issue=updated["issue_id"])
                except Exception as exc:
                    log(f"Re-implementation failed: {exc}", issue=orig["issue_id"], file=sys.stderr)


def cmd_ask_clarification(args: argparse.Namespace) -> None:
    """Post a clarification comment on Linear and update status."""
    issue_id = args.issue_ids[0]
    tracking = load_tracking(issue_id)
    if not tracking:
        log(f"Not found in tracking", issue=issue_id)
        return
    if tracking["status"] != "needs_clarification":
        log(f"Status is '{tracking['status']}', expected 'needs_clarification'", issue=issue_id)
        return

    reasoning = tracking.get("reasoning", "")
    if not reasoning:
        log("No reasoning available to post", issue=issue_id)
        return

    comment_body = (
        "**🏐 Spiky needs clarification for auto-implementation:**\n\n"
        f"{reasoning}\n\n"
        "_Please reply with the requested details so we can proceed._"
    )

    log(f"Posting clarification comment...", issue=issue_id)
    if not post_linear_comment(issue_id, comment_body):
        log("Failed to post comment", issue=issue_id, file=sys.stderr)
        return

    tracking["status"] = "asked_clarification"
    tracking["clarification_asked_at"] = now_iso()
    save_tracking(issue_id, tracking)
    log("Clarification comment posted — awaiting reply", issue=issue_id)


def cmd_run(args: argparse.Namespace) -> None:
    cmd_triage(args)
    log("=" * 60)
    log("Triage complete. Now implementing approved issues...")
    cmd_implement(args)


def cmd_push(args: argparse.Namespace) -> None:
    """Push worktrees and create PRs for implemented issues."""
    if args.all:
        tracked = load_all_tracking()
        issue_ids = [t["issue_id"] for t in tracked.values() if t["status"] == "implemented"]
        if not issue_ids:
            log("No implemented issues to push.")
            return
    else:
        issue_ids = args.issue_ids

    log(f"Pushing {len(issue_ids)} issue(s)...")

    for issue_id in issue_ids:
        tracking = load_tracking(issue_id)
        if not tracking:
            log(f"Not found in tracking", issue=issue_id)
            continue
        if tracking["status"] not in ("implemented", "pushed"):
            log(f"Status is '{tracking['status']}', expected 'implemented'", issue=issue_id)
            continue

        repos = tracking.get("repos", ["backend"])
        implementations = tracking.get("implementations", {})
        prs = tracking.get("pull_requests", {})

        for repo_key in repos:
            impl = implementations.get(repo_key, {})
            if not impl or impl.get("exit_code") != 0:
                log(f"{repo_key}: not successfully implemented, skipping", issue=issue_id)
                continue
            if repo_key in prs:
                log(f"{repo_key}: PR already created: {prs[repo_key]['url']}", issue=issue_id)
                continue

            worktree_path = impl["worktree_path"]
            branch = impl["worktree_branch"]

            log(f"{repo_key}: pushing {branch}...", issue=issue_id)
            push_result = subprocess.run(
                ["git", "-C", worktree_path, "push", "-u", "origin", branch],
                capture_output=True, text=True,
            )
            if push_result.returncode != 0:
                log(f"{repo_key}: push failed: {push_result.stderr.strip()}", issue=issue_id)
                continue

            pr_title = _make_pr_title(tracking)
            pr_body = _make_pr_body(tracking, repo_key,
                                    link_issue=getattr(args, "link_issue", False))

            log(f"{repo_key}: creating PR...", issue=issue_id)
            pr_result = subprocess.run(
                ["gh", "pr", "create",
                 "--title", pr_title,
                 "--body", pr_body,
                 "--head", branch],
                capture_output=True, text=True, cwd=worktree_path,
            )
            if pr_result.returncode != 0:
                log(f"{repo_key}: PR creation failed: {pr_result.stderr.strip()}", issue=issue_id)
                continue

            pr_url = pr_result.stdout.strip()
            log(f"{repo_key}: PR created: {pr_url}", issue=issue_id)
            prs[repo_key] = {"url": pr_url, "created_at": now_iso()}

        tracking["pull_requests"] = prs
        if prs:
            tracking["status"] = "pushed"
        save_tracking(issue_id, tracking)


def _make_pr_title(tracking: dict) -> str:
    """Generate a conventional commit style PR title: <type>: <lowercase title>"""
    title = tracking["title"]
    pr_type = tracking.get("pr_type", "feat")
    if pr_type not in ("feat", "fix", "chore"):
        pr_type = "feat"
    lower_title = title[0].lower() + title[1:] if title else title
    if lower_title.endswith("."):
        lower_title = lower_title[:-1]
    return f"{pr_type}: {lower_title}"


def _make_pr_body(tracking: dict, repo_key: str, *, link_issue: bool = False) -> str:
    """Generate PR body from tracking data."""
    issue_url = tracking.get("url", "")
    repos = tracking.get("repos", [])

    lines = []
    if link_issue and issue_url:
        lines.append(f"Fixes: {issue_url}")
    lines.append("")
    lines.append(f"🏐 Auto-generated by Spiky from {tracking['issue_id']}.")
    lines.append("")

    summary = tracking.get("plan_summary", "")
    if summary:
        lines.append(summary)
        lines.append("")

    if len(repos) > 1 and repo_key == "frontend":
        lines.append("### Note")
        lines.append("")
        lines.append("This issue also has backend changes in a separate PR. "
                      "This PR may have compilation failures until the backend PR is merged and published.")
        lines.append("")

    lines.append("### Testing")
    lines.append("")
    lines.append("- [ ] Review agent-generated changes")
    lines.append("- [ ] Verify compilation and tests pass")

    return "\n".join(lines)


def _merge_upstream(worktree_path: str, issue_id: str, repo_key: str) -> bool:
    """Fetch and merge upstream main into the worktree branch.

    Returns True if there are unresolved merge conflicts (agent should fix them),
    False if the merge was clean or a fast-forward.
    """
    log(f"{repo_key}: fetching upstream...", issue=issue_id)
    subprocess.run(["git", "-C", worktree_path, "fetch", "origin"],
                   capture_output=True, text=True)

    default_branch = _get_default_branch(worktree_path)
    remote_ref = f"origin/{default_branch}"

    log(f"{repo_key}: merging {remote_ref}...", issue=issue_id)
    merge_result = subprocess.run(
        ["git", "-C", worktree_path, "merge", remote_ref, "--no-edit"],
        capture_output=True, text=True,
    )

    if merge_result.returncode == 0:
        log(f"{repo_key}: merge clean", issue=issue_id)
        return False

    if "CONFLICT" in merge_result.stdout or "CONFLICT" in merge_result.stderr:
        conflict_files = subprocess.run(
            ["git", "-C", worktree_path, "diff", "--name-only", "--diff-filter=U"],
            capture_output=True, text=True,
        )
        files = conflict_files.stdout.strip()
        log(f"{repo_key}: merge conflicts in: {files}", issue=issue_id)
        return True

    log(f"{repo_key}: merge failed unexpectedly: {merge_result.stderr.strip()}", issue=issue_id)
    return False


def _revise_repo(tracking: dict, repo_key: str, extra_context: str = "", rev_num: int = 0) -> dict | None:
    """Revise a single repo for an issue. Returns revision record or None if skipped/failed."""
    issue_id = tracking["issue_id"]
    prs = tracking.get("pull_requests", {})
    pr_info = prs.get(repo_key, {})
    pr_url = pr_info.get("url", "")
    if not pr_url:
        return None

    log(f"{repo_key}: checking PR {pr_url}", issue=issue_id)
    comments = fetch_pr_comments(pr_url)

    if not comments:
        log(f"{repo_key}: no unresolved comments, skipping", issue=issue_id)
        return None

    log(f"{repo_key}: found {len(comments)} unresolved comment(s)", issue=issue_id)

    impl = tracking.get("implementations", {}).get(repo_key, {})
    worktree_path = impl.get("worktree_path")
    if not worktree_path or not Path(worktree_path).is_dir():
        log(f"{repo_key}: worktree not found at {worktree_path}, skipping", issue=issue_id)
        return None

    merge_conflict = _merge_upstream(worktree_path, issue_id, repo_key)

    comments_formatted = _format_comments_for_prompt(comments)
    if merge_conflict:
        comments_formatted += (
            "\n\nADDITIONAL TASK: There are merge conflicts with the main branch that "
            "have been left as conflict markers in the working tree. Resolve all merge "
            "conflicts, then run compilation/tests."
        )
    if extra_context:
        comments_formatted += f"\n\n{extra_context}"

    prior_log = ""
    impl_log_path = impl.get("output_log", "")
    if impl_log_path and Path(impl_log_path).exists():
        content = Path(impl_log_path).read_text()
        prior_log = content[-4000:] if len(content) > 4000 else content

    triage = tracking.get("triage", {})
    triage_text = triage.get("summary", "") if isinstance(triage, dict) else str(triage)

    prompt = REVISE_PROMPT.format(
        id=issue_id, title=tracking["title"],
        description=tracking.get("description", ""),
        triage=triage_text or "No triage available",
        plan=tracking.get("plan", "No plan available"),
        prior_log=prior_log or "No prior log available",
        comments_formatted=comments_formatted,
    )

    image_texts: list[tuple[str, str]] = [(tracking.get("description", ""), "issue description")]
    for c in comments:
        image_texts.append((c.get("body", ""), f"PR comment by {c.get('author', 'unknown')}"))
    revise_images = extract_and_download_images(image_texts, worktree_path, issue_id=issue_id)
    images_section = format_images_for_prompt(revise_images)
    if images_section:
        prompt += images_section

    model = os.environ.get("IMPL_MODEL") or None
    log_name = f"{issue_id}/{repo_key}-revise-{rev_num}"
    exit_code, _output = run_cursor_agent(
        prompt, workspace=worktree_path, yolo=True, sandbox=True,
        log_name=log_name, timeout=IMPL_TIMEOUT, model=model,
    )

    log_file = LOGS_DIR / f"{log_name}.log"

    if exit_code != 0:
        log(f"{repo_key}: revision agent FAILED (exit={exit_code})", issue=issue_id)
        if log_file.exists():
            log(f"  Log: {log_file}", issue=issue_id)
        return None

    log(f"{repo_key}: pushing revision...", issue=issue_id)
    push_result = subprocess.run(
        ["git", "-C", worktree_path, "push"],
        capture_output=True, text=True,
    )
    if push_result.returncode != 0:
        log(f"{repo_key}: push failed: {push_result.stderr.strip()}", issue=issue_id)
        return None

    log(f"{repo_key}: revision pushed successfully", issue=issue_id)

    thread_ids = [c.get("thread_id") for c in comments if c.get("thread_id")]
    if thread_ids:
        resolved = resolve_pr_threads(thread_ids, issue_id=issue_id)
        log(f"{repo_key}: resolved {resolved}/{len(thread_ids)} review thread(s)", issue=issue_id)

    ic_ids = [c.get("database_id") for c in comments if c.get("type") == "issue_comment" and c.get("database_id")]
    if ic_ids:
        reacted = react_to_pr_comments(pr_url, ic_ids, issue_id=issue_id)
        log(f"{repo_key}: reacted to {reacted}/{len(ic_ids)} PR comment(s)", issue=issue_id)

    return {
        "repo": repo_key,
        "comments_addressed": len(comments),
        "revised_at": now_iso(),
        "exit_code": exit_code,
        "log": str(log_file),
    }


def _revise_issue(tracking: dict, max_revisions: int = DEFAULT_MAX_REVIEW_REVISIONS) -> dict:
    """Check for unresolved PR comments and run agents to address them.

    Processes backend first so that if BE changes occur, the updated APIs
    can be linked into the frontend worktree before the FE agent runs.
    After each successful PR revision, runs the self-review loop.
    """
    issue_id = tracking["issue_id"]
    prs = tracking.get("pull_requests", {})
    if not prs:
        return tracking

    revisions = tracking.get("revisions", [])
    revisions_before = len(revisions)
    impls = tracking.get("implementations", {})

    # Process BE first so we can link updated APIs into FE
    repo_order = sorted(prs.keys(), key=lambda r: (r != "backend", r))

    be_revised = False
    for repo_key in repo_order:
        extra_context = ""

        # If BE was just revised, re-link into FE so it compiles against updated types
        if repo_key == "frontend" and be_revised:
            be_wt = impls.get("backend", {}).get("worktree_path")
            fe_wt = impls.get("frontend", {}).get("worktree_path")
            if be_wt and fe_wt and Path(be_wt).is_dir() and Path(fe_wt).is_dir():
                log("Re-linking backend APIs into frontend after BE revision...", issue=issue_id)
                if link_be_to_fe(be_wt, fe_wt):
                    extra_context = (
                        "NOTE: The backend was also revised in this cycle. The updated backend "
                        "APIs have been rebuilt and linked locally into this worktree via "
                        "scout:link. If the backend changes affect types or endpoints you use, "
                        "make sure the frontend code is consistent with the new backend APIs."
                    )
                else:
                    log("WARNING: Failed to re-link backend APIs", issue=issue_id)

        revision = _revise_repo(tracking, repo_key, extra_context=extra_context, rev_num=len(revisions) + 1)
        if revision:
            revisions.append(revision)
            if repo_key == "backend":
                be_revised = True

            remaining = max_revisions - tracking.get("revision_count", 0)
            if remaining > 0:
                _review_revise_loop(tracking, repo_key, remaining)

    tracking["revisions"] = revisions
    if revisions_before != len(revisions):
        tracking["revision_count"] = tracking.get("revision_count", 0) + 1
    return tracking


def _check_pr_states(candidates: list[dict]) -> list[dict]:
    """Check PR states and move merged/closed issues out of 'pushed'. Returns remaining candidates."""
    remaining = []
    for t in candidates:
        prs = t.get("pull_requests", {})
        pr_urls = [pr.get("url") for pr in prs.values() if pr and pr.get("url")]
        if not pr_urls:
            remaining.append(t)
            continue

        states = [fetch_pr_state(url) for url in pr_urls]
        issue_id = t["issue_id"]

        if all(s == "merged" for s in states):
            log(f"{issue_id}: all PRs merged, marking completed", issue=issue_id)
            t["status"] = "completed"
            t["completed_at"] = now_iso()
            save_tracking(issue_id, t)
        elif any(s == "closed" for s in states):
            log(f"{issue_id}: PR closed/canceled, marking canceled", issue=issue_id)
            t["status"] = "canceled"
            t["canceled_at"] = now_iso()
            save_tracking(issue_id, t)
        else:
            remaining.append(t)
    return remaining


def cmd_revise(args: argparse.Namespace) -> None:
    """Address PR review feedback for pushed issues."""
    tracked = load_all_tracking()

    if args.issue_ids:
        candidates = [tracked[iid] for iid in args.issue_ids if iid in tracked]
    else:
        candidates = [t for t in tracked.values() if t["status"] == "pushed"]

    candidates = _check_pr_states(candidates)

    max_revisions = getattr(args, "max_revisions", None)
    if max_revisions is not None:
        before = len(candidates)
        candidates = [t for t in candidates if t.get("revision_count", 0) < max_revisions]
        skipped = before - len(candidates)
        if skipped:
            log(f"Skipped {skipped} issue(s) already at {max_revisions}+ revisions")

    if not candidates:
        log("No pushed issues with PRs to revise.")
        return

    total = len(candidates)
    workers = getattr(args, "workers", DEFAULT_WORKERS)
    log(f"Revising {total} issue(s) with {workers} worker(s)...")
    t0 = time.monotonic()
    done = 0

    review_max = max_revisions if max_revisions is not None else DEFAULT_MAX_REVIEW_REVISIONS

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_revise_issue, t, review_max): t for t in candidates}
        for future in as_completed(futures):
            orig = futures[future]
            try:
                tracking = future.result()
                save_tracking(tracking["issue_id"], tracking)
            except Exception as exc:
                log(f"Revision failed with exception: {exc}",
                    issue=orig["issue_id"], file=sys.stderr)
            done += 1
            log(f"Progress: {done}/{total} revised")

    elapsed = time.monotonic() - t0
    log(f"Revision complete in {_fmt_duration(elapsed)}")


def cmd_show(args: argparse.Namespace) -> None:
    """Show full details for a specific issue."""
    tracking = load_tracking(args.issue_id)
    if not tracking:
        print(f"Issue {args.issue_id} not found in tracking.")
        return
    print(json.dumps(tracking, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _add_workers_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Max parallel agents (default: {DEFAULT_WORKERS})")


def main():
    parser = argparse.ArgumentParser(
        description="Spiky: triage, plan, and implement Linear issues via cursor agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_triage = sub.add_parser("triage", help="Fetch and triage open Linear issues")
    p_triage.add_argument("--limit", type=int, default=None, help="Max issues to triage")
    _add_workers_arg(p_triage)
    p_triage.set_defaults(func=cmd_triage)

    p_status = sub.add_parser("status", help="Show status of all tracked issues")
    p_status.set_defaults(func=cmd_status)

    p_approve = sub.add_parser("approve", help="Approve issues for implementation")
    p_approve.add_argument("issue_ids", nargs="+", help="Issue identifiers to approve")
    p_approve.set_defaults(func=cmd_approve)

    p_ask = sub.add_parser("ask_clarification", help="Post clarification comment on Linear")
    p_ask.add_argument("issue_ids", nargs=1, help="Issue identifier")
    p_ask.set_defaults(func=cmd_ask_clarification)

    p_implement = sub.add_parser("implement", help="Implement approved issues")
    p_implement.add_argument("issue_ids", nargs="*", help="Issue identifiers to implement (default: all approved)")
    p_implement.add_argument("--max-revisions", type=int, default=DEFAULT_MAX_REVIEW_REVISIONS,
                             help=f"Max self-review revision cycles (default: {DEFAULT_MAX_REVIEW_REVISIONS})")
    _add_workers_arg(p_implement)
    p_implement.set_defaults(func=cmd_implement)

    p_run = sub.add_parser("run", help="Full cycle: triage then implement approved")
    p_run.add_argument("--limit", type=int, default=None, help="Max issues to triage")
    p_run.add_argument("--max-revisions", type=int, default=DEFAULT_MAX_REVIEW_REVISIONS,
                       help=f"Max self-review revision cycles (default: {DEFAULT_MAX_REVIEW_REVISIONS})")
    _add_workers_arg(p_run)
    p_run.set_defaults(func=cmd_run)

    p_push = sub.add_parser("push", help="Push worktrees and create PRs for implemented issues")
    p_push.add_argument("issue_ids", nargs="*", help="Issue identifiers to push")
    p_push.add_argument("--all", action="store_true", help="Push all implemented issues")
    p_push.add_argument("--link-issue", action="store_true",
                        help="Include 'Fixes: <linear-url>' in PR body")
    p_push.set_defaults(func=cmd_push)

    p_revise = sub.add_parser("revise", help="Address PR review feedback for pushed issues")
    p_revise.add_argument("issue_ids", nargs="*", help="Issue identifiers to revise (default: all pushed)")
    p_revise.add_argument("--max-revisions", type=int, default=None,
                          help="Skip issues that already have this many revisions")
    _add_workers_arg(p_revise)
    p_revise.set_defaults(func=cmd_revise)

    p_retry = sub.add_parser("retry", help="Retry failed triage or implementation")
    p_retry.add_argument("issue_ids", nargs="*", help="Issue identifiers to retry (default: all failed)")
    _add_workers_arg(p_retry)
    p_retry.set_defaults(func=cmd_retry)

    p_show = sub.add_parser("show", help="Show full details for an issue")
    p_show.add_argument("issue_id", help="Issue identifier")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
