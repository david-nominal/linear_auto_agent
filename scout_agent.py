#!/usr/bin/env python3
"""Linear Auto-Agent: fetch open issues, triage, plan, and implement via cursor agent CLI."""

import argparse
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
    "backend": {"env_var": "BACKEND_REPO", "symlink_name": "scout", "description": "Java backend"},
    "frontend": {"env_var": "FRONTEND_REPO", "symlink_name": "galaxy", "description": "TypeScript frontend"},
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
# Repo configuration
# ---------------------------------------------------------------------------

def get_repo_path(repo_key: str) -> str:
    cfg = REPOS[repo_key]
    path = os.environ.get(cfg["env_var"], "")
    if not path:
        print(f"ERROR: {cfg['env_var']} not set in .env", file=sys.stderr)
        sys.exit(1)
    if not Path(path).is_dir():
        print(f"ERROR: {cfg['env_var']}={path} is not a directory", file=sys.stderr)
        sys.exit(1)
    return path


def ensure_workspace_symlinks() -> Path:
    """Create data/workspace/ with symlinks to both repos."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    for repo_key, cfg in REPOS.items():
        link = WORKSPACE_DIR / cfg["symlink_name"]
        target = Path(get_repo_path(repo_key)).resolve()
        if link.is_symlink():
            if link.resolve() == target:
                continue
            link.unlink()
        elif link.exists():
            print(f"ERROR: {link} exists and is not a symlink", file=sys.stderr)
            sys.exit(1)
        link.symlink_to(target)
    return WORKSPACE_DIR


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
    log(f"API key set: {len(api_key)} chars")
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
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / f"{log_name}.log"

    issue_hint = ""
    if log_name:
        issue_hint = log_name.split("-triage")[0].split("-plan")[0].split("-backend")[0].split("-frontend")[0].split("-revise")[0]

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

## Context

The workspace contains two repos:
- scout/ — Java backend (Gradle, Conjure APIs, Postgres, ClickHouse, Temporal)
- galaxy/ — TypeScript frontend (React)

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
                                             log_name=f"{issue['id']}-triage",
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
- scout/ — Java backend (Gradle, Conjure APIs, Postgres, ClickHouse, Temporal). Conventions in scout/AGENTS.md.
- galaxy/ — TypeScript frontend (React)

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
    model = os.environ.get("PLAN_MODEL") or None
    exit_code, output = run_cursor_agent(prompt, mode="plan", workspace=workspace,
                                         log_name=f"{issue['id']}-plan",
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
    log("  Linking scout APIs in frontend worktree...")
    result = subprocess.run(
        ["pnpm", "scout:link", be_worktree],
        capture_output=True, text=True, cwd=fe_worktree, timeout=120,
    )
    if result.returncode != 0:
        log(f"  scout:link failed: {result.stderr[:500]}")
        return False
    log(f"  scout:link done ({_fmt_duration(time.monotonic() - t2)})")

    t3 = time.monotonic()
    log("  Running proto codegen...")
    result = subprocess.run(
        ["pnpm", "generate:proto"],
        capture_output=True, text=True, cwd=fe_worktree, timeout=120,
    )
    if result.returncode != 0:
        log(f"  generate:proto failed (non-fatal): {result.stderr[:500]}")
    else:
        log(f"  generate:proto done ({_fmt_duration(time.monotonic() - t3)})")

    log(f"  FE/BE linking complete ({_fmt_duration(time.monotonic() - t0)})")
    return True


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

IMPLEMENT_PROMPT_BACKEND = """\
You are implementing a Linear issue on a git worktree of the Java backend repo (scout).

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
You are implementing a Linear issue on a git worktree of the TypeScript frontend repo (galaxy).

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
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

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

    model = os.environ.get("IMPL_MODEL") or None
    started_at = now_iso()
    exit_code, output = run_cursor_agent(prompt, workspace=worktree_path, yolo=True, sandbox=True,
                                         log_name=worktree_name, timeout=IMPL_TIMEOUT,
                                         model=model)
    completed_at = now_iso()

    log_file = LOGS_DIR / f"{worktree_name}.log"

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

## Unresolved review comments

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
            "thread_id": thread.get("id"),
            "path": first.get("path"),
            "line": first.get("line"),
            "author": first.get("author", {}).get("login", "unknown"),
            "body": "\n".join(body_parts),
        })

    return comments


def _format_comments_for_prompt(comments: list[dict]) -> str:
    """Format PR comments into a readable string for the revision prompt."""
    if not comments:
        return "(no comments)"

    parts = []
    for i, c in enumerate(comments, 1):
        location = ""
        if c.get("path"):
            location = f" in `{c['path']}`"
            if c.get("line"):
                location += f" (line {c['line']})"
        author = c.get("author", "unknown")
        parts.append(f"{i}. **{author}**{location}:\n   {c['body']}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _triage_and_plan_issue(issue: dict, workspace: str) -> dict:
    """Triage a single issue and generate a plan if easy/medium. Returns tracking data."""
    iid = issue["id"]
    log(f"Triaging: {issue['title']}", issue=iid)

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
    workspace = str(ensure_workspace_symlinks())

    log("Fetching issues from Linear...")
    issues = fetch_issues()
    log(f"Found {len(issues)} open issues")

    tracked = load_all_tracking()
    untracked = [i for i in issues if i["id"] not in tracked]
    log(f"{len(untracked)} untracked issues")

    if args.limit:
        untracked = untracked[:args.limit]
        log(f"Limited to {args.limit} issues")

    if not untracked:
        log("No new issues to triage.")
        return

    total = len(untracked)
    workers = getattr(args, "workers", DEFAULT_WORKERS)
    log(f"Triaging {total} issue(s) with {workers} worker(s)...")
    t0 = time.monotonic()
    done = 0
    counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_triage_and_plan_issue, issue, workspace): issue
                   for issue in untracked}
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
        "needs_clarification": 8, "complex_skip": 9,
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


def _implement_issue(tracking: dict) -> dict:
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

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_implement_issue, t): t for t in approved}
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


def cmd_retry(args: argparse.Namespace) -> None:
    """Retry failed triage or implementation for specific issues (or all failed)."""
    tracked = load_all_tracking()
    workspace = str(ensure_workspace_symlinks())

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
    lines.append(f"Auto-generated by linear-auto-agent from {tracking['issue_id']}.")
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


def _revise_issue(tracking: dict) -> dict:
    """Check for unresolved PR comments and run agents to address them."""
    issue_id = tracking["issue_id"]
    prs = tracking.get("pull_requests", {})
    if not prs:
        return tracking

    revisions = tracking.get("revisions", [])

    for repo_key, pr_info in prs.items():
        pr_url = pr_info.get("url", "")
        if not pr_url:
            continue

        log(f"{repo_key}: checking PR {pr_url}", issue=issue_id)
        comments = fetch_pr_comments(pr_url)

        if not comments:
            log(f"{repo_key}: no unresolved comments, skipping", issue=issue_id)
            continue

        log(f"{repo_key}: found {len(comments)} unresolved comment(s)", issue=issue_id)

        impl = tracking.get("implementations", {}).get(repo_key, {})
        worktree_path = impl.get("worktree_path")
        if not worktree_path or not Path(worktree_path).is_dir():
            log(f"{repo_key}: worktree not found at {worktree_path}, skipping", issue=issue_id)
            continue

        merge_conflict = _merge_upstream(worktree_path, issue_id, repo_key)

        comments_formatted = _format_comments_for_prompt(comments)
        if merge_conflict:
            comments_formatted += (
                "\n\nADDITIONAL TASK: There are merge conflicts with the main branch that "
                "have been left as conflict markers in the working tree. Resolve all merge "
                "conflicts, then run compilation/tests."
            )

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

        model = os.environ.get("IMPL_MODEL") or None
        log_name = f"{issue_id}-{repo_key}-revise"
        exit_code, _output = run_cursor_agent(
            prompt, workspace=worktree_path, yolo=True, sandbox=True,
            log_name=log_name, timeout=IMPL_TIMEOUT, model=model,
        )

        log_file = LOGS_DIR / f"{log_name}.log"

        if exit_code != 0:
            log(f"{repo_key}: revision agent FAILED (exit={exit_code})", issue=issue_id)
            if log_file.exists():
                log(f"  Log: {log_file}", issue=issue_id)
            continue

        log(f"{repo_key}: pushing revision...", issue=issue_id)
        push_result = subprocess.run(
            ["git", "-C", worktree_path, "push"],
            capture_output=True, text=True,
        )
        if push_result.returncode != 0:
            log(f"{repo_key}: push failed: {push_result.stderr.strip()}", issue=issue_id)
            continue

        log(f"{repo_key}: revision pushed successfully", issue=issue_id)

        thread_ids = [c.get("thread_id") for c in comments if c.get("thread_id")]
        if thread_ids:
            resolved = resolve_pr_threads(thread_ids, issue_id=issue_id)
            log(f"{repo_key}: resolved {resolved}/{len(thread_ids)} review thread(s)", issue=issue_id)
        revisions.append({
            "repo": repo_key,
            "comments_addressed": len(comments),
            "revised_at": now_iso(),
            "exit_code": exit_code,
            "log": str(log_file),
        })

    tracking["revisions"] = revisions
    return tracking


def cmd_revise(args: argparse.Namespace) -> None:
    """Address PR review feedback for pushed issues."""
    tracked = load_all_tracking()

    if args.issue_ids:
        candidates = [tracked[iid] for iid in args.issue_ids if iid in tracked]
    else:
        candidates = [t for t in tracked.values() if t["status"] == "pushed"]

    if not candidates:
        log("No pushed issues with PRs to revise.")
        return

    total = len(candidates)
    workers = getattr(args, "workers", DEFAULT_WORKERS)
    log(f"Revising {total} issue(s) with {workers} worker(s)...")
    t0 = time.monotonic()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_revise_issue, t): t for t in candidates}
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
        description="Linear Auto-Agent: triage, plan, and implement Linear issues via cursor agent",
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

    p_implement = sub.add_parser("implement", help="Implement all approved issues")
    _add_workers_arg(p_implement)
    p_implement.set_defaults(func=cmd_implement)

    p_run = sub.add_parser("run", help="Full cycle: triage then implement approved")
    p_run.add_argument("--limit", type=int, default=None, help="Max issues to triage")
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
