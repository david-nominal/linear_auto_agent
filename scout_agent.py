#!/usr/bin/env python3
"""Linear Auto-Agent: fetch open issues, triage, plan, and implement via cursor agent CLI."""

import argparse
import json
import os
import re
import subprocess
import sys
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
                duplicate: { null: true }
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
        resp.raise_for_status()
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
                     yolo: bool = False, sandbox: bool = False) -> tuple[int, str]:
    """Run cursor agent CLI and return (exit_code, output).

    Security:
    - --print mode: MCPs require interactive approval, which is impossible headlessly,
      so all MCP servers are effectively blocked (unless --approve-mcps is passed, which we never do).
    - --mode ask/plan: read-only, no file edits, no shell commands.
    - --sandbox enabled: OS-level (macOS Seatbelt) restriction — writes only to workspace + /tmp,
      no network access. Used for implementation agents to hard-scope writes to the worktree.
    """
    cmd = ["cursor", "agent", "--print", "--trust"]

    if mode:
        cmd += ["--mode", mode]
    if workspace:
        cmd += ["--workspace", workspace]
    if yolo:
        cmd.append("--yolo")
    if sandbox:
        cmd += ["--sandbox", "enabled"]

    cmd.append(prompt)

    print(f"  Running cursor agent ({mode or 'default'} mode, sandbox={'on' if sandbox else 'off'})...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    return result.returncode, output


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
2. If yes, is it an easy task (isolated change, < ~1 hour of focused work, no migrations, no cross-service breaking changes) or complex?
3. Which repo(s) need changes?

You MUST output ONLY a JSON block (no other text) in this exact format:

```json
{{"decision": "<needs_clarification|complex_skip|easy>", "repos": ["backend", "frontend"], "reasoning": "<1-2 sentence explanation>"}}
```

"repos" should only include repos that need changes. Use "backend" for scout, "frontend" for galaxy.
"""


def triage_issue(issue: dict, workspace: str) -> dict:
    """Triage a single issue. Returns the parsed decision dict."""
    prompt = TRIAGE_PROMPT.format(**issue)
    exit_code, output = run_cursor_agent(prompt, mode="ask", workspace=workspace)

    parsed = parse_json_from_output(output)
    if parsed and "decision" in parsed:
        if "repos" not in parsed:
            parsed["repos"] = ["backend"]
        return parsed

    return {
        "decision": "needs_clarification",
        "repos": [],
        "reasoning": f"Failed to parse agent output (exit={exit_code})",
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

PLAN_PROMPT = """\
You are creating an implementation plan for a Linear issue that has been triaged as easy/automatable.

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

Create a concrete implementation plan. For each affected repo, provide:
1. Which files need to be created or modified (specific paths)
2. What the changes are (be specific)
3. What tests need to be added or updated
4. Any risks or edge cases

If both repos are affected, clearly separate the plan into "## Backend (scout)" and "## Frontend (galaxy)" sections.
Note: if both are affected, the frontend changes may depend on the backend changes being merged and published first.
The frontend PR may have compilation failures until then — that's expected and should be noted.

Output your plan as a clear, numbered list per repo. Be specific about file paths.
"""


def plan_issue(issue: dict, reasoning: str, repos: list[str], workspace: str) -> str:
    """Generate an implementation plan. Returns the plan text."""
    repos_str = ", ".join(repos)
    prompt = PLAN_PROMPT.format(**issue, reasoning=reasoning, repos_str=repos_str)
    exit_code, output = run_cursor_agent(prompt, mode="plan", workspace=workspace)

    if exit_code != 0:
        return f"[Plan generation failed with exit code {exit_code}]\n\n{output[-2000:]}"

    return output.strip()


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


def implement_issue_for_repo(issue_id: str, tracking: dict, repo_key: str) -> dict:
    """Create a worktree in the given repo and run the implementation agent. Returns impl record."""
    repo_path = get_repo_path(repo_key)
    branch_name = f"agent/{issue_id}"
    worktree_name = f"{issue_id}-{repo_key}"
    worktree_path = str((WORKTREES_DIR / worktree_name).resolve())

    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    default_branch = _get_default_branch(repo_path)
    wt_cmd = ["git", "-C", repo_path, "worktree", "add", worktree_path, "-b", branch_name, default_branch]
    wt_result = subprocess.run(wt_cmd, capture_output=True, text=True)
    if wt_result.returncode != 0:
        if "already exists" in wt_result.stderr:
            print(f"  Worktree/branch {branch_name} already exists, reusing")
        else:
            return {
                "repo": repo_key,
                "error": f"Failed to create worktree: {wt_result.stderr}",
                "exit_code": -1,
                "started_at": now_iso(),
                "completed_at": now_iso(),
            }

    repos_affected = tracking.get("repos", [])
    be_note = BE_DEPENDENCY_NOTE if repo_key == "frontend" and "backend" in repos_affected else ""

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

    started_at = now_iso()
    exit_code, output = run_cursor_agent(prompt, workspace=worktree_path, yolo=True, sandbox=True)
    completed_at = now_iso()

    log_file = LOGS_DIR / f"{worktree_name}.log"
    log_file.write_text(output)

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
# CLI commands
# ---------------------------------------------------------------------------

def cmd_triage(args: argparse.Namespace) -> None:
    workspace = str(ensure_workspace_symlinks())

    print("Fetching issues from Linear...")
    issues = fetch_issues()
    print(f"  Found {len(issues)} open issues")

    tracked = load_all_tracking()
    untracked = [i for i in issues if i["id"] not in tracked]
    print(f"  {len(untracked)} untracked issues")

    if args.limit:
        untracked = untracked[:args.limit]
        print(f"  Limited to {args.limit} issues")

    for issue in untracked:
        print(f"\n--- Triaging {issue['id']}: {issue['title']}")

        decision = triage_issue(issue, workspace)
        repos = decision.get("repos", [])
        repos_str = ", ".join(repos) if repos else "none"
        print(f"  Decision: {decision['decision']} | repos: {repos_str} — {decision['reasoning']}")

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
            "reasoning": decision["reasoning"],
            "status": decision["decision"],
            "plan": None,
            "implementations": {},
        }

        if decision["decision"] == "easy":
            print("  Generating implementation plan...")
            plan = plan_issue(issue, decision["reasoning"], repos, workspace)
            tracking_data["plan"] = plan
            tracking_data["status"] = "awaiting_approval"
            print("  Plan saved — awaiting approval")

        save_tracking(issue["id"], tracking_data)

    print("\nTriage complete. Run 'status' to review.")


def cmd_status(_args: argparse.Namespace) -> None:
    tracked = load_all_tracking()
    if not tracked:
        print("No tracked issues yet. Run 'triage' first.")
        return

    status_order = {
        "awaiting_approval": 0, "approved": 1, "implemented": 2,
        "failed": 3, "easy": 4, "needs_clarification": 5, "complex_skip": 6,
    }
    sorted_items = sorted(tracked.values(), key=lambda t: status_order.get(t["status"], 99))

    id_w = max(len(t["issue_id"]) for t in sorted_items)
    st_w = max(len(t["status"]) for t in sorted_items)

    print(f"{'Issue':<{id_w}}  {'Status':<{st_w}}  {'Repos':<12}  Summary")
    print(f"{'-' * id_w}  {'-' * st_w}  {'-' * 12}  {'-' * 50}")

    for t in sorted_items:
        repos = ",".join(t.get("repos", []))[:12] or "-"
        summary = ""
        if t["status"] == "awaiting_approval" and t.get("plan"):
            summary = t["plan"][:70].replace("\n", " ")
        elif t.get("reasoning"):
            summary = t["reasoning"][:70]
        impls = t.get("implementations", {})
        if impls:
            branches = [v.get("worktree_branch", "") for v in impls.values() if isinstance(v, dict)]
            if branches:
                summary = f"[{', '.join(branches)}] {summary}"
        print(f"{t['issue_id']:<{id_w}}  {t['status']:<{st_w}}  {repos:<12}  {summary}")


def cmd_approve(args: argparse.Namespace) -> None:
    for issue_id in args.issue_ids:
        tracking = load_tracking(issue_id)
        if not tracking:
            print(f"  {issue_id}: not found in tracking")
            continue
        if tracking["status"] != "awaiting_approval":
            print(f"  {issue_id}: status is '{tracking['status']}', expected 'awaiting_approval'")
            continue
        tracking["status"] = "approved"
        tracking["approved_at"] = now_iso()
        save_tracking(issue_id, tracking)
        print(f"  {issue_id}: approved")


def cmd_implement(_args: argparse.Namespace) -> None:
    tracked = load_all_tracking()
    approved = [t for t in tracked.values() if t["status"] == "approved"]

    if not approved:
        print("No approved issues to implement. Run 'approve <issue-id>' first.")
        return

    for tracking in approved:
        issue_id = tracking["issue_id"]
        repos = tracking.get("repos", ["backend"])
        print(f"\n--- Implementing {issue_id}: {tracking['title']} (repos: {', '.join(repos)})")

        implementations = tracking.get("implementations", {})
        all_ok = True

        for repo_key in repos:
            if repo_key in implementations and implementations[repo_key].get("exit_code") == 0:
                print(f"  {repo_key}: already implemented, skipping")
                continue

            print(f"  {repo_key}: creating worktree and running agent...")
            impl = implement_issue_for_repo(issue_id, tracking, repo_key)
            implementations[repo_key] = impl

            if impl.get("exit_code", -1) != 0:
                all_ok = False
                print(f"  {repo_key}: FAILED (exit={impl.get('exit_code')})")
                if impl.get("output_log"):
                    print(f"    Log: {impl['output_log']}")
            else:
                print(f"  {repo_key}: done")
                if impl.get("output_log"):
                    print(f"    Log: {impl['output_log']}")

        tracking["implementations"] = implementations
        tracking["status"] = "implemented" if all_ok else "failed"
        save_tracking(issue_id, tracking)


def cmd_run(args: argparse.Namespace) -> None:
    cmd_triage(args)
    print("\n" + "=" * 60)
    print("Triage complete. Now implementing approved issues...\n")
    cmd_implement(args)


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

def main():
    parser = argparse.ArgumentParser(
        description="Linear Auto-Agent: triage, plan, and implement Linear issues via cursor agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_triage = sub.add_parser("triage", help="Fetch and triage open Linear issues")
    p_triage.add_argument("--limit", type=int, default=None, help="Max issues to triage")
    p_triage.set_defaults(func=cmd_triage)

    p_status = sub.add_parser("status", help="Show status of all tracked issues")
    p_status.set_defaults(func=cmd_status)

    p_approve = sub.add_parser("approve", help="Approve issues for implementation")
    p_approve.add_argument("issue_ids", nargs="+", help="Issue identifiers to approve")
    p_approve.set_defaults(func=cmd_approve)

    p_implement = sub.add_parser("implement", help="Implement all approved issues")
    p_implement.set_defaults(func=cmd_implement)

    p_run = sub.add_parser("run", help="Full cycle: triage then implement approved")
    p_run.add_argument("--limit", type=int, default=None, help="Max issues to triage")
    p_run.set_defaults(func=cmd_run)

    p_show = sub.add_parser("show", help="Show full details for an issue")
    p_show.add_argument("issue_id", help="Issue identifier")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
