# Linear Auto-Agent

Automated triage and implementation of Linear issues using Cursor Agent CLI.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Authenticate cursor agent: `cursor agent login`
3. Copy `.env.example` to `.env` and fill in:
   - `LINEAR_API_KEY` — personal API key from https://linear.app/settings/account/security
   - `BACKEND_REPO` — absolute path to the backend repo (e.g. `/Users/you/git/scout`)
   - `FRONTEND_REPO` — absolute path to the frontend repo (e.g. `/Users/you/git/galaxy`)

## Usage

```bash
# Triage open issues (fetches from Linear, runs read-only agent for each)
./run.sh triage
./run.sh triage --limit 5    # only process 5 issues

# Review what the agent found
./run.sh status

# Show full details for a specific issue
./run.sh show ENG-123

# Approve issues for implementation
./run.sh approve ENG-123 ENG-456

# Implement all approved issues (creates git worktrees, runs coding agent)
./run.sh implement

# Full cycle: triage then implement previously-approved issues
./run.sh run --limit 3
```

## How it works

### Two-repo model

The agent works across two repos simultaneously:
- **scout** (Java backend) — Gradle, Conjure APIs, Postgres, ClickHouse, Temporal
- **galaxy** (TypeScript frontend) — React

For read-only steps (triage + planning), a symlinked workspace at `data/workspace/` gives the agent
visibility into both repos in a single call. For implementation, separate git worktrees are created
per affected repo for isolation.

If an issue requires changes to both repos, the agent creates two worktrees and two separate PRs.
The frontend PR may have compilation failures until the backend PR is merged and published — this
is expected and noted in the tracking file.

### Security model

| Step       | Agent mode                    | Permissions                        |
|------------|-------------------------------|------------------------------------|
| Fetch      | Python (no agent)             | Linear API read-only               |
| Triage     | `cursor agent --mode ask`     | Read-only, no MCP, no edits        |
| Plan       | `cursor agent --mode plan`    | Read-only, no MCP, no edits        |
| Implement  | `cursor agent --yolo`         | Edit + shell, scoped to worktree, no MCP |

### Workflow

1. **Triage** — fetches open Linear issues, asks a read-only agent (with visibility into both repos) whether each is well-defined and easy or complex, and which repos are affected
2. **Plan** — for easy issues, a read-only agent produces a concrete implementation plan with separate sections per repo
3. **Review** — you review plans via `status`/`show` and `approve` the ones you want implemented
4. **Implement** — a coding agent implements approved issues on isolated git worktrees (no push)
5. **Review** — you inspect worktrees and push/PR manually

### Data layout

```
data/
├── workspace/         # Symlinks to repos (auto-created)
│   ├── scout -> /path/to/scout
│   └── galaxy -> /path/to/galaxy
├── tracking/          # Per-issue JSON decision records
│   └── ENG-123.json
├── worktrees/         # Git worktrees for implementation
│   ├── ENG-123-backend/
│   └── ENG-123-frontend/
└── logs/              # Agent output logs
    ├── ENG-123-backend.log
    └── ENG-123-frontend.log
```

## Scheduling

Add to crontab for periodic triage:

```bash
0 */6 * * * cd /path/to/linear_auto_agent && ./run.sh triage --limit 10 >> data/logs/cron.log 2>&1
```
