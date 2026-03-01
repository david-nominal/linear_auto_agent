# Spiky

Automated triage and implementation of Linear issues using Cursor Agent CLI.

## Why Spiky?

Existing AI coding tools (Cursor, Devin, Greptile) each handle a slice of the engineering workflow but nothing connects them end-to-end:

- **No end-to-end loop.** Today the cycle is sometimes: tell Cursor/Devin to implement → push → Greptile leaves review comments → copy-paste comments back to Cursor → repeat. That feedback loop is entirely manual.
- **No structured mass-tasking.** Cursor and Devin are on-demand for individual issues. They require lots of hand-holding from the person who triggered them (who might not even be the best suited person for the job like MissionOps) — no self-review, no triage.
- **Nothing cross-repo.** AI can reason across backend + frontend, but current tools are scoped to a single repo. A split codebase shouldn't slow things down.

Spiky fills these gaps with a **full, more deterministic pipeline**: Linear ticket → triage (easy / hard / needs clarification) → plan → implement → self-review + external review (Greptile, Devin, humans) → PR ready.
Triage + Planning is done against both the BE and FE repo. A single ticket can produce coordinated BE + FE PRs where the frontend is validated against the actual backend types, not stale ones. This works by linking the FE locally to the new BE typings.

The bet is that AI isn't ready for fully autonomous long-horizon agents — structured, observable, stage-based orchestration works better today. Forcing a compilation, self-review, CI, and lint check is better than just instructing the agent to do it. Each stage (triage, plan, implement, revise) is a distinct checkpoint you can inspect, approve, or reject — not one long agent session that may drift. Every decision and log is visible via a web dashboard. 

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Authenticate cursor agent: `cursor agent login`
3. Copy `.env.example` to `.env` and fill in keys.

Optional model overrides (uses cursor agent's default model if unset):
   - `TRIAGE_MODEL` — model for triage (e.g. `gpt-4o-mini` for speed/cost)
   - `PLAN_MODEL` — model for planning
   - `IMPL_MODEL` — model for implementation and revision

## Usage

```bash
# Triage open issues (fetches from Linear, runs read-only agent for each)
./run.sh triage
./run.sh triage --limit 5              # only process 5 issues
./run.sh triage --limit 10 --workers 5 # triage 10 issues, 5 in parallel

# Review what the agent found
./run.sh status

# Show full details for a specific issue
./run.sh show ENG-123

# Approve issues for implementation
./run.sh approve ENG-123 ENG-456

# Implement all approved issues (creates git worktrees, runs coding agent)
./run.sh implement
./run.sh implement --workers 2         # implement 2 issues in parallel

# Full cycle: triage then implement previously-approved issues
./run.sh run --limit 3

# Push implemented issues and create PRs
./run.sh push --all
./run.sh push ENG-123

# Address PR review feedback (fetches unresolved comments, runs agent, pushes)
./run.sh revise                        # all pushed PRs with unresolved comments
./run.sh revise ENG-123                # specific issue only
```

### Web UI

```bash
uv run server.py
# Open http://localhost:8111
```

## How it works

### Two-repo model

The agent works across two repos simultaneously:
- **scout** (Java backend) — Gradle, Conjure APIs, Postgres, ClickHouse, Temporal
- **galaxy** (TypeScript frontend) — React

Repos are cloned into `data/workspace/` automatically on first run. For read-only steps (triage +
planning), this workspace gives the agent visibility into both repos in a single call. For
implementation, separate git worktrees are created per affected repo for isolation.

If an issue requires changes to both repos, the backend is implemented first. Then the scout
Conjure TypeScript and proto zip are built locally and linked into the galaxy worktree via
`pnpm scout:link`, so the frontend agent can compile against the new backend types.

### Security model

| Step           | Agent mode                    | Timeout | Permissions                        |
|----------------|-------------------------------|---------|-------------------------------------|
| Fetch          | Python (no agent)             | —       | Linear API read-only               |
| Triage         | `cursor agent --mode ask`     | 60s     | Read-only, no MCP, no edits        |
| Plan           | `cursor agent --mode plan`    | 120s    | Read-only, no MCP, no edits        |
| Implement      | `cursor agent --yolo`         | 600s    | Edit + shell, scoped to worktree, no MCP |
| Compile + Lint | Python (no agent)             | 300s    | Shell only, scoped to worktree     |
| Revise         | `cursor agent --yolo`         | 600s    | Edit + shell, scoped to worktree, no MCP |

### Workflow

1. **Triage** — fetches open Linear issues, classifies each as easy/medium/complex and which repos are affected (parallelized)
2. **Plan** — for easy and medium issues, produces a concrete implementation plan with a short summary for PR descriptions
3. **Review** — you review plans via `status`/`show` and `approve` the ones you want implemented
4. **Implement** — implements approved issues on isolated git worktrees; for cross-repo issues, builds and links backend APIs locally before running the frontend agent (parallelized across issues)
5. **Compile + Lint** — after each implementation or revision, runs `gradlew compileJava` + `spotlessCheck` (backend) or `pnpm build` + `pnpm lint` (frontend). If it fails, the agent is re-invoked with the error output to fix it (up to 2 retries)
6. **Push** — pushes branches and creates PRs with concise descriptions
7. **Revise** — fetches unresolved PR review comments (from humans, Devin, Greptile, etc.), runs an agent to address them, runs compile+lint again, and pushes the fixes

### Data layout

```
data/
├── workspace/         # Cloned repos (auto-created)
│   ├── scout/
│   └── galaxy/
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

## Future Extensions

**Broader pipeline coverage:**
- Expand the definition of E2E to the left — auto-create Linear tickets from Slack threads, Notion docs, or codebase crawling
- Take over existing PRs (not just ones Spiky created)
- Respond to PR review comments conversationally, not just fix them
- Run user-initated and not just mass from linear

**Smarter agent behavior:**
- Push back against review comments when they're wrong or out of scope
- Self-cancel tasks that become too complex mid-implementation
- Self-review that flags excessive complexity before requesting human review

**Integrations:**
- Auto-tag the correct reviewer based on code ownership
- Post screenshots/videos of changes and use them as part of review
- Slack notifications for pipeline progress and review requests
- Run 24/7 on AWS as a persistent service
