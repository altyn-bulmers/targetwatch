# TargetWatch

Drug-target watchlist for a biotech VC / pharma BD analyst. FastAPI + one Tailwind page, one container on
Aiven Runtime, Postgres via `DATABASE_URL`.

Flow: add target + indication → snapshot Open Targets (GraphQL) + ClinicalTrials.gov (API v2) → **Analyst**
agent (Claude tool use: `get_evidence`, `get_trials`, `get_previous_snapshot`, `get_past_decisions`) drafts
a verdict (advance / watch / kill) with 3–5 cited claims → a citation check and the **Skeptic** agent remove
every claim not backed by a number in the evidence → Advance / Kill with reason chips → **What changed?**
re-fetches, diffs against the previous snapshot and re-runs both agents. Each step is written to the `runs`
table (`steps`, `latency_ms`, tokens, `claims_removed`) and streamed to the page by polling.

## Environment

| Variable | |
|---|---|
| `DATABASE_URL` | Postgres URI (`postgres://…?sslmode=require` on Aiven). Tables are created at startup. |
| `ANTHROPIC_API_KEY` | Claude API key. |
| `ANTHROPIC_MODEL` | Default `claude-opus-5-5`. |
| `ANALYST_EFFORT` / `SKEPTIC_EFFORT` | Default `medium` / `low`. |
| `PORT` | Default `8080`. |

## Deploy on Aiven Runtime

Runtime → Deploy application → this repo, Dockerfile `Dockerfile`, port 8080 (HTTP). Connect the PostgreSQL
service (or add `DATABASE_URL` as a secret) and add `ANTHROPIC_API_KEY` as a secret. A push does not
redeploy: use "Deploy latest commit".

## Demo

1. Add `CLDN18.2` + `gastric cancer` (the `.2` isoform is kept as a trial-search alias).
2. **Simulated baseline (90 days ago)**: the first snapshot drops trials updated in the last 90 days and is
   labelled SIMULATED (Open Targets values come from today's release; there is no historical API).
3. **Run Analyst + Skeptic**, then **Advance / Kill** with a reason chip.
4. **What changed?**: fresh fetch, diff (new / not-in-baseline / halted trials, count and score shifts),
   and whether the verdict changes.

Link straight to a target with `/#target=<id>`.

## Eval entry point

```python
from app import run_targetwatch

out = run_targetwatch("CLDN18.2", "gastric cancer")  # live fetch + both agents, no database needed
```

`out` holds `verdict`, `headline`, `claims`, `removed_claims`, `claims_removed`, `steps`, token counts and
these six fields:

| Field | Values | Definition |
|---|---|---|
| `active_trials_bin` | `0`, `1-5`, `6-20`, `>20` | trials RECRUITING / NOT_YET_RECRUITING / ACTIVE_NOT_RECRUITING / ENROLLING_BY_INVITATION |
| `highest_phase` | `none`, `NA`, `EARLY_PHASE1` … `PHASE4` | highest phase among matched trials |
| `recent_termination` | `y` / `n` | a matched trial TERMINATED / WITHDRAWN / SUSPENDED with last update ≤ 365 days ago |
| `ot_association` | `strong` / `moderate` / `weak` | Open Targets overall (indirect) score ≥ 0.5 / ≥ 0.2 / below |
| `genetic_support` | `y` / `n` | Open Targets genetic_association, genetic_literature or somatic_mutation score > 0 |
| `antibody_tractable` | `y` / `n` | any Open Targets antibody tractability bucket is true |

Smoke-test the fetchers without Claude: `python -m app.fetch CLDN18.2 "gastric cancer" [days_ago]`.
