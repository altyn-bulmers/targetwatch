import asyncio
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import agents, db, fetch

STATIC = Path(__file__).parent / "static"
KEY_FACTS = ["ot.assoc.overall", "ot.assoc.genetic_association", "ot.assoc.clinical", "ot.tract.AB",
             "ot.safety.count", "ot.drugs.max_stage_indication", "depmap.frac_below_-0.5",
             "ct.total", "ct.active", "ct.phase.PHASE3", "ct.sponsors", "ct.halted_365d"]
jobs: set[asyncio.Task] = set()
db_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_error
    try:
        await db.open_pool()
    except Exception as e:  # keep serving the page and /api/health so the error is visible
        db_error = f"{type(e).__name__}: {e}"
    yield
    await db.close_pool()


app = FastAPI(title="TargetWatch", lifespan=lifespan)


def need_db() -> None:
    if db.pool is None:
        raise HTTPException(503, db_error or "DATABASE_URL is not set")


async def target_row(target_id: int) -> dict:
    need_db()
    row = await db.fetchone("select * from targets where id = %s", target_id)
    if not row:
        raise HTTPException(404, "No such target")
    return row


async def save_snapshot(target_id: int, snap: dict) -> int:
    row = await db.fetchone(
        "insert into snapshots (target_id, as_of, simulated, data) values (%s, %s, %s, %s) returning id",
        target_id, date.fromisoformat(snap["as_of"]), snap["simulated"], db.j(snap))
    await db.fetch("insert into events (target_id, kind, payload) values (%s, 'snapshot', %s)", target_id,
                   db.j({"snapshot_id": row["id"], "as_of": snap["as_of"], "simulated": snap["simulated"],
                         "trials": len(snap["trials"])}))
    return row["id"]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    if db.pool is None:
        database = db_error or "not configured"
    else:
        try:
            row = await db.fetchone("select version() as v")
            database = "ok: " + row["v"].split(" on ")[0]
        except Exception as e:
            database = f"error: {e}"
    return {"status": "ok", "database": database, "model": agents.MODEL,
            "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.get("/api/targets")
async def list_targets():
    need_db()
    return await db.fetch("""
        select t.id, t.symbol, t.approved_symbol, t.disease_name,
               (select output->>'verdict' from runs r where r.target_id = t.id and r.status = 'done'
                order by r.id desc limit 1) as verdict,
               (select count(*) from snapshots s where s.target_id = t.id) as snapshots
        from targets t order by t.id desc""")


class NewTarget(BaseModel):
    symbol: str
    indication: str


@app.post("/api/targets")
async def add_target(body: NewTarget):
    need_db()
    try:
        t = await fetch.resolve(body.symbol, body.indication)
    except LookupError as e:
        raise HTTPException(422, str(e))
    row = await db.fetchone("""
        insert into targets (symbol, indication, ensembl_id, approved_symbol, approved_name, disease_id,
                             disease_name, aliases)
        values (%s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (ensembl_id, disease_id) do update
          set symbol = excluded.symbol, indication = excluded.indication, aliases = excluded.aliases
        returning *""", t["symbol"], t["indication"], t["ensembl_id"], t["approved_symbol"],
        t["approved_name"], t["disease_id"], t["disease_name"], db.j(t["aliases"]))
    await db.fetch("insert into events (target_id, kind, payload) values (%s, 'target_added', %s)",
                   row["id"], db.j(t))
    return row


@app.get("/api/targets/{target_id}")
async def get_target(target_id: int):
    target = await target_row(target_id)
    snapshots = await db.fetch("select id, as_of, simulated, created_at from snapshots where target_id = %s "
                               "order by id desc", target_id)
    latest = await db.fetchone("select data from snapshots where target_id = %s order by id desc limit 1", target_id)
    runs = await db.fetch("select id, kind, status, created_at, output->>'verdict' as verdict from runs "
                          "where target_id = %s order by id desc limit 10", target_id)
    decisions = await db.fetch("select id, decision, reasons, note, agent_verdict, created_at from decisions "
                               "where target_id = %s order by id desc", target_id)
    snap = latest["data"] if latest else None
    if snap:
        snap = {**snap, "trials": snap["trials"][:25], "trials_total": len(snap["trials"])}
    return {"target": target, "snapshots": snapshots, "snapshot": snap, "runs": runs, "decisions": decisions}


class SnapshotRequest(BaseModel):
    simulated: bool = False


@app.post("/api/targets/{target_id}/snapshots")
async def create_snapshot(target_id: int, body: SnapshotRequest):
    target = await target_row(target_id)
    if body.simulated:
        n = await db.fetchone("select count(*) as n from snapshots where target_id = %s", target_id)
        if n["n"]:
            raise HTTPException(409, "A simulated baseline can only be the first snapshot")
    try:
        snap = await fetch.build_snapshot(target, 90 if body.simulated else 0)
    except Exception as e:
        raise HTTPException(502, f"Fetch failed: {type(e).__name__}: {e}")
    snapshot_id = await save_snapshot(target_id, snap)
    return {"id": snapshot_id, "as_of": snap["as_of"], "simulated": snap["simulated"], "trials": len(snap["trials"])}


class RunRequest(BaseModel):
    kind: str = "memo"  # memo | whats_changed


@app.post("/api/targets/{target_id}/runs")
async def start_run(target_id: int, body: RunRequest):
    await target_row(target_id)
    if body.kind not in ("memo", "whats_changed"):
        raise HTTPException(422, "kind must be memo or whats_changed")
    n = await db.fetchone("select count(*) as n from snapshots where target_id = %s", target_id)
    if not n["n"]:
        raise HTTPException(409, "Take a snapshot first")
    running = await db.fetchone("select id from runs where target_id = %s and status = 'running'", target_id)
    if running:
        return {"id": running["id"]}
    row = await db.fetchone("insert into runs (target_id, kind) values (%s, %s) returning id", target_id, body.kind)
    task = asyncio.create_task(run_job(row["id"], target_id, body.kind))
    jobs.add(task)
    task.add_done_callback(jobs.discard)
    return {"id": row["id"]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: int):
    need_db()
    row = await db.fetchone("select * from runs where id = %s", run_id)
    if not row:
        raise HTTPException(404, "No such run")
    return row


async def run_job(run_id: int, target_id: int, kind: str) -> None:
    t0 = time.monotonic()

    def elapsed() -> int:
        return int((time.monotonic() - t0) * 1000)

    async def log(step: dict) -> None:
        step = {"t_ms": elapsed(), **step}
        await db.fetch("update runs set steps = steps || %s, input_tokens = input_tokens + %s, "
                       "output_tokens = output_tokens + %s, updated_at = now() where id = %s",
                       db.j([step]), step.get("input_tokens", 0), step.get("output_tokens", 0), run_id)

    try:
        target = await db.fetchone("select * from targets where id = %s", target_id)
        if kind == "whats_changed":
            await log({"agent": "fetcher", "type": "fetch", "msg": "fresh fetch: Open Targets + ClinicalTrials.gov"})
            snap = await fetch.build_snapshot(target)
            snapshot_id = await save_snapshot(target_id, snap)
            await log({"agent": "fetcher", "type": "snapshot",
                       "msg": f"snapshot #{snapshot_id}: {len(snap['trials'])} trials, as of {snap['as_of']}"})
        snaps = await db.fetch("select data from snapshots where target_id = %s order by id desc limit 2", target_id)
        previous_run = await db.fetchone("select output->>'verdict' as verdict from runs where target_id = %s "
                                         "and status = 'done' order by id desc limit 1", target_id)
        decisions = await db.fetch("""
            select d.id, d.decision, d.reasons, d.note, d.agent_verdict, d.key_facts, d.created_at,
                   t.approved_symbol, t.disease_name
            from decisions d join targets t on t.id = d.target_id order by d.id desc limit 20""")
        ctx = agents.RunContext(target, snaps[0]["data"], snaps[1]["data"] if len(snaps) > 1 else None,
                                previous_run["verdict"] if previous_run else None, decisions)
        out = await agents.run_pipeline(ctx, log)
        await db.fetch("update runs set status = 'done', output = %s, claims_removed = %s, latency_ms = %s, "
                       "updated_at = now() where id = %s", db.j(out), out["claims_removed"], elapsed(), run_id)
    except Exception as e:
        await db.fetch("update runs set status = 'error', error = %s, latency_ms = %s, updated_at = now() "
                       "where id = %s", f"{type(e).__name__}: {e}", elapsed(), run_id)


class DecisionRequest(BaseModel):
    decision: str
    reasons: list[str] = []
    note: str | None = None
    run_id: int | None = None


@app.post("/api/targets/{target_id}/decisions")
async def decide(target_id: int, body: DecisionRequest):
    await target_row(target_id)
    if body.decision not in ("advance", "kill"):
        raise HTTPException(422, "decision must be advance or kill")
    if body.decision == "kill" and not body.reasons:
        raise HTTPException(422, "Pick at least one reason to kill")
    snap = await db.fetchone("select id, data->'facts' as facts from snapshots where target_id = %s "
                             "order by id desc limit 1", target_id)
    run = (await db.fetchone("select output->>'verdict' as verdict from runs where id = %s", body.run_id)
           if body.run_id else None)
    facts = (snap or {}).get("facts") or {}
    row = await db.fetchone("""
        insert into decisions (target_id, snapshot_id, run_id, decision, reasons, note, agent_verdict, key_facts)
        values (%s, %s, %s, %s, %s, %s, %s, %s) returning *""",
        target_id, snap["id"] if snap else None, body.run_id, body.decision, body.reasons, body.note,
        run["verdict"] if run else None, db.j({k: facts[k] for k in KEY_FACTS if k in facts}))
    await db.fetch("insert into events (target_id, kind, payload) values (%s, 'decision', %s)", target_id,
                   db.j({"decision_id": row["id"], "decision": body.decision, "reasons": body.reasons}))
    return row
