"""Analyst (tool use) + Skeptic (claim filter) on Claude, and run_targetwatch() for eval/."""
import asyncio
import json
import os
import time

import anthropic

from app import fetch

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
ANALYST_EFFORT = os.environ.get("ANALYST_EFFORT", "medium")
SKEPTIC_EFFORT = os.environ.get("SKEPTIC_EFFORT", "low")
MAX_TURNS = 8

FIELDS = {
    "active_trials_bin": ["0", "1-5", "6-20", ">20"],
    "highest_phase": ["none", "NA", "EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"],
    "recent_termination": ["y", "n"],
    "ot_association": ["strong", "moderate", "weak"],
    "genetic_support": ["y", "n"],
    "antibody_tractable": ["y", "n"],
}
FIELD_RULES = """Field definitions (derive each one from the facts exactly):
- active_trials_bin: from ct.active: "0", "1-5", "6-20" or ">20".
- highest_phase: ct.highest_phase.
- recent_termination: "y" if ct.halted_365d > 0, else "n".
- ot_association: from ot.assoc.overall: "strong" if >= 0.5, "moderate" if >= 0.2, else "weak".
- genetic_support: "y" if ot.assoc.genetic_association, ot.assoc.genetic_literature or ot.assoc.somatic_mutation is > 0, else "n".
- antibody_tractable: "y" if ot.tract.AB > 0, else "n"."""

ANALYST_SYSTEM = f"""You are the Analyst in TargetWatch, a diligence tool for a biotech VC / pharma BD analyst. You assess one drug target in one indication and write a short memo.

Start by calling all four tools; they are cheap, so call them together. Then answer.

verdict: "advance" = worth active diligence now; "watch" = keep tracking, not compelling yet; "kill" = stop spending time on it.

claims: 3 to 5. Each is one or two sentences on one of: genetics, clinical validation, competition / crowding, safety, tractability, essentiality, momentum, precedent. Every claim cites at least one fact id with its value copied exactly as the tool returned it, and every number in the claim text must be one of the cited values. Never use numbers from memory. An Open Targets datatype score of 0 means no evidence of that type. severity: red = argues against, amber = mixed or uncertain, green = supports.

precedents: if get_past_decisions returns decisions, summarise the analyst's revealed criteria in one or two sentences and say how they apply here (cite dec. facts in a claim when a precedent drives your verdict). Otherwise "none".

verdict_change: if get_previous_snapshot returns found=true, give the previous verdict, whether yours differs, and which diff. facts drove that; say so if the previous snapshot was simulated. Otherwise "".

{FIELD_RULES}"""

SKEPTIC_SYSTEM = f"""You are the Skeptic in TargetWatch. You receive an Analyst's draft memo and the complete evidence the Analyst saw, as fact id -> value.

Remove every claim that is not backed by a number in the evidence: remove it if any number, count, score, phase or date in its text is not in the evidence, if its cited facts do not support what the text says, or if it leans on outside knowledge. Keep claims that are fully supported. Refer to claims by their 0-based index in draft.claims.

Then check the six fields against these definitions and the evidence, and correct any that are wrong.

{FIELD_RULES}

Return only removals and corrections; empty lists mean everything checks out."""


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


CITATION = _obj({"fact": {"type": "string"}, "value": {"type": "string"}})
ANALYST_SCHEMA = _obj({
    "verdict": {"type": "string", "enum": ["advance", "watch", "kill"]},
    "headline": {"type": "string"},
    "claims": {"type": "array", "items": _obj({
        "text": {"type": "string"},
        "severity": {"type": "string", "enum": ["red", "amber", "green"]},
        "citations": {"type": "array", "items": CITATION}})},
    **{name: {"type": "string", "enum": values} for name, values in FIELDS.items()},
    "precedents": {"type": "string"},
    "verdict_change": {"type": "string"},
})
SKEPTIC_SCHEMA = _obj({
    "removed": {"type": "array", "items": _obj({"index": {"type": "integer"}, "reason": {"type": "string"}})},
    "field_corrections": {"type": "array", "items": _obj({
        "field": {"type": "string", "enum": list(FIELDS)},
        "value": {"type": "string"},
        "reason": {"type": "string"}})},
})


def _tool(name: str, description: str) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}}


TOOLS = [
    _tool("get_evidence", "Open Targets evidence for this target in this indication from the latest snapshot: "
          "association scores (overall = indirect, direct, and by evidence datatype), tractability bucket counts by "
          "modality, safety liabilities, DepMap essentiality and drugs against the target. Returns facts (id -> value) to cite."),
    _tool("get_trials", "ClinicalTrials.gov trials for this target in this indication from the latest snapshot: counts "
          "by status and phase, sponsors, recent terminations, and one fact per trial (trial.<NCT> = PHASE|STATUS|last update)."),
    _tool("get_previous_snapshot", "The snapshot before the latest one: its facts (prefixed prev.), whether it was "
          "simulated, the previous run's verdict, and the computed diff (diff. facts: new, halted and changed trials, "
          "score shifts). found=false if there is none."),
    _tool("get_past_decisions", "The analyst's recorded Advance/Kill decisions across all tracked targets, newest first, "
          "with reason chips (safety, crowded, weak genetics), notes, the agent verdict at the time and key facts at "
          "decision time (dec.<id>. facts)."),
]


class RunContext:
    """What the Analyst's tools can see: the latest snapshot, the one before it, and past decisions."""

    def __init__(self, target, snapshot, previous=None, previous_verdict=None, decisions=()):
        self.target, self.snapshot, self.previous = target, snapshot, previous
        self.previous_verdict = previous_verdict
        self.decisions = list(decisions)
        self.diff = fetch.diff(previous, snapshot) if previous else None

    def brief(self) -> str:
        t, s = self.target, self.snapshot
        simulated = " (SIMULATED baseline: trials updated after this date were removed)" if s["simulated"] else ""
        return (f"Target: {t['approved_symbol']} ({t['ensembl_id']}, {s['ot']['approved_name']}). "
                f"Indication: {t['disease_name']} ({t['disease_id']}); the analyst typed '{t['indication']}'. "
                f"Latest snapshot as of {s['as_of']}{simulated}; Open Targets release {s['ot']['release']}. "
                "Write the memo.")

    def tool(self, name: str) -> dict:
        s, facts = self.snapshot, self.snapshot["facts"]
        meta = {"as_of": s["as_of"], "simulated": s["simulated"], "ot_release": s["ot"]["release"]}
        if name == "get_evidence":
            ot = s["ot"]
            return {"snapshot": meta,
                    "facts": {k: v for k, v in facts.items() if k.startswith(("ot.", "depmap."))},
                    "details": {"tractable_buckets": ot["tractability"], "safety_liabilities": ot["safety"][:15],
                                "drugs": ot["drugs"][:10],
                                "depmap_indication_keywords": ot["depmap"]["indication_keywords"]}}
        if name == "get_trials":
            trials = s["trials"]
            return {"snapshot": meta, "query": s["ct_query"],
                    "facts": {k: v for k, v in facts.items() if k.startswith(("ct.", "trial."))},
                    "details": {"halted": [t for t in trials if t["status"] in fetch.HALTED][:15],
                                "most_recently_updated": [
                                    {k: t[k] for k in ("nct", "title", "phase", "status", "last_update",
                                                       "sponsor", "interventions")} for t in trials[:15]]}}
        if name == "get_previous_snapshot":
            if not self.previous:
                return {"found": False}
            p, d = self.previous, self.diff
            return {"found": True, "as_of": p["as_of"], "simulated": p["simulated"],
                    "previous_verdict": self.previous_verdict or "none",
                    "diff": {k: d[k] for k in ("new", "not_in_baseline", "halted", "status_changed", "dropped", "shifts")},
                    "facts": {f"prev.{k}": v for k, v in p["facts"].items() if not k.startswith("trial.")} | d["facts"]}
        if name == "get_past_decisions":
            dec_facts = {}
            for dec in self.decisions:
                dec_facts[f"dec.{dec['id']}.decision"] = dec["decision"]
                dec_facts |= {f"dec.{dec['id']}.{k}": v for k, v in (dec["key_facts"] or {}).items()}
            return {"decisions": [{"id": dec["id"], "target": dec["approved_symbol"], "indication": dec["disease_name"],
                                   "decision": dec["decision"], "reasons": dec["reasons"], "note": dec["note"],
                                   "agent_verdict_at_the_time": dec["agent_verdict"],
                                   "date": str(dec["created_at"])[:10]} for dec in self.decisions],
                    "facts": dec_facts}
        return {"error": f"unknown tool {name}"}


async def _call(client, step, agent: str, **kwargs):
    t0 = time.monotonic()
    resp = await client.beta.messages.create(model=MODEL, max_tokens=16000,
                                             betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    u = resp.usage
    await step({"agent": agent, "type": "llm", "latency_ms": int((time.monotonic() - t0) * 1000),
                "input_tokens": u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
                "output_tokens": u.output_tokens, "stop_reason": resp.stop_reason, "model": resp.model})
    if resp.stop_reason == "refusal":
        category = resp.stop_details.category if resp.stop_details else None
        raise RuntimeError(f"{agent}: the model declined the request (refusal, category {category})")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"{agent}: response hit max_tokens")
    return resp


def _json_text(resp, agent: str) -> dict:
    text = "".join(b.text for b in resp.content if b.type == "text")
    if not text:
        raise RuntimeError(f"{agent}: response had no JSON text")
    return json.loads(text)


async def _analyst(client, ctx: RunContext, step) -> tuple[dict, dict]:
    messages = [{"role": "user", "content": ctx.brief()}]
    evidence = {}
    for _ in range(MAX_TURNS):
        resp = await _call(client, step, "analyst", system=ANALYST_SYSTEM, tools=TOOLS, messages=messages,
                           output_config={"effort": ANALYST_EFFORT,
                                          "format": {"type": "json_schema", "schema": ANALYST_SCHEMA}})
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            return _json_text(resp, "analyst"), evidence
        results = []
        for block in uses:
            t0 = time.monotonic()
            out = ctx.tool(block.name)
            evidence.update(out.get("facts", {}))
            await step({"agent": "analyst", "type": "tool", "name": block.name, "facts": len(out.get("facts", {})),
                        "latency_ms": int((time.monotonic() - t0) * 1000)})
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"analyst: no memo after {MAX_TURNS} turns")


_MISSING = object()


def _matches(actual, cited: str) -> bool:
    if actual is _MISSING:
        return False
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        try:
            return abs(float(cited) - actual) <= 0.005
        except ValueError:
            return False
    return str(actual).strip().lower() == str(cited).strip().lower()


def citation_check(claims: list[dict], evidence: dict) -> tuple[list[dict], list[dict]]:
    """Drop claims that cite nothing, or cite a fact id/value no tool returned."""
    kept, removed = [], []
    for claim in claims:
        bad = next((c for c in claim["citations"]
                    if not _matches(evidence.get(c["fact"], _MISSING), c["value"])), None)
        if not claim["citations"]:
            removed.append({**claim, "removed_by": "citation check", "reason": "no citation"})
        elif bad:
            removed.append({**claim, "removed_by": "citation check",
                            "reason": f"cites {bad['fact']} = {bad['value']}; "
                                      f"evidence has {evidence.get(bad['fact'], 'no such fact')}"})
        else:
            kept.append(claim)
    return kept, removed


async def _skeptic(client, draft: dict, evidence: dict, step) -> dict:
    resp = await _call(client, step, "skeptic", system=SKEPTIC_SYSTEM,
                       messages=[{"role": "user", "content": json.dumps({"draft": draft, "evidence": evidence}, default=str)}],
                       output_config={"effort": SKEPTIC_EFFORT,
                                      "format": {"type": "json_schema", "schema": SKEPTIC_SCHEMA}})
    return _json_text(resp, "skeptic")


async def run_pipeline(ctx: RunContext, log) -> dict:
    """Analyst drafts with tools, code drops uncitable claims, Skeptic removes unsupported ones. `log` gets each step."""
    t0 = time.monotonic()
    usage = {"input_tokens": 0, "output_tokens": 0}

    async def step(s: dict) -> None:
        usage["input_tokens"] += s.get("input_tokens", 0)
        usage["output_tokens"] += s.get("output_tokens", 0)
        await log(s)

    async with anthropic.AsyncAnthropic() as client:
        draft, evidence = await _analyst(client, ctx, step)
        await step({"agent": "analyst", "type": "draft", "verdict": draft["verdict"], "claims": len(draft["claims"])})
        kept, removed = citation_check(draft["claims"], evidence)
        for claim in removed:
            await step({"agent": "skeptic", "type": "removed", "by": claim["removed_by"],
                        "claim": claim["text"], "reason": claim["reason"]})
        review = await _skeptic(client, {**draft, "claims": kept}, evidence, step)

    drop = {r["index"]: r["reason"] for r in review["removed"] if 0 <= r["index"] < len(kept)}
    claims = []
    for i, claim in enumerate(kept):
        if i in drop:
            removed.append({**claim, "removed_by": "skeptic", "reason": drop[i]})
            await step({"agent": "skeptic", "type": "removed", "by": "skeptic", "claim": claim["text"], "reason": drop[i]})
        else:
            claims.append(claim)
    fields = {name: draft[name] for name in FIELDS}
    for fix in review["field_corrections"]:
        if fix["value"] in FIELDS[fix["field"]] and fields[fix["field"]] != fix["value"]:
            await step({"agent": "skeptic", "type": "field_corrected", "field": fix["field"],
                        "from": fields[fix["field"]], "to": fix["value"], "reason": fix["reason"]})
            fields[fix["field"]] = fix["value"]
    await step({"agent": "skeptic", "type": "done", "kept": len(claims), "removed": len(removed)})

    t, s = ctx.target, ctx.snapshot
    return {"target": t["approved_symbol"], "indication": t["disease_name"], "disease_id": t["disease_id"],
            **fields,
            "verdict": draft["verdict"], "headline": draft["headline"],
            "claims": claims, "removed_claims": removed, "claims_removed": len(removed),
            "precedents": draft["precedents"], "verdict_change": draft["verdict_change"],
            "snapshot": {"as_of": s["as_of"], "simulated": s["simulated"], "ot_release": s["ot"]["release"]},
            "model": MODEL, "latency_ms": int((time.monotonic() - t0) * 1000), **usage}


async def run_targetwatch_async(target: str, indication: str) -> dict:
    resolved = await fetch.resolve(target, indication)
    snapshot = await fetch.build_snapshot(resolved)
    steps = []

    async def log(s: dict) -> None:
        steps.append(s)

    out = await run_pipeline(RunContext(resolved, snapshot), log)
    return {**out, "steps": steps}


def run_targetwatch(target: str, indication: str) -> dict:
    """Live fetch + Analyst + Skeptic for one target/indication; no database needed. Returns the final JSON."""
    return asyncio.run(run_targetwatch_async(target, indication))
