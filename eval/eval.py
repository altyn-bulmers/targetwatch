# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic==2.1.0", "httpx2==2.16.0", "matplotlib==3.11.2"]
# ///
"""TargetWatch eval: six dossier fields for 15 target/indication pairs, from TargetWatch (app.run_targetwatch) and
from the same Claude model answering without tools, scored against ground truth from this file's own Open Targets
and ClinicalTrials.gov fetchers (no app code).

    uv run eval/eval.py                # full run, needs ANTHROPIC_API_KEY
    uv run eval/eval.py --truth-only   # fetch (or read) and print the ground truth, no model calls
    uv run eval/eval.py --limit 2      # pilot on the first two pairs

Writes eval/cache.json (ground truth), eval/results.json (every dossier, with each baseline prompt as sent) and
eval/eval.png.
"""
import argparse
import asyncio
import inspect
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import anthropic
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE, RESULTS, FIGURE = HERE / "cache.json", HERE / "results.json", HERE / "eval.png"
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
CT_URL = "https://clinicaltrials.gov/api/v2/studies"
OFFICIAL_API = "https://api.anthropic.com"

# Same variables and defaults as app/agents.py, so the baseline runs the Analyst's model and effort.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("ANALYST_EFFORT", "medium")
PRICES = {"claude-opus-5-5": {"input_tokens": 4.0, "output_tokens": 20.0,  # $ per million tokens
                              "cache_creation_input_tokens": 5.0, "cache_read_input_tokens": 0.20}}
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
IN_FLIGHT = 4
TIMEOUT_S = 900  # per TargetWatch dossier

# (target, indication as an analyst types it, Open Targets disease id). The disease ids are pinned because the
# search runners-up are wrong for some indications (AML -> heart failure).
PAIRS = [
    ("CLDN18", "gastric cancer", "MONDO_0001056"),
    ("ERBB2", "breast cancer", "MONDO_0007254"),
    ("EGFR", "NSCLC", "MONDO_0005233"),
    ("KRAS", "NSCLC", "MONDO_0005233"),
    ("BRAF", "melanoma", "MONDO_0005105"),
    ("TACSTD2", "breast cancer", "MONDO_0007254"),
    ("IDO1", "melanoma", "MONDO_0005105"),
    ("TIGIT", "NSCLC", "MONDO_0005233"),
    ("CD47", "AML", "MONDO_0018874"),
    ("FGFR2", "cholangiocarcinoma", "MONDO_0019087"),
    ("NECTIN4", "urothelial carcinoma", "MONDO_0040679"),
    ("DLL3", "small cell lung cancer", "MONDO_0008433"),
    ("LAG3", "melanoma", "MONDO_0005105"),
    ("GPC3", "hepatocellular carcinoma", "MONDO_0007256"),
    ("CDH17", "colorectal cancer", "MONDO_0005575"),
]
# Names trial records use for each target besides its symbol.
ALIASES = {
    "CLDN18": ["CLDN18.2", "claudin 18", "claudin 18.2", "claudin18.2"],
    "ERBB2": ["HER2", "HER-2"],
    "EGFR": [],
    "KRAS": ["K-RAS"],
    "BRAF": ["B-RAF"],
    "TACSTD2": ["TROP2", "TROP-2"],
    "IDO1": ["IDO-1", "IDO"],
    "TIGIT": [],
    "CD47": [],
    "FGFR2": [],
    "NECTIN4": ["Nectin-4", "PVRL4"],
    "DLL3": ["delta-like ligand 3"],
    "LAG3": ["LAG-3"],
    "GPC3": ["glypican-3", "glypican 3"],
    "CDH17": ["cadherin-17", "cadherin 17"],
}

# TargetWatch's field names and values (app/agents.py FIELDS); its "NA" phase reads as "none".
FIELDS = {
    "active_trials_bin": ["0", "1-5", "6-20", ">20"],
    "highest_phase": ["none", "EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"],
    "recent_termination": ["y", "n"],
    "ot_association": ["strong", "moderate", "weak"],
    "genetic_support": ["y", "n"],
    "antibody_tractable": ["y", "n"],
}
ACTIVE = {"RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"}
STOPPED = {"TERMINATED", "WITHDRAWN"}
PHASES = FIELDS["highest_phase"][1:]  # ascending; NA and missing phases are ignored
GENETIC = ("genetic_association", "genetic_literature", "somatic_mutation")

OT_TARGET = """query($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 10}) { hits { id name } } }"""
OT_PAIR = """query($ensg: String!, $efo: String!) {
  meta { dataVersion { year month } }
  disease(efoId: $efo) {
    name
    associatedTargets(Bs: [$ensg], enableIndirect: true) { rows { score datatypeScores { id score } } } }
  target(ensemblId: $ensg) {
    tractability { modality label value }
    drugAndClinicalCandidates { rows { drug { name } } } } }"""
CT_FIELDS = "NCTId,OverallStatus,Phase,CompletionDate,CompletionDateType,LastUpdatePostDate,WhyStopped"


# ---------- ground truth ----------

def http_json(url: str, payload: dict | None = None) -> dict:
    """GET, or POST a JSON payload; three tries on rate limits, server errors and network failures."""
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "User-Agent": "TargetWatch-eval"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == 2:
                raise RuntimeError(f"HTTP {e.code} from {url[:70]}: {e.read()[:400]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt + random.random())


def ot(query: str, **variables) -> dict:
    body = http_json(OT_URL, {"query": query, "variables": variables})
    if body.get("errors"):
        raise RuntimeError("Open Targets: " + "; ".join(e["message"] for e in body["errors"]))
    return body["data"]


def ct_studies(indication: str, term: str) -> list[dict]:
    params = {"query.cond": indication, "query.term": term, "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
              "fields": CT_FIELDS, "pageSize": 1000}
    studies = []
    while True:
        page = http_json(f"{CT_URL}?{urllib.parse.urlencode(params)}")
        studies += [s["protocolSection"] for s in page.get("studies", [])]
        if not page.get("nextPageToken"):
            return studies
        params["pageToken"] = page["nextPageToken"]


def quote(term: str) -> str:
    return '"' + term.replace('"', "") + '"'


def day(value: str) -> date:
    """ClinicalTrials.gov dates are YYYY-MM-DD, YYYY-MM or YYYY."""
    return date.fromisoformat(value + {4: "-01-01", 7: "-01"}.get(len(value), ""))


def fetch_truth(target: str, indication: str, disease_id: str, as_of: date) -> dict:
    """Raw ground-truth values for one pair; truth_labels() turns them into field values."""
    hits = ot(OT_TARGET, q=target)["search"]["hits"]
    ensg = next((h["id"] for h in hits if h["name"].upper() == target), None)
    if ensg is None:
        raise LookupError(f"Open Targets has no target named {target}")
    data = ot(OT_PAIR, ensg=ensg, efo=disease_id)
    disease, tgt = data["disease"], data["target"]
    if disease is None:
        raise LookupError(f"Open Targets has no disease {disease_id} any more; update PAIRS")
    rows = disease["associatedTargets"]["rows"]
    drugs = sorted({r["drug"]["name"] for r in (tgt["drugAndClinicalCandidates"] or {}).get("rows") or []})

    # A trial counts when its title, conditions or interventions name the target, or one of its interventions is a
    # drug against the target. Mentions elsewhere (eligibility, outcomes) are mostly biomarkers.
    names = " OR ".join(quote(n) for n in [target, *ALIASES[target]])
    interventions = " OR ".join([names, *map(quote, drugs)])
    studies = ct_studies(indication, f"AREA[TitleSearch]({names}) OR AREA[ConditionSearch]({names}) "
                                     f"OR AREA[InterventionSearch]({interventions})")

    since = as_of - timedelta(days=365)
    top, halted = None, []
    for s in studies:
        nct, status = s["identificationModule"]["nctId"], s["statusModule"]
        phases = [p for p in s.get("designModule", {}).get("phases", []) if p in PHASES]
        if phases and (top is None or max(map(PHASES.index, phases)) > PHASES.index(top[0])):
            top = (max(phases, key=PHASES.index), nct)
        if status["overallStatus"] in STOPPED:
            # When it stopped: the actual completion date; the last update is often a later results posting.
            done = status.get("completionDateStruct") or {}
            stopped = done.get("date") if done.get("type") == "ACTUAL" else None
            stopped = stopped or (status.get("lastUpdatePostDateStruct") or {}).get("date")
            if stopped and day(stopped) >= since:
                halted.append({"nct": nct, "status": status["overallStatus"], "stopped": stopped,
                               "why": (status.get("whyStopped") or "")[:200]})

    version = data["meta"]["dataVersion"]
    return {"ensembl_id": ensg, "disease": {"id": disease_id, "name": disease["name"]},
            "ot_release": f"{version['year']}.{version['month']}",
            "ot_score": rows[0]["score"] if rows else 0.0,
            "datatypes": {x["id"]: x["score"] for x in rows[0]["datatypeScores"]} if rows else {},
            "ab_buckets": [x["label"] for x in tgt["tractability"] or [] if x["modality"] == "AB" and x["value"]],
            "drugs": drugs, "trials_matched": len(studies),
            "active": sum(s["statusModule"]["overallStatus"] in ACTIVE for s in studies),
            "highest_phase": top[0] if top else "none", "highest_phase_nct": top[1] if top else None,
            "halted_365d": halted}


def truth_labels(raw: dict) -> dict:
    n, score = raw["active"], raw["ot_score"]
    return {"active_trials_bin": "0" if n == 0 else "1-5" if n <= 5 else "6-20" if n <= 20 else ">20",
            "highest_phase": raw["highest_phase"],
            "recent_termination": "y" if raw["halted_365d"] else "n",
            "ot_association": "strong" if score >= 0.5 else "moderate" if score >= 0.2 else "weak",
            "genetic_support": "y" if any(raw["datatypes"].get(k, 0) > 0 for k in GENETIC) else "n",
            "antibody_tractable": "y" if raw["ab_buckets"] else "n"}


def near_cutoff(raw: dict) -> set[str]:
    """Fields whose truth sits within 2 trials or 0.03 of a cut-off, where a day's drift can flip the label."""
    near = set()
    if any(abs(raw["active"] - edge) <= 2 for edge in (0.5, 5.5, 20.5)):
        near.add("active_trials_bin")
    if any(abs(raw["ot_score"] - edge) <= 0.03 for edge in (0.2, 0.5)):
        near.add("ot_association")
    return near


def pair_key(target: str, indication: str) -> str:
    return f"{target}|{indication}"


def load_truth(refresh: bool) -> dict:
    """The cached ground truth; fetched again with --refresh or when a pair is missing."""
    if CACHE.exists() and not refresh:
        truth = json.loads(CACHE.read_text())
        if all(pair_key(t, i) in truth["pairs"] for t, i, _ in PAIRS):
            return truth
    as_of = date.today()
    print(f"Fetching ground truth for {len(PAIRS)} pairs from Open Targets and ClinicalTrials.gov", flush=True)
    truth = {"fetched_on": as_of.isoformat(), "pairs": {}}
    for target, indication, disease_id in PAIRS:
        raw = fetch_truth(target, indication, disease_id, as_of)
        truth["pairs"][pair_key(target, indication)] = raw
        print(f"  {target} / {indication}: {raw['trials_matched']} trials", flush=True)
    CACHE.write_text(json.dumps(truth, indent=1) + "\n")
    return truth


def print_truth(truth: dict, pairs: list) -> None:
    release = truth["pairs"][pair_key(*pairs[0][:2])]["ot_release"]
    print(f"\nGround truth: Open Targets {release} + ClinicalTrials.gov, fetched {truth['fetched_on']} "
          f"({CACHE.relative_to(ROOT)}); ≈ marks a value next to a cut-off")
    print(f"{'pair':36}{'disease':16}{'active':>12}  {'phase':14}{'term.':7}{'OT':>16}  {'gen.':6}AB")
    for target, indication, _ in pairs:
        raw = truth["pairs"][pair_key(target, indication)]
        label, near = truth_labels(raw), near_cutoff(raw)
        mark = {f: " ≈" if f in near else "" for f in FIELDS}
        active = f"{raw['active']} {label['active_trials_bin']}{mark['active_trials_bin']}"
        assoc = f"{raw['ot_score']:.3f} {label['ot_association']}{mark['ot_association']}"
        print(f"{target + ' / ' + indication:36}{raw['disease']['id']:16}{active:>12}  {label['highest_phase']:14}"
              f"{label['recent_termination']:7}{assoc:>16}  {label['genetic_support']:6}{label['antibody_tractable']}")


# ---------- the two systems ----------

BASELINE_SYSTEM = ("You are a drug-discovery analyst. You have no tools and no internet access: answer from your own "
                   "knowledge, and give your best estimate for every field even when you are unsure.")
BASELINE_PROMPT = """Target: {target}
Indication: {indication}
As of: {as_of}

Fill in six fields for this target in this indication, as of the date above. Matched trials are interventional \
studies on ClinicalTrials.gov in this indication that name the target (gene symbol or protein name) in their title, \
conditions or interventions, or whose interventions include a drug acting on the target.
- active_trials_bin: matched trials that are recruiting, not yet recruiting, active and not recruiting, or enrolling \
by invitation: "0", "1-5", "6-20" or ">20".
- highest_phase: the highest phase of any matched trial, whatever its status: "EARLY_PHASE1", "PHASE1", "PHASE2", \
"PHASE3" or "PHASE4"; "none" if no matched trial has a phase.
- recent_termination: "y" if a matched trial was terminated or withdrawn in the 12 months before the date above, \
else "n".
- ot_association: the Open Targets Platform overall association score between the target and this disease, as shown \
on the disease's page (evidence for its subtypes included): "strong" if at least 0.5, "moderate" if at least 0.2, \
else "weak".
- genetic_support: "y" if Open Targets has genetic association, genetic literature or somatic mutation evidence for \
the target in this disease, else "n".
- antibody_tractable: "y" if Open Targets lists the target as tractable by antibody in any antibody tractability \
category, else "n"."""
SCHEMA = {"type": "object", "properties": {f: {"type": "string", "enum": v} for f, v in FIELDS.items()},
          "required": list(FIELDS), "additionalProperties": False}


def baseline_prompt(target: str, indication: str, as_of: str) -> str:
    """All the baseline sees besides BASELINE_SYSTEM: the pair, the date and the field definitions."""
    return BASELINE_PROMPT.format(target=target, indication=indication, as_of=as_of)


def ground_truth_in(prompt: str, raw: dict) -> list[str]:
    """Values only the ground-truth fetch could supply (ids, trial ids, the score) that appear in the prompt."""
    values = [raw["ensembl_id"], raw["disease"]["id"], f"{raw['ot_score']:.3f}", raw["highest_phase_nct"],
              *(h["nct"] for h in raw["halted_365d"])]
    return [v for v in values if v and v in prompt]


def cost(usage: dict) -> float | None:
    """Dollars for a token count at the model's prices; None for a model missing from PRICES."""
    price = PRICES.get(MODEL)
    return sum(usage.get(k, 0) * rate for k, rate in price.items()) / 1e6 if price else None


async def baseline_dossier(client: anthropic.AsyncAnthropic, gate: asyncio.Semaphore, target: str, indication: str,
                           as_of: str) -> dict:
    """One no-tools call; no fallbacks, so every scored answer comes from MODEL."""
    prompt = baseline_prompt(target, indication, as_of)
    async with gate:
        start = time.monotonic()
        try:
            response = await client.messages.create(
                model=MODEL, max_tokens=16000, system=BASELINE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}})
        except anthropic.AnthropicError as e:
            return {"prompt": prompt, "status": "error", "error": f"{type(e).__name__}: {e}"}
        latency = time.monotonic() - start
    usage = {k: getattr(response.usage, k, 0) or 0 for k in USAGE_KEYS}
    row = {"prompt": prompt, "latency_s": round(latency, 2), "usage": usage, "cost_usd": cost(usage),
           "model": response.model, "stop_reason": response.stop_reason}
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        return {**row, "status": "refusal", "error": f"declined (category {category})"}
    if response.stop_reason != "end_turn" or not response.model.startswith(MODEL):
        return {**row, "status": "error", "error": f"stop_reason {response.stop_reason}, served by {response.model}"}
    text = "".join(b.text for b in response.content if b.type == "text")
    try:
        return {**row, "status": "ok", "output": json.loads(text)}
    except json.JSONDecodeError:
        return {**row, "status": "error", "error": f"output is not JSON: {text[:200]!r}"}


def load_targetwatch():
    """app.run_targetwatch from this repository. There is no stand-in: the eval stops if it can't be imported."""
    sys.path.insert(0, str(ROOT))
    sys.dont_write_bytecode = True  # importing the app must not write into app/__pycache__
    try:
        from app import run_targetwatch
    except ImportError as e:
        sys.exit(f"Cannot import run_targetwatch from the app in {ROOT}: {e}\n"
                 "If a dependency is missing, run: uv run --with-requirements requirements.txt eval/eval.py")
    return run_targetwatch


async def in_daemon_thread(fn, *args):
    """fn(*args) in a daemon thread, so a dossier abandoned at the timeout cannot keep the process alive."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def settle(result, error):
        if future.done():  # given up on by the timeout
            return
        if error:
            future.set_exception(error)
        else:
            future.set_result(result)

    def work():
        try:
            outcome = (fn(*args), None)
        except Exception as e:
            outcome = (None, e)
        try:
            loop.call_soon_threadsafe(settle, *outcome)
        except RuntimeError:  # the event loop has closed
            pass

    threading.Thread(target=work, daemon=True).start()
    return await future


async def targetwatch_dossier(run, gate: asyncio.Semaphore, target: str, indication: str) -> dict:
    """One TargetWatch dossier: the app's live fetch plus both agents, timed from start to finish."""
    async with gate:
        start = time.monotonic()
        try:
            out = await asyncio.wait_for(in_daemon_thread(run, target, indication), TIMEOUT_S)
        except TimeoutError as e:
            return {"status": "error", "error": str(e) or f"no dossier within {TIMEOUT_S} s"}
        except Exception as e:
            return {"status": "refusal" if "refusal" in str(e) else "error", "error": f"{type(e).__name__}: {e}"}
        latency = time.monotonic() - start
    if not isinstance(out, dict):
        return {"status": "error", "error": f"run_targetwatch returned {type(out).__name__}"}
    # The app reports input tokens with any cache reads folded in; it sets no cache_control, so all are uncached.
    usage = {k: out[k] for k in ("input_tokens", "output_tokens") if k in out}
    return {"status": "ok", "output": out, "latency_s": round(latency, 2), "usage": usage,
            "cost_usd": cost(usage) if usage else None, "model": out.get("model"),
            "served_by": sorted({s.get("model") for s in out.get("steps", []) if s.get("type") == "llm"})}


async def collect(system: str, jobs: list) -> list[dict]:
    """Runs one system's (target, indication, coroutine) jobs, printing a line as each finishes."""
    async def one(target, indication, job):
        row = {"system": system, "target": target, "indication": indication, **await job}
        took = f" in {row['latency_s']:.0f} s" if "latency_s" in row else ""
        print(f"  {target} / {indication}: {row['status']}{took}", flush=True)
        return row

    return await asyncio.gather(*(one(*job) for job in jobs))


# ---------- scoring and report ----------

def normalize(field: str, value) -> str | None:
    """The answer as one of FIELDS[field]; None when missing or invalid (scored wrong, never read as "n")."""
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if field == "highest_phase":
        value = "none" if value in ("na", "none") else value.upper()
    return value if value in FIELDS[field] else None


def grade(row: dict, truth: dict) -> None:
    """Scores an ok row's answers against the fetched ground truth for its pair."""
    if row["status"] == "ok":
        row["answers"] = {f: normalize(f, row["output"].get(f)) for f in FIELDS}
        row["correct"] = {f: row["answers"][f] == truth[f] for f in FIELDS}


COLUMNS = {"active_trials_bin": ("active", 11), "highest_phase": ("phase", 16), "recent_termination": ("term.", 11),
           "ot_association": ("OT", 13), "genetic_support": ("genetic", 11), "antibody_tractable": ("AB", 6)}


def print_side_by_side(pairs: list, labels: list, rows: dict) -> None:
    """For each pair, the ground truth and then each system's answers, one row each; * marks a wrong answer."""
    print(f"\nAnswers next to the ground truth in {CACHE.relative_to(ROOT)} (* = wrong, ≈ = truth next to a cut-off)")
    print(f"{'pair':34}{'':13}" + "".join(f"{name:{width}}" for name, width in COLUMNS.values()))
    for i, (target, indication, _) in enumerate(pairs):
        cells = {f: labels[i][f] + (" ≈" if f in labels[i]["near"] else "") for f in FIELDS}
        print(f"{target + ' / ' + indication:34}{'truth':13}" + "".join(f"{cells[f]:{COLUMNS[f][1]}}" for f in FIELDS))
        for system, name in (("baseline", "baseline"), ("targetwatch", "TargetWatch")):
            row = rows[system][i]
            if row["status"] != "ok":
                print(f"{'':34}{name:13}{row['status']}: {row['error'][:100]}")
                continue
            cells = {f: (row["answers"][f] or "missing") + ("" if row["correct"][f] else " *") for f in FIELDS}
            print(f"{'':34}{name:13}" + "".join(f"{cells[f]:{COLUMNS[f][1]}}" for f in FIELDS))


def bootstrap(a: list[float], b: list[float], resamples: int = 2000) -> list[tuple[float, float]]:
    """95% intervals over resampled pairs for mean(a), mean(b) and mean(a) - mean(b), paired."""
    rng, n, draws = random.Random(0), len(a), []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        mean_a, mean_b = sum(a[i] for i in idx) / n, sum(b[i] for i in idx) / n
        draws.append((mean_a, mean_b, mean_a - mean_b))
    intervals = []
    for k in range(3):
        values = sorted(d[k] for d in draws)
        intervals.append((values[int(0.025 * resamples)], values[int(0.975 * resamples) - 1]))
    return intervals


def summarize(labels: list[dict], targetwatch: list[dict], baseline: list[dict]) -> dict:
    """Accuracy on the pairs both systems answered; latency and cost over each system's ok dossiers."""
    both = [i for i in range(len(labels)) if targetwatch[i]["status"] == baseline[i]["status"] == "ok"]
    summary = {"pairs": len(labels), "scored": len(both)}
    for name, rows in (("targetwatch", targetwatch), ("baseline", baseline)):
        ok = [r for r in rows if r["status"] == "ok"]
        costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
        per_pair = [sum(rows[i]["correct"].values()) / len(FIELDS) for i in both]
        summary[name] = {"fields": {f: sum(rows[i]["correct"][f] for i in both) for f in FIELDS},
                         "per_pair": per_pair, "overall": statistics.fmean(per_pair) if both else None,
                         "latency_s": statistics.fmean(r["latency_s"] for r in ok) if ok else None,
                         "cost_usd": statistics.fmean(costs) if costs else None,
                         "status": Counter(r["status"] for r in rows)}
    summary["majority"] = {f: Counter(labels[i][f] for i in both).most_common(1)[0][1] / len(both) if both else None
                           for f in FIELDS}
    if both:
        tw_ci, bl_ci, diff_ci = bootstrap(summary["targetwatch"]["per_pair"], summary["baseline"]["per_pair"])
        summary["targetwatch"]["ci"], summary["baseline"]["ci"], summary["difference_ci"] = tw_ci, bl_ci, diff_ci
        summary["difference"] = summary["targetwatch"]["overall"] - summary["baseline"]["overall"]
        summary["majority"]["overall"] = statistics.fmean(summary["majority"][f] for f in FIELDS)
    return summary


def print_report(s: dict, notes: list) -> None:
    n, tw, bl = s["scored"], s["targetwatch"], s["baseline"]
    print(f"\nTargetWatch vs. baseline ({MODEL}, effort {EFFORT}, no tools): "
          f"{n} of {s['pairs']} pairs answered by both")

    def acc(k: int, total: int) -> str:
        return f"{k}/{total} {k / total:5.0%}" if total else "-"

    def seconds(x):
        return f"{x:.1f} s" if x is not None else "n/a"

    def dollars(x):
        return f"${x:.3f}" if x is not None else "n/a"

    print(f"{'':26}{'TargetWatch':24}{'Baseline':24}majority label")
    for f in FIELDS:
        majority = f"{s['majority'][f]:.0%}" if n else "-"
        print(f"{f:26}{acc(tw['fields'][f], n):24}{acc(bl['fields'][f], n):24}{majority}")
    if n:
        cells = n * len(FIELDS)
        ci = {k: f"[{s[k]['ci'][0]:.0%}-{s[k]['ci'][1]:.0%}]" for k in ("targetwatch", "baseline")}
        print(f"{'overall':26}{acc(sum(tw['fields'].values()), cells) + ' ' + ci['targetwatch']:24}"
              f"{acc(sum(bl['fields'].values()), cells) + ' ' + ci['baseline']:24}{s['majority']['overall']:.0%}")
        lo, hi = s["difference_ci"]
        print(f"TargetWatch - Baseline: {100 * s['difference']:+.0f} points "
              f"(95% CI {100 * lo:+.0f} to {100 * hi:+.0f})")
    print(f"{'latency per dossier':26}{seconds(tw['latency_s']) + ' (fetch + agents)':24}{seconds(bl['latency_s']):24}")
    print(f"{'cost per dossier':26}{dollars(tw['cost_usd']):24}{dollars(bl['cost_usd']):24}")
    count = {k: " / ".join(str(s[k]["status"].get(st, 0)) for st in ("ok", "refusal", "error"))
             for k in ("targetwatch", "baseline")}
    print(f"{'ok / refusal / error':26}{count['targetwatch']:24}{count['baseline']:24}")
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  {note}")


# Chart colors: categorical slots 1-2 of the dataviz palette (validated) and its light-surface ink tokens.
SERIES = {"targetwatch": "#2a78d6", "baseline": "#eb6834"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


def plot(s: dict, truth: dict, release: str, endpoint: str) -> None:
    n, rows = s["scored"], [*FIELDS, "overall"]
    fig = plt.figure(figsize=(12, 5.8), dpi=150, facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], left=0.15, right=0.97, top=0.78, bottom=0.08,
                            wspace=0.35, hspace=0.75)
    ax = fig.add_subplot(grid[:, 0])
    for j, row in enumerate(rows):
        y = len(rows) - 1 - j
        for system, offset in (("targetwatch", 0.15), ("baseline", -0.15)):
            st = s[system]
            value = 100 * (st["overall"] if row == "overall" else st["fields"][row] / n)
            ax.barh(y + offset, value, height=0.26, color=SERIES[system])
            tip = value
            if row == "overall":
                lo, hi = (100 * x for x in st["ci"])
                ax.plot([lo, hi], [y + offset] * 2, color=INK_2, lw=1, marker="|", markersize=6)
                tip = max(value, hi)
            ax.text(tip + 1.5, y + offset, f"{value:.0f}%", va="center", fontsize=8, color=INK)
        majority = 100 * s["majority"][row]
        ax.plot([majority] * 2, [y - 0.36, y + 0.36], color=MUTED, lw=1.6, solid_capstyle="butt")
    ax.set_yticks(range(len(rows)), [r.replace("_", " ").replace("ot ", "OT ") for r in reversed(rows)], fontsize=9,
                  color=INK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.legend(handles=[Patch(color=SERIES["targetwatch"], label="TargetWatch"),
                       Patch(color=SERIES["baseline"], label=f"Baseline: {MODEL}, no tools"),
                       Line2D([], [], color=MUTED, marker="|", linestyle="none", markersize=11, markeredgewidth=1.6,
                              label="majority label (constant guess)")],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)

    for k, (key, title, fmt) in enumerate((("latency_s", "Mean latency per dossier", "{:.1f} s"),
                                           ("cost_usd", "Mean cost per dossier", "${:.3f}"))):
        side = fig.add_subplot(grid[k, 1])
        values = {"targetwatch": s["targetwatch"][key], "baseline": s["baseline"][key]}
        top = max((v for v in values.values() if v is not None), default=0) or 1
        for y, system in ((1, "targetwatch"), (0, "baseline")):
            value = values[system]
            if value is None:
                side.text(0, y, " n/a", va="center", fontsize=8, color=MUTED)
                continue
            side.barh(y, value, height=0.5, color=SERIES[system])
            side.text(value + 0.03 * top, y, fmt.format(value), va="center", fontsize=8, color=INK)
        side.set_xlim(0, 1.45 * top)
        side.set_ylim(-0.6, 1.6)
        side.set_yticks([1, 0], ["TargetWatch", "Baseline"], fontsize=8.5, color=INK)
        side.set_xticks([])
        side.set_title(title, loc="left", fontsize=9.5, color=INK)
        style(side)

    fig.text(0.015, 0.95, f"TargetWatch vs. {MODEL} without tools: field accuracy on {n} target/indication pairs",
             fontsize=13, color=INK, weight="bold")
    fig.text(0.015, 0.905, f"Effort {EFFORT}. Ground truth: Open Targets {release} + ClinicalTrials.gov, fetched "
             f"{truth['fetched_on']}. Whiskers on overall: 95% bootstrap CI over pairs. Latency and cost: mean over "
             "answered dossiers.", fontsize=8.5, color=INK_2)
    if not endpoint.startswith(OFFICIAL_API):
        fig.text(0.015, 0.87, f"Claude calls went to {endpoint}, not {OFFICIAL_API}.", fontsize=8.5, color=INK,
                 weight="bold")
    fig.savefig(FIGURE, facecolor=SURFACE)
    plt.close(fig)


# ---------- main ----------

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="fetch the ground truth again")
    parser.add_argument("--limit", type=int, help="only the first N pairs (pilot)")
    parser.add_argument("--truth-only", action="store_true", help="print the ground truth and stop")
    args = parser.parse_args()
    pairs = PAIRS[:args.limit] if args.limit else PAIRS

    truth = load_truth(args.refresh)
    print_truth(truth, pairs)
    if args.truth_only:
        return
    raws = [truth["pairs"][pair_key(t, i)] for t, i, _ in pairs]
    labels = [{**truth_labels(raw), "near": near_cutoff(raw)} for raw in raws]
    as_of, release = truth["fetched_on"], raws[0]["ot_release"]

    run = load_targetwatch()
    for (target, indication, _), raw in zip(pairs, raws, strict=True):
        if leaked := ground_truth_in(baseline_prompt(target, indication, as_of), raw):
            sys.exit(f"The baseline prompt for {target} / {indication} contains ground truth: {', '.join(leaked)}")

    gate = asyncio.Semaphore(IN_FLIGHT)
    async with anthropic.AsyncAnthropic(max_retries=3) as client:
        endpoint = str(client.base_url)
        print(f"\nClaude API: {endpoint}" + ("" if endpoint.startswith(OFFICIAL_API) else f" (not {OFFICIAL_API})"))
        print(f"Baseline: {MODEL}, effort {EFFORT}, no tools; its prompt holds the pair, the date and the field "
              "definitions, no ground truth")
        try:
            baseline = await collect("baseline",
                                     [(t, i, baseline_dossier(client, gate, t, i, as_of)) for t, i, _ in pairs])
        except TypeError as e:  # raised by the SDK when it finds no API key, token or profile
            if "authentication" not in str(e):
                raise
            sys.exit(f"No Claude credentials ({e}). Set ANTHROPIC_API_KEY; --truth-only needs none.")
    print(f"\nTargetWatch: {run.__module__}.{run.__name__} from {inspect.getsourcefile(run)}")
    targetwatch = await collect("targetwatch", [(t, i, targetwatch_dossier(run, gate, t, i)) for t, i, _ in pairs])

    rows = {"baseline": baseline, "targetwatch": targetwatch}
    for system_rows in rows.values():
        for row, label in zip(system_rows, labels, strict=True):
            grade(row, label)
    print_side_by_side(pairs, labels, rows)
    summary = summarize(labels, targetwatch, baseline)

    notes = []
    if not endpoint.startswith(OFFICIAL_API):
        notes.append(f"Claude calls went to {endpoint} (ANTHROPIC_BASE_URL), not {OFFICIAL_API}.")
    if as_of != date.today().isoformat():
        notes.append(f"The ground truth is from {as_of} while TargetWatch fetched live today; --refresh to refetch.")
    for system, system_rows in rows.items():
        answer_sets = {json.dumps(r["answers"], sort_keys=True) for r in system_rows if r["status"] == "ok"}
        if len(answer_sets) == 1 and len(pairs) > 1:
            notes.append(f"{system} gave the same six answers for every pair; check that it really ran.")
    answered = [(r, disease_id) for r, (_, _, disease_id) in zip(targetwatch, pairs, strict=True)
                if r["status"] == "ok"]
    for r, disease_id in answered:
        pair = f"{r['target']} / {r['indication']}"
        resolved = r["output"].get("disease_id")
        if resolved and resolved != disease_id:
            notes.append(f"TargetWatch resolved {pair} to {resolved}; the truth uses {disease_id}.")
        others = [m for m in r.get("served_by", []) if m and not m.startswith(MODEL)]
        if others:
            notes.append(f"TargetWatch {pair}: steps served by {', '.join(others)} (fallback).")
    configured = {r["model"] for r, _ in answered if r.get("model")} - {MODEL}
    if configured:
        notes.append(f"TargetWatch runs {', '.join(sorted(configured))}; the baseline runs {MODEL}.")
    if any(r.get("usage") for r, _ in answered):
        notes.append(f"TargetWatch cost = its input/output tokens at {MODEL} prices (the app uses no prompt caching).")
    print_report(summary, notes)

    RESULTS.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": MODEL, "effort": EFFORT,
        "endpoint": endpoint, "targetwatch": inspect.getsourcefile(run), "truth_fetched_on": as_of,
        "summary": summary, "notes": notes, "rows": [*targetwatch, *baseline]}, indent=1, default=str) + "\n")
    if summary["scored"]:
        plot(summary, truth, release, endpoint)
        print(f"\nWrote {RESULTS.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")
    else:
        print(f"\nWrote {RESULTS.relative_to(ROOT)}; no pair was answered by both systems, so no chart.")


if __name__ == "__main__":
    asyncio.run(main())
