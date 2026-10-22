"""Fixed fetchers: Open Targets GraphQL + ClinicalTrials.gov v2 -> snapshot with flat, citable facts."""
import asyncio
import re
import statistics
from datetime import date, datetime, timedelta, timezone

import httpx2

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
CT_URL = "https://clinicaltrials.gov/api/v2/studies"
HEADERS = {"User-Agent": "TargetWatch/0.1 (hackathon demo)"}

DATATYPES = ["genetic_association", "genetic_literature", "somatic_mutation", "clinical",
             "affected_pathway", "rna_expression", "literature", "animal_model"]
ACTIVE = {"RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"}
HALTED = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
PHASE_RANK = {"NA": 0, "EARLY_PHASE1": 1, "PHASE1": 2, "PHASE2": 3, "PHASE3": 4, "PHASE4": 5}
STAGE_RANK = {"UNKNOWN": 0, "EARLY_PHASE_1": 1, "PHASE_1": 2, "PHASE_1_2": 3, "PHASE_2": 4,
              "PHASE_2_3": 5, "PHASE_3": 6, "APPROVAL": 7}
GENERIC_WORDS = {"cancer", "carcinoma", "adenocarcinoma", "tumor", "tumour", "neoplasm", "disease",
                 "malignant", "syndrome", "type", "with"}
# ClinicalTrials.gov's condition expansion misses many GEJ / gastric adenocarcinoma records.
GASTRIC = 'gastric OR stomach OR gastroesophageal OR esophagogastric OR "gastro-esophageal"'
CONDITION_EXPANSIONS = {"gastric": GASTRIC, "stomach": GASTRIC}
CT_FIELDS = ("NCTId,BriefTitle,OverallStatus,Phase,LastUpdatePostDate,StudyFirstPostDate,StartDate,"
             "LeadSponsorName,LeadSponsorClass,InterventionName,EnrollmentCount,WhyStopped")

SEARCH = """query S($q: String!, $e: [String!]!) {
  search(queryString: $q, entityNames: $e, page: {index: 0, size: 5}) { hits { id name description } } }"""

# Indirect association comes from the disease side: target-side enableIndirect returns unrelated rows.
OT_QUERY = """query TD($ensg: String!, $efo: String!) {
  meta { dataVersion { year month } }
  target(ensemblId: $ensg) {
    approvedSymbol approvedName
    tractability { modality label value }
    safetyLiabilities { event datasource biosamples { tissueLabel } literature }
    isEssential
    depMapEssentiality { screens { diseaseFromSource geneEffect } }
    associatedDiseases(Bs: [$efo], enableIndirect: false) { rows { score } }
    drugAndClinicalCandidates { rows { maxClinicalStage
      drug { name drugType mechanismsOfAction { rows { mechanismOfAction } } }
      clinicalReports { clinicalStage diseases { disease { id } } } } }
  }
  disease(efoId: $efo) {
    descendants
    associatedTargets(Bs: [$ensg], enableIndirect: true) { rows { score datatypeScores { id score } } }
  }
}"""


async def gql(client: httpx2.AsyncClient, query: str, variables: dict) -> dict:
    r = await client.post(OT_URL, json={"query": query, "variables": variables})
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError("Open Targets: " + "; ".join(e["message"] for e in body["errors"]))
    return body["data"]


async def resolve(symbol: str, indication: str) -> dict:
    raw = symbol.strip()
    base = re.sub(r"\.\d+$", "", raw)  # CLDN18.2 -> CLDN18; the isoform stays as a trial-search alias
    async with httpx2.AsyncClient(timeout=30, headers=HEADERS) as client:
        t, d = await asyncio.gather(gql(client, SEARCH, {"q": base, "e": ["target"]}),
                                    gql(client, SEARCH, {"q": indication, "e": ["disease"]}))
    target_hits, disease_hits = t["search"]["hits"], d["search"]["hits"]
    if not target_hits or target_hits[0]["name"].upper() != base.upper():
        raise LookupError(f"No exact Open Targets target for {base!r}. Candidates: "
                          + (", ".join(h["name"] for h in target_hits) or "none"))
    if not disease_hits:
        raise LookupError(f"No Open Targets disease matches {indication!r}")
    return {"symbol": raw, "indication": indication.strip(),
            "ensembl_id": target_hits[0]["id"], "approved_symbol": target_hits[0]["name"],
            "approved_name": target_hits[0].get("description"),
            "disease_id": disease_hits[0]["id"], "disease_name": disease_hits[0]["name"],
            "aliases": [raw] if raw != base else []}


def summarize_ot(data: dict, target: dict) -> dict:
    t, d = data["target"], data["disease"]
    if t is None or d is None:
        raise LookupError(f"Open Targets has no record for {target['ensembl_id']} / {target['disease_id']}")
    direct_rows = t["associatedDiseases"]["rows"]
    rows = d["associatedTargets"]["rows"]
    datatypes = dict.fromkeys(DATATYPES, 0.0)  # missing datatypes are omitted by the API
    for x in rows[0]["datatypeScores"] if rows else []:
        datatypes[x["id"]] = round(x["score"], 4)

    screens = [s for tissue in t["depMapEssentiality"] or [] for s in tissue["screens"] or []
               if s["geneEffect"] is not None]
    effects = [s["geneEffect"] for s in screens]
    words = re.findall(r"[a-z]+", f"{target['indication']} {target['disease_name']}".lower())
    keywords = sorted({w for w in words if len(w) > 3 and w not in GENERIC_WORDS})
    ind_effects = [s["geneEffect"] for s in screens
                   if any(k in (s["diseaseFromSource"] or "").lower() for k in keywords)]

    disease_ids = {target["disease_id"], *(d["descendants"] or [])}
    drugs = []
    for row in (t["drugAndClinicalCandidates"] or {}).get("rows") or []:
        stages = [r["clinicalStage"] for r in row["clinicalReports"] or []
                  if any((x.get("disease") or {}).get("id") in disease_ids for x in r["diseases"] or [])]
        moa = ((row["drug"] or {}).get("mechanismsOfAction") or {}).get("rows") or []
        drugs.append({"name": row["drug"]["name"], "type": row["drug"]["drugType"],
                      "max_stage": row["maxClinicalStage"],
                      "max_stage_indication": max(stages, key=lambda s: STAGE_RANK.get(s, 0)) if stages else None,
                      "mechanism": moa[0]["mechanismOfAction"] if moa else None})
    drugs.sort(key=lambda x: STAGE_RANK.get(x["max_stage"], 0), reverse=True)

    version = data["meta"]["dataVersion"]
    return {
        "release": f"{version['year']}.{version['month']}",
        "approved_name": t["approvedName"],
        "overall": round(rows[0]["score"], 4) if rows else 0.0,
        "direct": round(direct_rows[0]["score"], 4) if direct_rows else 0.0,
        "datatypes": datatypes,
        "tractability": [{"modality": x["modality"], "label": x["label"]}
                         for x in t["tractability"] or [] if x["value"]],
        "safety": [{"event": x["event"], "datasource": x["datasource"],
                    "tissue": ", ".join(b["tissueLabel"] for b in x["biosamples"] or [] if b.get("tissueLabel"))}
                   for x in t["safetyLiabilities"] or []],
        "depmap": {"is_essential": bool(t["isEssential"]), "lines": len(effects),
                   "median_effect": round(statistics.median(effects), 3) if effects else None,
                   "frac_below_-0.5": round(sum(e < -0.5 for e in effects) / len(effects), 3) if effects else None,
                   "indication_keywords": keywords, "indication_lines": len(ind_effects),
                   "indication_median_effect": round(statistics.median(ind_effects), 3) if ind_effects else None},
        "drugs": drugs,
    }


def trial_terms(target: dict, ot: dict) -> list[str]:
    terms = [target["approved_symbol"], ot["approved_name"] or "", *target["aliases"]]
    name = re.fullmatch(r"([A-Za-z]+) (\d+)", (ot["approved_name"] or "").strip())  # e.g. "claudin 18"
    for alias in target["aliases"]:
        # ClinicalTrials.gov treats "CLDN18.2", "CLDN 18.2" and "claudin18.2" as different concepts.
        iso = re.fullmatch(r"([A-Za-z]+)(\d+)\.(\d+)", alias)
        if iso:
            letters, number, sub = iso.groups()
            terms.append(f"{letters} {number}.{sub}")
            if name:
                terms += [f"{name[1]}{name[2]}.{sub}", f"{name[1]} {name[2]}.{sub}"]
    terms += [drug["name"] for drug in ot["drugs"][:8]]
    seen, out = set(), []
    for term in terms:
        term = term.replace('"', "").strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out


def norm_trial(study: dict) -> dict:
    p = study["protocolSection"]
    ident, status = p["identificationModule"], p["statusModule"]
    design = p.get("designModule", {})
    sponsor = p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    phases = design.get("phases") or []
    return {
        "nct": ident["nctId"],
        "title": ident.get("briefTitle", ""),
        "status": status.get("overallStatus"),
        "phase": max(phases, key=lambda x: PHASE_RANK.get(x, 0)) if phases else "NA",
        "last_update": status.get("lastUpdatePostDateStruct", {}).get("date"),
        "first_posted": status.get("studyFirstPostDateStruct", {}).get("date"),
        "start": status.get("startDateStruct", {}).get("date"),
        "sponsor": sponsor.get("name"),
        "sponsor_class": sponsor.get("class"),
        "interventions": [i["name"] for i in p.get("armsInterventionsModule", {}).get("interventions", [])][:4],
        "enrollment": design.get("enrollmentInfo", {}).get("count"),
        "why_stopped": status.get("whyStopped"),
    }


async def fetch_trials(client: httpx2.AsyncClient, target: dict, ot: dict) -> tuple[list[dict], dict]:
    key = next((k for k in CONDITION_EXPANSIONS if k in target["indication"].lower()), None)
    params = {"query.cond": CONDITION_EXPANSIONS[key] if key else target["indication"],
              "query.term": " OR ".join(f'"{t}"' for t in trial_terms(target, ot)),
              "fields": CT_FIELDS, "pageSize": 1000, "sort": "LastUpdatePostDate:desc"}
    studies, token = [], None
    for _ in range(3):
        r = await client.get(CT_URL, params={**params, **({"pageToken": token} if token else {})})
        if r.status_code != 200:
            raise RuntimeError(f"ClinicalTrials.gov {r.status_code}: {r.text[:300]}")
        body = r.json()
        studies += body.get("studies", [])
        token = body.get("nextPageToken")
        if not token:
            break
    return [norm_trial(s) for s in studies], {"cond": params["query.cond"], "term": params["query.term"]}


def compute_facts(snap: dict) -> dict:
    ot, trials = snap["ot"], snap["trials"]
    as_of = date.fromisoformat(snap["as_of"])
    f = {"ot.release": ot["release"], "ot.assoc.overall": ot["overall"], "ot.assoc.direct": ot["direct"]}
    f |= {f"ot.assoc.{k}": v for k, v in ot["datatypes"].items()}
    for modality in ("SM", "AB", "PR", "OC"):
        f[f"ot.tract.{modality}"] = sum(x["modality"] == modality for x in ot["tractability"])
    f["ot.safety.count"] = len(ot["safety"])
    f["ot.drugs.count"] = len(ot["drugs"])
    f["ot.drugs.max_stage"] = ot["drugs"][0]["max_stage"] if ot["drugs"] else "none"
    in_indication = [x["max_stage_indication"] for x in ot["drugs"] if x["max_stage_indication"]]
    f["ot.drugs.max_stage_indication"] = (max(in_indication, key=lambda s: STAGE_RANK.get(s, 0))
                                          if in_indication else "none")
    dm = ot["depmap"]
    f |= {"depmap.is_essential": int(dm["is_essential"]), "depmap.lines": dm["lines"],
          "depmap.median_effect": dm["median_effect"], "depmap.frac_below_-0.5": dm["frac_below_-0.5"],
          "depmap.indication_lines": dm["indication_lines"],
          "depmap.indication_median_effect": dm["indication_median_effect"]}
    year_ago = (as_of - timedelta(days=365)).isoformat()
    recent = (as_of - timedelta(days=90)).isoformat()
    f |= {"ct.total": len(trials),
          "ct.active": sum(t["status"] in ACTIVE for t in trials),
          "ct.halted": sum(t["status"] in HALTED for t in trials),
          "ct.halted_365d": sum(t["status"] in HALTED and (t["last_update"] or "") >= year_ago for t in trials),
          "ct.updated_90d": sum((t["last_update"] or "") >= recent for t in trials),
          "ct.highest_phase": max((t["phase"] for t in trials), key=lambda p: PHASE_RANK.get(p, 0), default="none"),
          "ct.sponsors": len({t["sponsor"] for t in trials if t["sponsor"]}),
          "ct.industry_trials": sum(t["sponsor_class"] == "INDUSTRY" for t in trials)}
    for phase in PHASE_RANK:
        f[f"ct.phase.{phase}"] = sum(t["phase"] == phase for t in trials)
    for t in trials:
        f[f"trial.{t['nct']}"] = f"{t['phase']}|{t['status']}|{t['last_update']}"
    return f


async def build_snapshot(target: dict, simulate_days: int = 0) -> dict:
    as_of = date.today() - timedelta(days=simulate_days)
    async with httpx2.AsyncClient(timeout=60, headers=HEADERS) as client:
        data = await gql(client, OT_QUERY, {"ensg": target["ensembl_id"], "efo": target["disease_id"]})
        ot = summarize_ot(data, target)
        trials, query = await fetch_trials(client, target, ot)
    if simulate_days:  # simulated baseline: drop trials updated after the as-of date
        trials = [t for t in trials if (t["last_update"] or "") <= as_of.isoformat()]
    snap = {"as_of": as_of.isoformat(), "simulated": bool(simulate_days),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ot": ot, "trials": trials, "ct_query": query}
    snap["facts"] = compute_facts(snap)
    return snap


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def diff(prev: dict, cur: dict) -> dict:
    before = {t["nct"]: t for t in prev["trials"]}
    after = {t["nct"]: t for t in cur["trials"]}
    added = [n for n in after if n not in before]
    new = [n for n in added if (after[n]["first_posted"] or "") > prev["as_of"]]
    halted = [n for n in after if after[n]["status"] in HALTED
              and (n not in before or before[n]["status"] not in HALTED)]
    status_changed = [{"nct": n, "from": before[n]["status"], "to": after[n]["status"]}
                      for n in after if n in before and before[n]["status"] != after[n]["status"]]
    shifts = {}
    for k, v in cur["facts"].items():
        old = prev["facts"].get(k)
        if k.startswith("trial.") or old == v:
            continue
        if isinstance(v, float) and _number(old) and abs(v - old) < 0.01:
            continue
        shifts[k] = {"from": old, "to": v}
    d = {"baseline_as_of": prev["as_of"], "baseline_simulated": prev["simulated"],
         "new": new, "not_in_baseline": [n for n in added if n not in new], "halted": halted,
         "status_changed": status_changed, "dropped": [n for n in before if n not in after], "shifts": shifts}
    d["facts"] = {"diff.trials.new": len(new), "diff.trials.not_in_baseline": len(d["not_in_baseline"]),
                  "diff.trials.halted": len(halted), "diff.trials.status_changed": len(status_changed),
                  "diff.trials.dropped": len(d["dropped"])}
    for k, s in shifts.items():
        d["facts"][f"diff.{k}"] = (round(s["to"] - s["from"], 4) if _number(s["to"]) and _number(s["from"])
                                   else f"{s['from']} -> {s['to']}")
    return d


if __name__ == "__main__":
    import json
    import sys

    async def _main():
        target = await resolve(sys.argv[1], sys.argv[2])
        snap = await build_snapshot(target, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        print(json.dumps({"target": target, "ct_query": snap["ct_query"],
                          "facts": {k: v for k, v in snap["facts"].items() if not k.startswith("trial.")}}, indent=1))

    asyncio.run(_main())
