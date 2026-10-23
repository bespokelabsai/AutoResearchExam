"""Final verifier: score the run from the grader ledger.

reward      hidden test score of the submission with the best validation score
            (ties go to the earlier submission); 0 if nothing was submitted.
auarc       area under the curve of that same quantity over the wall-clock budget,
            using the time each validation score became available, divided by the
            budget. 0 before the first graded submission.
"""

import json
from pathlib import Path

STATE = Path("/var/lib/are")
OUT = Path("/logs/verifier")
OUT.mkdir(parents=True, exist_ok=True)

meta = json.loads((STATE / "meta.json").read_text()) if (STATE / "meta.json").exists() else {}
budget = float(meta.get("budget_seconds") or 1.0)

subs: dict[int, dict] = {}
ledger = STATE / "ledger.jsonl"
for line in (ledger.read_text().splitlines() if ledger.exists() else []):
    ev = json.loads(line)
    s = subs.setdefault(ev["id"], {"id": ev["id"]})
    if ev["event"] == "submitted":
        s["submitted_t"] = ev["t"]
    elif ev["event"] == "val":
        s["val"] = ev["score"]
        s["val_t"] = ev["t"]
    elif ev["event"] == "test":
        s["test"] = ev["score"]

graded = sorted((s for s in subs.values() if "val" in s), key=lambda s: s["val_t"])

# Step function of the hidden score of the best-validation pick over time.
auc, best, last_t, cur = 0.0, None, 0.0, 0.0
timeline = []
for s in graded:
    t = min(s["val_t"], budget)
    auc += cur * (t - last_t)
    last_t = t
    if best is None or s["val"] > best["val"] or (s["val"] == best["val"] and s["id"] < best["id"]):
        best = s
        cur = float(best.get("test") or 0.0)
    timeline.append({"t": s["val_t"], "id": s["id"], "val": s["val"], "test": s.get("test"),
                     "pick": best["id"]})
auc += cur * (budget - last_t) if last_t < budget else 0.0

reward = {
    "reward": float(best.get("test") or 0.0) if best else 0.0,
    "auarc": auc / budget,
    "best_val": float(best["val"]) if best else 0.0,
    "n_submissions": float(len(subs)),
    "n_test_graded": float(sum(s.get("test") is not None for s in subs.values())),
}
(OUT / "reward.json").write_text(json.dumps(reward))
(OUT / "timeline.json").write_text(json.dumps({"budget": budget, "selected": best and best["id"],
                                               "submissions": timeline}, indent=1))
print(json.dumps(reward, indent=1))
