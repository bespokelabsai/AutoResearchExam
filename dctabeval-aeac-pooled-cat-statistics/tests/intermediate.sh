#!/bin/bash
set -u

PY=/usr/local/bin/python3

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

reward_fallback() {
  if [ ! -s /logs/verifier/reward.txt ]; then
    mkdir -p /logs/verifier
    printf '0\n' > /logs/verifier/reward.txt
  fi
}
trap reward_fallback EXIT

chmod -R go-rwx /tests

export GRADE_SPLIT=intermediate

cd /tests
"$PY" -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA \
  > /logs/verifier/intermediate_pytest.log 2>&1
"$PY" /tests/compute_reward.py >> /logs/verifier/intermediate_pytest.log 2>&1

"$PY" - <<'PY'
import json
from pathlib import Path

try:
    r = json.loads(Path("/logs/verifier/metric.json").read_text())
except Exception as exc:
    print(f"evaluation did not complete: {type(exc).__name__}")
    raise SystemExit(0)

print("evaluation of /app/output/solution.py")
if not r.get("valid"):
    print(f"  deliverable unusable: {r.get('invalid_reason')}")
    raise SystemExit(0)
print(f"  seeds scored               : {r['seeds_ok']} / {len(r['seeds'])}")
print(f"  seeds that failed to score : {r['seeds_failed']}")
print(f"  mean ROC-AUC over the panel: {r['metric']:.4f}")
if r["seeds_failed"]:
    kinds = sorted({str(d.get("status", "")).split(":")[0]
                    for d in r.get("failure_detail", [])})
    print(f"  failure kinds              : {', '.join(k for k in kinds if k)}")
PY

exit 0
