#!/bin/bash
set -u

PY=/usr/local/bin/python3

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

trap 'if [ ! -s /logs/verifier/reward.txt ]; then echo 0 > /logs/verifier/reward.txt; fi' EXIT

chmod -R go-rwx /tests

export GRADE_SPLIT=intermediate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /tests
"$PY" -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA \
  > /logs/verifier/pytest_intermediate.log 2>&1
"$PY" /tests/compute_reward.py > /logs/verifier/compute_reward.log 2>&1

"$PY" - <<'PY'
import json
from pathlib import Path

try:
    rec = json.loads(Path("/logs/verifier/metric.json").read_text())
except Exception:
    rec = {}
try:
    score = float(Path("/logs/verifier/reward.txt").read_text().strip())
except Exception:
    score = 0.0

total = int(rec.get("cells_total", 0))
invalid = int(rec.get("cells_invalid", total))
reasons = rec.get("invalid_reasons", {}) or {}
note = rec.get("note", "") or ""

print("evaluation cells:      %d" % total)
print("cells scored:          %d" % (total - invalid))
print("cells scored as zero:  %d" % invalid)
if reasons:
    print("zeroed because:        " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
if note:
    print("note:                  %s" % note)
print("score:                 %.6f" % score)
PY

exit 0
