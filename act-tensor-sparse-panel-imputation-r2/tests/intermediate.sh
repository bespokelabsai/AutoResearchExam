#!/bin/bash
set -u

trap 'if [ ! -s /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; printf "0\n" > /logs/verifier/reward.txt; fi' EXIT

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

chmod -R go-rwx /tests

cd /tests
export GRADER_SPLIT=intermediate
/usr/local/bin/python3 -m pytest --ctrf /logs/verifier/ctrf.json \
    /tests/test_state.py -rA > /logs/verifier/intermediate_pytest.log 2>&1
/usr/local/bin/python3 /tests/compute_reward.py >> /logs/verifier/intermediate_pytest.log 2>&1

/usr/local/bin/python3 - <<'PY'
import json
from pathlib import Path
try:
    d = json.loads(Path("/logs/verifier/metric.json").read_text())
    print(f"panels_scored={d['n_instances']} ran_without_error={d['n_scored_ok']}")
    print(f"mean_R2_imp={d['metric']:+.4f}")
    if not d["submission_ok"]:
        print(f"submission_rejected: {d['submission_note']}")
except Exception:
    print("panels_scored=0 ran_without_error=0")
    print("mean_R2_imp=n/a (the submission produced no gradeable result)")
PY

exit 0
