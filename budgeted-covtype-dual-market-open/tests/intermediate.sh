#!/bin/bash
set -u

mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json
printf '0\n' > /logs/verifier/reward.txt
write_default_reward() {
  [ -s /logs/verifier/reward.txt ] || printf '0\n' > /logs/verifier/reward.txt
}
trap write_default_reward EXIT HUP INT TERM
chmod -R go-rwx /tests

export BUDGETED_SPLIT=intermediate
cd /tests
/usr/local/bin/python3 -m pytest \
  --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
/usr/local/bin/python3 /tests/compute_reward.py
if [ -f /logs/verifier/metric.json ]; then
  /usr/local/bin/python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("/logs/verifier/metric.json").read_text())
print(json.dumps({
    "reward": data.get("reward", 0.0),
    "balanced_accuracy": data.get("balanced_accuracy"),
    "valid": data.get("valid", False),
}))
PY
fi
exit 0
