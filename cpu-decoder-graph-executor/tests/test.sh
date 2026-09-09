#!/bin/bash
set -u

PY=/usr/local/bin/python3

mkdir -p /app/output /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

trap '[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt' EXIT

chmod -R go-rwx /tests

cd /tests
export GRADE_SPLIT=final
"$PY" -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
"$PY" /tests/compute_reward.py

exit 0
