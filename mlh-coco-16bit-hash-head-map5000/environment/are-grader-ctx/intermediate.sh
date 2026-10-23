#!/bin/bash
set -u

PY=/usr/local/bin/python3

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

reward_floor() { [ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt; }
trap reward_floor EXIT

chmod -R go-rwx /tests

cd /tests
export GRADE_SPLIT=intermediate
"$PY" -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
"$PY" /tests/compute_reward.py > /logs/verifier/intermediate_reward.log 2>&1

exit 0
