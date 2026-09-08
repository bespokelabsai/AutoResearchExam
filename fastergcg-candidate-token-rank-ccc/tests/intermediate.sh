#!/bin/bash
set -u

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

reward_fallback() { [ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt; }
trap reward_fallback EXIT

chmod -R go-rwx /tests

export PANEL=intermediate
cd /tests
/usr/bin/python3 -m pytest --ctrf /logs/verifier/ctrf.json \
    /tests/test_state.py::test_submission_scores -rA
/usr/bin/python3 /tests/compute_reward.py

exit 0
