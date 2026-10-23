#!/bin/bash
set -u

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

trap 'if [ ! -s /logs/verifier/reward.txt ]; then echo 0 > /logs/verifier/reward.txt; fi' EXIT TERM INT

chmod -R go-rwx /tests

export GRADER_SPLIT=final

cd /tests
/usr/local/bin/python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
/usr/local/bin/python3 /tests/compute_reward.py

exit 0
