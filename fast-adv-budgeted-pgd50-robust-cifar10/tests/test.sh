#!/bin/bash
set -u

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

trap 'if [ ! -s /logs/verifier/reward.txt ]; then printf "0\n" > /logs/verifier/reward.txt; fi' EXIT

chmod -R go-rwx /tests

cd /tests
export GRADED_SPLIT=final
/usr/bin/python3 -m pytest -p no:cacheprovider -q /tests/test_state.py
/usr/bin/python3 /tests/compute_reward.py

exit 0
