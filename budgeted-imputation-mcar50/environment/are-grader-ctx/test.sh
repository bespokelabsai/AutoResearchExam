#!/bin/bash
set -u

PY=/usr/local/bin/python3

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

trap 'if [ ! -s /logs/verifier/reward.txt ]; then echo 0 > /logs/verifier/reward.txt; fi' EXIT

chmod -R go-rwx /tests

export GRADE_SPLIT=final
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /tests
"$PY" -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
"$PY" /tests/compute_reward.py

exit 0
