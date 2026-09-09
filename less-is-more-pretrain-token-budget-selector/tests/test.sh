#!/bin/bash
set -u

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

default_reward() {
  if [ ! -s /logs/verifier/reward.txt ]; then printf "0\n" > /logs/verifier/reward.txt; fi
}
trap 'default_reward' EXIT
trap 'default_reward; exit 0' HUP INT TERM

chmod -R go-rwx /tests

export HIDDEN_SPLIT=final
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /tests
/usr/bin/python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
/usr/bin/python3 /tests/compute_reward.py

exit 0
