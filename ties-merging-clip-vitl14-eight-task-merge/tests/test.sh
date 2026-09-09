#!/bin/bash
set -u

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

chmod -R go-rwx /tests

export PDK_PANEL=final
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=false

cd /tests
PYTEST_CAP=6600
if command -v timeout >/dev/null 2>&1; then
  timeout -k 30 "$PYTEST_CAP" /usr/bin/python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
else
  /usr/bin/python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
fi
/usr/bin/python3 /tests/compute_reward.py

exit 0
