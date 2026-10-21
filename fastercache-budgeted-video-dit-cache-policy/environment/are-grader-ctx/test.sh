#!/bin/bash
set -u

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

chmod -R go-rwx /tests

export HARBOR_GRADED_SPLIT=final
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd /tests
/usr/bin/python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA

if [ ! -s /logs/verifier/metric.json ]; then
  cat > /logs/verifier/metric.json <<'JSON'
{
  "metric": 0.0,
  "metric_name": "mean_psnr_db_vs_uncached_reference",
  "metric_is_measured": false,
  "reward": 0.0,
  "valid": false,
  "failure_reason": "the grader did not run to completion",
  "split": "final"
}
JSON
fi

/usr/bin/python3 /tests/compute_reward.py

exit 0
