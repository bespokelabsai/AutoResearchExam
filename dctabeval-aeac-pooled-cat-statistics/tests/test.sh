#!/bin/bash
set -u

PY=/usr/local/bin/python3

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

reward_fallback() {
  if [ ! -s /logs/verifier/reward.txt ]; then
    mkdir -p /logs/verifier
    printf '0\n' > /logs/verifier/reward.txt
    echo "reward=0 (trap fallback: grader exited without writing a reward)"
  fi
}
trap reward_fallback EXIT

chmod -R go-rwx /tests

export GRADE_SPLIT=final

cd /tests
"$PY" -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
"$PY" /tests/compute_reward.py

exit 0
