#!/bin/bash
set -u

mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier 2>/dev/null || true
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/metric.json

write_default_reward() {
  if [ ! -s /logs/verifier/reward.txt ]; then
    printf '0\n' > /logs/verifier/reward.txt
  fi
}
trap write_default_reward EXIT HUP INT TERM

chmod -R go-rwx /tests

export GRADE_SPLIT=intermediate
cd /tests
/usr/local/bin/python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -q -s --no-header
/usr/local/bin/python3 /tests/compute_reward.py

exit 0
