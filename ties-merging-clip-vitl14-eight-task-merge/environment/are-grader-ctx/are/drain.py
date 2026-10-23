"""Verifier collect hook, run after the agent stops.

Grades the final workspace as one last submission (unless it equals the last one),
then waits until every submission has both grades recorded.
"""

import json
import time
import urllib.request
from pathlib import Path

TOKEN = Path("/run/are-secret/token").read_text().strip()
H = {"x-are-token": TOKEN}

req = urllib.request.Request("http://localhost:8000/final", method="POST", headers=H)
with urllib.request.urlopen(req, timeout=120) as resp:
    print("final:", json.loads(resp.read()), flush=True)

while True:
    with urllib.request.urlopen(urllib.request.Request("http://localhost:8000/drain", headers=H), timeout=30) as resp:
        state = json.loads(resp.read())
    print(state, flush=True)
    if state["val_pending"] == 0 and state["test_pending"] == 0:
        break
    time.sleep(5)
