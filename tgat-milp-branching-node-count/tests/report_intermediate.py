from __future__ import annotations

import json
import statistics
from pathlib import Path

try:
    payload = json.loads(Path("/logs/verifier/metric.json").read_text())
except Exception as exc:
    print(f"no grading result to report ({exc})")
    raise SystemExit(0)

if not payload.get("valid"):
    print(f"submission rejected: {payload.get('invalid_reason')}")
    raise SystemExit(0)

nodes = sorted(r["nodes"] for r in payload["runs"])
print("=== public evaluation panel ===")
print(f"runs                     : {payload['n_runs']}")
print(f"proved optimality         : {payload['n_proved_optimal']}")
print(f"charged the cap           : {payload['n_charged_cap']}")
print(f"nodes min / median / max  : {nodes[0]} / "
      f"{int(statistics.median(nodes))} / {nodes[-1]}")
print(f"1-shifted geometric mean  : {payload['metric']}")
