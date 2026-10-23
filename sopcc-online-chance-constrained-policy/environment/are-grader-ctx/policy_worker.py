from __future__ import annotations

import importlib.util
import json
import os
import resource
import sys
import traceback


def _fail(reply, stage, exc):
    reply({"ok": False, "stage": stage, "error": f"{type(exc).__name__}: {exc}",
           "traceback": traceback.format_exc(limit=6)[-1200:]})


def _apply_limits(limits: dict) -> None:
    """Cap this process before any candidate code is imported.

    Set here rather than by the caller so there is no window between exec and the cap, and
    the hard limit is set too: candidate code can lower these but never raise them.
    """
    cpu = int(limits["cpu_seconds"])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu + 1, cpu + 3))
    for name, key in (("RLIMIT_AS", "address_space_bytes"),
                      ("RLIMIT_FSIZE", "file_size_bytes"),
                      ("RLIMIT_NOFILE", "open_files")):
        value = int(limits[key])
        soft, hard = resource.getrlimit(getattr(resource, name))
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(getattr(resource, name), (value, value))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main() -> int:
    solution_dir = os.path.abspath(sys.argv[1])
    entry = os.path.join(solution_dir, "policy.py")
    _apply_limits(json.loads(sys.argv[2]))

    reply_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    channel = os.fdopen(reply_fd, "w", buffering=1)

    def reply(obj):
        channel.write(json.dumps(obj) + "\n")
        channel.flush()

    import numpy as np
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    policy = None
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        msg = json.loads(raw)
        cmd = msg.get("cmd")

        if cmd == "init":
            if not os.path.isfile(entry):
                reply({"ok": False, "stage": "import", "error": f"no entry point at {entry}"})
                return 0
            try:
                if solution_dir not in sys.path:
                    sys.path.insert(0, solution_dir)
                spec = importlib.util.spec_from_file_location("candidate_policy", entry)
                module = importlib.util.module_from_spec(spec)
                sys.modules["candidate_policy"] = module
                spec.loader.exec_module(module)
                policy_cls = getattr(module, "Policy")
            except Exception as exc:
                _fail(reply, "import", exc)
                return 0
            try:
                policy = policy_cls(int(msg["seed"]))
            except Exception as exc:
                _fail(reply, "construct", exc)
                return 0
            try:
                instance = {
                    "coords": np.asarray(msg["instance"]["coords"], dtype=np.float64),
                    "rewards": np.asarray(msg["instance"]["rewards"], dtype=np.float64),
                    "dist": np.asarray(msg["instance"]["dist"], dtype=np.float64),
                    "start": int(msg["instance"]["start"]),
                    "goal": int(msg["instance"]["goal"]),
                    "budget": float(msg["instance"]["budget"]),
                    "p_fail": float(msg["instance"]["p_fail"]),
                    "cost_model": str(msg["instance"]["cost_model"]),
                }
                policy.prepare(instance)
            except Exception as exc:
                _fail(reply, "prepare", exc)
                return 0
            reply({"ok": True})

        elif cmd == "act":
            obs = msg["obs"]
            obs = {
                "current": int(obs["current"]),
                "visited": np.asarray(obs["visited"], dtype=bool),
                "remaining_budget": float(obs["remaining_budget"]),
                "collected_reward": float(obs["collected_reward"]),
                "step": int(obs["step"]),
            }
            try:
                action = policy.act(obs)
            except Exception as exc:
                reply({"ok": False, "stage": "act",
                       "error": f"{type(exc).__name__}: {exc}"})
                continue
            if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
                reply({"ok": True, "action": None,
                       "kind": type(action).__name__})
            else:
                reply({"ok": True, "action": int(action)})

        elif cmd == "bye":
            return 0

        else:
            reply({"ok": False, "stage": "protocol", "error": f"unknown command {cmd!r}"})
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
