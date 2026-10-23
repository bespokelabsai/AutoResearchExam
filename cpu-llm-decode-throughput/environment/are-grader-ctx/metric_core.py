from __future__ import annotations

import os
import shutil
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner

VOCAB_SIZE = 50257

LOAD_BUDGET_S = 180.0

RUN_BUDGET_MULTIPLIER = 2.0
RUN_BUDGET_FLOOR_S = 60.0
REFERENCE_RUN_BUDGET_S = 900.0

AGREEMENT_MIN = 0.99


def validate_outputs(outputs, requests):
    """Return (ok, detail). Outputs are untrusted data, so every shape is checked."""
    if not isinstance(outputs, list):
        return False, "engine did not return a list"
    if len(outputs) != len(requests):
        return False, f"engine returned {len(outputs)} lists, expected {len(requests)}"
    for i, (row, req) in enumerate(zip(outputs, requests)):
        want = int(req["max_new_tokens"])
        if not isinstance(row, list):
            return False, f"element {i} is not a list"
        if len(row) != want:
            return False, f"element {i} has {len(row)} token ids, expected {want}"
        for tid in row:
            if not isinstance(tid, int) or isinstance(tid, bool):
                return False, f"element {i} contains a non-integer token id"
            if not 0 <= tid < VOCAB_SIZE:
                return False, f"element {i} contains out-of-range token id {tid}"
    return True, ""


def count_agreement(outputs, reference):
    """Position-wise agreement between two valid output sets."""
    matched = total = 0
    for row, ref in zip(outputs, reference):
        total += len(ref)
        matched += sum(1 for a, b in zip(row, ref) if a == b)
    return matched, total


def tokens_per_second(requests, elapsed_s):
    return sum(int(r["max_new_tokens"]) for r in requests) / elapsed_s


def _fresh_dir(path, owner=None):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if owner is not None:
        os.chown(path, owner[0], owner[1])
    return path


def run_rounds(
    reference_engine_path,
    candidate_engine_path,
    model_dir,
    warmup_path,
    workloads,
    work_root,
    cmd_prefix=(),
    runner_path=None,
    run_dir_owner=None,
    reap_uid=None,
    log=print,
):
    """Interleave reference and candidate over one distinct workload per round.

    Each round is a fresh process pair on a workload the previous round never saw, so
    nothing carried over from an earlier round can stand in for the work. The reference
    run's directory is destroyed before the candidate is launched, so the candidate
    cannot read the answer off the run that just produced it.

    Returns a list of per-round dicts; a round with ``ok`` false stops the comparison.
    """
    rounds = []
    for index, (name, requests) in enumerate(workloads):
        ref_dir = _fresh_dir(os.path.join(work_root, f"round{index}_reference"), run_dir_owner)
        cand_dir = _fresh_dir(os.path.join(work_root, f"round{index}_candidate"), run_dir_owner)

        ref = runner.run_timed(
            reference_engine_path,
            model_dir,
            warmup_path,
            requests,
            ref_dir,
            LOAD_BUDGET_S,
            REFERENCE_RUN_BUDGET_S,
            cmd_prefix=cmd_prefix,
            runner_path=runner_path,
            reap_uid=reap_uid,
        )
        if not ref.ok:
            return rounds + [
                {"round": index, "workload": name, "ok": False,
                 "detail": f"reference run failed: {ref.detail}"}
            ]
        ref_ok, ref_detail = validate_outputs(ref.outputs, requests)
        if not ref_ok:
            return rounds + [
                {"round": index, "workload": name, "ok": False,
                 "detail": f"reference run invalid: {ref_detail}"}
            ]
        shutil.rmtree(ref_dir, ignore_errors=True)

        budget = max(RUN_BUDGET_FLOOR_S, RUN_BUDGET_MULTIPLIER * ref.elapsed_s)
        cand = runner.run_timed(
            candidate_engine_path,
            model_dir,
            warmup_path,
            requests,
            cand_dir,
            LOAD_BUDGET_S,
            budget,
            cmd_prefix=cmd_prefix,
            runner_path=runner_path,
            reap_uid=reap_uid,
        )
        record = {
            "round": index,
            "workload": name,
            "new_tokens": sum(int(r["max_new_tokens"]) for r in requests),
            "reference_s": round(ref.elapsed_s, 4),
            "reference_tok_s": round(tokens_per_second(requests, ref.elapsed_s), 3),
            "budget_s": round(budget, 2),
        }
        if not cand.ok:
            record.update({"ok": False, "detail": cand.detail})
            rounds.append(record)
            log(f"round {index} ({name}): candidate failed - {cand.detail}")
            return rounds

        ok, detail = validate_outputs(cand.outputs, requests)
        if not ok:
            record.update({"ok": False, "detail": detail})
            rounds.append(record)
            log(f"round {index} ({name}): candidate invalid - {detail}")
            return rounds

        matched, total = count_agreement(cand.outputs, ref.outputs)
        record.update(
            {
                "ok": True,
                "detail": "",
                "candidate_s": round(cand.elapsed_s, 4),
                "candidate_tok_s": round(tokens_per_second(requests, cand.elapsed_s), 3),
                "speedup": round(ref.elapsed_s / cand.elapsed_s, 4),
                "matched_positions": matched,
                "total_positions": total,
            }
        )
        rounds.append(record)
        log(
            f"round {index} ({name}): reference {record['reference_tok_s']} tok/s, "
            f"candidate {record['candidate_tok_s']} tok/s, "
            f"speedup {record['speedup']}x, agreement {matched}/{total}"
        )
        shutil.rmtree(cand_dir, ignore_errors=True)
    return rounds


def summarise(rounds):
    """Fold per-round records into the graded speedup S and the agreement rate."""
    if not rounds or not all(r.get("ok") for r in rounds):
        failed = next((r for r in rounds if not r.get("ok")), None)
        return {
            "speedup": 0.0,
            "agreement": 0.0,
            "valid": False,
            "detail": (failed or {}).get("detail", "no rounds were run"),
            "rounds": rounds,
        }
    matched = sum(r["matched_positions"] for r in rounds)
    total = sum(r["total_positions"] for r in rounds)
    agreement = matched / total if total else 0.0
    speedups = [r["speedup"] for r in rounds]
    if agreement < AGREEMENT_MIN:
        return {
            "speedup": 0.0,
            "agreement": agreement,
            "valid": True,
            "detail": f"token agreement {agreement:.4f} is below {AGREEMENT_MIN}",
            "rounds": rounds,
        }
    return {
        "speedup": statistics.median(speedups),
        "agreement": agreement,
        "valid": True,
        "detail": "",
        "per_round_speedup": speedups,
        "rounds": rounds,
    }
