from __future__ import annotations

import argparse
import os
import random

import numpy as np

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's "
    "questions."
)
POST_TEXT = " ASSISTANT:"

MODEL_PATH = "/opt/assets/vicuna-7b-v1.5"
ALLOWLIST_PATH = "/app/data/token_allowlist.npy"
EMBEDDING_PATH = "/opt/assets/embedding_matrix.npy"

TOPK = 64
STEPS = 100
SNAPSHOT_EVERY = 10


def _pin_determinism(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_allowlist(tokenizer) -> np.ndarray:
    """Token ids whose single-token decoding is non-empty printable ASCII.

    Special tokens and ids >= vocab_size are excluded. Deterministic: the same
    tokenizer always yields the same array.
    """
    special = set(tokenizer.all_special_ids)
    keep = []
    for tid in range(tokenizer.vocab_size):
        if tid in special:
            continue
        piece = tokenizer.decode([tid])
        if piece and piece.isascii() and piece.isprintable():
            keep.append(tid)
    return np.asarray(keep, dtype=np.int64)


class StateGenerator:
    """Holds the target model and produces optimisation states for behaviours."""

    def __init__(self, model_path: str = MODEL_PATH,
                 allowlist_path: str = ALLOWLIST_PATH, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _pin_determinism(seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16).eval().cuda()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.embed = self.model.get_input_embeddings().weight
        self.embed_f32 = self.embed.detach().float()
        self.vocab_size = int(self.embed.shape[0])
        self.allowlist = np.load(allowlist_path)
        self.allow_t = torch.as_tensor(self.allowlist, device="cuda")
        self.post_ids = self.tokenizer(
            POST_TEXT, add_special_tokens=False)["input_ids"]

    def encode_behaviour(self, goal: str, target: str):
        prefix_ids = self.tokenizer(
            f"{VICUNA_SYSTEM} USER: {goal}", add_special_tokens=True)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        return list(prefix_ids), list(self.post_ids), list(target_ids)

    def _target_nll(self, logits, target_ids_t, n_target: int):
        """Token-mean cross-entropy of the target block, per batch element."""
        torch = self.torch
        pred = logits[:, -n_target - 1:-1, :].float()
        logprobs = torch.log_softmax(pred, dim=-1)
        gathered = logprobs.gather(
            2, target_ids_t.view(1, -1, 1).expand(pred.shape[0], -1, 1))
        return -gathered.squeeze(2).mean(dim=1)

    def candidate_losses(self, prefix_ids, post_ids, target_ids, token_ids,
                         batch_size: int = 64):
        """Exact token-mean target NLL for each replacement token id."""
        torch = self.torch
        dev = "cuda"
        pre = torch.as_tensor(prefix_ids, device=dev)
        post = torch.as_tensor(post_ids + target_ids, device=dev)
        tgt = torch.as_tensor(target_ids, device=dev)
        n_target = len(target_ids)
        out = torch.empty(len(token_ids), dtype=torch.float64)
        with torch.inference_mode():
            for start in range(0, len(token_ids), batch_size):
                chunk = torch.as_tensor(
                    token_ids[start:start + batch_size], device=dev)
                b = chunk.shape[0]
                ids = torch.cat([
                    pre.unsqueeze(0).expand(b, -1),
                    chunk.view(b, 1),
                    post.unsqueeze(0).expand(b, -1),
                ], dim=1)
                logits = self.model(input_ids=ids).logits
                out[start:start + b] = self._target_nll(
                    logits, tgt, n_target).double().cpu()
        return out.numpy()

    def one_hot_grad(self, prefix_ids, post_ids, target_ids, current_token_id):
        """d(loss)/d(one-hot) at the optimisable position, float32 (V,)."""
        torch = self.torch
        dev = "cuda"
        one_hot = torch.zeros(1, self.vocab_size, device=dev, dtype=torch.float32)
        one_hot[0, current_token_id] = 1.0
        one_hot.requires_grad_(True)
        suffix_emb = (one_hot @ self.embed_f32).to(self.embed.dtype).unsqueeze(0)
        pre = torch.as_tensor(prefix_ids, device=dev).unsqueeze(0)
        post = torch.as_tensor(post_ids + target_ids, device=dev).unsqueeze(0)
        embeds = torch.cat([
            self.embed[pre].detach(), suffix_emb, self.embed[post].detach()],
            dim=1)
        logits = self.model(inputs_embeds=embeds).logits
        loss = self._target_nll(
            logits, torch.as_tensor(target_ids, device=dev), len(target_ids))[0]
        grad, = torch.autograd.grad(loss, one_hot)
        return grad[0].detach().float().cpu().numpy(), float(loss.item())

    def top_candidates(self, grad_row: np.ndarray, current_token_id: int,
                       topk: int = TOPK, exclude=()) -> np.ndarray:
        """The topk allowlist tokens with the smallest gradient entry.

        The current token, and any token in `exclude`, is never a candidate.
        Returned ascending by token id, so the order handed to a ranker carries
        no gradient information.
        """
        blocked = np.asarray(sorted({current_token_id, *exclude}), dtype=np.int64)
        allowed = self.allowlist[~np.isin(self.allowlist, blocked)]
        scores = grad_row[allowed]
        order = np.argsort(scores, kind="stable")[:topk]
        return np.sort(allowed[order])

    def trajectory(self, goal: str, target: str, init_token_id: int,
                   steps: int = STEPS, snapshot_every: int = SNAPSHOT_EVERY,
                   topk: int = TOPK, behaviour_row: int = -1,
                   verbose: bool = False):
        prefix_ids, post_ids, target_ids = self.encode_behaviour(goal, target)
        current = int(init_token_id)
        visited = {current}
        snapshots = []
        for step in range(1, steps + 1):
            grad_row, current_loss = self.one_hot_grad(
                prefix_ids, post_ids, target_ids, current)
            cands = self.top_candidates(grad_row, current, topk, exclude=visited)
            cand_losses = self.candidate_losses(
                prefix_ids, post_ids, target_ids, cands.tolist())
            true_delta = cand_losses - current_loss
            if step % snapshot_every == 0:
                snapshots.append({
                    "behaviour_row": behaviour_row,
                    "step": step,
                    "current_token_id": current,
                    "current_loss": current_loss,
                    "grad_row": grad_row,
                    "candidate_ids": cands,
                    "prefix_ids": np.asarray(prefix_ids, dtype=np.int64),
                    "post_ids": np.asarray(post_ids, dtype=np.int64),
                    "target_ids": np.asarray(target_ids, dtype=np.int64),
                    "true_delta": true_delta,
                })
            current = int(cands[int(np.argmin(true_delta))])
            visited.add(current)
            if verbose:
                print(f"  step {step:3d} loss {current_loss:.5f} "
                      f"best_delta {float(true_delta.min()):+.5f} "
                      f"-> token {current}", flush=True)
        return snapshots


def _pad(rows, dtype=np.int64):
    width = max(len(r) for r in rows)
    out = np.zeros((len(rows), width), dtype=dtype)
    lens = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
        lens[i] = len(r)
    return out, lens


def save_states(snapshots, path: str) -> None:
    """Write a list of snapshot dicts to an .npz in the documented schema."""
    arrays = {
        "behaviour_row": np.asarray([s["behaviour_row"] for s in snapshots], np.int64),
        "step": np.asarray([s["step"] for s in snapshots], np.int64),
        "current_token_id": np.asarray(
            [s["current_token_id"] for s in snapshots], np.int64),
        "current_loss": np.asarray(
            [s["current_loss"] for s in snapshots], np.float64),
        "grad_row": np.stack([s["grad_row"] for s in snapshots]).astype(np.float32),
        "candidate_ids": np.stack(
            [s["candidate_ids"] for s in snapshots]).astype(np.int64),
        "true_delta": np.stack(
            [s["true_delta"] for s in snapshots]).astype(np.float64),
    }
    for key in ("prefix_ids", "post_ids", "target_ids"):
        padded, lens = _pad([s[key] for s in snapshots])
        arrays[key] = padded
        arrays[key.replace("_ids", "_len")] = lens
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez(path, **arrays)


def load_states(path: str, embedding_path: str = EMBEDDING_PATH):
    """Read an .npz written by save_states into a list of feature dicts.

    Each element carries exactly the keys the ranker's `features` argument
    carries, plus `true_delta`, which the sealed grading states do not expose.
    `embedding_matrix` is the same read-only array object in every element; pass
    embedding_path=None to leave it out.
    """
    z = np.load(path)
    embedding = None
    if embedding_path is not None:
        embedding = np.load(embedding_path)
        embedding.setflags(write=False)
    out = []
    for i in range(z["step"].shape[0]):
        item = {
            "behaviour_row": int(z["behaviour_row"][i]),
            "step": int(z["step"][i]),
            "current_token_id": int(z["current_token_id"][i]),
            "current_loss": float(z["current_loss"][i]),
            "grad_row": z["grad_row"][i],
            "candidate_ids": z["candidate_ids"][i],
            "prefix_ids": z["prefix_ids"][i][: int(z["prefix_len"][i])].tolist(),
            "post_ids": z["post_ids"][i][: int(z["post_len"][i])].tolist(),
            "target_ids": z["target_ids"][i][: int(z["target_len"][i])].tolist(),
        }
        if embedding is not None:
            item["embedding_matrix"] = embedding
        if "true_delta" in z:
            item["true_delta"] = z["true_delta"][i]
        out.append(item)
    return out


def draw_initial_tokens(allowlist: np.ndarray, rows, seed: int) -> dict:
    """One initial suffix token per behaviour row, drawn uniformly, seeded."""
    rng = np.random.default_rng(seed)
    return {int(r): int(allowlist[rng.integers(0, allowlist.shape[0])])
            for r in rows}


def main() -> None:
    import csv

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="/app/data/dev_behaviors.csv")
    ap.add_argument("--rows", required=True,
                    help="comma separated row indices, e.g. 0,3,7")
    ap.add_argument("--seed", type=int, required=True,
                    help="seeds the initial suffix token of every row")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--allowlist", default=ALLOWLIST_PATH)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows = [int(r) for r in args.rows.split(",") if r.strip() != ""]
    with open(args.csv, newline="") as fh:
        table = list(csv.DictReader(fh))

    gen = StateGenerator(args.model, args.allowlist, seed=args.seed)
    init = draw_initial_tokens(gen.allowlist, rows, args.seed)
    snapshots = []
    for row in rows:
        if not args.quiet:
            print(f"[row {row}] init token {init[row]} "
                  f"{gen.tokenizer.decode([init[row]])!r}", flush=True)
        snapshots += gen.trajectory(
            table[row]["goal"], table[row]["target"], init[row],
            steps=args.steps, snapshot_every=args.snapshot_every,
            topk=args.topk, behaviour_row=row, verbose=not args.quiet)
    save_states(snapshots, args.out)
    print(f"wrote {len(snapshots)} snapshots to {args.out}", flush=True)


if __name__ == "__main__":
    main()
