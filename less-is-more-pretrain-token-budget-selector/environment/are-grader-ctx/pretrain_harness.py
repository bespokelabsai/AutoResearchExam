from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

EOT_TOKEN = 50256



def set_determinism(seed: int) -> None:
    """Pin every source of run-to-run variation this harness can reach."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)



class Pool:
    """A token pool laid out as docs.jsonl + tokens.npy + offsets.npy."""

    def __init__(self, pool_dir: str | os.PathLike):
        self.dir = Path(pool_dir)
        self.offsets = np.load(self.dir / "offsets.npy", mmap_mode="r")
        self.tokens = np.load(self.dir / "tokens.npy", mmap_mode="r")
        self.n_docs = int(self.offsets.shape[0] - 1)
        self.ntokens = np.diff(np.asarray(self.offsets, dtype=np.int64))

    def span(self, doc_id: int) -> np.ndarray:
        a = int(self.offsets[doc_id])
        b = int(self.offsets[doc_id + 1])
        return np.asarray(self.tokens[a:b], dtype=np.int64)


def sanitize_selection(raw, ntokens: np.ndarray, budget_tokens: int):
    """Turn an untrusted return value into the accepted document list.

    Contract (identical to the one stated in the task instructions):
      * the value must be a list (or tuple) of ints;
      * ids outside [0, N) are dropped;
      * a repeat of an already-accepted id is dropped;
      * documents are walked IN THE ORDER RETURNED and accepted whenever the running
        token total plus that document's ntokens stays within budget_tokens; a document
        that would overflow is skipped, never fatal.

    Returns (accepted_ids, tokens_used, stats).  Raises ValueError when the value is not
    a list of ints at all -- that is the invalid path, not a zero-length selection.
    """
    if isinstance(raw, tuple):
        raw = list(raw)
    if not isinstance(raw, list):
        raise ValueError(f"select() must return a list of ints, got {type(raw).__name__}")
    n = int(ntokens.shape[0])
    accepted: list[int] = []
    seen: set[int] = set()
    used = 0
    n_bad_type = 0
    n_out_of_range = 0
    n_duplicate = 0
    n_overflow = 0
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            n_bad_type += 1
            continue
        i = int(item)
        if i < 0 or i >= n:
            n_out_of_range += 1
            continue
        if i in seen:
            n_duplicate += 1
            continue
        cost = int(ntokens[i])
        if used + cost > budget_tokens:
            n_overflow += 1
            continue
        seen.add(i)
        accepted.append(i)
        used += cost
    stats = {
        "n_returned": len(raw),
        "n_accepted": len(accepted),
        "tokens_used": used,
        "n_bad_type": n_bad_type,
        "n_out_of_range": n_out_of_range,
        "n_duplicate": n_duplicate,
        "n_overflow": n_overflow,
    }
    return accepted, used, stats


def build_block_stream(pool: Pool, doc_ids, block_size: int, perm_seed: int) -> np.ndarray:
    """Concatenate the selected documents in the order given, chunk into contiguous
    blocks of block_size tokens, drop the final partial block, then permute the blocks
    once with numpy.random.default_rng(perm_seed)."""
    if len(doc_ids) == 0:
        return np.zeros((0, block_size), dtype=np.uint16)
    total = int(sum(int(pool.ntokens[i]) for i in doc_ids))
    flat = np.empty(total, dtype=np.uint16)
    at = 0
    for i in doc_ids:
        a = int(pool.offsets[i])
        b = int(pool.offsets[i + 1])
        k = b - a
        flat[at:at + k] = pool.tokens[a:b]
        at += k
    n_blocks = total // block_size
    if n_blocks == 0:
        return np.zeros((0, block_size), dtype=np.uint16)
    blocks = flat[: n_blocks * block_size].reshape(n_blocks, block_size)
    perm = np.random.default_rng(perm_seed).permutation(n_blocks)
    return np.ascontiguousarray(blocks[perm])


def load_eval_blocks(eval_dir: str | os.PathLike, block_size: int) -> np.ndarray:
    """The evaluation shard, already concatenated document-by-document at build time
    (each document followed by one EOT), chunked into contiguous blocks with the final
    partial block dropped.  No permutation: the evaluation set is a fixed set."""
    toks = np.load(Path(eval_dir) / "tokens.npy", mmap_mode="r")
    n_blocks = int(toks.shape[0]) // block_size
    arr = np.asarray(toks[: n_blocks * block_size], dtype=np.uint16)
    return arr.reshape(n_blocks, block_size)



class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.c_attn = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=cfg["bias"])
        self.c_proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=cfg["bias"])
        self.c_proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg["n_embd"], bias=cfg["bias"])
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg["n_embd"], bias=cfg["bias"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.wpe = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"], bias=cfg["bias"])
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * cfg["n_layer"])
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        return self.lm_head(self.ln_f(x))


def n_parameters(model: GPT) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total



def _lr_at(step: int, cfg) -> float:
    warm = cfg["warmup_steps"]
    if step < warm:
        return cfg["learning_rate"] * (step + 1) / warm
    prog = (step - warm) / max(1, cfg["max_steps"] - warm)
    prog = min(1.0, max(0.0, prog))
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])


def train_model(cfg, blocks: np.ndarray, device: str = "cuda", log=print) -> GPT:
    """Train a fresh model from random initialisation on `blocks`, consumed cyclically.

    Deterministic: seeded init, seeded data order, deterministic kernels, no TF32.
    """
    set_determinism(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = GPT(cfg).to(device)
    log(f"[train] parameters={n_parameters(model)}")

    decay, nodecay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"]), eps=1e-8, fused=False,
    )

    micro = cfg["micro_batch_size"]
    seqs_per_step = cfg["tokens_per_step"] // cfg["block_size"]
    assert seqs_per_step % micro == 0, "tokens_per_step/block_size must be a multiple of micro_batch_size"
    accum = seqs_per_step // micro
    n_blocks = int(blocks.shape[0])
    if n_blocks == 0:
        raise ValueError("empty training stream")

    cursor = 0
    model.train()
    for step in range(cfg["max_steps"]):
        lr = _lr_at(step, cfg)
        for group in optim.param_groups:
            group["lr"] = lr
        optim.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = (cursor + np.arange(micro)) % n_blocks
            cursor = (cursor + micro) % n_blocks
            batch = torch.from_numpy(blocks[idx].astype(np.int64)).to(device, non_blocking=False)
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)
                )
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optim.step()
        if step % 100 == 0 or step == cfg["max_steps"] - 1:
            log(f"[train] step {step}/{cfg['max_steps']} lr {lr:.3e} loss {loss.item():.4f}")
    return model


@torch.no_grad()
def eval_perplexity(model: GPT, eval_blocks: np.ndarray, cfg, device: str = "cuda"):
    """Token-weighted mean cross-entropy over predicted positions 1..block_size-1.

    Position 0 of each block has no context and is excluded.  Perplexity is
    exp(total_ce / total_predicted_positions) -- one mean over the whole shard, not a
    mean of per-block perplexities.
    """
    model.eval()
    micro = cfg["micro_batch_size"]
    total_ce = 0.0
    total_pos = 0
    for start in range(0, int(eval_blocks.shape[0]), micro):
        chunk = eval_blocks[start:start + micro]
        batch = torch.from_numpy(chunk.astype(np.int64)).to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum"
        )
        total_ce += float(ce.item())
        total_pos += int(y.numel())
    if total_pos == 0:
        raise ValueError("empty evaluation set")
    mean_ce = total_ce / total_pos
    return {"total_ce": total_ce, "n_positions": total_pos,
            "mean_ce": mean_ce, "perplexity": math.exp(mean_ce)}


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_cycle(cfg, pool: Pool, doc_ids, eval_blocks: np.ndarray, device="cuda", log=print):
    """One full train-then-evaluate cycle over the given document ids."""
    blocks = build_block_stream(pool, doc_ids, cfg["block_size"], cfg["block_permutation_seed"])
    log(f"[cycle] docs={len(doc_ids)} blocks={blocks.shape[0]}")
    model = train_model(cfg, blocks, device=device, log=log)
    out = eval_perplexity(model, eval_blocks, cfg, device=device)
    del model
    torch.cuda.empty_cache()
    out["n_blocks"] = int(blocks.shape[0])
    out["n_docs"] = len(doc_ids)
    log(f"[cycle] perplexity={out['perplexity']:.6f}")
    return out
