from __future__ import annotations

import json
import math
import os

import torch
from safetensors.torch import load_file

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

_BLOCK_KEYS = (
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
)


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """The tanh-approximated GELU the checkpoint was trained with."""
    return 0.5 * x * (1.0 + torch.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


class Engine:
    """Serves generation requests one at a time."""

    def __init__(self, model_dir: str) -> None:
        with open(os.path.join(model_dir, "config.json")) as fh:
            cfg = json.load(fh)
        self.n_layer = int(cfg["n_layer"])
        self.n_head = int(cfg["n_head"])
        self.n_embd = int(cfg["n_embd"])
        self.n_positions = int(cfg["n_positions"])
        self.eps = float(cfg["layer_norm_epsilon"])
        self.head_dim = self.n_embd // self.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        sd = load_file(os.path.join(model_dir, "model.safetensors"))
        self.wte = sd["wte.weight"].float().contiguous()
        self.wpe = sd["wpe.weight"].float().contiguous()
        self.ln_f_w = sd["ln_f.weight"].float().contiguous()
        self.ln_f_b = sd["ln_f.bias"].float().contiguous()
        self.blocks = [
            {k: sd[f"h.{i}.{k}"].float().contiguous() for k in _BLOCK_KEYS}
            for i in range(self.n_layer)
        ]


    def _forward(self, ids: torch.Tensor, past: list, start_pos: int) -> torch.Tensor:
        """One forward pass over `ids` ([1, T]); returns logits for the last position."""
        bsz, seq = ids.shape
        pos = torch.arange(start_pos, start_pos + seq, dtype=torch.long)
        h = self.wte[ids] + self.wpe[pos]

        past_len = 0 if past[0] is None else past[0][0].shape[2]
        total = past_len + seq
        mask = None
        if seq > 1:
            allowed = torch.arange(total)[None, :] <= torch.arange(
                past_len, past_len + seq
            )[:, None]
            mask = torch.zeros(seq, total)
            mask.masked_fill_(~allowed, float("-inf"))

        for i, b in enumerate(self.blocks):
            x = torch.nn.functional.layer_norm(
                h, (self.n_embd,), b["ln_1.weight"], b["ln_1.bias"], self.eps
            )
            qkv = torch.addmm(
                b["attn.c_attn.bias"],
                x.view(bsz * seq, self.n_embd),
                b["attn.c_attn.weight"],
            ).view(bsz, seq, 3 * self.n_embd)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = q.view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
            if past[i] is not None:
                k = torch.cat((past[i][0], k), dim=2)
                v = torch.cat((past[i][1], v), dim=2)
            past[i] = (k, v)

            att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                att = att + mask
            att = torch.softmax(att, dim=-1)
            y = torch.matmul(att, v)
            y = y.transpose(1, 2).reshape(bsz * seq, self.n_embd)
            y = torch.addmm(b["attn.c_proj.bias"], y, b["attn.c_proj.weight"])
            h = h + y.view(bsz, seq, self.n_embd)

            x = torch.nn.functional.layer_norm(
                h, (self.n_embd,), b["ln_2.weight"], b["ln_2.bias"], self.eps
            )
            x = torch.addmm(
                b["mlp.c_fc.bias"], x.view(bsz * seq, self.n_embd), b["mlp.c_fc.weight"]
            )
            x = gelu_new(x)
            x = torch.addmm(b["mlp.c_proj.bias"], x, b["mlp.c_proj.weight"])
            h = h + x.view(bsz, seq, self.n_embd)

        h = torch.nn.functional.layer_norm(
            h[:, -1, :], (self.n_embd,), self.ln_f_w, self.ln_f_b, self.eps
        )
        return torch.matmul(h, self.wte.t())


    @torch.inference_mode()
    def generate(self, requests: list) -> list:
        """Serve every request; returns one list of new token ids per request, in order."""
        out = []
        for req in requests:
            prompt = list(req["prompt_token_ids"])
            n_new = int(req["max_new_tokens"])
            past = [None] * self.n_layer
            ids = torch.tensor([prompt], dtype=torch.long)
            logits = self._forward(ids, past, 0)
            produced = []
            pos = len(prompt)
            for _ in range(n_new):
                nxt = int(torch.argmax(logits[0]).item())
                produced.append(nxt)
                if len(produced) == n_new:
                    break
                logits = self._forward(
                    torch.tensor([[nxt]], dtype=torch.long), past, pos
                )
                pos += 1
            out.append(produced)
        return out
