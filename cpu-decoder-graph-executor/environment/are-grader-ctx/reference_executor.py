import numpy as np


def _rmsnorm(x, w, eps, out, sq):
    np.square(x, out=sq)
    ss = np.mean(sq, axis=-1, keepdims=True)
    np.multiply(x, np.reciprocal(np.sqrt(ss + np.float32(eps))), out=out)
    np.multiply(out, w, out=out)
    return out


def _rope(t, cos, sin, out):
    """Rotary embedding over the last axis, applied per head.  t: (H, T, head_dim)."""
    half = t.shape[-1] // 2
    a = t[..., :half]
    b = t[..., half:]
    np.subtract(a * cos, b * sin, out=out[..., :half])
    np.add(a * sin, b * cos, out=out[..., half:])
    return out


class _Executor:
    def __init__(self, spec, weights, n_workers):
        self.spec = spec
        self.w = weights
        self.n_workers = int(n_workers)

        T = spec["T"]
        d = spec["d_model"]
        d_ff = spec["d_ff"]
        hd = spec["head_dim"]
        H = spec["n_heads"]
        KV = spec["n_kv_heads"]
        self.eps = float(spec["rms_eps"])
        self.scale = np.float32(hd ** -0.5)
        self.rep = H // KV

        self.buf = {
            "h": np.empty((T, d), dtype=np.float32),
            "n1": np.empty((T, d), dtype=np.float32),
            "n2": np.empty((T, d), dtype=np.float32),
            "q": np.empty((T, H * hd), dtype=np.float32),
            "k": np.empty((T, KV * hd), dtype=np.float32),
            "v": np.empty((T, KV * hd), dtype=np.float32),
            "att": np.empty((T, H * hd), dtype=np.float32),
            "ao": np.empty((T, d), dtype=np.float32),
            "g": np.empty((T, d_ff), dtype=np.float32),
            "u": np.empty((T, d_ff), dtype=np.float32),
            "f": np.empty((T, d_ff), dtype=np.float32),
            "fo": np.empty((T, d), dtype=np.float32),
        }
        self.qr = np.empty((H, T, hd), dtype=np.float32)
        self.kr = np.empty((KV, T, hd), dtype=np.float32)
        self.kkT = np.empty((H, hd, T), dtype=np.float32)
        self.vv = np.empty((H, T, hd), dtype=np.float32)
        self.scores = np.empty((H, T, T), dtype=np.float32)
        self.probs = np.empty((H, T, T), dtype=np.float32)
        self.ctx = np.empty((H, T, hd), dtype=np.float32)
        self.sq = np.empty((T, d), dtype=np.float32)

        self.cos = weights["rope_cos"]
        self.sin = weights["rope_sin"]
        self.mask = weights["attn_mask"]

        self.plan = []
        for layer in spec["layers"]:
            ops = []
            for node in layer:
                w = weights[node["weight"]] if node["weight"] else None
                ops.append((node["op"], node["out"], node["inputs"], w))
            self.plan.append(ops)
        self.final_w = weights[spec["final"]["weight"]]

    def _attention(self, q, k, v, out):
        H, T, hd = self.qr.shape
        KV = self.kr.shape[0]
        _rope(q.reshape(T, H, hd).transpose(1, 0, 2), self.cos, self.sin, self.qr)
        _rope(k.reshape(T, KV, hd).transpose(1, 0, 2), self.cos, self.sin, self.kr)
        self.kkT.reshape(KV, self.rep, hd, T)[...] = self.kr.transpose(0, 2, 1)[:, None]
        self.vv.reshape(KV, self.rep, T, hd)[...] = \
            v.reshape(T, KV, hd).transpose(1, 0, 2)[:, None]
        np.matmul(self.qr, self.kkT, out=self.scores)
        np.multiply(self.scores, self.scale, out=self.scores)
        np.add(self.scores, self.mask, out=self.scores)
        np.subtract(self.scores, np.max(self.scores, axis=-1, keepdims=True),
                    out=self.probs)
        np.exp(self.probs, out=self.probs)
        np.divide(self.probs, np.sum(self.probs, axis=-1, keepdims=True), out=self.probs)
        np.matmul(self.probs, self.vv, out=self.ctx)
        out.reshape(T, H, hd)[...] = self.ctx.transpose(1, 0, 2)
        return out

    def __call__(self, x):
        buf = self.buf
        h = buf["h"]
        h[...] = x
        for ops in self.plan:
            for op, out_name, inp, W in ops:
                if op == "matmul":
                    np.matmul(buf[inp[0]], W, out=buf[out_name])
                elif op == "rmsnorm":
                    _rmsnorm(buf[inp[0]], W, self.eps, buf[out_name], self.sq)
                elif op == "attention":
                    self._attention(buf[inp[0]], buf[inp[1]], buf[inp[2]], buf[out_name])
                elif op == "add":
                    np.add(buf[inp[0]], buf[inp[1]], out=buf[out_name])
                elif op == "swiglu":
                    g = buf[inp[0]]
                    f = buf[out_name]
                    np.negative(g, out=f)
                    np.exp(f, out=f)
                    np.add(f, np.float32(1.0), out=f)
                    np.divide(g, f, out=f)
                    np.multiply(f, buf[inp[1]], out=f)
                else:
                    raise ValueError("unknown op %r" % op)
        return _rmsnorm(h, self.final_w, self.eps, buf["n1"], self.sq)


def build_executor(graph_spec, weights, n_workers):
    """Build an executor for `graph_spec`.  See the module docstring for the contract."""
    return _Executor(graph_spec, weights, n_workers)
