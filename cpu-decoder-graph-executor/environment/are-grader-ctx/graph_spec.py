import numpy as np

D_MODEL_CHOICES = (192, 256, 384)
FF_RATIO_CHOICES = (2.6875, 4.0)
N_LAYER_CHOICES = (16, 32)
N_HEAD_CHOICES = (4, 8)
KV_DIV_CHOICES = (1, 2, 4)
T_CHOICES = (1, 1, 1, 8, 8, 32)

RMS_EPS = 1e-5
ROPE_THETA = 10000.0


def sample_spec(seed):
    """Return the graph_spec dict for `seed`."""
    rng = np.random.default_rng(seed)
    d_model = int(rng.choice(D_MODEL_CHOICES))
    ratio = float(rng.choice(FF_RATIO_CHOICES))
    d_ff = int(round(d_model * ratio / 32.0)) * 32
    n_layers = int(rng.choice(N_LAYER_CHOICES))
    n_heads = int(rng.choice(N_HEAD_CHOICES))
    head_dim = d_model // n_heads
    n_kv_heads = n_heads // int(rng.choice(KV_DIV_CHOICES))
    T = int(rng.choice(T_CHOICES))

    layers = []
    for li in range(n_layers):
        p = "blk.%d." % li
        layers.append([
            {"op": "rmsnorm", "out": "n1", "inputs": ["h"], "weight": p + "attn_norm"},
            {"op": "matmul", "out": "q", "inputs": ["n1"], "weight": p + "attn_q"},
            {"op": "matmul", "out": "k", "inputs": ["n1"], "weight": p + "attn_k"},
            {"op": "matmul", "out": "v", "inputs": ["n1"], "weight": p + "attn_v"},
            {"op": "attention", "out": "att", "inputs": ["q", "k", "v"], "weight": None},
            {"op": "matmul", "out": "ao", "inputs": ["att"], "weight": p + "attn_out"},
            {"op": "add", "out": "h", "inputs": ["h", "ao"], "weight": None},
            {"op": "rmsnorm", "out": "n2", "inputs": ["h"], "weight": p + "ffn_norm"},
            {"op": "matmul", "out": "g", "inputs": ["n2"], "weight": p + "ffn_gate"},
            {"op": "matmul", "out": "u", "inputs": ["n2"], "weight": p + "ffn_up"},
            {"op": "swiglu", "out": "f", "inputs": ["g", "u"], "weight": None},
            {"op": "matmul", "out": "fo", "inputs": ["f"], "weight": p + "ffn_down"},
            {"op": "add", "out": "h", "inputs": ["h", "fo"], "weight": None},
        ])

    return {
        "seed": int(seed),
        "n_layers": n_layers,
        "d_model": d_model,
        "d_ff": d_ff,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "T": T,
        "rms_eps": RMS_EPS,
        "layers": layers,
        "final": {"op": "rmsnorm", "out": "h", "inputs": ["h"], "weight": "output_norm"},
    }


def _normal(rng, shape, scale):
    return (rng.standard_normal(shape, dtype=np.float32) * np.float32(scale))


def build_weights(spec):
    """Return the tensor dict for `spec`.  All arrays are C-contiguous float32."""
    rng = np.random.default_rng(spec["seed"] + 1_000_003)
    d = spec["d_model"]
    d_ff = spec["d_ff"]
    hd = spec["head_dim"]
    n_q = spec["n_heads"] * hd
    n_kv = spec["n_kv_heads"] * hd
    T = spec["T"]

    w = {}
    for li in range(spec["n_layers"]):
        p = "blk.%d." % li
        w[p + "attn_norm"] = np.ascontiguousarray(1.0 + 0.02 * _normal(rng, (d,), 1.0))
        w[p + "attn_q"] = np.ascontiguousarray(_normal(rng, (d, n_q), d ** -0.5))
        w[p + "attn_k"] = np.ascontiguousarray(_normal(rng, (d, n_kv), d ** -0.5))
        w[p + "attn_v"] = np.ascontiguousarray(_normal(rng, (d, n_kv), d ** -0.5))
        w[p + "attn_out"] = np.ascontiguousarray(_normal(rng, (n_q, d), n_q ** -0.5))
        w[p + "ffn_norm"] = np.ascontiguousarray(1.0 + 0.02 * _normal(rng, (d,), 1.0))
        w[p + "ffn_gate"] = np.ascontiguousarray(_normal(rng, (d, d_ff), d ** -0.5))
        w[p + "ffn_up"] = np.ascontiguousarray(_normal(rng, (d, d_ff), d ** -0.5))
        w[p + "ffn_down"] = np.ascontiguousarray(_normal(rng, (d_ff, d), d_ff ** -0.5))
    w["output_norm"] = np.ascontiguousarray(1.0 + 0.02 * _normal(rng, (d,), 1.0))

    half = hd // 2
    inv = (ROPE_THETA ** (-np.arange(half, dtype=np.float64) / half))
    ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
    w["rope_cos"] = np.ascontiguousarray(np.cos(ang).astype(np.float32))
    w["rope_sin"] = np.ascontiguousarray(np.sin(ang).astype(np.float32))

    mask = np.zeros((T, T), dtype=np.float32)
    mask[np.triu_indices(T, k=1)] = -np.inf
    w["attn_mask"] = np.ascontiguousarray(mask)
    return w


def build_inputs(spec, n):
    """Return `n` distinct input activations of shape (T, d_model), float32.

    Drawn from fresh OS entropy, never from the instance seed: the same list is fed
    to both executors within a run, but no executor can precompute the output for an
    input it has not yet been sent.
    """
    rng = np.random.default_rng()
    T, d = spec["T"], spec["d_model"]
    return [np.ascontiguousarray(rng.standard_normal((T, d), dtype=np.float32))
            for _ in range(n)]
