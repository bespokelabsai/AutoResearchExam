from __future__ import annotations

import math

import numpy as np

Z_FAMILIES = ("dirichlet", "dirichlet_sparse", "logitnormal")
PERT_FAMILIES = ("mono", "fourier", "mixed")




def draw_z(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n rows of predicted probability vectors, shape (n, K), rows sum to 1."""
    K = int(spec["K"])
    fam = spec["z_family"]
    w_unif = float(fam["w_unif"])
    is_unif = rng.random(n) < w_unif
    out = np.empty((n, K), dtype=np.float64)

    n_u = int(is_unif.sum())
    if n_u:
        out[is_unif] = rng.dirichlet(np.ones(K), size=n_u)
    n_c = n - n_u
    if n_c:
        kind = fam["kind"]
        if kind in ("dirichlet", "dirichlet_sparse"):
            alpha = np.asarray(fam["alpha"], dtype=np.float64)
            out[~is_unif] = rng.dirichlet(alpha, size=n_c)
        elif kind == "logitnormal":
            mu = np.asarray(fam["mu"], dtype=np.float64)
            sigma = float(fam["sigma"])
            v = mu + sigma * rng.standard_normal((n_c, K))
            v -= v.max(axis=1, keepdims=True)
            np.exp(v, out=v)
            out[~is_unif] = v / v.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"unknown z family {kind!r}")
    return out




def perturbation(spec: dict, zt: np.ndarray) -> np.ndarray:
    """Smooth shape function h(z_(1:k)) with values in [0, 1], shape (n, k)."""
    p = spec["pert"]
    kind = p["kind"]
    if kind == "mono":
        b = np.asarray(p["b"], dtype=np.float64)
        c = np.asarray(p["c"], dtype=np.float64)
        return 0.5 * (1.0 + np.tanh(b * (zt - c)))
    if kind == "fourier":
        freq = np.asarray(p["freq"], dtype=np.float64)
        phase = np.asarray(p["phase"], dtype=np.float64)
        return 0.5 * (1.0 + np.sin(2.0 * np.pi * (zt @ freq.T + phase)))
    if kind == "mixed":
        b = np.asarray(p["b"], dtype=np.float64)
        c = np.asarray(p["c"], dtype=np.float64)
        freq = np.asarray(p["freq"], dtype=np.float64)
        phase = np.asarray(p["phase"], dtype=np.float64)
        mono = 0.5 * (1.0 + np.tanh(b * (zt - c)))
        four = 0.5 * (1.0 + np.sin(2.0 * np.pi * (zt @ freq.T + phase)))
        return 0.5 * mono + 0.5 * four
    raise ValueError(f"unknown perturbation family {kind!r}")


def _top_k(Z: np.ndarray, k: int):
    order = np.argsort(-Z, axis=1, kind="stable")
    zt = np.take_along_axis(Z, order[:, :k], axis=1)
    return order, zt


def scale_factor(zt: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Per-rank cap c_j(z), shape (n, k).

    Understating rank j (sigma_j = -1) may take away at most its own probability z_j;
    overstating it may add at most an equal share r = (1 - s)/k of the mass the
    non-top classes hold.  Both caps hold simultaneously for every eps in [0, 1], so
    eta is always a valid conditional distribution.
    """
    r = np.maximum((1.0 - zt.sum(axis=1)) / zt.shape[1], 0.0)[:, None]
    return np.where(signs[None, :] < 0.0, zt, np.broadcast_to(r, zt.shape))


def a_integrand(spec: dict, Z: np.ndarray) -> np.ndarray:
    """The integrand whose mean is A: sum_j c_j(z)^2 h_j(z)^2."""
    _, zt = _top_k(Z, int(spec["k"]))
    h = perturbation(spec, zt)
    c = scale_factor(zt, np.asarray(spec["pert"]["sign"], dtype=np.float64))
    return ((c * h) ** 2).sum(axis=1)


def estimate_a(spec: dict, n_draws: int, seed: int, block: int = 500_000):
    """Monte-Carlo estimate of A and its standard error, with a fixed seed."""
    rng = np.random.default_rng(seed)
    total = 0.0
    total_sq = 0.0
    done = 0
    while done < n_draws:
        m = min(block, n_draws - done)
        vals = a_integrand(spec, draw_z(spec, m, rng))
        total += float(vals.sum())
        total_sq += float((vals * vals).sum())
        done += m
    mean = total / n_draws
    var = max(total_sq / n_draws - mean * mean, 0.0)
    return mean, float(np.sqrt(var / n_draws))


def sigma0_sq(K: int, k: int, n_draws: int = 4_000_000, seed: int = 20240819):
    """sigma_0^2 = 2 * int_{Delta(K,k)} (||z||_2^2 - 2||z||_3^3 + ||z||_2^4) dz.

    Monte Carlo over the enclosing box [0,1]^k restricted to the truncated Weyl
    chamber; the Lebesgue measure of the region is estimated by the same draws.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    total_sq = 0.0
    count = 0
    done = 0
    block = 200_000
    while done < n_draws:
        m = min(block, n_draws - done)
        z = rng.random((m, k))
        z = -np.sort(-z, axis=1)
        s = z.sum(axis=1)
        keep = (s >= k / K) & (s <= 1.0)
        zz = z[keep]
        val = (zz * zz).sum(1) - 2.0 * (zz**3).sum(1) + ((zz * zz).sum(1)) ** 2
        total += float(val.sum())
        total_sq += float((val * val).sum())
        count += int(keep.sum())
        done += m
    fact = float(math.factorial(k))
    volume = count / n_draws / fact
    integral = 2.0 * (total / n_draws) / fact
    se = 2.0 * float(np.sqrt(max(total_sq / n_draws - (total / n_draws) ** 2, 0.0)) / np.sqrt(n_draws)) / fact
    return integral, volume, se




def draw_replication(spec: dict, theta: float, rng: np.random.Generator):
    """Draw one calibration data set whose true ECE^2_{1:k} equals `theta`.

    Returns (Z, Y) with Z of shape (n, K), C-contiguous float64, rows summing to 1,
    and Y of shape (n,), int64, holding realised class indices in [0, K).
    """
    K = int(spec["K"])
    k = int(spec["k"])
    n = int(spec["n"])
    A = float(spec["A"])
    theta = float(theta)
    eps = 0.0 if theta <= 0.0 else float(np.sqrt(theta / A))
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"eps={eps} outside [0, 1]; theta={theta} too large for A={A}")

    Z = draw_z(spec, n, rng)
    order, zt = _top_k(Z, k)
    signs = np.asarray(spec["pert"]["sign"], dtype=np.float64)
    h = perturbation(spec, zt)
    eta_top = zt + eps * signs[None, :] * scale_factor(zt, signs) * h

    P = np.zeros((n, K), dtype=np.float64)
    rows = np.arange(n)
    for j in range(k):
        P[rows, order[:, j]] = eta_top[:, j]
    rest_idx = order[:, k:]
    z_rest = np.take_along_axis(Z, rest_idx, axis=1)
    denom = z_rest.sum(axis=1)
    frac = np.where(
        denom[:, None] > 0.0,
        z_rest / np.maximum(denom, 1e-300)[:, None],
        1.0 / max(K - k, 1),
    )
    remainder = np.maximum(1.0 - eta_top.sum(axis=1), 0.0)
    np.put_along_axis(P, rest_idx, frac * remainder[:, None], axis=1)

    np.clip(P, 0.0, 1.0, out=P)
    P /= P.sum(axis=1, keepdims=True)
    cdf = np.cumsum(P, axis=1)
    u = rng.random((n, 1))
    Y = (u > cdf).sum(axis=1).astype(np.int64)
    np.clip(Y, 0, K - 1, out=Y)
    return np.ascontiguousarray(Z), Y


def theta_for(spec: dict, rep_rng: np.random.Generator) -> float:
    """Pick this replication's true squared calibration error from the setting grid."""
    grid = np.asarray(spec["theta_grid"], dtype=np.float64)
    return float(grid[rep_rng.integers(grid.size)])
