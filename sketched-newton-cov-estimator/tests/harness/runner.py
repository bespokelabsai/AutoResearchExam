import numpy as np
from scipy.linalg import solve_triangular

_BLOCK = 8192


def make_config(d, T, tau, w=None):
    """The `config` dict handed to `estimate`. Contains no instance secrets."""
    if w is None:
        w = np.full(d, 1.0 / d)
    return {
        "d": int(d),
        "T": int(T),
        "tau": int(tau),
        "q": 1,
        "sketch": "kaczmarz",
        "beta": 0.505,
        "c_beta": 1.0,
        "loss": "squared_error_linear_regression",
        "x0": np.zeros(d),
        "B0": np.eye(d),
        "w": np.asarray(w, dtype=np.float64),
    }


def draw_samples(instance, seed):
    """Draw the T samples of one replication from an instance's parameters.

    `instance` needs the keys d, T, sigma, x_star and either Sigma_a or (equi_r).
    Returns (A, b) with A of shape (T, d) and b of shape (T,).
    """
    d, T = int(instance["d"]), int(instance["T"])
    Sigma_a = instance_covariance(instance)
    L = np.linalg.cholesky(Sigma_a)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((T, d)) @ L.T
    eps = rng.standard_normal(T) * float(instance["sigma"])
    b = A @ np.asarray(instance["x_star"], dtype=np.float64) + eps
    return A, b


def instance_covariance(instance):
    """Equi-correlation design: [Sigma_a]_ii = 1, [Sigma_a]_ij = r for i != j."""
    d = int(instance["d"])
    r = float(instance["equi_r"])
    Sigma = np.full((d, d), r)
    np.fill_diagonal(Sigma, 1.0)
    return Sigma


def stream_from_samples(A, b, config, alg_seed):
    """Yield one record per iteration. `A` is (T, d), `b` is (T,).

    Each record is a fresh dict holding read-only arrays. The runner never writes into an
    array it has already yielded, so a record stays valid for as long as you keep it.
        t            int, the iteration index
        x            (d,)   x_{t+1}, the iterate AFTER this step's update
        alpha        float  the realized stepsize alpha_t
        g            (d,)   the stochastic gradient at x_t
        H            (d, d) the sample Hessian at x_t
        B            (d, d) the Hessian average B_t used by this step's inner solve
        sketch_idx   (tau,) the Kaczmarz indices used by this step's inner solve
    """
    d = int(config["d"])
    T = int(config["T"])
    tau = int(config["tau"])
    beta = float(config["beta"])
    c_beta = float(config["c_beta"])

    x = np.array(config["x0"], dtype=np.float64)
    B = np.array(config["B0"], dtype=np.float64)
    rng = np.random.default_rng(alg_seed)

    idx_block = None
    u_block = None
    for t in range(T):
        k = t % _BLOCK
        if k == 0:
            n = min(_BLOCK, T - t)
            idx_block = rng.integers(0, d, size=(n, tau))
            u_block = rng.random(n)

        a = A[t]
        g = a * (a @ x - b[t])

        idx = idx_block[k]
        sel = B[:, idx]
        nrm = np.sqrt(np.einsum("ij,ij->j", sel, sel))
        F = sel / nrm
        dx = F @ solve_triangular(np.tril(F.T @ F), -g[idx] / nrm, lower=True,
                                  unit_diagonal=True, check_finite=False, overwrite_b=True)

        beta_t = c_beta / (t + 1.0) ** beta
        alpha = beta_t + u_block[k] * beta_t * beta_t
        x = x + alpha * dx

        H = np.outer(a, a)
        for arr in (x, g, H, B, idx):
            arr.flags.writeable = False
        yield {
            "t": t,
            "x": x,
            "alpha": float(alpha),
            "g": g,
            "H": H,
            "B": B,
            "sketch_idx": idx,
        }

        B = ((t + 1.0) * B + H) / (t + 2.0)


def make_stream(instance, config, sample_seed, alg_seed):
    """Convenience wrapper: draw an instance's samples, then stream one replication."""
    A, b = draw_samples(instance, sample_seed)
    return stream_from_samples(A, b, config, alg_seed)
