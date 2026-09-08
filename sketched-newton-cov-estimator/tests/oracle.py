import numpy as np


def _projectors(B):
    """P[i] = b_i b_i^T / ||b_i||^2 for every column b_i of B."""
    nrm2 = np.einsum("ij,ij->j", B, B)
    return np.einsum("ik,jk->kij", B, B) / nrm2[:, None, None]


def xi_star(Sigma_a, sigma2, tau):
    """The limiting covariance of the last iterate, exactly."""
    Sigma_a = np.asarray(Sigma_a, dtype=np.float64)
    d = Sigma_a.shape[0]
    Omega = float(sigma2) * np.linalg.inv(Sigma_a)

    Q = np.eye(d)[None, :, :] - _projectors(Sigma_a)
    C = np.linalg.matrix_power(Q.mean(axis=0), int(tau))

    A = Omega.copy()
    for _ in range(int(tau)):
        A = np.einsum("kij,jl,kml->im", Q, A, Q) / d

    RHS = Omega - C @ Omega - Omega @ C + A
    RHS = 0.5 * (RHS + RHS.T)

    M = np.eye(d) - C
    M = 0.5 * (M + M.T)
    s, U = np.linalg.eigh(M)
    if s.min() <= 1e-12:
        raise ValueError("I - C* is singular; Xi* does not exist for this instance")
    Theta = 1.0 / (s[:, None] + s[None, :])
    Xi = U @ (Theta * (U.T @ RHS @ U)) @ U.T
    return 0.5 * (Xi + Xi.T)


def sandwich(Sigma_a, sigma2):
    """Omega*/2, the exact-solver (tau -> infinity) limit. Used only by build-time checks."""
    return 0.5 * float(sigma2) * np.linalg.inv(np.asarray(Sigma_a, dtype=np.float64))
