import numpy as np

N_ACTIONS = 24
OBS_DIM = 514


class ReferencePolicy:
    """Loads the frozen weights and evaluates the policy's pre-mask logits."""

    def __init__(self, weights_path):
        z = np.load(weights_path)
        self.layers = []
        i = 1
        while ("W%d" % i) in z:
            self.layers.append((z["W%d" % i].astype(np.float32), z["b%d" % i].astype(np.float32)))
            i += 1
        if not self.layers:
            raise ValueError("no layers found in %s" % weights_path)

    def logits_batch(self, obs):
        """Pre-mask logits for a batch. obs is (N, 514) float32; returns (N, 24) float32."""
        h = np.asarray(obs, dtype=np.float32)
        if h.ndim != 2 or h.shape[1] != OBS_DIM:
            raise ValueError("expected (N, %d) observations" % OBS_DIM)
        for k, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if k + 1 < len(self.layers):
                np.maximum(h, 0.0, out=h)
        return h

    def logits(self, obs):
        """Pre-mask logits for one observation. obs is (514,); returns (24,) float32."""
        return self.logits_batch(np.asarray(obs, dtype=np.float32)[None, :])[0]

    def masked_probs(self, obs, mask):
        """Action distribution after forcing the logits of masked-out actions to -inf.

        `mask` is a length-24 boolean array; True means the action is left available.
        Returns a (24,) float32 array that sums to 1, or all zeros if `mask` is empty.
        """
        mask = np.asarray(mask, dtype=bool)
        z = self.logits(obs)
        if not mask.any():
            return np.zeros(N_ACTIONS, dtype=np.float32)
        z = np.where(mask, z, -np.inf)
        z = z - z.max()
        p = np.exp(z)
        return (p / p.sum()).astype(np.float32)
