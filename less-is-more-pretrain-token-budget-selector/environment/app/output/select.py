import numpy as np


def select(pool_dir: str, budget_tokens: int, workdir: str) -> list[int]:
    offsets = np.load(f"{pool_dir}/offsets.npy", mmap_mode="r")
    n_docs = int(offsets.shape[0]) - 1
    return list(range(n_docs))
