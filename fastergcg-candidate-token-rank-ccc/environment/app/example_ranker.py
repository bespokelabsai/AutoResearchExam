import numpy as np


def rank(features: dict) -> np.ndarray:
    """Preference scores for the 64 candidate substitutions.

    Lower score = predicted to reduce the loss more.

    features keys:
        grad_row          float32 (V,)   d(loss)/d(one-hot) at the optimisable
                                         position, over the whole vocabulary
        candidate_ids     int64   (64,)  ascending token ids
        current_token_id  int            the token currently at that position
        current_loss      float          loss of the current prompt
        prefix_ids        list[int]      tokens before the optimisable position
        post_ids          list[int]      tokens between it and the target
        target_ids        list[int]      the target tokens the loss is taken over
        step              int            optimisation step of this snapshot
        embedding_matrix  float32 (V, d) read-only input embeddings
    """
    return np.arange(features["candidate_ids"].shape[0], dtype=np.float64)
