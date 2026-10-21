from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

POOL_SIZE = 12_500
KEEP = 3_125
LABEL_BUDGET = 125
N_ITERATIONS = 5


class InvalidSelection(ValueError):
    """The pruner returned something the protocol does not allow."""


class _ConstantModel:
    """Fallback for a labelled set that contains a single class."""

    def __init__(self, label, classes):
        self.label = int(label)
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        return np.full(len(X), self.label, dtype=np.int64)

    def predict_proba(self, X):
        p = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        p[:, int(np.searchsorted(self.classes_, self.label))] = 1.0
        return p


def fit_model(X, y):
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return _ConstantModel(y[0], np.unique(y))
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=0)
    clf.fit(X, y)
    return clf


def macro_f1(y_true, y_pred):
    """Macro-averaged F1, in percent."""
    return 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)


def validate_selection(selection, available_set, keep):
    """Normalise and check one pruner return value.  Raises InvalidSelection."""
    if isinstance(selection, np.ndarray):
        if selection.ndim != 1 or not np.issubdtype(selection.dtype, np.integer):
            raise InvalidSelection("numpy return must be a 1-D integer array")
        items = selection.tolist()
    elif isinstance(selection, (list, tuple)):
        items = list(selection)
    else:
        raise InvalidSelection(f"return type {type(selection).__name__} is not accepted")
    out = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise InvalidSelection("selection contains a non-integer element")
        out.append(int(value))
    if not 1 <= len(out) <= keep:
        raise InvalidSelection(f"selection length {len(out)} outside [1, {keep}]")
    if len(set(out)) != len(out):
        raise InvalidSelection("selection contains duplicate indices")
    if not available_set.issuperset(out):
        raise InvalidSelection("selection contains an index that is not available")
    return np.asarray(out, dtype=np.int64)


def run_al(pool_labels, pool_X, eval_y, eval_X, seed, call_prune,
           keep=KEEP, budget=LABEL_BUDGET, iterations=N_ITERATIONS):
    """Run one seeded end-to-end AL loop.  `call_prune` takes
    (available, labeled_indices, labeled_labels, keep, iteration, rng_seed) and returns the
    retained pool indices.  Returns the per-iteration macro-F1 list."""
    n = len(pool_labels)
    rng = np.random.default_rng(seed)
    labeled: list[int] = []
    model = None
    curve = []

    for iteration in range(iterations):
        labeled_set = set(labeled)
        available = [i for i in range(n) if i not in labeled_set]
        selection = call_prune(available, list(labeled),
                               [int(pool_labels[i]) for i in labeled],
                               keep, iteration, int(seed))
        sel = np.sort(validate_selection(selection, set(available), keep))

        take = min(budget, len(sel))
        if model is None:
            newly = sel[rng.permutation(len(sel))[:take]]
        else:
            confidence = model.predict_proba(pool_X[sel]).max(axis=1)
            order = np.lexsort((sel, confidence))
            newly = sel[order[:take]]

        labeled.extend(int(i) for i in newly)
        model = fit_model(pool_X[labeled], pool_labels[labeled])
        curve.append(macro_f1(eval_y, model.predict(eval_X)))

    return curve
