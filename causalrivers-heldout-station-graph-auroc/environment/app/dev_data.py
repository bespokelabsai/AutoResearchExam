import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path("/app/data")
REGIONS = ("a", "b")
FREQ_MINUTES = 15


def load_series(region, mmap=True):
    """Return the (T, n_stations) float32 discharge matrix of a development region."""
    path = DATA_DIR / f"dev_{region}_series.npy"
    return np.load(path, mmap_mode="r" if mmap else None)


def load_edges(region):
    """Return the true directed edges of a development region as (cause, effect) pairs."""
    path = DATA_DIR / f"dev_{region}_edges.csv"
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return [(int(row["cause"]), int(row["effect"])) for row in reader]
