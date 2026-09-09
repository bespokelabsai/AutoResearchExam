import numpy as np

N_BLOCKS = 16
N_FLAGS = 4
TILE_CHANNELS = N_BLOCKS + N_FLAGS
CROP = 5
VIEW_DIM = CROP * CROP * TILE_CHANNELS
EXTRA_DIM = 14
OBS_DIM = VIEW_DIM + EXTRA_DIM
N_ACTIONS = 24
N_EXTRAS_RAW = 12

BLOCK_VOID = 0
BLOCK_FLOOR = 1
BLOCK_WALL = 2
BLOCK_STONE = 3
BLOCK_TREE = 4
BLOCK_COAL = 5
BLOCK_IRON = 6
BLOCK_GEM = 7
BLOCK_WATER = 8
BLOCK_LAVA = 9
BLOCK_TABLE = 10
BLOCK_FURNACE = 11
BLOCK_DOWN_LADDER = 12
BLOCK_UP_LADDER = 13
BLOCK_PLANT = 14
BLOCK_SAND = 15

FLAG_MELEE = 1
FLAG_PASSIVE = 2
FLAG_RANGED = 4
FLAG_ITEM = 8

MAX_STEPS = 500


def build_obs_batch(crop_block, crop_flags, extras_raw):
    """Vectorised encoder over (N, 25) uint8, (N, 25) uint8 and (N, 12) uint16 inputs.

    Returns a float32 array of shape (N, 514).
    """
    crop_block = np.asarray(crop_block, dtype=np.uint8)
    crop_flags = np.asarray(crop_flags, dtype=np.uint8)
    extras_raw = np.asarray(extras_raw, dtype=np.uint16)
    n = crop_block.shape[0]

    view = np.zeros((n, CROP * CROP, TILE_CHANNELS), dtype=np.float32)
    rows = np.arange(n)[:, None]
    cols = np.arange(CROP * CROP)[None, :]
    view[rows, cols, crop_block.astype(np.intp)] = 1.0
    for bit in range(N_FLAGS):
        view[:, :, N_BLOCKS + bit] = ((crop_flags >> bit) & 1).astype(np.float32)

    e = extras_raw.astype(np.float32)
    flags = extras_raw[:, 10].astype(np.uint16)
    extras = np.empty((n, EXTRA_DIM), dtype=np.float32)
    extras[:, 0:5] = np.sqrt(e[:, 0:5]) / 10.0
    extras[:, 5] = e[:, 5] / 3.0
    extras[:, 6] = e[:, 6] / 3.0
    extras[:, 7] = e[:, 7] / 10.0
    extras[:, 8] = e[:, 8] / 10.0
    extras[:, 9] = np.sqrt(e[:, 9]) / 10.0
    extras[:, 10] = (flags & 1).astype(np.float32)
    extras[:, 11] = ((flags >> 1) & 1).astype(np.float32)
    extras[:, 12] = ((flags >> 2) & 1).astype(np.float32)
    extras[:, 13] = e[:, 11] / float(MAX_STEPS)

    return np.concatenate([view.reshape(n, VIEW_DIM), extras], axis=1)


def build_obs(crop_block, crop_flags, extras_raw):
    """Single-observation encoder. Returns a float32 array of shape (514,)."""
    return build_obs_batch(
        np.asarray(crop_block, dtype=np.uint8)[None, :],
        np.asarray(crop_flags, dtype=np.uint8)[None, :],
        np.asarray(extras_raw, dtype=np.uint16)[None, :],
    )[0]
