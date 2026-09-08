from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

import numpy as np


MODELS = {
    "pretrained": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
    "sun397": ("tanganke/clip-vit-large-patch14_sun397", "3b814ef1bd89d1f4044018d41a09bf468a5e4d34"),
    "cars": ("tanganke/clip-vit-large-patch14_stanford-cars", "e863c1cb4999e1f630868d349de346db775e8242"),
    "resisc45": ("tanganke/clip-vit-large-patch14_resisc45", "18076dfd48e5a590057ded82835288bfa3f27ff9"),
    "eurosat": ("tanganke/clip-vit-large-patch14_eurosat", "61f5f8b50e9804c9f2b6482fd421d44fba7ffb66"),
    "svhn": ("tanganke/clip-vit-large-patch14_svhn", "c43d85753b6aecd2435acca76bcc8aae8f8061b2"),
    "gtsrb": ("tanganke/clip-vit-large-patch14_gtsrb", "6eacfe7af7432984b5352a676735d00e5a1ae9c4"),
    "mnist": ("tanganke/clip-vit-large-patch14_mnist", "721ec5961afb977071c4124d43cf7f5fa4ba76af"),
    "dtd": ("tanganke/clip-vit-large-patch14_dtd", "c55d192f53fe864a1955168593b7f03ab474daa4"),
}

DATASETS = {
    "sun397": ("tanganke/sun397", "9c4afd6174afa3974f209a0ef099dbbe244dfd27", "data/test-*.parquet"),
    "cars": ("tanganke/stanford_cars", "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40", "data/test-*.parquet"),
    "resisc45": ("tanganke/resisc45", "5c484982a196dcb501b8e79c6cf4680dc6f2ccdf", "data/test-*.parquet"),
    "eurosat": ("tanganke/eurosat", "43750fc422c7fbd4289e2df3f7473617a5937acc", "data/test-*.parquet"),
    "svhn": ("ufldl-stanford/svhn", "f9e1717d73324ebbc1ece84267ccd80d0aca0690", "cropped_digits/test-*.parquet"),
    "gtsrb": ("tanganke/gtsrb", "ea889fff70dda7b09e94a4b207dded438deca549", "data/test-*.parquet"),
    "mnist": ("ylecun/mnist", "77f3279092a1c1579b2250db8eafed0ad422088c", "mnist/test-*.parquet"),
    "dtd": ("tanganke/dtd", "d2afa97d9f335b1a6b3b09c637aef667f98f966e", "data/test-*.parquet"),
}

TASKS = ["sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]

TEMPLATES = {
    "sun397": ["a photo of a {}.", "a photo of the {}."],
    "cars": [
        "a photo of a {}.", "a photo of the {}.", "a photo of my {}.", "i love my {}!",
        "a photo of my dirty {}.", "a photo of my clean {}.", "a photo of my new {}.",
        "a photo of my old {}.",
    ],
    "resisc45": [
        "satellite imagery of {}.", "aerial imagery of {}.", "satellite photo of {}.",
        "aerial photo of {}.", "satellite view of {}.", "aerial view of {}.",
        "satellite imagery of a {}.", "aerial imagery of a {}.", "satellite photo of a {}.",
        "aerial photo of a {}.", "satellite view of a {}.", "aerial view of a {}.",
        "satellite imagery of the {}.", "aerial imagery of the {}.", "satellite photo of the {}.",
        "aerial photo of the {}.", "satellite view of the {}.", "aerial view of the {}.",
    ],
    "eurosat": [
        "a centered satellite photo of {}.", "a centered satellite photo of a {}.",
        "a centered satellite photo of the {}.",
    ],
    "svhn": ['a photo of the number: "{}".'],
    "gtsrb": [
        'a zoomed in photo of a "{}" traffic sign.',
        'a centered photo of a "{}" traffic sign.',
        'a close up photo of a "{}" traffic sign.',
    ],
    "mnist": ['a photo of the number: "{}".'],
    "dtd": [
        "a photo of a {} texture.", "a photo of a {} pattern.", "a photo of a {} thing.",
        "a photo of a {} object.", "a photo of the {} texture.", "a photo of the {} pattern.",
        "a photo of the {} thing.", "a photo of the {} object.",
    ],
}

IMAGE_SIDE = 224
CHUNK = 512
NEAR_DUP_AUDIT_BITS = 6
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

_PROC = None


def _clean(name: str) -> str:
    return str(name).replace("_", " ").replace("/", " ").strip()




def _init_worker(pretrained_dir: str) -> None:
    global _PROC
    from transformers import CLIPImageProcessor

    _PROC = CLIPImageProcessor.from_pretrained(pretrained_dir)


def _decode_one(raw: bytes) -> np.ndarray:
    """bytes -> uint8 [3, 224, 224] via the pinned CLIPImageProcessor resize + center crop."""
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        out = _PROC(images=im, do_rescale=False, do_normalize=False,
                    return_tensors="np")["pixel_values"][0]
    arr = np.asarray(out)
    if arr.shape[0] != 3:
        arr = np.transpose(arr, (2, 0, 1))
    assert arr.shape == (3, IMAGE_SIDE, IMAGE_SIDE), arr.shape
    return np.rint(arr).clip(0, 255).astype(np.uint8)


def dhash(arr: np.ndarray) -> np.ndarray:
    """480-bit difference hash of one uint8 [3, 224, 224] image, packed into 60 bytes.

    A 16x16 grid of block means, then the horizontal and vertical comparisons between adjacent
    cells. Far more discriminative than an 8x8 average hash, which is why the audit below can
    report a meaningful distance instead of a coin flip.
    """
    grey = arr.astype(np.float32).mean(axis=0)
    grid = grey.reshape(16, IMAGE_SIDE // 16, 16, IMAGE_SIDE // 16).mean(axis=(1, 3))
    bits = np.concatenate([(grid[:, 1:] > grid[:, :-1]).ravel(),
                           (grid[1:, :] > grid[:-1, :]).ravel()])
    return np.packbits(bits)


def min_hamming(sealed: np.ndarray, dev: np.ndarray) -> np.ndarray:
    """For every sealed hash row, the minimum Hamming distance to any dev hash row."""
    out = np.full(len(sealed), 480, dtype=np.int32)
    if len(dev) == 0 or len(sealed) == 0:
        return out
    for start in range(0, len(sealed), 128):
        blk = sealed[start:start + 128]
        x = np.bitwise_xor(blk[:, None, :], dev[None, :, :])
        out[start:start + 128] = _POPCOUNT[x].sum(axis=2).min(axis=1)
    return out


def content_key(arr: np.ndarray) -> bytes:
    """SHA-256 of the decoded pixels: an exact duplicate and nothing looser."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).digest()




def build_heads(assets: Path) -> None:
    import torch
    from transformers import CLIPModel, CLIPTokenizerFast

    torch.manual_seed(0)
    pre = assets / "pretrained"
    model = CLIPModel.from_pretrained(str(pre), dtype=torch.float32).eval()
    tok = CLIPTokenizerFast.from_pretrained(str(pre))

    heads = {}
    with torch.no_grad():
        for task in TASKS:
            names = json.loads((assets / "classes" / f"{task}.json").read_text())
            rows = []
            for name in names:
                prompts = [t.format(_clean(name)) for t in TEMPLATES[task]]
                batch = tok(prompts, padding=True, return_tensors="pt")
                out = model.get_text_features(**batch)
                emb = (out if isinstance(out, torch.Tensor) else out.pooler_output).float()
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb = emb.mean(dim=0)
                rows.append(emb / emb.norm())
            head = torch.stack(rows).contiguous().float()
            assert head.shape == (len(names), 768), (task, head.shape)
            heads[task] = head
            print(f"head {task}: {tuple(head.shape)}", flush=True)
        proj = model.visual_projection.weight.detach().clone().float()
    assert proj.shape == (768, 1024), proj.shape
    torch.save({"heads": heads, "visual_projection": proj}, assets / "heads.pt")
    print("heads.pt written", flush=True)




def panel_indices(n_rows: int, stride: int) -> dict[str, np.ndarray]:
    idx = np.arange(n_rows)
    out = {"dev": idx[idx % 2 == 0],
           "intermediate": idx[idx % 4 == 1],
           "final": idx[idx % 4 == 3]}
    if stride > 1:
        out = {k: v[::stride] for k, v in out.items()}
    return out


def _fetch_split(task: str, cache: Path):
    from huggingface_hub import snapshot_download
    import datasets as hfds

    repo, rev, pattern = DATASETS[task]
    root = snapshot_download(repo_id=repo, repo_type="dataset", revision=rev,
                             local_dir=str(cache / task), allow_patterns=[pattern])
    files = sorted(glob.glob(os.path.join(root, pattern)))
    if not files:
        raise RuntimeError(f"{task}: no parquet matched {pattern} at {rev}")
    return hfds.load_dataset("parquet", data_files={"test": files}, split="test")


def materialize(args: argparse.Namespace) -> None:
    import datasets as hfds

    assets = Path(args.assets)
    (assets / "classes").mkdir(parents=True, exist_ok=True)
    cache = Path(args.dataset_cache)
    verifier = args.role == "verifier"

    for task in TASKS:
        t0 = time.time()
        ds = _fetch_split(task, cache)
        image_col = "image" if "image" in ds.column_names else ds.column_names[0]
        label_col = "label" if "label" in ds.column_names else ds.column_names[-1]
        names = [_clean(x) for x in ds.features[label_col].names]
        (assets / "classes" / f"{task}.json").write_text(json.dumps(names, indent=1))

        labels_all = np.asarray(ds[label_col], dtype=np.int64)
        n_rows = len(labels_all)
        panels = panel_indices(n_rows, args.stride)
        assert not (set(panels["dev"]) & set(panels["intermediate"]))
        assert not (set(panels["dev"]) & set(panels["final"]))
        assert not (set(panels["intermediate"]) & set(panels["final"]))
        raw = ds.cast_column(image_col, hfds.Image(decode=False))

        need = sorted(set(panels["dev"]).union(panels["intermediate"], panels["final"])) \
            if verifier else sorted(panels["dev"])
        hashes: dict[int, np.ndarray] = {}
        keys: dict[int, bytes] = {}
        arrays: dict[int, np.ndarray] = {}
        store = set(panels["dev"]) if not verifier else \
            set(panels["intermediate"]) | set(panels["final"])
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(str(assets / "pretrained"),)) as pool:
            for start in range(0, len(need), CHUNK):
                rows = need[start:start + CHUNK]
                blobs = [b["bytes"] for b in raw[rows][image_col]]
                for row, arr in zip(rows, pool.map(_decode_one, blobs, chunksize=8)):
                    if verifier:
                        hashes[row] = dhash(arr)
                        keys[row] = content_key(arr)
                    if row in store:
                        arrays[row] = arr

        keep, audit = {}, {}
        wanted = ["intermediate", "final"] if verifier else ["dev"]
        dev_h = np.stack([hashes[int(r)] for r in panels["dev"]]) if verifier else None
        dev_keys = {keys[int(r)] for r in panels["dev"]} if verifier else set()
        for panel in wanted:
            sel = np.asarray(panels[panel], dtype=np.int64)
            dropped, remaining, achieved, near = 0, 0, 480, 0
            if verifier:
                mask = np.array([keys[int(r)] not in dev_keys for r in sel], dtype=bool)
                dropped = int((~mask).sum())
                sel = sel[mask]
                remaining = int(sum(1 for r in sel if keys[int(r)] in dev_keys))
                d = min_hamming(np.stack([hashes[int(r)] for r in sel]), dev_h) \
                    if len(sel) else np.zeros(0, dtype=np.int32)
                achieved = int(d.min()) if len(sel) else 480
                near = int((d <= NEAR_DUP_AUDIT_BITS).sum())
                rng = np.random.default_rng(args.panel_seed + TASKS.index(task))
                rng.shuffle(sel)
            keep[panel] = sel
            audit[panel] = {"task": task, "split_rows": int(n_rows),
                            "panel_rows_before_dedup": int(len(panels[panel])),
                            "panel_rows": int(len(sel)),
                            "dropped_exact_duplicates_of_dev": dropped,
                            "exact_duplicates_remaining": remaining,
                            "min_dhash_bits_to_dev": achieved,
                            "sealed_rows_within_audit_bits_of_dev": near,
                            "near_dup_audit_bits": NEAR_DUP_AUDIT_BITS,
                            "num_classes": len(names), "stride": args.stride}

        for panel, sel in keep.items():
            if panel == "dev":
                d = Path(args.out) / "data" / "dev" / task
                ld = d
            else:
                d = Path(args.eval_out) / panel / task
                ld = Path(args.label_out) / panel / task
            d.mkdir(parents=True, exist_ok=True)
            ld.mkdir(parents=True, exist_ok=True)
            mm = np.lib.format.open_memmap(
                d / "images.npy", mode="w+", dtype=np.uint8,
                shape=(len(sel), 3, IMAGE_SIDE, IMAGE_SIDE))
            for pos, row in enumerate(sel):
                mm[pos] = arrays[int(row)]
            mm.flush()
            del mm
            np.save(ld / "labels.npy", labels_all[sel])
            (ld / "classes.json").write_text(json.dumps(names, indent=1))
            (ld / "split_audit.json").write_text(json.dumps(audit[panel], indent=1))
            if panel == "dev":
                view = Path(args.out) / "data" / "dev_unlabeled" / task
                view.mkdir(parents=True, exist_ok=True)
                link = view / "images.npy"
                if not link.is_symlink():
                    os.symlink(os.path.relpath(d / "images.npy", view), link)
            print(f"{task}/{panel}: {audit[panel]}", flush=True)
        arrays.clear()
        hashes.clear()
        print(f"{task}: {n_rows} rows in {time.time() - t0:.1f}s", flush=True)




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["agent", "verifier"], required=True)
    ap.add_argument("--assets", default="/opt/assets")
    ap.add_argument("--out", default="/app")
    ap.add_argument("--eval-out", default="/opt/eval")
    ap.add_argument("--label-out", default="/tests/hidden_data")
    ap.add_argument("--dataset-cache", default="/tmp/ds_cache")
    ap.add_argument("--panel-seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()
    if args.role == "verifier" and args.panel_seed == 0:
        raise SystemExit("--panel-seed must be set for the verifier role")

    assets = Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    if not args.skip_models:
        from huggingface_hub import snapshot_download

        for name, (repo, rev) in MODELS.items():
            allow = ["config.json", "model.safetensors"]
            if name == "pretrained":
                allow += ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                          "merges.txt", "special_tokens_map.json", "preprocessor_config.json"]
            t0 = time.time()
            snapshot_download(repo_id=repo, revision=rev, local_dir=str(assets / name),
                              allow_patterns=allow)
            print(f"fetched {name} in {time.time() - t0:.1f}s", flush=True)

    if not args.skip_data:
        materialize(args)
    build_heads(assets)
    print("BUILD_ASSETS_DONE", flush=True)


if __name__ == "__main__":
    main()
