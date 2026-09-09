# Merge eight vision models into one encoder

Create one CLIP ViT-L/14 vision encoder that performs well across eight image classification datasets.

## Goal

You receive one pretrained vision encoder and eight copies fine-tuned on:

`sun397`, `cars`, `resisc45`, `eurosat`, `svhn`, `gtsrb`, `mnist`, and `dtd`.

Return one state dictionary for the same architecture. The result is used as a single frozen encoder on all eight datasets. It may not route by dataset or keep separate model branches.

The metric is mean top-1 accuracy across the eight hidden evaluation sets. Each dataset has equal weight. Higher is better.

A mean accuracy of 65.23 percent or lower receives zero reward. This is the pretrained model baseline. Above that floor, reward increases with accuracy.

## Assets

Model weights are stored as safetensors:

- Pretrained model: `/opt/assets/pretrained/model.safetensors`
- Fine-tuned models: `/opt/assets/{task}/model.safetensors`

All models have the same 391 keys and tensor shapes.

## Deliverable

Write `/app/output/merge.py`. It must define:

```python
def merge(
    pretrained: dict[str, torch.Tensor],
    experts: dict[str, dict[str, torch.Tensor]],
    unlabeled_dir: str,
    device: str,
) -> dict[str, torch.Tensor]:
    ...
```

The grader calls `merge` once with positional arguments.

- `pretrained` is the pretrained state dictionary. Its values are float32 CPU tensors.
- `experts` maps each of the eight task names to its fine-tuned state dictionary.
- `unlabeled_dir` contains `unlabeled_dir/{task}/images.npy`. Each file is a `uint8` array with shape `[N, 3, 224, 224]`.
- `device` is `"cuda"`.

Return a dictionary with exactly the same keys and shapes as `pretrained`. Every value must be convertible to a finite float32 tensor. Float32 CPU tensors are the safest form.

Only files under `/app/output` are submitted. You may include helper modules or data there. Do not use symbolic links, hard links, special files, or more than 20,000 files. Load bundled files relative to `merge.py` because the grader copies the output directory before running it.

## Evaluation

For each dataset, the grader loads your returned weights into `CLIPVisionModel`. It applies the frozen CLIP visual projection and the dataset's frozen zero-shot classification head. It then computes top-1 accuracy on hidden labeled images.

The heads and projection are available at `/opt/assets/heads.pt`. Hidden labels are not available to your code. The images passed through `unlabeled_dir` are the same images used for scoring, but they have no labels and their order has no meaning.

`merge` must return within 1,200 seconds. The full scoring process has a 1,800 second limit. A missing file, import error, exception, timeout, wrong key or shape, non-finite value, or invalid submitted tree receives zero reward.

## Development data

Use:

- `/app/data/dev/{task}/images.npy`
- `/app/data/dev/{task}/labels.npy`
- `/app/data/dev/{task}/classes.json`
- `/app/data/dev_unlabeled/{task}/images.npy`
- `/app/eval_dev.py`

Run:

```bash
/usr/bin/python3 /app/eval_dev.py
```

The development and hidden rows are separate halves of the same test splits. Pixel-exact duplicates were removed from the hidden half.

## Environment

The grading machine has one NVIDIA H100 80 GB GPU, 8 physical CPU cores, 64 GiB RAM, a 300 GB disk, and no internet access. PyTorch, Transformers, Datasets, Safetensors, NumPy, and Pillow are installed.

The input state dictionaries use about 11 GB of host memory before `merge` starts. The output tree is read-only during grading. Write temporary files to the process working directory.

CPU library threads default to one. Random seeds and deterministic settings are fixed. TF32 is disabled. The hidden evaluation uses the same settings on every run.
