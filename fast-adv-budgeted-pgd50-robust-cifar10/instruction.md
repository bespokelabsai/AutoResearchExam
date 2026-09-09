# Train an adversarially robust CIFAR 10 model

Train a `PreActResNet18` within 180 seconds. The grader attacks the trained model and measures how many unseen CIFAR 10 images remain correctly classified.

## Deliverable

Write `/app/output/train.py`. It must be a single Python file no larger than 102,400 bytes.

The grader runs:

```text
python train.py --data /app/data/cifar10_train.npz --out <scratch>/model.pt
```

Your program must save a PyTorch `state_dict` at the `--out` path. The file must be no larger than 200 MiB. The grader loads it into its own `PreActResNet18()` using `strict=True`. It grades the latest complete checkpoint present when the 180 second limit ends.

Only `/app/output/train.py` moves to the grading environment. The grader places `preact_resnet.py` beside it before running the program.

## Training data

`/app/data/cifar10_train.npz` contains the CIFAR 10 training split:

* `images`: `uint8` with shape `(50000, 32, 32, 3)` in RGB order.
* `labels`: `int64` with shape `(50000,)` and values from `0` to `9`.

You may choose how to form a validation set from this data. The unseen evaluation images are not available during training.

## Scoring

The target metric is robust accuracy under a fixed PGD 50 attack with perturbation size `8/255`. Higher is better.

The attack uses 50 steps, step size `2/255`, and 10 random restarts. Images use CIFAR 10 channel means `(0.4914, 0.4822, 0.4465)` and standard deviations `(0.2471, 0.2435, 0.2616)`. The attack runs in float32 and uses a fixed random seed.

Before the attack, the grader divides logits by `T = max(1, s / 20)`, where `s` is their mean standard deviation across clean evaluation images. This leaves clean predictions unchanged and prevents logit scaling from weakening the attack.

The score reported during research iterations uses a validation half of the CIFAR 10 test split. The final score uses a separate hidden half. Clean accuracy is recorded but does not affect the reward.

Every increase in robust accuracy receives a higher reward. A missing or invalid checkpoint, incompatible state dictionary, nonfinite weights, input independent model, crash, or timeout receives zero.

## Runtime

The full program, including imports, data loading, training, and checkpoint writing, has 180 seconds of wall clock time.

The environment has one H100 GPU, 4 physical CPU cores, 16,384 MiB of memory, and no general internet access. PyTorch 2.10 with CUDA 12.8 is installed.
