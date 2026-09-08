# Improve W4A4 PixArt Sigma image quality

Improve the image quality of a W4A4 quantized PixArt Sigma model. Write your solution to `/app/output/quantize.py`. This must be the only file you create or modify.

The file must define `quantize_model(pipe, rank=32) -> None`. The grader passes in a fresh FP16 `diffusers.PixArtSigmaPipeline` loaded from `/opt/pixart-sigma`. `rank` must be an integer from `0` to `32`. You may use the calibration prompts in `/app/data/calibration_prompts.jsonl`.

## Required module format

Replace every `torch.nn.Linear` in `pipe.transformer` in place. Register these tensors in the order shown. Do not register any other tensor parameters or buffers.

* `qweight`: `torch.int8` with shape `(out_features, in_features)`. Values must be from `-7` to `7`.
* `scales`: `torch.float16` with shape `(out_features, ceil(in_features / 64))`. Values must be finite and positive.
* `l1`: `torch.float16` with shape `(in_features, r)`. Values must be finite.
* `l2`: `torch.float16` with shape `(r, out_features)`. Values must be finite. `r` must be from `0` to `rank`, and `rank` must be from `0` to `32`.
* `smooth`: `torch.float16` with shape `(in_features,)`. Values must be finite and positive.
* `bias`: Include this exactly when the original layer has a bias. It must be finite `torch.float16` with shape `(out_features,)`.

Do not keep the original weight. `quantize_model` must return `None`.

## Required computation

Use symmetric groupwise INT4 quantization over groups of 64 input channels. A shorter final group is allowed. Each layer must compute:

```text
x_s = x / smooth
output = Q4(x_s) @ dequantize(qweight, scales).T
       + (x_s @ l1) @ l2
       + bias
```

Quantize each input row and group using `max(abs(group)) / 7`. Round with PyTorch's default round to even rule, clip values to `-7` to `7`, and use a scale of `1` for an all zero group. Multiply each stored code by its group scale when dequantizing the residual weight.

Run the computation in FP16. You may calculate scales in FP32.

## Scoring

The grader compares images from your quantized model with images from the original FP16 model on unseen prompts. The score is mean PSNR in dB. Higher is better.

A basic round to nearest W4A4 model defines the zero reward baseline. Scores above that baseline receive higher rewards. Invalid or nonrunning submissions receive zero.

## Runtime

Importing your file, loading the model, running `quantize_model`, checking the replacement modules, and saving their tensor state must finish within 900 seconds.

The environment has one H100 GPU, 4 physical CPU cores, 32,768 MiB of memory, and no internet access. The grader runs separately and receives only `/app/output/quantize.py`, so the solution must be self contained.
