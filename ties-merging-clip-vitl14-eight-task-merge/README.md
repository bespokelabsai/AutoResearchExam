# Merging Several CLIP ViTs for Zero-Shot Classification

*Category: Model training. Subcategory: Model merging and distillation.*

We have a pretrained CLIP ViT L 14 vision encoder and eight copies fine tuned on different image classification datasets. The objective is to merge their weights into one frozen encoder. The resulting encoder is evaluated as a zero-shot classifier over the original datasets; the final score is the unweighted average of accuracy on all eight datasets. One may start by averaging the eight fine tuned model weights. Because the datasets are provided, it is also possible to fine tune the aggregate model on individual datasets.
