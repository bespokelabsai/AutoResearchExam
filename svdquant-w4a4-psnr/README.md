# Fast image generation model quantization

*Category: Systems and efficiency. Subcategory: Compression.*

This is a standard GPU quantization task. The agent starts with PixArt Sigma, an image generator that uses 16 bit quantization, and a set of sample text prompts. The agent must replace every linear layer in its transformer with 4 bit weights and intermediate values while keeping the images close to those output from the original model. Rounding errors can build across many layers, and a scale that works for most values can damage unusual channels. One may start by scaling small groups of values and rounding them to the nearest 4 bit level. Further gains may come from giving different channels smoother scales and adding small learned corrections for common errors.
