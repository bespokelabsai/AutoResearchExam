# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Batch the whole workload.** The reference is batch-1; a continuous-batching engine — prefills grouped by prompt length, then one shared decode loop over all live requests with a common KV cache — dwarfs every other optimisation.
- **Keep the active set contiguous.** Requests finish at different times; swap finished slots to the end so live rows stay a prefix and every KV slice is a zero-copy view instead of a gather.
- **Construction and warm-up are untimed.** Pre-allocate and pre-touch the KV pool and all large buffers there, then use `out=` destinations and in-place residual adds inside `generate`.
- The **full-vocab projection** is a top hotspot: chunk it over vocabulary blocks with a running max, or fuse projection and argmax.
- Replace hand-rolled activations with the fused kernel; use the fused attention primitive with a head-first KV layout so no per-step permutes happen.
- Decode does small-M matmuls where the generic BLAS path is weak — pre-pack the decode-only linear weights once at construction, keeping the default path for prefill.
- **Refuted repeatedly:** low precision (breaks the token-agreement gate and is slow without native support), graph compilers (dynamic shrinking batch), and switching thread counts between phases.
