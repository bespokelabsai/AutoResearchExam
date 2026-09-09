# Hint

From prior autonomous-research runs on this task. Directions only, no tuned constants.

- **Specialise by sequence length.** Single-token instances are DRAM-bandwidth-bound (weights streamed once, almost no arithmetic); multi-token instances are GEMM-bound. Different code paths, different levers.
- **Do algebra at build time:** fuse projections sharing an input, fold the normalisation scale into the next projection, fold the attention scale into the query weights. At length one there is no history, so the attention block collapses — pre-compose the value and output projections into one matrix.
- **Layout decides GEMM cost.** Carrying activations transposed so the weight is the streamed operand avoids repacking the large operand on every call. At length one use the matrix-vector path on 1-D views, not a matmul on an `(n,1)` buffer.
- The largest gains came from a small hand-written vector kernel loaded over a C ABI: weights repacked into sequential panels, a **persistent** worker pool pinned one thread per **physical** core (the default oversubscribes SMT siblings), fused epilogues (residual-accumulating projections, fused norm+residual, fused gated activation, fused attention), and eventually one native call per forward pass.
- Once the single-token path sits at the memory roofline, **cutting bytes per weight converts almost 1:1 into speed** and the tolerance is far looser than such formats cost.
- Guard the fast path with a fallback, and prove the deliverable is self-contained — only the submission directory reaches the grading environment.
