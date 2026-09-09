# Low-discrepancy subset selection

You get a cloud of `n` points in the unit cube `[0,1]^3` and a subset size `k`. Return the `k`
points that fill the cube most evenly, in the L-infinity star discrepancy sense.

For a set `P` of `k` points, the star discrepancy is:

$$d^*(P) = \sup_{q \in [0,1]^3} \left| \frac{\#\{p \in P : p < q\}}{k} - q_1 q_2 q_3 \right|$$

This is the worst gap, over every box `[0,q)` anchored at the origin, between the fraction of `P`
inside the box and the box's volume. Lower is better.

## Submission format

Write `/app/output/select_subset.py`, defining exactly one entry point:

```python
select(points, k, seed, time_budget_s) -> sequence of k ints
```

- `points`: C-contiguous float64 array, shape `(n, 3)`, values in `[0, 1]`, `800 <= n <= 4000`.
- `k`: int, one of `30`, `50`, `65`; always `k < n`.
- `seed`: int. Use it for any randomness so a call is reproducible.
- `time_budget_s`: float, `10.0` at grade time. `select` must return within it.
- Return value: any iterable of exactly `k` distinct integer indices into `points`, each in
  `[0, n)`. Values can be Python ints, numpy integers, or floats with an integral value.

## Data notes

- Example clouds live in `/app/data/examples/`.
- `/app/data/examples/manifest.json` lists each cloud's file name, `n`, and `k`.
- Grading uses sealed clouds drawn the same way; your code sees them for the first time at grade
  time.

## Metric and scoring

- `select` is called once per (cloud, seed) pair.
- You're scored on the mean exact `d*` of the returned subsets across the whole panel.
- A subset no better than an unguided, uniformly random `k`-subset scores zero.
- Lower discrepancy always scores higher, with no cutoff where further improvement stops
  counting.

## Execution environment

- Grading runs in a separate environment from this one, and only `/app/output` carries over. So
  your solution must be fully self-contained there: if `select_subset.py` imports a helper module
  you wrote, that module must also live under `/app/output`.
- Each call runs in a fresh process pinned to a single CPU.
- `/app/output` is mounted read-only at grade time. A scratch working directory is provided and
  deleted after each call, so nothing written during one call is visible to any other.
- The process is SIGKILLed `time_budget_s + 5` seconds after it starts. Interpreter startup and
  importing your module count against that extra 5 seconds.
- Available packages: the Python standard library, numpy, and scipy. No network access.
- `/app/output` is writable now; everything else is read-only.
- Everything under `/app/output` must be a regular file. A symbolic or hard link anywhere in that
  tree makes the whole submission unscoreable.

## Invalid submissions

- If a call is killed, raises, or returns anything other than exactly `k` distinct in-range
  integral indices, that cloud is scored as if no subset had been selected, the worst credit
  available.
- If your module fails to import, you score zero overall.
