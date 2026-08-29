# PyDigest2

A pure-Python port of **digest2**, the Minor Planet Center's statistical-ranging
tool for classifying short-arc asteroid astrometry (NEO, Mars-crosser, main-belt,
Trojan, Jupiter-family-comet, etc.) from as few as two observations of a tracklet.

This is an independent line-by-line reimplementation of the algorithm in
[Bill Gray / MPC's C `digest2`](https://github.com/Smithsonian/mpc-public/tree/main/digest2)
(current as of the source snapshot this port was validated against, August 2026)
— no C code and no C extension: numpy holds the population model, and an
optional JIT plus cross-tracklet parallelism carry the throughput. It is a different thing from the Smithsonian
`digest2` PyPI package, which wraps the *same C engine* as a compiled extension
(and is therefore bit-identical to the CLI by construction); this project
reimplements the numerics themselves in Python, which is a much better fit for
reading, modifying, or embedding the algorithm without a C toolchain — at the
cost of not being bit-identical in every case. See **Validated accuracy** below
for exactly what that costs.

## Install

```bash
pip install -e .          # pure Python; numpy only
pip install -e ".[fast]"  # + numba JIT: ~30x faster, recommended
```

Requires only `numpy`; `numba` is an optional accelerator (see
[Performance](#performance)). The population model (`digest2.model.csv`) and
observatory-code table (`digest2.obscodes`) ship bundled under
`pydigest2/data/`, so it works out of the box with no extra downloads.

## Usage

### Command line

```bash
pydigest2 sample.obs
pydigest2 -c digest2.config sample.obs sample.xml   # formats may be mixed
cat sample.obs | pydigest2 -
```

Options mirror the reference CLI: `-c/--config`, `-m/--model`, `-o/--obscodes`,
`-p/--config-path`, `-u/--cpu`, `-l/--limit <class>/<raw|noid>=<N>`,
`-h/--help`, `-v/--version`. `digest2.config` keywords (`headings`, `rms`,
`rmsPrime`, `noThreshold`, `raw`, `noid`, `repeatable`, `random`, `obserr`,
`poss`, and orbit-class names) all work the same way — see `pydigest2 -h` for
the quick reference. Input is MPC 80-column (`.obs`) or ADES XML (`.xml`);
ADES PSV (`.psv`) is also accepted as a bonus (no C reference exists for that
format — see [Deliberate differences](#deliberate-differences-from-the-c-cli)).

### Python API

```python
from pydigest2 import Digest2Engine
from pydigest2.observations import parse_mpc80_file

engine = Digest2Engine.load()          # bundled model/obscodes, defaults
engine.config.repeatable = True        # deterministic scoring, if wanted

for desig, olist in parse_mpc80_file("sample.obs").items():
    result = engine.score(olist)
    print(desig, result.rms, result.raw_scores[1], result.noid_scores[1])  # class 1 = NEO
```

For batch throughput, `pydigest2.engine.score_many()` fans tracklets out
across worker threads or processes (see [Performance](#performance)).

## Validated accuracy

Every piece of this port that has a C counterpart was checked against a real
build of the reference `digest2` C sources (compiled from the same
`Smithsonian/mpc-public` snapshot, both as a standalone CLI and via a small
harness linked directly against `d2lib.c`), fed byte-identical inputs:

- **Motion-vector synthesis** (`twoObs`/`oneObs`, including the multi-arc-split
  code path for long/multi-site tracklets) and the great-circle RMS match the
  C reference to full double precision on every case checked.
- **The adaptive orbit search itself** — in "repeatable" mode (fixed LCG seed)
  — reproduces the reference's angle-search parameters (`ang1`/`ang2`) and the
  resulting orbit elements (q, e, i) to 10+ significant digits at the first
  several recursion nodes checked by hand, and the *linear congruential
  generator* driving the jitter is bit-identical.
- **End-to-end scores**, over **5,339 real NEOCP tracklets** run through
  both CLIs with identical config (repeatable mode): **99.46% produce
  byte-identical printed scores**, 99.87% agree on NEO to within 1 point
  (on the 0–100 scale), 100% agree on MB1 to within 1 point, and only
  **7 tracklets (0.13%) differ by more than 1 point** on any score.

**Why isn't it 100%?** This is a chaotic, recursive, floating-point algorithm:
which orbit bins get tagged depends on hard threshold comparisons (bin edges,
a `d3 > 0.1 rad` recursion cutoff) evaluated after ~10⁵–10⁶ transcendental
function calls per tracklet. A single sub-ULP rounding difference anywhere in
that chain — and one genuinely exists, in the sun-observer vector's tiny
out-of-plane component, which is computed as a difference of two much larger
terms (catastrophic cancellation, present in the *original* algorithm, not
introduced by this port) — can occasionally flip one bin-boundary decision and
cascade into a several-point score difference for a *sparse* tracklet, where
each bin carries more relative weight. This is not fixable by "trying harder"
to match the C build line-for-line; it would require literally identical
instruction-level floating-point behavior, which two independent
implementations in different languages cannot generally guarantee. It's worth
noting the reference program itself is non-deterministic by default (it only
becomes reproducible with the `repeatable` config keyword, reseeding a
random-per-run generator) — so treating "within about a point, the large
majority of the time" as the right notion of "identical" is not a compromise
specific to this port, it's how the algorithm is meant to be read even against
itself.

Re-running this validation (against your own C build) is straightforward and
documented inline in `tests/test_integration.py` and `tests/test_cli.py`,
which encode the exact reference values this was checked against.

## Performance

Speed is roughly at parity with the compiled C reference. Measured on the
same machine (4 cores), scoring the same 5,339 real NEOCP tracklets
through each program's CLI:

| | 1 core | all 4 cores |
|---|---|---|
| C `digest2` | 10.3 ms/tracklet | 2.61 ms/tracklet |
| **pydigest2** (with `[fast]`) | 10.9 ms/tracklet | 2.96 ms/tracklet |
| pydigest2 (pure Python) | ~600 ms/tracklet | ~150 ms/tracklet |

Getting there needed three things, in order of how much they mattered:

1. **An optional JIT.** `pip install pydigest2[fast]` pulls in numba,
   which compiles the search kernel and makes it ~30x faster. It is
   never required: without it, the *same function* runs interpreted, and
   everything still works — just slowly. Both paths are built from one
   source (`pydigest2/_search.py`) precisely so they cannot drift, and
   `tests/test_kernel.py` asserts they produce bit-identical scores.
2. **Bitmask bin tagging.** The reference implementation tracks, per
   orbit class, two sets of tagged (q,e,i,H) bins, and its tag-merge
   step loops over every class for every tagged bin. Representing "which
   classes is this bin tagged for" as one integer bitmask collapses that
   from O(bins x classes) to O(bins), and reduces the innermost
   bookkeeping to a couple of integer ops. This is a genuine algorithmic
   improvement over the C original, and it is why the Python version can
   land this close despite the language gap.
3. **Fusing the inner loop.** The angle search runs ~350k–500k times per
   tracklet; in CPython the per-call and attribute-lookup overhead of a
   nicely decomposed version was ~60% of total runtime. The kernel is
   therefore one long function with the recursion flattened into an
   explicit stack. That is a real readability cost, taken deliberately in
   exactly one place, and the readable one-concept-per-function form of
   every piece still exists and is still what the unit tests check
   (`pydigest2/classes.py`, `pydigest2/binning.py`).

What is *not* used is NumPy vectorization of the search itself. That
looks tempting and doesn't work: the search is a data-dependent recursive
tree walk whose branching depends on whether each trial orbit lands in a
previously-untagged bin, and whose midpoints are jittered by an LCG that
must be consumed in exactly that traversal order to reproduce the
reference program's results. There is no fixed set of steps to batch
across. NumPy earns its place holding the population model; the
parallelism that exists is *across* tracklets, which are fully
independent.

`--cpu`/`-u` selects worker threads when the compiled kernel is present
(it releases the GIL, so threads give real parallelism and share the
loaded model outright) and worker processes otherwise. As in the
reference program the default is every core, and output order and scores
are identical regardless of worker count.

## Deliberate differences from the C CLI

- The model/obscodes files fall back to the bundled copies in
  `pydigest2/data/` when not found alongside the input, rather than requiring
  a local `digest2.model.csv`/`digest2.obscodes` (the C tool auto-downloads
  only the latter). Scoring itself is unaffected — only where the files are
  found when the caller didn't say.
- `-m`'s cache file is this package's own `.npz` format, not the C tool's raw
  `fwrite` binary layout — an internal speed optimization in both
  implementations, not an interchange format.
- `.psv` (ADES pipe-separated-value) input is supported as a bonus; there is
  no C reference for this format (the compiled reference CLI reads MPC80 and
  ADES XML only).
- `--cpu`/`-u` selects worker threads (with the compiled kernel, which releases
  the GIL) or worker processes (without it), rather than the C program's OS
  threads. Same meaning, same default of "every core".
- Malformed `digest2.config` lines raise a clear error rather than silently
  misparsing; in particular, CRLF line endings (which the real MPC.config
  distributed alongside the model data actually has, and which the C reader's
  `\n`-only stripping chokes on) are handled transparently.
- Output line order always matches input order. The C CLI's default
  multi-threaded mode prints tracklets in *completion* order, which is
  therefore not reproducible even between two runs of the same binary; `-u 1`
  restores file order there too, which is what this port's outputs were
  checked against.

**A pre-existing reference bug, found in the course of validating this port:**
the compiled C CLI segfaults on any `--limit` invocation in this source
snapshot (reproduced with default config, restricted-class config, single- and
multi-observation files — not something this port needs to reproduce).
Worth flagging upstream; `pydigest2 -l ...` is implemented from the documented
behavior and covered by tests.

## Package layout

| Module | Purpose |
|---|---|
| `constants.py`, `classes.py`, `binning.py` | Physical/model constants, the 15 orbit-class predicates, (q,e,i,H) bin lookups |
| `rng.py` | The exact LCG driving "repeatable" mode |
| `model.py`, `obscodes.py` | Population-model CSV/cache and observatory-code table loading |
| `observations.py` | MPC80 / ADES XML / ADES PSV parsing |
| `geometry.py` | Solar ephemeris, great-circle fit/RMS |
| `tracklet.py` | Motion-vector synthesis (`twoObs`/`oneObs`) |
| `_search.py` | The search kernel: one function, run interpreted or JIT-compiled |
| `ranging.py` | Scoring driver: motion vector -> search -> class percentages |
| `config.py` | `digest2.config` parsing, CLI config state |
| `engine.py` | High-level scoring engine, incl. parallel batch scoring |
| `intake.py` | Streaming tracklet grouping matching the reference CLI exactly |
| `output.py` | Score-table formatting |
| `cli.py` | The `pydigest2` command-line entry point |

Each module's docstring names the C source function(s) it ports.

## License

Public domain, matching the original digest2 source.
