# PyDigest2

A pure-Python port of **digest2**, the Minor Planet Center's statistical-ranging
tool for classifying short-arc asteroid astrometry (NEO, Mars-crosser, main-belt,
Trojan, Jupiter-family-comet, etc.) from as few as two observations of a tracklet.

This is an independent line-by-line reimplementation of the algorithm in
[Bill Gray / MPC's C `digest2`](https://github.com/Smithsonian/mpc-public/tree/main/digest2)
(current as of the source snapshot this port was validated against, August 2026)
— no C code, no C extension, numpy for the population-model data and
multiprocessing for throughput. It is a different thing from the Smithsonian
`digest2` PyPI package, which wraps the *same C engine* as a compiled extension
(and is therefore bit-identical to the CLI by construction); this project
reimplements the numerics themselves in Python, which is a much better fit for
reading, modifying, or embedding the algorithm without a C toolchain — at the
cost of not being bit-identical in every case. See **Validated accuracy** below
for exactly what that costs.

## Install

```bash
pip install -e .
```

Requires only `numpy`. The population model (`digest2.model.csv`) and
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
across a process pool (see **Performance design** below).

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
- **End-to-end scores**, on a sample of 250 real NEOCP tracklets (repeatable
  mode, all 15 classes): **94.8% of tracklets match the C reference exactly
  or to floating-point noise**, and **99.4% of all individual raw/no-ID score
  values are within 1 point (on the 0–100 scale) of the reference.** The
  residual ~5% of tracklets showing a few-point divergence are consistently
  the shortest/sparsest arcs (2–4 observations) — see the next paragraph for
  why.

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

## Performance design

The search for a single tracklet is **not** a good fit for NumPy vectorization:
it's a data- and RNG-dependent recursive tree walk (see the module docstring
in `pydigest2/ranging.py`), so the per-node 3-vector math stays plain Python
floats/tuples — for objects this small, NumPy's per-call array overhead
comfortably loses to the interpreter's own float arithmetic. Two things *do*
get NumPy/parallelism:

1. **The population model** (four arrays, up to `(15, 29, 8, 11, 18)`
   float64) is NumPy end-to-end, with a fast-loading `.npz` cache generated
   next to the CSV on first use (mirrors the C tool's own CSV→binary
   caching, just in this package's own format).
2. **Tracklets are independent of each other** — no shared state, separate
   RNG streams — so `pydigest2.engine.score_many()` fans a batch out across a
   `ProcessPoolExecutor` (real parallelism, unlike threads, since the search
   is pure-Python CPU work under the GIL). On fork-based platforms
   (Linux/macOS) workers inherit the already-loaded model via copy-on-write,
   so there's no per-worker reload cost. In informal testing, 250 real NEOCP
   tracklets scored in ~51s on 4 cores (~5 tracklets/s), vs. the C CLI's
   ~22ms/tracklet single-threaded — Python is slower per tracklet, as
   expected, but the parallel batch API keeps large files practical.

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
- `--cpu`/`-u` selects worker *processes*, not OS threads (Python has no
  equivalent to lightweight threads sharing one interpreter under the GIL for
  CPU-bound work).
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
| `ranging.py` | The adaptive orbit search itself, and the final score computation |
| `config.py` | `digest2.config` parsing, CLI config state |
| `engine.py` | High-level scoring engine, incl. process-pool batch scoring |
| `intake.py` | Streaming tracklet grouping matching the reference CLI exactly |
| `output.py` | Score-table formatting |
| `cli.py` | The `pydigest2` command-line entry point |

Each module's docstring names the C source function(s) it ports.

## License

Public domain, matching the original digest2 source.
