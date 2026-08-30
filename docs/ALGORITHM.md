# How digest2 works

A walkthrough of the algorithm this package implements, cross-referenced to
the reference C source and to the modules here.

The authoritative description is:

> Keys, S., Vereš, P., Payne, M. J., Holman, M. J., Jedicke, R., Williams, G. V.,
> Spahr, T., Asher, D. J., & Hergenrother, C. (2019).
> **The digest2 NEO Classification Code.**
> *Publications of the Astronomical Society of the Pacific*, 131(1000), 064501.
> [arXiv:1904.09188](https://arxiv.org/abs/1904.09188)

Section numbers below (§3.1, Appendix A, …) refer to that paper. A 2023
follow-up, [arXiv:2309.16407](https://arxiv.org/abs/2309.16407), adds the
per-detection ADES uncertainty handling that is also implemented here.

---

## 1. The problem

A *tracklet* — a few detections of one object over minutes to hours —
constrains the object's sky position (α, δ) and its apparent motion
(α̇, δ̇). It says essentially nothing about the topocentric distance ρ or
the radial velocity ρ̇. Those two unknowns are exactly what an orbit needs.
This is the fundamental difficulty of short-arc orbit determination.

So digest2 does not solve for an orbit. It **enumerates every orbit the
tracklet is compatible with, and asks what fraction of the modelled Solar
System population looks like those orbits.**

Milani et al. (2004) call the set of (ρ, ρ̇) yielding a heliocentrically
bound orbit the *admissible region*. digest2 sweeps that region, bins each
candidate orbit into a census of the Solar System, and reports the share of
the matching population belonging to a class of interest. The resulting
score, **D2**, runs 0–100. It is a population-weighted plausibility, *not* a
probability.

Lineage: R. McNaught's PANGLOSS (late 1980s) → the FORTRAN "223.f" used in
Jedicke (1996) → digest2. It predates, and is independent of, both
statistical ranging (Virtanen et al. 2001) and systematic ranging
(Chesley 2005), while sharing their central move.

---

## 2. The pipeline

### 2.1 Endpoint synthesis — §3.1

*C:* `twoObs`, `oneObs` in `d2math.c`  ·  *here:* `pydigest2/tracklet.py`

Reduce N detections to exactly **two**, defining a single motion vector.
Three cases:

- **Short arc, single site.** Fit a great circle to all detections, then
  synthesize endpoints at the **17th and 83rd percentile** of the arc rather
  than at its extremes — so a single bad measurement at either end cannot
  swing the admissible region. (The paper notes in a footnote that the
  rationale for those particular percentiles "is unknown to the surviving
  authors".)
- **Any space-based detection.** Parallax from the spacecraft's own orbit
  curves the apparent path, so great-circle interpolation is meaningless.
  Two *real* observations nearest those percentiles are used instead.
- **Long arc, or multiple sites.** Split off two sub-arcs, each under 3 hours
  and all from one site, and reduce each separately.

The great-circle fit also yields the **RMS** residual reported in the output.
A high RMS means either poor astrometry or genuine departure from great-circle
motion (diurnal parallax on a nearby object); digest2 cannot distinguish the
two, which is why it is shown rather than acted on.

### 2.2 Dithering for observational error — §3.1.1

*C:* `offsetMotionVector`  ·  *here:* inlined in `pydigest2/_search.py`

Each endpoint is displaced by ±0.5σ in RA and/or Dec, giving the nominal
tracklet plus 8 variants — **9 in total**. The two endpoints are displaced in
*opposite* senses, which maximises the spread of resulting motion vectors.

σ is the per-observatory astrometric uncertainty from the config file
(default 1.0″; e.g. F51/F52 0.2″, G96 0.3″, 703/704 0.7″). With ADES input,
per-detection `rmsRA`/`rmsDec` are used instead, clamped to [0.7σ, 5σ] unless
the `noThreshold` keyword lifts the ceiling.

The whole search below runs once per variant.

### 2.3 Sweep the distance — Appendix A

*C:* `setupDistanceDependentVectors`, `dRange`  ·  *here:* `_search.py`

For a nominated observer–object distance **D**, between **0.05 and 100 AU**:

- Δ⃗₁ = D·d̂₁ gives the topocentric vector; the heliocentric position follows
  as r⃗₁ = R⃗₁ + Δ⃗₁, with R⃗₁ the Sun→observer vector from local sidereal time
  and the site's parallax constants. Everything is worked in ecliptic
  Cartesian coordinates.
- **Angle limits.** Only the transverse velocity component is observable, so
  the true velocity could lie anywhere in a wedge. The bounding case is the
  parabolic orbit, ε = v²/2 − U/r₁ = 0 (U = k², k the Gaussian gravitational
  constant). Substituting into the cosine rule gives a quadratic in Δ₂ whose
  two roots are the escape-velocity angles α₁, α₂ — the edges of the
  admissible region at this distance.
- **Absolute magnitude.** From the mean V magnitude of the tracklet, in the
  IAU H–G system with G = 0.15:

  H = V − 5·log₁₀(Δ·r) + 2.5·log₁₀(0.85·Φ₁ + 0.15·Φ₂)

  with Φ₁ = exp(−3.33·tan^0.63(Φ/2)) and Φ₂ = exp(−1.87·tan^1.22(Φ/2)).
  If the tracklet carries no photometry at all, V = 21 is assumed — roughly
  the limiting magnitude of current surveys.

### 2.4 Sweep the angle — Appendix A

*C:* `solveAngleRange`, `aRange`, `tagAngle`  ·  *here:* `_search.py`

For an angle α inside (α₁, α₂), the law of sines gives

Δ₂ = Δ₂₁ · sin α / sin(π − α − θ),  then  v⃗ = (Δ₂·d̂₂ − Δ⃗₂₁) / (t₂ − t₁)

which completes the state vector (r⃗, v⃗) and hence the Keplerian elements
q, e, i. Together with H from §2.3 that is a full candidate orbit.

### 2.5 Bin it — §3.3, §3.4

*C:* `qeiToBin`, `hToBin` in `d2model.c`  ·  *here:* `pydigest2/binning.py`

The orbit indexes a bin of a binned Solar System census: **29 × 8 × 11 × 18**
bins in (q, e, i, H), non-uniformly spaced — finer where the population is
dense, which also keeps near-empty bins rare in sparse regions.

Binning on **perihelion q rather than semi-major axis a** is deliberate: it
puts a bin edge exactly at q = 1.3 AU, the NEO definition, so the NEO /
non-NEO split falls on a bin boundary instead of being smeared across bins.

Two flavours of the model ship together:

| flavour | meaning |
|---|---|
| **raw** | the full modelled population — the Pan-STARRS Synthetic Solar System Model (Grav et al. 2011), ~14 M orbits |
| **no-ID** | full population *minus* already-identifiable known objects |

The no-ID reduction decrements bins using the `astorb` catalogue, keeping
orbits whose field-24 metric (peak ephemeris uncertainty over 10 years) is
under ~1′ — i.e. objects secure enough that identifying a tracklet with them
would be trivial. The no-ID score is the operationally meaningful one, since
you would only be scoring a tracklet you had already failed to identify.

### 2.6 Search strategy — Appendix C

This is the part most often misread. digest2 does **not** raster a fixed grid
and does **not** use χ² or RMS goodness-of-fit. It performs a **recursive
binary subdivision that terminates on information gain**:

- Subdivide the angle range, solve the orbit at the midpoint, and recurse into
  both halves *only if that orbit tagged a bin not already tagged*.
- One level up, the same for distance: search a distance, and recurse only if
  new bins were tagged there.

Two mechanisms present in the code are **not described in the paper**, and
both materially affect coverage:

1. Subdivision is *forced* while the interval is still large — angle span
   > 0.1 rad, distance span > 0.2 AU — regardless of whether anything was
   tagged.
2. An "age" counter grants one extra level of subdivision after tagging has
   stopped, so the search does not halt at the first barren midpoint.

Midpoints are also **jittered** by a pseudo-random draw rather than taken
exactly halfway, which is what makes the search stochastic (see §4 below).

### 2.7 Score — §3.5

*C:* `score()` in `d2math.c`  ·  *here:* `pydigest2/ranging.py`

Per class, two tag sets are accumulated: **in-class** (a candidate orbit met
the class definition) and **out-of-class** (one did not). Then

**D2 = 100 · Σ_class / (Σ_class + Σ_other)**

where Σ_class sums model population over in-class-tagged bins, and Σ_other
sums it over out-of-class-tagged bins. Fifteen classes are defined
(`pydigest2/classes.py`): Int, NEO, N22, N18, MC, Hun, Pho, MB1, Pal, Han,
MB2, MB3, Hil, JTr, JFC.

The MPC posts a tracklet to the [NEOCP](https://minorplanetcenter.net/iau/NEO/toconfirm_tabular.html)
when its NEO no-ID score reaches **D2,crit = 65** — the threshold in use since
2012.

---

## 3. How well it separates NEOs — §4

From the paper's population studies:

- 14% of NEO tracklets are below 65 *at discovery*, but **99.6% exceed 65 at
  some point** over a simulated 10-year window, and 94% reach D2 = 100.
- **98.5% of non-NEO tracklets** stay below 65.
- Score depends strongly on **rate of motion and ecliptic latitude**. NEOs
  near the ecliptic at low solar elongation get confused with main-belt
  objects — which is exactly where NEOs are missed.

---

## 4. The score is stochastic by design — §4.4.1

Because midpoints are jittered, two runs on the same tracklet can sample
different bins and return different scores. The paper quantifies this over
1000 tracklets × 100 runs each, in the **default random-seed mode that the
MPC actually uses in production** (deliberately, "to avoid bias"):

- only **82%** of tracklets reproduce the same integer D2 run-to-run
- **0.7%** vary by more than 3 points
- one real candidate (`P10Gj15`) scored anywhere from below 60 to 66 across
  1000 runs — flipping its NEOCP eligibility around the threshold of 65

Adding the `repeatable` keyword reseeds the generator with a constant per
tracklet, making runs reproducible. That is the mode this port is validated
in, and the mode to use for any comparison between implementations.

This matters for judging *any* reimplementation: "identical" can only mean
identical to within the algorithm's own run-to-run spread.

---

## 5. Where the code has moved past the paper

Noticed while porting line-by-line. The code, not the paper, is what this
package reproduces.

| | paper (2019) | current code |
|---|---|---|
| Search termination | tag-based only (App. C) | tag-based **plus** forced subdivision on large intervals **plus** an "age" counter granting one extra level |
| Photometric bands | −0.8 for B, +0.4 for anything else, 0 for V | a 23-entry table (g −0.28, r +0.23, w −0.16, o +0.33, u +2.5, …) covering modern survey filters |
| Class boundaries | strict inequalities in Table 8 (e.g. `e < 0.18`) | non-strict (rejects only `e > 0.18`) — a measure-zero difference, but the code is what is ported |
| Per-detection uncertainty | listed as future work (§5.1) | implemented, via ADES `rmsRA`/`rmsDec` and the `noThreshold` keyword (the 2023 paper) |
| Python API | listed as future work (§5.1) | shipped, as the MPC's C-extension wrapper |

The paper's §5.1 also anticipates a smooth 4-D population function replacing
the discrete bins, and a full statistical treatment of all detections
replacing the two-endpoint reduction. Neither has landed; both would change
scores substantially if they did.

---

## 6. Module map

| Concept | C source | This package |
|---|---|---|
| Endpoint synthesis, great-circle RMS | `d2math.c` (`twoObs`, `oneObs`, `gcFit`) | `tracklet.py`, `geometry.py` |
| Solar ephemeris, sidereal time | `d2math.c` (`se2000`, `lst`) | `geometry.py` |
| Dithering, distance/angle search, tagging | `d2math.c` (`offsetMotionVector` … `tagAngle`) | `_search.py` |
| Score assembly | `d2math.c` (`score`) | `ranging.py` |
| Orbit-class predicates | `d2model.c` (`isNeo` …) | `classes.py` |
| (q, e, i, H) bin lookup | `d2model.c` (`qeiToBin`, `hToBin`) | `binning.py` |
| Population model I/O | `d2modelio.c` | `model.py` |
| Observatory codes, MPC80 parsing | `d2mpc.c`, `common.c` | `obscodes.py`, `observations.py` |
| ADES parsing | `d2ades.c` | `observations.py` |
| Config, CLI, output formatting | `d2cli.c`, `digest2.c` | `config.py`, `cli.py`, `output.py`, `intake.py` |

`_search.py` is one deliberately un-decomposed function, for performance
reasons explained in its own docstring and in the README. The readable,
one-concept-per-function form of the predicates and bin lookups lives in
`classes.py` and `binning.py`, which are separately unit-tested and remain the
authoritative definitions.
