"""The adaptive Monte-Carlo orbit search: digest2's actual algorithm.

Ported from ``d2math.c``: ``tagAngle``, ``aRange``, ``solveAngleRange``,
``searchAngles``, ``offsetMotionVector``, ``setupDistanceDependentVectors``,
``searchDistance``, ``dRange``, and the final score computation in
``score()``.

Why this is scalar Python, not NumPy, per tracklet
----------------------------------------------------
This is a *data-dependent recursive* search: at each node, whether (and
in what order) the two halves get explored depends on whether the
midpoint orbit lands in a previously untagged model bin -- and in
"repeatable" mode, that midpoint itself is jittered by a per-tracklet
LCG stream that must be consumed in exactly this recursion's call
order to reproduce the reference program's bin selection. That rules
out batching a single tracklet's search into flat NumPy array ops:
there is no fixed set of "steps" to vectorize over, and reordering
calls would desync the RNG.

What *is* embarrassingly parallel is the tracklet dimension: every
tracklet's search is fully independent (its own LCG state, its own
per-class tag sets), and a real input file scores thousands of them.
That's where this package gets its throughput -- see
:mod:`pydigest2.engine`, which fans this function out across
processes -- while the handful of 3-vector ops in one search node stay
plain Python floats/tuples, which for objects this small outrun NumPy's
per-call array overhead by roughly an order of magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, atan2, cos, degrees, exp, log10, pi, sin, sqrt
from typing import List, Optional, Sequence, Set, Tuple

from .binning import h_to_bin, qei_to_bin
from .classes import CLASS_TESTS
from .constants import (
    AGE_LIMIT, ARCSEC_RAD, INV_K, MAX_DISTANCE, MIN_ANGLE_STEP,
    MIN_DISTANCE, MIN_DISTANCE_STEP, U,
)
from .geometry import Vec3, cross3, dot3, ec_rotate, lst, se2000, sub3
from .model import Model
from .obscodes import SiteTable, parse_cod3
from .observations import Observation
from .rng import Lcg
from .tracklet import clip_err, two_obs

BinKey = Tuple[int, int, int, int]

import sys
if sys.getrecursionlimit() < 2000:
    sys.setrecursionlimit(2000)  # comfortable headroom over observed search depths


@dataclass
class PerClass:
    sum_all_in_class: float = 0.0
    sum_unk_in_class: float = 0.0
    sum_all_out_of_class: float = 0.0
    sum_unk_out_of_class: float = 0.0
    tag_in_class: Set[BinKey] = field(default_factory=set)
    tag_out_of_class: Set[BinKey] = field(default_factory=set)
    d_in_class: Set[BinKey] = field(default_factory=set)
    d_out_of_class: Set[BinKey] = field(default_factory=set)
    raw_score: float = 0.0
    noid_score: float = 0.0


@dataclass
class TrackletState:
    obs_pair: Tuple[Observation, Observation]
    obs_err: Tuple[float, float]           # radians
    no_obs_err: bool
    rand: Lcg
    vmag: float
    sun_observer: Tuple[Vec3, Vec3]
    dt: float
    invdt: float
    invdtsq: float
    soe: float
    coe: float
    is_ades: bool
    no_threshold: bool
    class_indices: Sequence[int]           # global class ids being computed
    model: Model

    per_class: List[PerClass] = field(default_factory=list)
    observer_object_unit: List[Optional[Vec3]] = field(default_factory=lambda: [None, None])

    # distance-dependent scratch, rewritten by setup_distance_dependent_vectors
    sun_object0: Vec3 = (0.0, 0.0, 0.0)
    sun_object0_mag: float = 0.0
    sun_object0_magsq: float = 0.0
    observer_object0: Vec3 = (0.0, 0.0, 0.0)
    observer_object0_mag: float = 0.0
    observer1_object0: Vec3 = (0.0, 0.0, 0.0)
    observer1_object0_mag: float = 0.0
    observer1_object0_magsq: float = 0.0
    tz: float = 0.0
    hmag: float = 0.0
    hmag_bin: int = 0

    d_any_tag: bool = False
    d_tag: Set[BinKey] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.per_class:
            self.per_class = [PerClass() for _ in self.class_indices]
        self._class_tests = [CLASS_TESTS[c] for c in self.class_indices]


def clear_d_tags(state: TrackletState) -> None:
    state.d_tag.clear()
    for cl in state.per_class:
        cl.d_in_class.clear()
        cl.d_out_of_class.clear()


def _update_rms_values(rms_ra: float, rms_dec: float, error_from_config: float,
                        no_threshold: bool) -> Tuple[float, float]:
    """Port of updateRMSValues (d2math.c). ``error_from_config`` here is
    always a local, per-call value (radians); mutating its C-side
    "effective" fallback never escapes this function."""
    efc = error_from_config if error_from_config else 1.0
    if not rms_ra and not rms_dec:
        return efc, efc
    if not rms_ra:
        rms_ra = rms_dec
    if not rms_dec:
        rms_dec = rms_ra
    if no_threshold:
        if efc > rms_ra:
            rms_ra = efc
        if efc > rms_dec:
            rms_dec = efc
        return rms_ra, rms_dec
    min_t, max_t = 0.7 * efc, 5.0 * efc
    if rms_ra < min_t:
        rms_ra = min_t
    if rms_dec < min_t:
        rms_dec = min_t
    if rms_ra > max_t:
        rms_ra = max_t
    if rms_dec > max_t:
        rms_dec = max_t
    if efc > rms_ra:
        rms_ra = efc
    if efc > rms_dec:
        rms_dec = efc
    return rms_ra, rms_dec


def offset_motion_vector(state: TrackletState, rx: int, dx: int) -> None:
    for i in (0, 1):
        obs = state.obs_pair[i]
        error_from_config = state.obs_err[i]
        if state.is_ades:
            rms_ra, rms_dec = _update_rms_values(
                obs.rms_ra, obs.rms_dec, error_from_config, state.no_threshold)
            dec = obs.dec + dx * rms_dec * 0.5
            cosdec = cos(dec)
            ra = obs.ra + rx * rms_ra * 0.5 * cosdec
        else:
            dec = obs.dec + dx * error_from_config * 0.5
            cosdec = cos(dec)
            ra = obs.ra + rx * error_from_config * 0.5 * cosdec
        v = (cos(ra) * cosdec, sin(ra) * cosdec, sin(dec))
        state.observer_object_unit[i] = ec_rotate(v, state.soe, state.coe)
        rx, dx = -rx, -dx


# Exponents in the H-magnitude phase-integral approximation (setupDistance-
# DependentVectors in d2math.c). The C code approximates x**0.63/x**1.22 with
# a 1024-point interpolated lookup table purely for speed; Python isn't
# calling this tens of millions of times in a tight loop, so we just use
# the exact power -- strictly more accurate, and the table's interpolation
# error is well under the 1% score tolerance this port targets anyway.
def setup_distance_dependent_vectors(state: TrackletState, d: float) -> None:
    unit0 = state.observer_object_unit[0]
    state.observer_object0_mag = d
    state.observer_object0 = (unit0[0] * d, unit0[1] * d, unit0[2] * d)

    so = state.sun_observer[0]
    oo0 = state.observer_object0
    state.sun_object0 = (so[0] + oo0[0], so[1] + oo0[1], so[2] + oo0[2])
    state.sun_object0_magsq = dot3(state.sun_object0, state.sun_object0)
    state.sun_object0_mag = sqrt(state.sun_object0_magsq)

    so1 = state.sun_observer[1]
    sun0 = state.sun_object0
    state.observer1_object0 = (sun0[0] - so1[0], sun0[1] - so1[1], sun0[2] - so1[2])
    state.observer1_object0_magsq = dot3(state.observer1_object0, state.observer1_object0)
    state.observer1_object0_mag = sqrt(state.observer1_object0_magsq)

    rdelta = state.observer_object0_mag * state.sun_object0_mag
    cospsi = dot3(state.observer_object0, state.sun_object0) / rdelta

    if cospsi > -0.9999:
        tanhalf = sqrt(1.0 - cospsi * cospsi) / (1.0 + cospsi)
        phi1 = exp(-3.33 * tanhalf ** 0.63)
        phi2 = exp(-1.87 * tanhalf ** 1.22)
        state.hmag = state.vmag - 5.0 * log10(rdelta) + 2.5 * log10(0.85 * phi1 + 0.15 * phi2)
    else:
        state.hmag = 30.0  # pointed straight at the sun; give it a valid but meaningless H

    state.hmag_bin = h_to_bin(state.hmag)


def tag_angle(state: TrackletState, an: float) -> bool:
    d2 = state.observer1_object0_mag * sin(an) / sin(pi - an - state.tz)

    unit1 = state.observer_object_unit[1]
    o1o0 = state.observer1_object0
    scale = state.invdt * INV_K
    v = (
        (d2 * unit1[0] - o1o0[0]) * scale,
        (d2 * unit1[1] - o1o0[1]) * scale,
        (d2 * unit1[2] - o1o0[2]) * scale,
    )

    hv = cross3(state.sun_object0, v)
    hsq = dot3(hv, hv)
    hm = sqrt(hsq)

    vsq = dot3(v, v)
    temp = 2.0 - state.sun_object0_mag * vsq
    if state.sun_object0_mag > temp * 100.0:
        return False

    orbit_a = state.sun_object0_mag / temp
    inva = temp / state.sun_object0_mag
    orbit_e = sqrt(1.0 - hsq * inva)
    if orbit_e > 0.99:
        return False

    izero = hv[2] >= hm
    orbit_i = 0.0 if izero else degrees(acos(hv[2] / hm))

    q = orbit_a * (1.0 - orbit_e)
    bin3 = qei_to_bin(q, orbit_e, orbit_i)
    if bin3 is None:
        return False
    key = (bin3[0], bin3[1], bin3[2], state.hmag_bin)

    new_tag = False
    hmag = state.hmag
    for cl, test in zip(state.per_class, state._class_tests):
        if test(q, orbit_e, orbit_i, hmag):
            if key not in cl.d_in_class:
                cl.d_in_class.add(key)
                new_tag = True
        else:
            if key not in cl.d_out_of_class:
                cl.d_out_of_class.add(key)
                new_tag = True

    if new_tag:
        state.d_any_tag = True
        state.d_tag.add(key)
    return new_tag


def a_range(state: TrackletState, ang1: float, ang2: float, age: int) -> None:
    d3 = (ang2 - ang1) / 3.0
    mid = ang1 + d3 + d3 * state.rand.next()
    if tag_angle(state, mid) or d3 > MIN_ANGLE_STEP:
        a_range(state, ang1, mid, 0)
        a_range(state, mid, ang2, 0)
        return
    if age < AGE_LIMIT:
        a_range(state, ang1, mid, age + 1)
        a_range(state, mid, ang2, age + 1)


def solve_angle_range(state: TrackletState) -> Optional[Tuple[float, float]]:
    th = dot3(state.observer1_object0, state.observer_object_unit[1]) / state.observer1_object0_mag
    state.tz = acos(th)

    aa = state.invdtsq
    bb = (-2.0 * state.observer1_object0_mag * th) * aa
    cc = state.observer1_object0_magsq * aa - 2.0 * U / state.sun_object0_mag
    dsc = bb * bb - 4 * aa * cc
    if not (dsc > 0.0):
        return None

    sd = sqrt(dsc)
    sd1 = -sd
    inv2aa = 0.5 / aa
    ang1 = ang2 = None
    while True:
        d2 = (-bb + sd1) * inv2aa
        d2s = d2 * d2
        nns = d2s + state.observer1_object0_magsq - 2.0 * d2 * state.observer1_object0_mag * th
        nn = sqrt(nns)
        ca = (nns + state.observer1_object0_magsq - d2s) / (2.0 * nn * state.observer1_object0_mag)
        sa = d2 * sin(state.tz) / nn
        ang2 = 2.0 * atan2(sa, 1.0 + ca)
        if sd1 == sd:
            break
        ang1 = ang2
        sd1 = sd
    return ang1, ang2


def search_angles(state: TrackletState) -> bool:
    r = solve_angle_range(state)
    if r is None:
        return False
    a_range(state, r[0], r[1], 0)

    if not state.d_any_tag:
        return False

    new_tag = False
    model = state.model
    for key in state.d_tag:
        iq, ie, ii, ih = key
        for cl, cls_id in zip(state.per_class, state.class_indices):
            if key in cl.d_in_class and key not in cl.tag_in_class:
                new_tag = True
                cl.tag_in_class.add(key)
                cl.sum_all_in_class += model.all_class[cls_id, iq, ie, ii, ih]
                cl.sum_unk_in_class += model.unk_class[cls_id, iq, ie, ii, ih]
            if key in cl.d_out_of_class and key not in cl.tag_out_of_class:
                new_tag = True
                cl.tag_out_of_class.add(key)
                all_ss = model.all_ss[iq, ie, ii, ih]
                unk_ss = model.unk_ss[iq, ie, ii, ih]
                cl.sum_all_out_of_class += all_ss - model.all_class[cls_id, iq, ie, ii, ih]
                cl.sum_unk_out_of_class += unk_ss - model.unk_class[cls_id, iq, ie, ii, ih]
    return new_tag


def search_distance(state: TrackletState, d: float) -> bool:
    clear_d_tags(state)
    state.d_any_tag = False
    new_tag = False
    for ri in (-1, 0, 1):
        for di in (-1, 0, 1):
            offset_motion_vector(state, ri, di)
            setup_distance_dependent_vectors(state, d)
            if search_angles(state):
                new_tag = True
            if state.no_obs_err:
                return new_tag
    return new_tag


def d_range(state: TrackletState, d1: float, d2: float, age: int) -> None:
    dmid = (d1 + d2) * 0.5
    if search_distance(state, dmid) or d2 - d1 > MIN_DISTANCE_STEP:
        d_range(state, d1, dmid, 0)
        d_range(state, dmid, d2, 0)
        return
    if age < AGE_LIMIT:
        d_range(state, d1, dmid, age + 1)
        d_range(state, dmid, d2, age + 1)


def gc_rms_prime_ades(olist: Sequence[Observation], obs_err0_rad: float,
                       no_threshold: bool) -> float:
    """d2math.c: gcRmsPrimeAdes -- RMS from the ADES-reported per-obs
    uncertainties, over *all* observations (not just the motion-vector pair)."""
    efc = obs_err0_rad / ARCSEC_RAD
    if not efc:
        efc = 1.0
    s = 0.0
    for obs in olist:
        rms_ra = obs.rms_ra / ARCSEC_RAD
        rms_dec = obs.rms_dec / ARCSEC_RAD
        if not no_threshold:
            min_t, max_t = 0.7 * efc, 5.0 * efc
            rms_ra = min(max(rms_ra, min_t), max_t)
            rms_dec = min(max(rms_dec, min_t), max_t)
        s += rms_ra * rms_ra + rms_dec * rms_dec
    return sqrt(s / len(olist))


def gc_rms_prime_mpc(obs_err0_rad: float) -> float:
    """d2math.c: gcRmsPrimeMPC. The per-obs term is a constant (the pair's
    obsErr[0]) repeated ``len(olist)`` times, so the RMS reduces exactly
    to ``sqrt(2) * obsErr[0]`` independent of tracklet length."""
    er = obs_err0_rad / ARCSEC_RAD
    return sqrt(2.0) * er


@dataclass
class ScoreResult:
    rms: float          # great-circle RMS of the tracklet, arcsec (0 if only 2 obs)
    rms_prime: float     # ADES rmsPrime, arcsec (0 if not ADES / not available)
    raw_scores: dict     # {class_index: score}
    noid_scores: dict    # {class_index: score}


def score_tracklet(olist: Sequence[Observation], *, vmag: float, is_ades: bool,
                    class_indices: Sequence[int], model: Model, site_table: SiteTable,
                    default_obserr_rad: float, no_threshold: bool,
                    repeatable: bool, seed: int = 3) -> ScoreResult:
    """Score one tracklet. Direct port of d2math.c's ``score()`` driver,
    combining motion-vector synthesis, the adaptive search, and the
    final raw/no-ID percentage computation.
    """
    mv = two_obs(olist, site_table)
    obs_pair = mv.obs_pair

    dt = obs_pair[1].mjd - obs_pair[0].mjd
    invdt = 1.0 / dt
    invdtsq = invdt * invdt

    obs_err = [0.0, 0.0]
    sun_observer: List[Vec3] = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    soe = coe = 0.0
    for i in (0, 1):
        obs = obs_pair[i]
        site = site_table[parse_cod3(obs.obscode)]
        obs_err[i] = clip_err(mv.obs_err_rms[i], site, default_obserr_rad)
        sun_earth, soe, coe = se2000(obs.mjd)  # tk->soe/coe end up holding pair[1]'s values
        if obs.spacebased:
            v = obs.earth_observer
        else:
            th = lst(obs.mjd, site.longitude)
            v = (site.rho_cos_phi * cos(th), site.rho_cos_phi * sin(th), site.rho_sin_phi)
        v = sub3(v, sun_earth)
        sun_observer[i] = ec_rotate(v, soe, coe)

    no_obs_err = obs_err[0] == 0.0 and obs_err[1] == 0.0

    state = TrackletState(
        obs_pair=obs_pair, obs_err=tuple(obs_err), no_obs_err=no_obs_err,
        rand=Lcg(seed if repeatable else _random_odd_seed()),
        vmag=vmag, sun_observer=tuple(sun_observer), dt=dt, invdt=invdt, invdtsq=invdtsq,
        soe=soe, coe=coe, is_ades=is_ades, no_threshold=no_threshold,
        class_indices=list(class_indices), model=model,
    )

    search_distance(state, MIN_DISTANCE)
    search_distance(state, MAX_DISTANCE)
    d_range(state, MIN_DISTANCE, MAX_DISTANCE, 0)

    rms_prime = 0.0
    if is_ades:
        rms_prime = gc_rms_prime_ades(olist, obs_err[0], no_threshold)
        if rms_prime == 0.0:
            rms_prime = gc_rms_prime_mpc(obs_err[0])

    raw_scores, noid_scores = {}, {}
    for cls_id, cl in zip(state.class_indices, state.per_class):
        denom_all = cl.sum_all_in_class + cl.sum_all_out_of_class
        raw = (100.0 * cl.sum_all_in_class / denom_all if denom_all > 0.0
               else (100.0 if cls_id < 2 else 0.0))
        denom_unk = cl.sum_unk_in_class + cl.sum_unk_out_of_class
        noid = (100.0 * cl.sum_unk_in_class / denom_unk if denom_unk > 0.0
                else (100.0 if cls_id < 2 else 0.0))
        # model lookups are numpy float64 scalars; the public result is
        # plain Python floats so callers never have to think about it.
        raw_scores[cls_id] = float(raw)
        noid_scores[cls_id] = float(noid)

    return ScoreResult(rms=mv.rms, rms_prime=rms_prime, raw_scores=raw_scores,
                        noid_scores=noid_scores)


def _random_odd_seed() -> int:
    import os
    return (int.from_bytes(os.urandom(7), "little") | 1)
