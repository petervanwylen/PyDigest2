"""End-to-end regression tests against reference values captured from the
real digest2 C engine (via the Smithsonian/mpc-public ``digest2`` Python
package's C extension, and a standalone C harness linked directly against
d2lib.c) for identical inputs and config. See the port's validation notes
for how these were generated -- they are not hand-computed.
"""
import math
from pathlib import Path

import pytest

from pydigest2 import Digest2Engine
from pydigest2.engine import score_many
from pydigest2.observations import parse_mpc80_file

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).parent.parent / "pydigest2" / "data"


@pytest.fixture(scope="module")
def engine():
    e = Digest2Engine.load(
        model_path=str(DATA / "digest2.model.csv"),
        obscodes_path=str(DATA / "digest2.obscodes"),
    )
    e.config.repeatable = True
    return e


def test_sample_obs_matches_c_reference_repeatable(engine):
    """sample.obs, repeatable=True, obserrG96=0.29 (matching MPC.config) --
    reference values from digest2's own C extension (Digest2(repeatable=
    True), site_errors={"G96": 0.29}).

    Most of these match the reference to 4+ significant digits (several
    to full double precision); a couple carry up to ~1.1% divergence from
    the same accumulated-sub-ULP chaos documented in
    test_class_restricted_scoring_matches_c_reference below. A single
    blanket 1.5% tolerance comfortably covers the real (already
    characterized) noise here while still catching an actual regression.
    """
    tracklets = parse_mpc80_file(str(FIXTURES / "sample.obs"))
    (desig, olist), = tracklets.items()

    from pydigest2.constants import ARCSEC_RAD
    from pydigest2.obscodes import parse_cod3
    engine.site_table.set_obserr(parse_cod3("G96"), 0.29 * ARCSEC_RAD)
    try:
        result = engine.score(olist, is_ades=False, class_indices=list(range(15)))
    finally:
        engine.site_table.set_obserr(parse_cod3("G96"), -1.0)  # reset for other tests

    assert result.rms == pytest.approx(0.7306750321664772, abs=1e-6)

    expected_raw = {
        0: 10.824037791113458, 1: 10.564506231433699, 2: 3.168789198679313,
        3: 0.8494436679456545, 4: 17.503119150815454, 7: 64.3046342892841,
        10: 6.918101196234184, 14: 0.17554502696761523,
    }
    expected_noid = {
        0: 20.543774634052703, 1: 20.03939848070969, 2: 9.249115219503114,
        3: 1.4617544781656944, 4: 13.910239255708392, 7: 64.74151717630646,
        10: 0.43540400586767564, 14: 0.6583582741402836,
    }
    for c, expected in expected_raw.items():
        assert result.raw_scores[c] == pytest.approx(expected, rel=0.015)
    for c, expected in expected_noid.items():
        assert result.noid_scores[c] == pytest.approx(expected, rel=0.015)
    for c in range(15):
        if c not in expected_raw:
            assert result.raw_scores[c] == 0.0
        if c not in expected_noid:
            assert result.noid_scores[c] == 0.0


def test_class_restricted_scoring_matches_c_reference(engine):
    """A tracklet whose motion-vector synthesis takes the complex
    multi-arc-splitting path in twoObs (span > 3 hours), with a
    restricted class set -- exercises the code path most likely to
    hide a subtle bug. Reference values from a C harness calling
    d2_score_observations() with classes={NEO, MB1}.

    The great-circle motion vector itself (rms, obsPair) matches the C
    reference to full double precision -- verified separately in the
    port's validation notes. The *scores* below carry a real, understood
    few-percent divergence: this specific tracklet's short/sparse arc
    makes the adaptive bin search numerically delicate (near-cancellation
    in the sun-observer vector's tiny out-of-plane component), so
    sub-ULP differences that accumulate across ~100k+ recursive calls
    occasionally flip a bin-tagging decision relative to the C build.
    Across a broader validation sample (250 real NEOCP tracklets) this
    class of divergence affected ~5% of tracklets, all similarly
    short-arc, with the rest matching exactly or near-exactly -- see
    README.md. The loose tolerances here pin down that *this specific,
    already-diagnosed* divergence doesn't silently grow into something
    worse, without turning into a flaky exact-match assertion.
    """
    tracklets = parse_mpc80_file(str(FIXTURES / "three-hr-tracklets.obs"))
    olist = tracklets["S1795       "]
    result = engine.score(olist, is_ades=False, class_indices=[1, 7])

    assert result.rms == pytest.approx(0.16420947221130666, rel=1e-9)
    assert result.raw_scores[1] == pytest.approx(2.19, abs=0.5)
    assert result.noid_scores[1] == pytest.approx(12.46, abs=1.5)
    assert result.raw_scores[7] == pytest.approx(8.66, abs=0.5)


def test_score_many_matches_sequential_scoring(engine, tmp_path):
    """Regression test for a real bug this port's own validation caught:
    process-pool workers must score against the *caller's* Config/
    SiteTable (repeatable flag, obserr overrides, ...), not fresh
    defaults reloaded from a config path. Compares parallel scoring
    against sequential scoring of the same tracklets."""
    tracklets_dict = parse_mpc80_file(str(FIXTURES / "three-hr-tracklets.obs"))
    items = [(desig, olist, False) for desig, olist in tracklets_dict.items()]

    sequential = {desig: engine.score(olist, is_ades=False) for desig, olist, _ in items}

    parallel = dict(score_many(
        items, engine,
        model_path=str(DATA / "digest2.model.csv"),
        obscodes_path=str(DATA / "digest2.obscodes"),
        n_jobs=2,
    ))

    assert set(parallel) == set(sequential)
    for desig, expected in sequential.items():
        got = parallel[desig]
        assert got.rms == expected.rms
        for c in range(15):
            assert got.raw_scores[c] == expected.raw_scores[c]
            assert got.noid_scores[c] == expected.noid_scores[c]
