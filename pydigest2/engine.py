"""High-level scoring engine: loads the model/obscodes/config once and
scores tracklets against them, optionally fanning out across processes.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import Config, load_config_file
from .constants import DEFAULT_V_MAG
from .model import Model, load_model
from .obscodes import SiteTable
from .observations import Observation
from .paths import find_model_path, find_obscodes_path
from .ranging import ScoreResult, score_tracklet


def compute_vmag(olist: Sequence[Observation]) -> float:
    """Composite tracklet magnitude: mean of reported mags, default 21.0
    (digest2.c: eval(), same rule as d2lib.c's lib_alloc_tracklet)."""
    vals = [o.vmag for o in olist if o.vmag > 0.0]
    return sum(vals) / len(vals) if vals else DEFAULT_V_MAG


@dataclass
class TrackletProblem:
    """Why a tracklet can't be scored (digest2.c: eval()'s guard checks)."""
    reason: str


def validate_tracklet(olist: Sequence[Observation]) -> Optional[TrackletProblem]:
    """Same three guard checks as digest2.c's eval(); reason strings match
    the reference program's messages verbatim (desig prefix is the
    caller's job -- see pydigest2.cli)."""
    if len(olist) < 2:
        return TrackletProblem("single observation. skipped.")
    if olist[-1].mjd < olist[0].mjd:
        return TrackletProblem("observations out of order.")
    if olist[0].ra == olist[-1].ra and olist[0].dec == olist[-1].dec:
        return TrackletProblem("tracklet shows no motion.")
    return None


class Digest2Engine:
    """Loaded model + obscodes + config, ready to score tracklets."""

    def __init__(self, model: Model, site_table: SiteTable, config: Optional[Config] = None):
        self.model = model
        self.site_table = site_table
        self.config = config or Config()

    @classmethod
    def load(cls, model_path: Optional[str] = None, obscodes_path: Optional[str] = None,
              config_path: Optional[str] = None) -> "Digest2Engine":
        model_path = model_path or find_model_path()
        obscodes_path = obscodes_path or find_obscodes_path()
        site_table = SiteTable.from_file(obscodes_path)
        config = load_config_file(config_path, site_table) if config_path else Config()
        model = load_model(model_path)
        return cls(model, site_table, config)

    def score(self, olist: Sequence[Observation], *, is_ades: bool = False,
              class_indices: Optional[Sequence[int]] = None,
              repeatable: Optional[bool] = None) -> ScoreResult:
        problem = validate_tracklet(olist)
        if problem is not None:
            raise ValueError(problem.reason)
        vmag = compute_vmag(olist)
        return score_tracklet(
            olist, vmag=vmag, is_ades=is_ades,
            class_indices=class_indices if class_indices is not None else self.config.class_compute,
            model=self.model, site_table=self.site_table,
            default_obserr_rad=self.config.default_obserr_rad,
            no_threshold=self.config.no_threshold,
            repeatable=self.config.repeatable if repeatable is None else repeatable,
        )


# --- process-pool fan-out for scoring many tracklets -----------------------
#
# Correctness note: a worker MUST score against the exact same Config and
# SiteTable as the calling engine -- including any obserr overrides or
# repeatable/class-set choices made programmatically after load(), which
# a config *file* path can't reconstruct. So the initializer is handed the
# already-built Config/SiteTable objects (small, cheap to pickle) directly,
# rather than a path to re-derive them from. On fork-based platforms
# (Linux/macOS) workers usually inherit a fully-populated engine via
# copy-on-write before the pool even starts, in which case the initializer
# is a no-op; the pickled fallback below only matters on spawn (Windows) or
# if the engine wasn't built until after the pool started.

_worker_engine: Optional[Digest2Engine] = None


def _init_worker(model_path: str, obscodes_path: str, site_table: SiteTable, config: Config) -> None:
    global _worker_engine
    if _worker_engine is not None:
        return  # already inherited a live engine via fork/copy-on-write
    model = load_model(model_path)
    _worker_engine = Digest2Engine(model, site_table, config)


def _score_one(args) -> Tuple[str, object]:
    desig, olist, is_ades = args
    assert _worker_engine is not None
    problem = validate_tracklet(olist)
    if problem is not None:
        return desig, problem
    try:
        return desig, _worker_engine.score(olist, is_ades=is_ades)
    except ArithmeticError as e:
        return desig, TrackletProblem(str(e))


def score_many(
    tracklets: Sequence[Tuple[str, Sequence[Observation], bool]],
    engine: Digest2Engine,
    model_path: str,
    obscodes_path: str,
    n_jobs: Optional[int] = None,
) -> List[Tuple[str, object]]:
    """Score many (designation, observations, is_ades) tracklets, using a
    process pool when there's enough work to be worth it.

    Every tracklet is fully independent (its own RNG stream, its own
    per-class tag sets), which is what makes this embarrassingly
    parallel -- this is the throughput lever for this port, since each
    individual tracklet's adaptive search is inherently sequential (see
    :mod:`pydigest2.ranging`). Workers score against ``engine``'s exact
    ``config``/``site_table`` (see the correctness note above); on
    Linux/macOS the already-loaded model is inherited via copy-on-write
    too, so there's no per-worker reload cost in the common case.
    """
    n_jobs = n_jobs or os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(tracklets) or 1))

    if n_jobs <= 1 or len(tracklets) < 8:
        return [_score_sequential(desig, olist, is_ades, engine) for desig, olist, is_ades in tracklets]

    global _worker_engine
    _worker_engine = engine  # visible to forked children immediately; see _init_worker

    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_init_worker,
        initargs=(model_path, obscodes_path, engine.site_table, engine.config),
    ) as pool:
        return list(pool.map(_score_one, tracklets, chunksize=max(1, len(tracklets) // (n_jobs * 4) or 1)))


def _score_sequential(desig, olist, is_ades, engine: Digest2Engine):
    problem = validate_tracklet(olist)
    if problem is not None:
        return desig, problem
    try:
        return desig, engine.score(olist, is_ades=is_ades)
    except ArithmeticError as e:
        return desig, TrackletProblem(str(e))
