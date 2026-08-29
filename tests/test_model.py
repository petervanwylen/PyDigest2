from pathlib import Path

from pydigest2.model import load_model

DATA = Path(__file__).parent.parent / "pydigest2" / "data" / "digest2.model.csv"


def test_load_real_model_csv_shapes():
    model = load_model(DATA, use_cache=False)
    from pydigest2.constants import D2CLASSES, EX, HX, IX, QX
    assert model.all_ss.shape == (QX, EX, IX, HX)
    assert model.unk_ss.shape == (QX, EX, IX, HX)
    assert model.all_class.shape == (D2CLASSES, QX, EX, IX, HX)
    assert model.unk_class.shape == (D2CLASSES, QX, EX, IX, HX)


def test_model_population_is_nonnegative_and_nonzero():
    model = load_model(DATA, use_cache=False)
    assert (model.all_ss >= 0).all()
    assert model.all_ss.sum() > 0
    assert model.unk_ss.sum() > 0


def test_class_population_never_exceeds_whole_solar_system():
    model = load_model(DATA, use_cache=False)
    # a class's "All" population at any bin can't exceed the SS total there
    assert (model.all_class <= model.all_ss[None, ...] + 1e-6).all()


def test_model_cache_round_trip(tmp_path):
    model = load_model(DATA, use_cache=False)
    from pydigest2.model import save_model_cache
    import numpy as np

    cache_path = tmp_path / "model.npz"
    save_model_cache(model, cache_path)
    reloaded = load_model(cache_path)
    assert np.array_equal(model.all_ss, reloaded.all_ss)
    assert np.array_equal(model.all_class, reloaded.all_class)
