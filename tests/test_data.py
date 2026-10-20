"""Dataset loading: sources are tried in order and every copy, cached or
downloaded, must match the expected fingerprint. Uses file:// URLs, no network."""
import numpy as np
import pandas as pd
import pytest

from functions.data import fingerprint, load_dataset


@pytest.fixture
def sample():
    # Multiples of 1/8 are exact in decimal and binary, so CSV copies parse back exactly.
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "PC1": rng.integers(-80, 80, size=20) / 8,
        "PC2": rng.integers(-80, 80, size=20) / 8,
        "FACTOR1": rng.choice(["V1", "V2", "V3"], size=20),
        "y": rng.integers(0, 2, size=20),
        "t": rng.integers(0, 2, size=20),
    })


def as_parquet(df, path):
    # OpenML stores y and t as uint8
    df.astype({"y": "uint8", "t": "uint8"}).to_parquet(path, index=False)
    return path.as_uri()


def as_csv(df, path):
    df.to_csv(path, index=False)
    return path.as_uri()


def test_parquet_and_csv_copies_have_the_same_fingerprint(sample, tmp_path):
    as_parquet(sample, tmp_path / "d.pq")
    as_csv(sample, tmp_path / "d.csv")
    parquet = pd.read_parquet(tmp_path / "d.pq")
    csv = pd.read_csv(tmp_path / "d.csv")

    assert fingerprint(parquet) == fingerprint(csv) == fingerprint(sample)


def test_cache_keeps_full_float_precision(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"PC1": rng.normal(size=50) / 3, "y": np.zeros(50, int), "t": np.ones(50, int)})
    source = as_parquet(df, tmp_path / "d.pq")
    cache = tmp_path / "cache.parquet"

    load_dataset(cache, sources=[source], expected=fingerprint(df))

    pd.testing.assert_frame_equal(load_dataset(cache, sources=[], expected=fingerprint(df)), df)


def test_fingerprint_detects_changed_values_order_and_columns(sample):
    changed = sample.copy()
    changed.loc[3, "PC2"] += 1e-12
    reordered = sample.iloc[::-1].reset_index(drop=True)
    renamed = sample.rename(columns={"PC1": "PCX"})

    reference = fingerprint(sample)
    assert len({reference, fingerprint(changed), fingerprint(reordered), fingerprint(renamed)}) == 4


def test_downloads_from_first_working_source_and_caches(sample, tmp_path):
    cache = tmp_path / "data" / "dataset.parquet"
    sources = [(tmp_path / "missing.pq").as_uri(), as_parquet(sample, tmp_path / "d.pq")]

    df = load_dataset(cache, sources=sources, expected=fingerprint(sample))

    pd.testing.assert_frame_equal(df, sample)
    assert cache.exists()
    pd.testing.assert_frame_equal(load_dataset(cache, sources=[], expected=fingerprint(sample)), sample)


def test_falls_back_when_a_source_has_different_content(sample, tmp_path):
    other = sample.copy()
    other.loc[0, "PC1"] = 99.0
    sources = [as_parquet(other, tmp_path / "other.pq"), as_csv(sample, tmp_path / "good.csv")]

    df = load_dataset(tmp_path / "cache.parquet", sources=sources, expected=fingerprint(sample))

    pd.testing.assert_frame_equal(df, sample)


def test_fails_loudly_when_no_source_matches(sample, tmp_path):
    other = sample.copy()
    other.loc[0, "y"] = 1 - other.loc[0, "y"]
    sources = [(tmp_path / "missing.csv").as_uri(), as_csv(other, tmp_path / "other.csv")]
    cache = tmp_path / "cache.parquet"

    with pytest.raises(RuntimeError) as error:
        load_dataset(cache, sources=sources, expected=fingerprint(sample))

    assert "missing.csv" in str(error.value) and "does not match" in str(error.value)
    assert not cache.exists()


def test_rejects_a_modified_cached_copy(sample, tmp_path):
    cache = tmp_path / "cache.parquet"
    modified = sample.copy()
    modified.loc[5, "PC1"] = 0.0
    modified.to_parquet(cache, index=False)

    with pytest.raises(ValueError, match="delete it"):
        load_dataset(cache, sources=[], expected=fingerprint(sample))
