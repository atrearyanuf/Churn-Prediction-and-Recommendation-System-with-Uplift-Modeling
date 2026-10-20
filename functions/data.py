"""Download and verify the Orange Belgium churn-uplift dataset.

Verhelst, Mercier, Shrestha & Bontempi (2023), arXiv:2312.07206.
Licence: CC BY-NC-ND 4.0 (as published on OpenML, dataset 45580).
"""
import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

SOURCES = (
    # OpenML dataset 45580, published with the paper
    "https://data.openml.org/datasets/0004/45580/dataset_45580.pq",
    # Copy used for the original course submission
    "https://www.dropbox.com/scl/fi/2jibh9twr8jml97mqtyki/churn_uplift_anonymized.csv"
    "?rlkey=907ib1npo2a9zk27ot5ra6zsg&dl=1",
)

# Fingerprint of the data the published notebook results were produced with.
EXPECTED_FINGERPRINT = "d0031a4fe8b7a1c5580f71a78d71ed7bd75360b5a0cd429796cfeed0f5d11c02"


def fingerprint(df):
    """SHA-256 of column names and values, in row order.

    y and t are compared as int64, so the parquet copy (uint8) and CSV copies agree.
    """
    df = df.astype({"y": "int64", "t": "int64"})
    digest = hashlib.sha256(",".join(df.columns).encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _download(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        content = io.BytesIO(response.read())
    # The published results came from pandas' default CSV parser on the Dropbox
    # copy; the OpenML parquet holds exactly those values.
    return pd.read_parquet(content) if url.endswith(".pq") else pd.read_csv(content)


def load_dataset(path, sources=SOURCES, expected=EXPECTED_FINGERPRINT):
    """Return the dataset, downloading it to the parquet file at `path` on first use.

    The cache is parquet because it stores floats exactly; a CSV re-parsed by
    pandas can change the last bit of some values.

    Sources are tried in order. Every copy, cached or downloaded, must match
    `expected`, so a changed or corrupted copy raises instead of silently
    producing different results.
    """
    path = Path(path)
    if path.exists():
        df = pd.read_parquet(path)
        if fingerprint(df) != expected:
            raise ValueError(f"{path} is not the expected dataset; delete it to download a fresh copy")
        return df

    errors = []
    for url in sources:
        try:
            df = _download(url)
        except (OSError, ValueError) as exc:  # network, HTTP or parse errors: try the next source
            errors.append(f"{url}: {exc}")
            continue
        if fingerprint(df) != expected:
            errors.append(f"{url}: content does not match the expected dataset")
            continue
        df = df.astype({"y": "int64", "t": "int64"})
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        df.to_parquet(partial, index=False)
        partial.replace(path)  # never leave a half-written cache behind
        return df
    raise RuntimeError("Could not get the dataset from any source:\n  " + "\n  ".join(errors))
