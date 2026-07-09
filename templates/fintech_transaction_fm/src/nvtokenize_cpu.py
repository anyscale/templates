"""CPU (pandas) mirror of the vendored NVIDIA tokenizer's preprocess + transform.

The vendored ``src/nvidia_tokenizer`` package is cuDF-based and physically requires a
GPU. This module reimplements exactly two of its functions on pandas so the data stages
can run — and scale — on CPU workers:

* :func:`preprocess_cpu` mirrors ``FinancialTokenizerPipeline.preprocess`` field for field.
* :func:`transform_cpu` mirrors ``FinancialTokenizerPipeline.transform`` for the standard
  12-step configuration (``amount_strategy="fixed"``, no time-delta step).

The vendored files are untouched; NVIDIA's GPU path remains the reference implementation.
Byte-identity of the two paths is proven by ``scripts/verify_cpu_tokenizer.py`` (Stage 0
of PLAN_RAY_DATA.md) before anything downstream relies on this module. The one
non-obvious equivalence: cuDF ``Series.hash_values()`` is MurmurHash3_x86_32 with seed 0,
reproduced here with ``mmh3.hash(utf8_bytes, signed=False)``.

Vocabulary/encoding need no mirror: ``FinancialTabularTokenizer`` builds its vocab from
static tables (pure Python, no cuDF ops), so constructing it and calling ``encode()``
already works on CPU nodes.
"""
import numpy as np
import pandas as pd

from .nvidia_tokenizer.financial_pipeline import (
    ALL_STATES,
    CHIP_MAPPING,
    INDUSTRY_RANGES,
    KNOWN_MCCS,
)

# transform() output columns, in the pipeline's step order.
TOKEN_COLS = ["amt_val", "merch_hash", "mcc_int", "mcc_str", "hour", "dow", "month",
              "card", "chip_upper", "zip3", "state_clean", "cust"]


def clean_merchant(raw: pd.Series) -> pd.Series:
    """Mirror of the merchant-name cleanup: upper-case, strip to ``[A-Z0-9\\s-]``."""
    return raw.astype(str).str.upper().str.replace(r"[^A-Z0-9\s\-]", "", regex=True)


def merchant_hash(clean: pd.Series) -> np.ndarray:
    """cuDF ``hash_values()`` equivalent: MurmurHash3_x86_32, seed 0, as uint32."""
    import mmh3
    return np.fromiter((mmh3.hash(s, signed=False) for s in clean.astype(str)),
                       dtype=np.uint32, count=len(clean))


def preprocess_cpu(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas mirror of ``FinancialTokenizerPipeline.preprocess`` (identical output).

    Takes the native TabFormer schema (``User, Card, Year, ..., Is Fraud?``) and returns
    the frame with normalized column names plus the derived per-field columns, globally
    sorted by ``(user, card, time_full)``. The sort is stable (mergesort): the raw CSV is
    already in per-card time order, so ties (same-minute transactions) keep source order.
    """
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    amt_f = df["amount"].astype(str).str.replace("$", "", regex=False).astype(float)
    df["amt_val"] = sum((amt_f >= t).astype("int32") for t in (10, 50, 100, 500, 1000, 5000))

    df["merch_hash"] = merchant_hash(clean_merchant(df["merchant_name"]))

    mcc = df["mcc"].fillna(-1).astype(int)
    df["mcc_int"] = mcc
    df["mcc_str"] = mcc.astype(str)

    date_str = (df["year"].astype(str) + "-"
                + df["month"].astype(str).str.zfill(2) + "-"
                + df["day"].astype(str).str.zfill(2) + " "
                + df["time"].fillna("00:00").astype(str))
    dt = pd.to_datetime(date_str, format="%Y-%m-%d %H:%M")
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month

    df["card"] = df["card"].astype(int).clip(0, 9)
    df["chip_upper"] = df["use_chip"].astype(str).str.upper()

    zip_col = df["zip"].fillna("00000").astype(str).str.replace(".0", "", regex=False)
    df["zip3"] = zip_col.str[:3].str.zfill(3).astype(int)

    state = df["merchant_state"].fillna("XX").astype(str).str.upper().str.strip()
    df["state_clean"] = state.where(state != "", "XX")

    df["cust"] = df["user"].astype(int).clip(0, 2999)

    if "time_full" not in df.columns:
        df["time_full"] = dt
    df = df.sort_values(["user", "card", "time_full"], kind="mergesort")
    td = df.groupby(["user", "card"])["time_full"].diff()
    df["time_delta_s"] = td.dt.total_seconds().fillna(0).clip(lower=0)

    return df.reset_index(drop=True)


def _fixed(prefix: str, vals: pd.Series, lo: int, hi: int, pad: int = 0) -> pd.Series:
    v = vals.astype("int32").clip(lo, hi)
    return prefix + "_" + v.astype(str).str.zfill(pad) if pad else prefix + "_" + v.astype(str)


def _direct(prefix: str, vals: pd.Series, mapping: dict, default: str) -> pd.Series:
    return prefix + "_" + vals.astype(str).map(mapping).fillna(default)


def _ranges(prefix: str, vals: pd.Series, ranges, default: str) -> pd.Series:
    v = vals.to_numpy(np.int64)
    out = np.full(len(v), f"{prefix}_{default}", dtype=object)
    for lo, hi, label in ranges:
        out[(v >= lo) & (v <= hi)] = f"{prefix}_{label}"
    return pd.Series(out, index=vals.index)


def transform_cpu(gp: pd.DataFrame, merchant_hash_size: int = 2000) -> pd.DataFrame:
    """Pandas mirror of ``FinancialTokenizerPipeline.transform`` (12 token-string columns)."""
    return pd.DataFrame({
        "amt_val": _fixed("AMT", gp["amt_val"], 0, 6),
        "merch_hash": "MERCH_" + (gp["merch_hash"].astype(np.uint32)
                                  % merchant_hash_size).astype(str),
        "mcc_int": _ranges("CAT", gp["mcc_int"], INDUSTRY_RANGES, "GENERAL"),
        "mcc_str": _direct("MCC", gp["mcc_str"], {str(m): str(m) for m in KNOWN_MCCS}, "-1"),
        "hour": _fixed("HOUR", gp["hour"], 0, 23, pad=2),
        "dow": _fixed("DOW", gp["dow"], 0, 6),
        "month": _fixed("MONTH", gp["month"], 1, 12, pad=2),
        "card": _fixed("CARD", gp["card"], 0, 9),
        "chip_upper": _direct("CHIP", gp["chip_upper"], CHIP_MAPPING, "UNK"),
        "zip3": _fixed("ZIP3", gp["zip3"], 0, 999, pad=3),
        "state_clean": _direct("STATE", gp["state_clean"], {s: s for s in ALL_STATES}, "XX"),
        "cust": _fixed("CUST", gp["cust"], 0, 2999),
    })
