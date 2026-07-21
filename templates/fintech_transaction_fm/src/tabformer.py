"""IBM TabFormer loader — the real-data path, built on Ray Data.

TabFormer (Padhi et al., ICASSP 2021, https://arxiv.org/abs/2011.01843) is the
de-facto public benchmark for transaction foundation models: 24.4M card
transactions, 2,000 users / ~6.1k cards, 1991-2020, 0.12% transaction-level
fraud. NVIDIA's transaction-FM blueprint and FATA-Trans both evaluate on it.

This module downloads the dataset (266MB tgz from IBM's GitHub, no auth
needed), then normalizes it into the canonical schema with a distributed Ray
Data pipeline — the CSV is read, transformed, aggregated, and written across
the cluster, so the driver never materializes the 24M rows (a single-node
pandas pass OOMs a 32GB head node).

Schema mapping notes (TabFormer -> canonical):

* ``card_id``       = User * 100 + Card (Card is a 0-8 per-user index)
* ``timestamp``     = Year/Month/Day + Time
* ``amount``        = "$57.20" -> 57.20 (sign preserved; refunds are negative)
* ``merchant_id``   = Merchant Name (already an opaque hashed integer)
* ``merchant_category`` = derived from MCC via a coarse range map ("other" for
  codes outside the named categories — fine, the MCC token keeps the detail)
* ``home_state``    = modal Merchant State per card (proxy; 2-letter US codes
  kept, countries -> FOREIGN, empty/online -> ONLINE)
* ``card_type``     = modal transaction channel per card (swipe/chip/online)
* ``issuer`` / ``bin_region`` = not present in TabFormer -> "UNKNOWN"

Per-transaction fields kept for the downstream raw baseline (NVIDIA's 13-feature
set uses these directly, not per-card summaries):

* ``use_chip``       = transaction channel (swipe/chip/online) — the strongest
  single fraud signal; NOT the same as the per-card ``card_type`` mode
* ``merchant_state`` = this transaction's Merchant State (2-letter / ONLINE / FOREIGN)
* ``merchant_city``  = this transaction's Merchant City (raw string)
* ``zip``            = this transaction's merchant Zip (NaN for online)
"""

from __future__ import annotations

import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd

from .paths import write_splits_meta

SOURCE_URL = (
    "https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz"
)
CSV_NAME = "card_transaction.v1.csv"

_CSV_COLUMNS = [
    "User", "Card", "Year", "Month", "Day", "Time",
    "Amount", "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
    "Zip", "MCC", "Is Fraud?",
]


def ensure_download(source_dir: str) -> str:
    """Download + extract the TabFormer csv if not already cached; return its path."""
    os.makedirs(source_dir, exist_ok=True)
    csv_path = os.path.join(source_dir, CSV_NAME)
    if os.path.exists(csv_path):
        return csv_path
    tgz_path = os.path.join(source_dir, "transactions.tgz")
    if not os.path.exists(tgz_path):
        print(f"[tabformer] downloading {SOURCE_URL} (~266MB) ...")
        urllib.request.urlretrieve(SOURCE_URL, tgz_path)
    print(f"[tabformer] extracting {tgz_path} ...")
    with tarfile.open(tgz_path) as tar:
        tar.extractall(source_dir)
    if not os.path.exists(csv_path):
        # The csv may sit in a subdirectory depending on archive layout.
        for root, _, files in os.walk(source_dir):
            if CSV_NAME in files:
                os.rename(os.path.join(root, CSV_NAME), csv_path)
                break
    assert os.path.exists(csv_path), f"extraction did not produce {CSV_NAME}"
    return csv_path


def _mcc_to_category(mcc: np.ndarray) -> np.ndarray:
    """Coarse MCC -> merchant_category mapping (same labels as the synthetic data)."""
    m = mcc.astype(np.int64)
    conds = [
        (m >= 3000) & (m < 4800) | (m == 7011),                      # airlines/hotels/transport
        (m >= 4800) & (m < 5000),                                    # telecom/utilities
        np.isin(m, [5411, 5422, 5441, 5451, 5462, 5499]),            # food stores
        (m >= 5800) & (m < 5900),                                    # eating places
        (m >= 8000) & (m < 9000) | np.isin(m, [5912, 5122]),         # medical + drug stores
        (m >= 7800) & (m < 8000) | np.isin(m, [7230, 7298]),         # entertainment/leisure
        np.isin(m, [4816, 5815, 5816, 5817, 5818, 5967, 5968]),      # digital goods/services
        (m >= 5000) & (m < 6000),                                    # remaining stores
    ]
    cats = [
        "travel", "utilities", "grocery", "restaurant",
        "healthcare", "entertainment", "online_services", "retail",
    ]
    return np.select(conds, cats, default="other")


def normalize_batch(b: pd.DataFrame) -> pd.DataFrame:
    """Per-batch TabFormer -> canonical-schema normalization (stateless).

    The ``map_batches`` callback the Part 2 notebook shows inline. The per-row
    munging here (parsing "$57.20", MCC->category, modal home state) is the
    dataset-specific part that stays out of the walkthrough.
    """
    hm = b["Time"].str.split(":", expand=True).astype(np.int64)
    ts = pd.to_datetime(
        dict(year=b["Year"], month=b["Month"], day=b["Day"], hour=hm[0], minute=hm[1])
    )
    state = b["Merchant State"].fillna("")
    return pd.DataFrame(
        {
            "card_id": (b["User"] * 100 + b["Card"]).astype(np.int64),
            "timestamp": ts,
            "amount": b["Amount"].str.replace("$", "", regex=False).astype(np.float64),
            "merchant_id": b["Merchant Name"].astype(np.int64),
            "merchant_category": _mcc_to_category(b["MCC"].to_numpy()),
            "mcc": b["MCC"].astype(np.int64),
            "hour": hm[0],
            "day_of_week": ts.dt.dayofweek.astype(np.int64),
            "is_fraud": (b["Is Fraud?"] == "Yes").astype(np.int64),
            # Per-transaction fields NVIDIA's raw baseline uses. "channel" and
            # "state_norm" also feed the per-card statics groupby below; both are
            # kept on every row (see attach_statics) so the downstream raw baseline
            # can match NVIDIA's 13-feature set. Channel (swipe/chip/ONLINE) is the
            # single strongest fraud signal — collapsing it to a per-card mode threw
            # it away.
            "channel": b["Use Chip"].str.split(" ").str[0].str.lower(),
            "state_norm": np.select(
                [state.str.len() == 2, state.str.len() == 0],
                [state, "ONLINE"],
                default="FOREIGN",
            ),
            "merchant_city": b["Merchant City"].fillna("").astype(str),
            "zip": b["Zip"].astype(np.float64),  # NaN for online txns; encoded downstream
        }
    )


def card_statics(g: pd.DataFrame) -> pd.DataFrame:
    """Modal home state + channel for one card (runs inside a Ray groupby)."""
    return pd.DataFrame(
        {
            "card_id": [int(g["card_id"].iloc[0])],
            "home_state": [g["state_norm"].mode().iat[0]],
            "card_type": [g["channel"].mode().iat[0]],
        }
    )


def tabformer_csv_convert_options():
    """PyArrow CSV read options: keep only the columns we use and read ``Time``
    as a string ("HH:MM", parsed in ``normalize_batch``)."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    return pacsv.ConvertOptions(
        include_columns=_CSV_COLUMNS,
        column_types={"Time": pa.string()},
    )


def sample_cards(ds, num_cards: int, seed: int = 42):
    """Down-sample to ``num_cards`` distinct cards (each card = one sequence).

    Returns ``ds`` unchanged when it has fewer cards than requested.
    """
    all_cards = np.sort(np.asarray(ds.unique("card_id")))
    if num_cards >= len(all_cards):
        return ds
    chosen = set(
        np.random.default_rng(seed).choice(all_cards, size=num_cards, replace=False).tolist()
    )
    return ds.map_batches(lambda b: b[b["card_id"].isin(chosen)], batch_format="pandas")


def attach_statics(ds, statics: pd.DataFrame):
    """Broadcast the small per-card statics table back onto every transaction.

    issuer/bin_region aren't in TabFormer, so they're a constant ``UNKNOWN``
    bucket; card_type/home_state come from the per-card ``card_statics`` groupby.
    """
    home_state = dict(zip(statics["card_id"], statics["home_state"]))
    card_type = dict(zip(statics["card_id"], statics["card_type"]))

    def _attach(b: pd.DataFrame) -> pd.DataFrame:
        # Promote the working columns to their canonical per-transaction names
        # (kept, not dropped — the downstream raw baseline needs them) and add the
        # per-card statics on top.
        b = b.rename(columns={"channel": "use_chip", "state_norm": "merchant_state"})
        b["issuer"] = "UNKNOWN"
        b["bin_region"] = "UNKNOWN"
        b["card_type"] = b["card_id"].map(card_type)
        b["home_state"] = b["card_id"].map(home_state)
        return b

    return ds.map_batches(_attach, batch_format="pandas")


def prepare_tabformer(
    raw_out: str,
    splits_out: str,
    num_cards: int,
    seed: int = 42,
    source_dir: str | None = None,
) -> str:
    """Normalize TabFormer to the canonical schema with Ray Data, sample cards,
    and write Parquet (``raw_out`` becomes a directory of shards)."""
    import ray

    source_dir = source_dir or os.path.join(os.path.dirname(os.path.dirname(raw_out)), "source")
    csv_path = ensure_download(source_dir)
    ray.init(ignore_reinit_error=True)

    print(f"[tabformer] reading {csv_path} with Ray Data ...")
    # The same pipeline the Part 2 notebook walks through inline, composed from
    # the public helpers so the headless and walkthrough paths can't drift.
    ds = ray.data.read_csv(
        csv_path, convert_options=tabformer_csv_convert_options()
    ).map_batches(normalize_batch, batch_format="pandas")
    ds = sample_cards(ds, num_cards, seed=seed).materialize()

    # Per-card static fields via a distributed groupby (issuer/bin_region don't
    # exist in TabFormer). The result is tiny (~6k rows) — broadcast it back.
    statics = (
        ds.groupby("card_id", num_partitions=128)
        .map_groups(card_statics, batch_format="pandas")
        .to_pandas()
    )
    final = attach_statics(ds, statics).materialize()
    final.write_parquet(raw_out)

    # Split cutoffs + stats. Only two slim columns come to the driver
    # (~16B/row) — exact quantiles without holding the whole table.
    slim = final.select_columns(["timestamp", "is_fraud"]).to_pandas()
    meta = write_splits_meta(
        splits_out,
        slim["timestamp"].to_numpy(),
        slim["is_fraud"].to_numpy(),
        source="tabformer",
        n_cards=len(statics),
    )
    print(
        f"[tabformer] {len(slim):,} transactions / {len(statics):,} cards "
        f"({slim['is_fraud'].mean() * 100:.3f}% fraud) -> {raw_out}\n"
        f"[tabformer] temporal split: train<{meta['train_end']} val<{meta['val_end']}"
    )
    return raw_out


# ---------------------------------------------------------------------------
# Exploration helpers for the native-column split (Part 2's Ray Data cells).
# ---------------------------------------------------------------------------

def add_analysis_columns(b: pd.DataFrame) -> pd.DataFrame:
    """Per-batch convenience columns for exploring the native-schema train split:
    ``card_id``, ``is_fraud``, ``amount`` (float), ``timestamp``, ``month`` (period str).
    The tokenizer in Part 3 uses the native columns directly — these are for analysis."""
    b = b.copy()
    b["card_id"] = b["User"] * 1000 + b["Card"]
    b["is_fraud"] = (b["Is Fraud?"].astype(str).str.lower() == "yes").astype(int)
    b["amount"] = (b["Amount"].astype(str).str.replace("$", "", regex=False)
                   .str.replace(",", "", regex=False).astype(float))
    b["timestamp"] = pd.to_datetime(
        b["Year"].astype(str) + "-" + b["Month"].astype(str).str.zfill(2) + "-"
        + b["Day"].astype(str).str.zfill(2) + " " + b["Time"].astype(str),
        errors="coerce")
    b["month"] = b["timestamp"].dt.to_period("M").astype(str)
    return b


def card_gap_hours(group: pd.DataFrame) -> pd.DataFrame:
    """Per-card inter-transaction gaps in hours (for the burstiness plot): one card's
    rows in, its positive time-ordered gaps out."""
    ts = group["timestamp"].sort_values()
    gaps = ts.diff().dt.total_seconds().div(3600.0).dropna()
    gaps = gaps[gaps > 0]
    return pd.DataFrame({"gap_hours": gaps.to_numpy()})
