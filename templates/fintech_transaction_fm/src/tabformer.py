"""IBM TabFormer loader — the real-data path.

TabFormer (Padhi et al., ICASSP 2021, https://arxiv.org/abs/2011.01843) is the
de-facto public benchmark for transaction foundation models: 24.4M card
transactions, 2,000 users / ~6.1k cards, 1991-2020, 0.12% transaction-level
fraud. NVIDIA's transaction-FM blueprint and FATA-Trans both evaluate on it.

This module downloads the dataset (266MB tgz from IBM's GitHub, no auth
needed), normalizes it into the same canonical schema the synthetic generator
produces, and samples down to ``num_cards`` so the smoke scale stays cheap.

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
"""

from __future__ import annotations

import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .paths import write_splits_meta

SOURCE_URL = (
    "https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz"
)
CSV_NAME = "card_transaction.v1.csv"


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


def _modal_per_card(df: pd.DataFrame, col: str) -> pd.Series:
    """Most frequent value of ``col`` per card_id (vectorized, no .apply(mode))."""
    counts = df.groupby(["card_id", col]).size().rename("n").reset_index()
    counts = counts.sort_values("n").drop_duplicates("card_id", keep="last")
    return counts.set_index("card_id")[col]


def prepare_tabformer(
    raw_out: str,
    splits_out: str,
    num_cards: int,
    seed: int = 42,
    source_dir: str | None = None,
) -> str:
    """Normalize TabFormer to the canonical schema, sample cards, write Parquet."""
    source_dir = source_dir or os.path.join(os.path.dirname(os.path.dirname(raw_out)), "source")
    csv_path = ensure_download(source_dir)

    print(f"[tabformer] reading {csv_path} ...")
    tbl = pacsv.read_csv(csv_path)
    card_key = pc.add(pc.multiply(pc.cast(tbl["User"], pa.int64()), 100), tbl["Card"])
    tbl = tbl.append_column("card_id", card_key)
    # Extract clock fields in Arrow (Time parses as time32[s]; cheap here, slow in pandas).
    tbl = tbl.append_column("hour", pc.hour(tbl["Time"]))
    tbl = tbl.append_column("minute", pc.minute(tbl["Time"]))

    all_cards = np.sort(pc.unique(tbl["card_id"]).to_numpy())
    n_take = min(num_cards, len(all_cards))
    if n_take < num_cards:
        print(f"[tabformer] only {len(all_cards)} cards available (requested {num_cards}); using all")
    chosen = np.random.default_rng(seed).choice(all_cards, size=n_take, replace=False)
    tbl = tbl.filter(pc.is_in(tbl["card_id"], value_set=pa.array(chosen)))

    df = tbl.select(
        ["card_id", "Year", "Month", "Day", "hour", "minute", "Amount",
         "Use Chip", "Merchant Name", "Merchant State", "MCC", "Is Fraud?"]
    ).to_pandas()

    ts = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"],
             hour=df["hour"], minute=df["minute"])
    )
    state = df["Merchant State"].fillna("")
    state_norm = np.select(
        [state.str.len() == 2, state.str.len() == 0], [state, "ONLINE"], default="FOREIGN"
    )
    df = df.assign(
        timestamp=ts,
        amount=df["Amount"].str.replace("$", "", regex=False).astype(np.float64),
        merchant_id=df["Merchant Name"].astype(np.int64),
        merchant_category=_mcc_to_category(df["MCC"].to_numpy()),
        mcc=df["MCC"].astype(np.int64),
        day_of_week=ts.dt.dayofweek.astype(np.int64),
        is_fraud=(df["Is Fraud?"] == "Yes").astype(np.int64),
        channel=df["Use Chip"].str.split(" ").str[0].str.lower(),
        state_norm=state_norm,
    )

    # Card-level static fields (issuer/bin_region don't exist in TabFormer).
    home_state = _modal_per_card(df, "state_norm").rename("home_state")
    card_type = _modal_per_card(df, "channel").rename("card_type")
    df = df.join(home_state, on="card_id").join(card_type, on="card_id")
    df["issuer"] = "UNKNOWN"
    df["bin_region"] = "UNKNOWN"

    out = df[
        ["card_id", "issuer", "card_type", "bin_region", "home_state",
         "timestamp", "amount", "merchant_id", "merchant_category", "mcc",
         "hour", "day_of_week", "is_fraud"]
    ].sort_values(["card_id", "timestamp"]).reset_index(drop=True)
    out["hour"] = out["hour"].astype(np.int64)

    os.makedirs(os.path.dirname(raw_out.rstrip("/")), exist_ok=True)
    out.to_parquet(raw_out, index=False)
    meta = write_splits_meta(
        splits_out, out["timestamp"].to_numpy(), out["is_fraud"].to_numpy(),
        source="tabformer", n_cards=n_take,
    )
    print(
        f"[tabformer] {len(out):,} transactions / {n_take:,} cards "
        f"({out['is_fraud'].mean() * 100:.3f}% fraud) -> {raw_out}\n"
        f"[tabformer] temporal split: train<{meta['train_end']} val<{meta['val_end']}"
    )
    return raw_out
