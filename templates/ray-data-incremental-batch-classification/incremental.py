"""Helpers for the incremental batch classification template.

Each function here is one lever from the notebook:

- ``make_synthetic_frames``   generates today's rows and the keys already classified
- ``anti_join_hash``          the default: a hash-shuffle ``left_anti`` join
- ``prior_key_hashes`` +
  ``ProbeFilter``             the alternative: broadcast 16-byte key hashes, filter
- ``download_model_once``     fetch the model to shared storage one time
- ``classify``                zero-shot classification with fractional GPUs
- ``write_partitioned``       partitioned Parquet write with one root ``_SUCCESS``
- ``enable_hang_detection``   turn on Ray Data's hanging-execution detector
"""

from __future__ import annotations

import faulthandler
import hashlib
import sys
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import ray
from ray.data import SaveMode

KEY_COLUMNS: tuple[str, ...] = ("doc_id", "company", "lang")
LABELS: tuple[str, ...] = ("positive", "negative", "neutral")
# Pass the hypothesis template explicitly. Leaving it out makes the
# zero-shot pipeline wrap every label in "This example is {}.", which changes
# which label wins on a noticeable share of real rows.
HYPOTHESIS_TEMPLATE = "The sentiment of this text is {}."

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# Only the files inference needs: the safetensors weights and the tokenizer.
# The repo also carries a PyTorch .bin copy and an ONNX export; skipping them
# keeps the download to about 578 MB instead of several GB.
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
]

_TEXT = {
    "en": [
        "{e} launched a new product and reviewers loved it.",
        "Customers are frustrated with {e} after the outage.",
        "{e} published its quarterly report today.",
    ],
    "es": [
        "{e} lanzó un nuevo producto y a los críticos les encantó.",
        "Los clientes están molestos con {e} tras la caída del servicio.",
        "{e} publicó hoy su informe trimestral.",
    ],
    "de": [
        "{e} hat ein neues Produkt vorgestellt, und die Kritiken sind begeistert.",
        "Kunden sind nach dem Ausfall verärgert über {e}.",
        "{e} hat heute seinen Quartalsbericht veröffentlicht.",
    ],
    "fr": [
        "{e} a lancé un nouveau produit et les critiques l'ont adoré.",
        "Les clients sont mécontents de {e} après la panne.",
        "{e} a publié aujourd'hui son rapport trimestriel.",
    ],
    "ja": [
        "{e}が新製品を発表し、レビューは絶賛している。",
        "障害の後、顧客は{e}に不満を抱いている。",
        "{e}は本日、四半期報告書を公表した。",
    ],
}


def make_synthetic_frames(
    num_rows: int,
    prior_fraction: float = 0.5,
    null_fraction: float = 0.02,
    num_shards: int = 40,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (today, prior).

    ``today`` has one row per (doc_id, company, lang) to classify, plus a text
    column and a ``shard`` column used as the write partition. ``prior``
    holds the key columns of rows an earlier run already classified: about
    ``prior_fraction`` of today's keys, extra keys that are not in today, and
    rows whose ``lang`` is NULL. NULL never equals NULL in a join, so every
    today row with a NULL key must survive the anti-join.
    """
    rng = np.random.default_rng(seed)
    langs = np.array(sorted(_TEXT))
    lang = rng.choice(langs, num_rows)
    company = np.array([f"company-{i:03d}" for i in rng.integers(0, 200, num_rows)])
    template_idx = rng.integers(0, 3, num_rows)
    text = [_TEXT[lg][t].format(e=c) for lg, t, c in zip(lang, template_idx, company)]
    today = pd.DataFrame(
        {
            "doc_id": [f"doc-{i:07d}" for i in range(num_rows)],
            "company": company,
            "lang": lang.astype(object),
            "shard": [f"shard-{i:03d}" for i in rng.integers(0, num_shards, num_rows)],
            "text": text,
        }
    )
    null_rows = rng.random(num_rows) < null_fraction
    today.loc[null_rows, "lang"] = None

    seen = today.loc[(~null_rows) & (rng.random(num_rows) < prior_fraction), list(KEY_COLUMNS)]
    extra = pd.DataFrame(
        {
            "doc_id": [f"old-{i:07d}" for i in range(max(1, num_rows // 10))],
            "company": "company-000",
            "lang": "en",
        }
    )
    # Same doc_id and company as today's NULL-lang rows, NULL lang here too.
    null_twins = today.loc[null_rows, list(KEY_COLUMNS)].copy()
    prior = pd.concat([seen, extra, null_twins], ignore_index=True)
    return today, prior


def _key_hashes(batch: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """16-byte blake2b per row over the key columns, and a mask of rows with no NULL key.

    128 bits, not 64: at hundreds of millions of prior keys a 64-bit collision
    becomes likely enough per run to silently drop a row that should have been
    classified.
    """
    keys = batch[list(cols)]
    valid = ~keys.isna().any(axis=1).to_numpy()
    joined = keys.astype(str).agg("\x1f".join, axis=1)
    hashes = np.array(
        [hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest() for s in joined],
        dtype="S16",
    )
    return hashes, valid


def anti_join_hash(today, prior, *, num_partitions: int, cols: Sequence[str] = KEY_COLUMNS):
    """The default: ``Dataset.join(..., join_type="left_anti")``.

    Correct, and it hash-shuffles BOTH sides through ``HashShuffleAggregator``
    actors sized by ``num_partitions``. Those actors hold shuffle state in
    memory, so losing one ends the whole dataset, and they keep GPU nodes busy
    (and un-reclaimable) while using no GPU.
    """
    return today.join(
        prior.select_columns(list(cols)),
        join_type="left_anti",
        num_partitions=num_partitions,
        on=tuple(cols),
    )


def prior_key_hashes(prior, cols: Sequence[str] = KEY_COLUMNS) -> np.ndarray:
    """Hash the prior keys on the workers; bring back only the 16-byte hashes.

    Hashing is a ``map_batches`` across the cluster. Draining the prior keys
    through the driver and hashing them there in one Python loop is the slow
    version of the same thing: it runs single-threaded while every GPU waits.
    """

    def to_hashes(batch: pd.DataFrame) -> pd.DataFrame:
        hashes, valid = _key_hashes(batch, cols)
        return pd.DataFrame({"h": list(hashes[valid])})

    hashed = prior.select_columns(list(cols)).map_batches(to_hashes, batch_format="pandas")
    parts = [np.asarray(b["h"], dtype="S16") for b in hashed.iter_batches(batch_format="numpy", batch_size=None)]
    if not parts:
        return np.array([], dtype="S16")
    return np.unique(np.concatenate(parts))


class ProbeFilter:
    """Keep rows whose key hash is NOT in the broadcast prior set.

    Rows with any NULL key column are always kept: NULL never equals NULL in a
    join, so a NULL-key row can never have been "already classified". Folding
    NULL to a sentinel value and probing it like any other key removes those
    rows, and a test that uses a hand-written reference with the same mistake
    passes anyway.
    """

    def __init__(self, prior_ref, cols: Sequence[str] = KEY_COLUMNS):
        self.prior = ray.get(prior_ref) if isinstance(prior_ref, ray.ObjectRef) else prior_ref
        self.cols = tuple(cols)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        hashes, valid = _key_hashes(batch, self.cols)
        already = np.isin(hashes, self.prior) & valid
        return batch.loc[~already]


def anti_join_probe(today, prior, *, cols: Sequence[str] = KEY_COLUMNS, concurrency: int = 4):
    """The alternative: broadcast the prior hashes, filter today in a streaming map.

    No shuffle, no aggregator actors, and the filter fuses into the stream, so
    GPU work downstream starts as soon as the first filtered blocks exist.
    Memory: about 16 bytes per prior key in every filter actor.
    """
    prior_ref = ray.put(prior_key_hashes(prior, cols))
    return today.map_batches(
        ProbeFilter,
        fn_constructor_kwargs={"prior_ref": prior_ref, "cols": tuple(cols)},
        batch_format="pandas",
        concurrency=concurrency,
    )


def key_set(ds, cols: Sequence[str] = KEY_COLUMNS) -> set[tuple]:
    """The set of key tuples in a dataset, NULLs included, for comparing two paths."""
    df = ds.select_columns(list(cols)).to_pandas()
    return set(map(tuple, df.astype(object).where(df.notna(), None).itertuples(index=False, name=None)))


def count_actors(class_name: str) -> int | None:
    """Actors of a class that exist or existed in this Ray session.

    Uses the state API (served by the Ray dashboard, always present on
    Anyscale). Without a dashboard it falls back to the GCS actor table, an
    internal API; returns None if neither is reachable.
    """
    try:
        from ray.util.state import list_actors

        return len(list_actors(filters=[("class_name", "=", class_name)], limit=10_000, raise_on_missing_output=False))
    except Exception:
        pass
    try:
        from ray._private import state

        return sum(1 for a in state.actors().values() if (a.get("ActorClassName") or "").endswith(class_name))
    except Exception:
        return None


def download_model_once(model_id: str, dest: str) -> str:
    """Download the model to shared storage once, from the driver.

    Without this, every classifier actor downloads its own copy when it starts.
    At hundreds of actors that is hundreds of simultaneous downloads of the
    same file, and the hub answers with HTTP 429 and the actors die in their
    constructors. For production, bake the weights into the image instead.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_dir=dest, allow_patterns=MODEL_FILES)


class ZeroShotClassifier:
    """One model per actor; classifies a batch of texts against fixed labels."""

    def __init__(
        self,
        model_dir: str,
        labels: Iterable[str] = LABELS,
        hypothesis_template: str = HYPOTHESIS_TEMPLATE,
        inference_batch_size: int = 32,
    ):
        import torch
        from transformers import pipeline

        on_gpu = torch.cuda.is_available()
        # bfloat16, not float16: DeBERTa-v3 overflows to NaN in float16.
        self.pipe = pipeline(
            "zero-shot-classification",
            model=model_dir,
            device=0 if on_gpu else -1,
            torch_dtype=torch.bfloat16 if on_gpu else torch.float32,
        )
        self.labels = list(labels)
        self.hypothesis_template = hypothesis_template
        self.inference_batch_size = inference_batch_size

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        results = self.pipe(
            list(batch["text"]),
            candidate_labels=self.labels,
            hypothesis_template=self.hypothesis_template,
            batch_size=self.inference_batch_size,
        )
        batch = batch.copy()
        batch["label"] = [r["labels"][0] for r in results]
        batch["score"] = [float(r["scores"][0]) for r in results]
        return batch


def classify(ds, model_dir: str, *, gpu_fraction: float = 1.0, num_actors: int | None = None, batch_size: int = 64):
    """Zero-shot classification, packing ``1 / gpu_fraction`` actors per GPU.

    A base-size classifier leaves most of an A10G or L4 idle with one actor.
    ``gpu_fraction=0.5`` puts two actors on each GPU. Size the actor count to
    the GPUs you have, never above: an actor pool larger than the GPU count
    leaves the extra actors pending for the whole run.
    """
    gpus = int(ray.cluster_resources().get("GPU", 0))
    kwargs = dict(
        fn_constructor_kwargs={"model_dir": model_dir},
        batch_format="pandas",
        batch_size=batch_size,
    )
    if gpus:
        actors = num_actors or max(1, int(gpus / gpu_fraction))
        return ds.map_batches(ZeroShotClassifier, num_gpus=gpu_fraction, concurrency=actors, **kwargs)
    return ds.map_batches(ZeroShotClassifier, num_cpus=1, concurrency=num_actors or 2, **kwargs)


def _write_marker(fs: pafs.FileSystem, directory: str) -> None:
    with fs.open_output_stream(f"{directory.rstrip('/')}/_SUCCESS"):
        pass


def write_partitioned(ds, path: str, partition_col: str, *, marker: str = "root") -> dict:
    """Partitioned Parquet write, then a completion marker.

    ``marker="root"`` writes one ``_SUCCESS`` at the output root.
    ``marker="per_partition"`` writes one in every partition directory, one
    object at a time from the driver: the habit Spark jobs carry over. With
    thousands of partitions on object storage that is thousands of sequential
    requests, minutes of wall clock after the data itself is written, and
    enough request rate to draw throttling on the same prefix.
    """
    fs, root = pafs.FileSystem.from_uri(path)
    t0 = time.perf_counter()
    ds.write_parquet(path, partition_cols=[partition_col], mode=SaveMode.OVERWRITE)
    t1 = time.perf_counter()
    if marker == "per_partition":
        infos = fs.get_file_info(pafs.FileSelector(root, recursive=True))
        dirs = sorted({i.path.rsplit("/", 1)[0] for i in infos if i.type == pafs.FileType.File and i.path.endswith(".parquet")})
        for d in dirs:
            _write_marker(fs, d)
        markers = len(dirs)
    elif marker == "root":
        _write_marker(fs, root)
        markers = 1
    else:
        raise ValueError(f"marker must be 'root' or 'per_partition', not {marker!r}")
    t2 = time.perf_counter()
    return {"markers": markers, "write_s": round(t1 - t0, 2), "marker_s": round(t2 - t1, 3)}


def count_markers(path: str) -> int:
    fs, root = pafs.FileSystem.from_uri(path)
    infos = fs.get_file_info(pafs.FileSelector(root, recursive=True))
    return sum(1 for i in infos if i.type == pafs.FileType.File and i.path.endswith("/_SUCCESS"))


def enable_hang_detection(ctx=None):
    """Add Ray Data's hanging-execution detector, which ships commented out of the defaults.

    It logs a warning naming the operator and the stuck task when a task runs
    far longer than its peers. It only warns: the job-level ``timeout_s`` is
    what actually ends a run that stops making progress. The import path is
    internal to Ray Data and may move between releases.
    """
    from ray.data import DataContext
    from ray.data._internal.issue_detection.detectors import HangingExecutionIssueDetector

    ctx = ctx or DataContext.get_current()
    detectors = ctx.issue_detectors_config.detectors
    if HangingExecutionIssueDetector not in detectors:
        detectors.append(HangingExecutionIssueDetector)
    return ctx


def arm_driver_stack_dump(after_s: float) -> None:
    """Dump every driver thread's stack to stderr once, ``after_s`` seconds from now.

    On clusters where no process can be ptrace-attached, this in-process dump
    is the only way to see where a silent driver is blocked. Use
    ``repeat=False``: a repeating dump reads Python frames without the GIL and
    can crash the driver it is meant to observe.
    """
    faulthandler.dump_traceback_later(after_s, repeat=False, exit=False, file=sys.stderr)
