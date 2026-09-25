"""Helpers for the incremental batch classification template.

The notebook's steps, in order:

- ``make_synthetic_frames``   today's rows and the keys already classified
- ``anti_join_hash``          the usual approach: a hash-shuffle ``left_anti`` join
- ``anti_join_probe``         the broadcast probe: ``prior_key_hashes``, then ``ProbeFilter``
- ``download_model_once``     fetch the model to shared storage once
- ``split_for_actors``        enough blocks that every classifier actor has work
- ``classify``                zero-shot classification with fractional GPUs
- ``write_partitioned``       partitioned Parquet write with one root ``_SUCCESS``
- ``enable_hang_detection``   turn on Ray Data's hanging-execution detector
- ``arm_driver_stack_dump``   one stack dump of the driver's threads, later

The rest support the notebook's comparisons.
"""

from __future__ import annotations

import faulthandler
import hashlib
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import ray
from ray.data import SaveMode

KEY_COLUMNS: tuple[str, ...] = ("doc_id", "company", "lang")
LABELS: tuple[str, ...] = ("positive", "negative", "neutral")
# What the model reads for each label, inside HYPOTHESIS_TEMPLATE. The wording
# decides the answer: the bare label names inside "The sentiment of this text is
# {}." labelled 0 of 643 neutral sentences neutral, and these phrases labelled all
# 2,000 of the same rows as written (laptop CPU, float32, 2026-09-24). Check any
# rewording against rows whose labels you know.
LABEL_PHRASES: dict[str, str] = {"positive": "good news", "negative": "bad news", "neutral": "routine news"}
# Pass the hypothesis template explicitly; the zero-shot pipeline's default is
# "This example is {}.".
HYPOTHESIS_TEMPLATE = "This text is {}."

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# Only the files inference needs: the safetensors weights and the tokenizer.
# Skipping the repo's PyTorch .bin copy and ONNX exports keeps the download to
# about 578 MB of 2.6 GB.
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


def label_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Count classified synthetic rows by the label each sentence was written to carry (rows)
    and the label the model gave it (columns)."""
    written = {t: label for sentences in _TEXT.values() for label, t in zip(LABELS, sentences)}
    expected = pd.Series([written[t.replace(c, "{e}")] for t, c in zip(df["text"], df["company"])], index=df.index)
    table = pd.crosstab(expected.rename("written as"), df["label"].rename("labelled"))
    return table.reindex(index=list(LABELS), columns=list(LABELS), fill_value=0)


def _key_hashes(batch: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """16-byte blake2b per row over the key columns, and a mask of rows with no NULL key.

    128 bits because a new row falsely matches with probability about
    (prior keys) / 2**bits. At 10**9 prior keys and 10**9 new rows per run, a
    64-bit hash silently drops about one row every 18 runs; a 128-bit hash,
    about 3e-21 rows per run. Both figures are arithmetic.
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
    """The usual approach: ``Dataset.join(..., join_type="left_anti")``.

    Correct, and it hash-shuffles both sides through ``HashShuffleAggregator``
    actors, one per partition up to the cluster's maximum CPU count. They hold
    the shuffle state in memory and are not restarted, so losing one fails the
    dataset. They reserve CPU and memory on GPU nodes while using no GPU, which
    keeps those nodes from scaling down. The join also turns off Ray Data's
    no-progress timeout for its plan.
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
    through the driver and hashing them in one Python loop gives the same set,
    single-threaded, while every GPU waits.
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
    """Keep rows whose key hash is absent from the broadcast prior set.

    Rows with any NULL key column are always kept: NULL never equals NULL in a
    join, so a NULL-key row can never have been "already classified". Folding
    NULL to a sentinel value and probing it like any other key removes those
    rows, and a test against a hand-written reference with the same mistake
    still passes.
    """

    def __init__(self, prior_ref, cols: Sequence[str] = KEY_COLUMNS):
        self.prior = ray.get(prior_ref) if isinstance(prior_ref, ray.ObjectRef) else prior_ref
        self.cols = tuple(cols)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        hashes, valid = _key_hashes(batch, self.cols)
        already = np.isin(hashes, self.prior) & valid
        return batch.loc[~already]


def anti_join_probe(today, prior, *, cols: Sequence[str] = KEY_COLUMNS, concurrency: int = 4):
    """The broadcast probe: broadcast the prior hashes, filter today in a streaming map.

    No shuffle and no aggregator actors, and the filter streams, so GPU work
    downstream starts as soon as the first filtered blocks exist. The broadcast
    set is 16 bytes per prior key.
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


def timed_materialize(fn, *args, count_class: str = "HashShuffleAggregator", **kwargs):
    """Materialize ``fn(*args, **kwargs)``.

    Returns the materialized dataset, the wall-clock seconds, and how many
    ``count_class`` actors the run started.
    """
    before = count_actors(count_class)
    t0 = time.perf_counter()
    ds = fn(*args, **kwargs).materialize()
    return ds, time.perf_counter() - t0, count_actors(count_class) - before


def download_model_once(model_id: str, dest: str) -> str:
    """Download the model to shared storage once, from the driver.

    Otherwise every classifier actor downloads its own copy when it starts. A
    large pool then sends the Hub a burst of requests for the same files, the
    Hub can answer with HTTP 429, and an actor whose download fails dies in its
    constructor. For production, bake the weights into the image instead.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_dir=dest, allow_patterns=MODEL_FILES)


class ZeroShotClassifier:
    """One model per actor; classifies a batch of texts against fixed labels.

    The model scores ``label_phrases``' values; the ``label`` column gets the
    matching key, so rewording a phrase does not rename the output.
    """

    def __init__(
        self,
        model_dir: str,
        label_phrases: Mapping[str, str] = LABEL_PHRASES,
        hypothesis_template: str = HYPOTHESIS_TEMPLATE,
        inference_batch_size: int = 32,
    ):
        import torch
        from transformers import pipeline

        on_gpu = torch.cuda.is_available()
        # bfloat16 on GPU: the model card says mDeBERTa does not support FP16.
        # float16 was not tried here.
        self.pipe = pipeline(
            "zero-shot-classification",
            model=model_dir,
            device=0 if on_gpu else -1,
            dtype=torch.bfloat16 if on_gpu else torch.float32,
        )
        self.label_of = {phrase: label for label, phrase in label_phrases.items()}
        self.hypothesis_template = hypothesis_template
        self.inference_batch_size = inference_batch_size

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        results = self.pipe(
            list(batch["text"]),
            candidate_labels=list(self.label_of),
            hypothesis_template=self.hypothesis_template,
            batch_size=self.inference_batch_size,
        )
        batch = batch.copy()
        batch["label"] = [self.label_of[r["labels"][0]] for r in results]
        batch["score"] = [float(r["scores"][0]) for r in results]
        return batch


# Blocks per classifier actor, for split_for_actors. Ray Data hands an actor one block per
# task. In Ray 2.58 each actor holds up to 2 tasks in flight and takes work as soon as it is
# ready, without waiting for the rest of the pool, so with one block per actor the first
# actors up can take every block and leave the rest idle. Four per actor leaves blocks for
# the actors that start later, and stays far from the thousands of small blocks that would
# each write one file per output partition.
BLOCKS_PER_ACTOR = 4


def classify_actors(gpu_fraction: float = 1.0, num_actors: int | None = None) -> int:
    """The number of classifier actors ``classify`` starts on this cluster.

    ``1 / gpu_fraction`` per GPU, sized to the GPUs present, never above: an
    actor pool larger than the GPU count leaves the extra actors pending for
    the whole run. Two CPU actors on a cluster with no GPU.
    """
    if num_actors:
        return num_actors
    gpus = int(ray.cluster_resources().get("GPU", 0))
    return max(1, int(gpus / gpu_fraction)) if gpus else 2


def split_for_actors(ds, actors: int):
    """Repartition ``ds`` into ``BLOCKS_PER_ACTOR`` blocks per classifier actor.

    ``repartition(num_blocks)`` is an all-to-all operation. It waits for all
    of ``ds`` before it splits, and it turns off Ray Data's no-progress
    timeout for any plan that contains it. Apply it to materialized rows, or
    ahead of the stages that should stream, as run_pipeline.py does; between
    the probe and the classifier it would hold the GPU stage until the probe
    finished.
    """
    return ds.repartition(actors * BLOCKS_PER_ACTOR)


def classify(ds, model_dir: str, *, gpu_fraction: float = 1.0, num_actors: int | None = None, batch_size: int = 64):
    """Zero-shot classification, packing ``1 / gpu_fraction`` actors per GPU.

    ``gpu_fraction=0.5`` puts two actors on each GPU, which pays when one actor
    leaves the GPU underused. Every actor needs blocks to work on: give ``ds``
    at least ``BLOCKS_PER_ACTOR`` blocks per actor (``split_for_actors``).
    """
    actors = classify_actors(gpu_fraction, num_actors)
    kwargs = dict(
        fn_constructor_kwargs={"model_dir": model_dir},
        batch_format="pandas",
        batch_size=batch_size,
        concurrency=actors,
    )
    if ray.cluster_resources().get("GPU", 0):
        return ds.map_batches(ZeroShotClassifier, num_gpus=gpu_fraction, **kwargs)
    return ds.map_batches(ZeroShotClassifier, num_cpus=1, **kwargs)


def _write_marker(fs: pafs.FileSystem, directory: str) -> None:
    with fs.open_output_stream(f"{directory.rstrip('/')}/_SUCCESS"):
        pass


def write_partitioned(ds, path: str, partition_col: str, *, marker: str = "root") -> dict:
    """Partitioned Parquet write, then a completion marker.

    Writes with ``SaveMode.OVERWRITE``, which deletes everything under
    ``path`` first. ``marker="root"`` writes one ``_SUCCESS`` at the output
    root. ``marker="per_partition"`` writes one in every partition directory,
    one object at a time from the driver: with thousands of partitions on
    object storage, that is thousands of sequential requests after the data is
    written, all on one prefix, which the store can throttle.
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
    """Add Ray Data's hanging-execution detector, which Ray 2.58 leaves out of the defaults.

    It warns, naming the operator and the task, when a task makes no progress
    for longer than its operator's mean task time plus 10 standard deviations,
    and only after the operator has finished 10 tasks. Each task it flags costs
    a State API call of up to a second on the executor thread, which is why it
    ships off. It only warns; the job's ``timeout_s`` ends a stalled run. The
    import path is internal to Ray Data and may move between releases.
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
    is the only way to see where a silent driver is blocked. Every dump reads
    Python frames without holding the GIL and can crash the driver it
    observes, so this one fires once (``repeat=False``).
    """
    faulthandler.dump_traceback_later(after_s, repeat=False, exit=False, file=sys.stderr)
