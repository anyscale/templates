"""
Preprocess NCT-CRC-HE (colorectal H&E patches, 9 tissue classes) into SkyRL's
multi-modal RL parquet schema.

Dataset source: 1aurent/NCT-CRC-HE (ungated, parquet-native).
Fields: 'image' (PIL image, 224x224 RGB) and 'label' (ClassLabel, 9 classes).

Modeled on examples/train/geometry3k/geometry_3k_dataset.py -- images ride
along inside the prompt as base64 JPEG data URIs, there is no separate image
column.

Why the CRC-VAL-HE-7K split is the sampling pool for BOTH train and val:
the NCT-CRC-HE-100K split (31 shards, 15GB) is stored *sorted by class* --
shard 00000 is 100% DEB, shard 00030 is 100% MUS -- so a class-balanced
subsample would mean downloading most of the 15GB. CRC-VAL-HE-7K is 3 shards
(1.1GB), holds all 9 classes, and 7,180 patches is ample for a 2,000/200 spike
subsample. The train and val subsamples drawn here are disjoint but come from
the same patient cohort, so val accuracy is a training-signal check, not a
generalization claim.
"""

import argparse
import base64
import io
import os
import random
from collections import defaultdict

import datasets
from PIL import Image


HF_REPO = "1aurent/NCT-CRC-HE"
VAL7K_SPLIT = "CRC_VAL_HE_7K"

# ClassLabel index -> human-readable class name used in the prompt and as the
# reward_spec ground truth. Order matches the dataset's ClassLabel names
# (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM).
CLASS_NAMES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]

SYSTEM_PROMPT = (
    "You are a pathology assistant. Examine the tissue patch and reason step by "
    "step before answering."
)

USER_PROMPT = (
    "What tissue type is shown? Choose one of: [" + ", ".join(CLASS_NAMES) + "]. "
    "Think step by step, then give your final answer as <answer>class_name</answer>."
)


def _pil_to_data_uri(img: Image.Image) -> str:
    """Convert a PIL Image to a base64 data URI string."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def make_map_fn(split):
    def process_fn(example, idx):
        class_name = CLASS_NAMES[int(example["label"])]
        content = [
            {"type": "image_url", "image_url": {"url": _pil_to_data_uri(example["image"])}},
            {"type": "text", "text": USER_PROMPT},
        ]
        return {
            "prompt": [
                # The system turn's content must be a content-part LIST, not a
                # bare string. Arrow cannot unify string content (system) with
                # list content (user) in one column, so HF datasets silently
                # coerces the whole `content` field to a JSON string and the
                # prompt reaches the generator as literal JSON text. geometry3k
                # never hits this -- it has a single user turn. Target schema,
                # matching geometry3k's parquet:
                #   prompt: list<struct<role: string,
                #                       content: list<extension<arrow.json>>>>
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ],
            "env_class": "nct_crc",
            "reward_spec": {
                "method": "rule",
                "ground_truth": class_name,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "label_id": int(example["label"]),
                "class_name": class_name,
            },
        }

    return process_fn


def balanced_indices(labels, per_class_train, per_class_val, seed):
    """Pick disjoint, class-balanced train/val index lists."""
    by_class = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[int(lab)].append(i)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for cls in sorted(by_class):
        pool = by_class[cls][:]
        rng.shuffle(pool)
        need = per_class_train + per_class_val
        if len(pool) < need:
            raise ValueError(
                f"class {cls} ({CLASS_NAMES[cls]}) has only {len(pool)} patches, "
                f"need {need}. Lower --train_size / --val_size."
            )
        train_idx.extend(pool[:per_class_train])
        val_idx.extend(pool[per_class_train : per_class_train + per_class_val])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/mnt/cluster_storage/data/nct_crc")
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    n_classes = len(CLASS_NAMES)
    per_class_train = args.train_size // n_classes
    per_class_val = max(1, args.val_size // n_classes)

    # Restrict to the CRC-VAL-HE-7K shards explicitly. Passing only
    # `split=VAL7K_SPLIT` to load_dataset still downloads every file in the
    # config (~31GB, all three splits) before selecting; a data_files glob
    # fetches just the 3 shards (~1.1GB) we actually sample from.
    print(f"Loading {HF_REPO} shards matching {VAL7K_SPLIT} ...")
    ds = datasets.load_dataset(
        HF_REPO,
        data_files={"pool": f"data/{VAL7K_SPLIT}-*.parquet"},
        split="pool",
        # The repo's dataset_info declares 3 splits; restricting to one via
        # data_files trips split verification, which is exactly what we want
        # to bypass here.
        verification_mode=datasets.VerificationMode.NO_CHECKS,
    )
    if not isinstance(ds.features.get("image"), datasets.Image):
        ds = ds.cast_column("image", datasets.Image())
    print(f"Loaded {len(ds)} patches; features: {ds.features}")

    # Read the label column only -- avoids decoding 7,180 images just to count.
    labels = ds.with_format(None).data.column("label").to_pylist()
    train_idx, val_idx = balanced_indices(labels, per_class_train, per_class_val, args.seed)
    print(
        f"Class-balanced subsample: {len(train_idx)} train "
        f"({per_class_train}/class), {len(val_idx)} val ({per_class_val}/class)"
    )

    os.makedirs(output_dir, exist_ok=True)
    num_proc = min(8, os.cpu_count() or 1)

    for name, idx in (("train", train_idx), ("val", val_idx)):
        subset = ds.select(idx)
        subset = subset.map(
            function=make_map_fn(name),
            with_indices=True,
            num_proc=num_proc,
            remove_columns=subset.column_names,
            desc=f"Processing {name}",
        )
        path = os.path.join(output_dir, f"{name}.parquet")
        subset.to_parquet(path)
        print(f"Saved {name} ({len(subset)} examples) to {path}")

    print(f"\nDataset preparation complete! Output directory: {output_dir}")


if __name__ == "__main__":
    main()
