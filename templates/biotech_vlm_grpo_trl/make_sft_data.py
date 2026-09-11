"""
Build a small, fake-but-shape-realistic SFT dataset: slide tiles + QA pairs, as JSONL.

PathAI's real SFT JSON shape is not known yet. Until it is, this script produces the
most common VLM-SFT layout in the wild (LLaVA-style `conversations`), and
train_sft_trl.py converts it through ONE function, `record_to_trl()`. When the real
shape arrives, change that function and nothing else.

Record shape written here:

    {
      "id": "nctcrc_000123",
      "slide_id": "FAKE-SLIDE-0007",                 # stand-in for a TCGA / PathAI slide id
      "image": "tiles/nctcrc_000123.png",            # path relative to the JSONL's directory
      "conversations": [
        {"from": "human", "value": "<image>\nWhat tissue type is shown in this tile?"},
        {"from": "gpt",   "value": "This tile shows lymphocytes: ..."}
      ],
      "metadata": {"label": "lymphocytes", "source": "NCT-CRC-HE CRC-VAL-HE-7K", "tile_px": 224}
    }

Tiles are real H&E: the 224x224 NCT-CRC-HE patches already on cluster storage
(the SkyRL arm's val.parquet, 198 class-balanced patches), so the VLM actually sees
tissue. The QA text is templated from the class label, so it is *synthetic*: this
dataset proves the plumbing, not the model.

Text-only records (for the LLM mode of train_sft_trl.py) drop the image and put a
templated morphology description in the question instead.

TCGA seam: `--source tcga` is a stub. Real whole-slide images (SVS, 0.5-3 GB each)
need `gdc-client` to download and `openslide-python` to tile at a chosen magnification;
neither is in this environment. See the README.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random

DESCRIPTIONS = {
    "adipose": "large clear vacuolated cells with thin cytoplasmic rims and peripherally displaced nuclei",
    "background": "empty glass with no tissue, only faint out-of-focus artifact",
    "debris": "amorphous granular eosinophilic material without intact cellular architecture",
    "lymphocytes": "densely packed small round cells with dark nuclei and scant cytoplasm",
    "mucus": "pale amorphous pools of extracellular mucin with few scattered cells",
    "smooth muscle": "elongated spindle cells with eosinophilic cytoplasm arranged in parallel bundles",
    "normal colon mucosa": "regularly spaced crypts lined by columnar epithelium with abundant goblet cells",
    "cancer-associated stroma": "loose fibrous matrix with scattered activated fibroblasts and irregular collagen",
    "colorectal adenocarcinoma epithelium": "crowded irregular glands with nuclear pleomorphism and loss of polarity",
}

QUESTIONS = [
    ("What tissue type is shown in this tile?", "This tile shows {label}: {desc}."),
    ("Describe the histology in this patch.", "The patch shows {desc}, consistent with {label}."),
    ("Is tumor epithelium present in this tile?", "{yesno}. The tile shows {desc}, consistent with {label}."),
]


def _tiles_from_parquet(parquet_path: str, tiles_dir: str, limit: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet_path, columns=["prompt", "reward_spec"])
    rows = list(zip(table.column("prompt").to_pylist(), table.column("reward_spec").to_pylist()))
    random.Random(seed).shuffle(rows)
    os.makedirs(tiles_dir, exist_ok=True)
    out = []
    for i, (prompt, spec) in enumerate(rows[: limit or None]):
        b64 = None
        for msg in prompt:
            content = msg["content"]
            if isinstance(content, str):
                content = json.loads(content)
            for part in content:
                if isinstance(part, str):
                    part = json.loads(part)
                if part["type"] == "image_url":
                    b64 = part["image_url"]["url"].split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        name = f"nctcrc_{i:06d}.png"
        img.save(os.path.join(tiles_dir, name))
        out.append({"tile": os.path.join(os.path.basename(tiles_dir), name), "label": spec["ground_truth"]})
    return out


def make_records(tiles: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for i, t in enumerate(tiles):
        label = t["label"]
        q, a_tmpl = rng.choice(QUESTIONS)
        is_tumor = label == "colorectal adenocarcinoma epithelium"
        answer = a_tmpl.format(label=label, desc=DESCRIPTIONS[label], yesno="Yes" if is_tumor else "No")
        records.append(
            {
                "id": f"nctcrc_{i:06d}",
                "slide_id": f"FAKE-SLIDE-{rng.randint(0, 19):04d}",
                "image": t["tile"],
                "conversations": [
                    {"from": "human", "value": "<image>\n" + q},
                    {"from": "gpt", "value": answer},
                ],
                "metadata": {"label": label, "source": "NCT-CRC-HE CRC-VAL-HE-7K", "tile_px": 224},
            }
        )
    return records


def make_text_records(records: list[dict]) -> list[dict]:
    """Text-only twin: same QA, the morphology description stands in for the image."""
    out = []
    for r in records:
        label = r["metadata"]["label"]
        q = r["conversations"][0]["value"].replace("<image>\n", "")
        out.append(
            {
                **{k: v for k, v in r.items() if k != "image"},
                "conversations": [
                    {"from": "human", "value": f"A colorectal H&E tile shows {DESCRIPTIONS[label]}. {q}"},
                    r["conversations"][1],
                ],
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["nctcrc", "tcga"], default="nctcrc")
    p.add_argument("--parquet", default="/mnt/cluster_storage/data/nct_crc/val.parquet")
    p.add_argument("--output_dir", default="/mnt/cluster_storage/data/pathai_sft_fake")
    p.add_argument("--limit", type=int, default=0, help="0 = all rows in the parquet")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.source == "tcga":
        raise NotImplementedError(
            "TCGA tiling is a seam, not implemented: download SVS slides with gdc-client, tile with "
            "openslide-python at 20x into 224/512 px PNGs, then emit the same record shape as `nctcrc`."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    tiles = _tiles_from_parquet(args.parquet, os.path.join(args.output_dir, "tiles"), args.limit, args.seed)
    records = make_records(tiles, args.seed)
    n_val = max(1, int(len(records) * args.val_frac))
    splits = {"train": records[n_val:], "val": records[:n_val]}
    for split, recs in splits.items():
        for suffix, rows in (("", recs), ("_text", make_text_records(recs))):
            path = os.path.join(args.output_dir, f"{split}{suffix}.jsonl")
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"wrote {len(rows):4d} records -> {path}")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()
