#!/usr/bin/env python3
"""Score variants with a DNA language model, on a GPU.

**What this is.** For each variant, the reference and alternate sequences of a
window around it are embedded by a nucleotide language model, and the distance
between the two embeddings is reported. A variant the model considers unremarkable
moves the embedding very little; one that disrupts a pattern the model learned
moves it more.

**What this is not.** Not a clinical score, not a validated pathogenicity
predictor, and not comparable to SpliceAI or CADD. Embedding distance is a proxy
that correlates with "this changes the sequence in a way the model noticed", which
is a weaker claim than any of those tools make. It is here because it is a real,
current, GPU-bound genomics workload with an honest implementation -- not because
the number should go in a report.

**Why it exists in this template.** It runs in two places from one implementation:

* ``pipeline/modules/local/annotate.nf`` calls the CLI below on a shard of the
  joint callset, as a GPU process inside the Nextflow DAG.
* the notebook's Ray Data step calls :class:`VariantScorer` directly over the
  whole callset, on the same cluster and the same GPUs.

That is the point of the comparison, so the two paths deliberately share this
file rather than reimplementing each other. If you change the scoring, both move.

No pysam, no pyfaidx: the reference is read through :class:`IndexedFasta` below,
which is thirty lines against the ``.fai`` samtools already wrote. One dependency
fewer in an image that has to agree with Ray on its interpreter.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterator

DEFAULT_MODEL = os.environ.get(
    "NF_RAY_SCORER_MODEL", "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
)

#: Bases either side of the variant. The model's tokenizer is 6-mer based with a
#: 2048-token limit, so ~12 kb is the ceiling; 1 kb is comfortably inside it and
#: keeps the batch small enough that an L4 is not the bottleneck.
DEFAULT_CONTEXT = 512


# -- reference access ---------------------------------------------------------


@dataclass(frozen=True)
class _FaiEntry:
    length: int
    offset: int
    line_bases: int
    line_width: int


class IndexedFasta:
    """Random access to a FASTA using its ``.fai`` index.

    samtools writes the index; this only has to do the arithmetic. Keeping it
    here rather than taking a dependency matters because every import in this
    file is also an import in a Ray worker.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._index: dict[str, _FaiEntry] = {}
        fai = path + ".fai"
        if not os.path.exists(fai):
            raise FileNotFoundError(
                f"{fai} not found. Index the reference first:  samtools faidx {path}"
            )
        with open(fai) as handle:
            for line in handle:
                name, length, offset, line_bases, line_width = line.split("\t")[:5]
                self._index[name] = _FaiEntry(
                    int(length), int(offset), int(line_bases), int(line_width)
                )
        self._fh = open(path, "rb")

    def __contains__(self, contig: str) -> bool:
        return contig in self._index

    def length(self, contig: str) -> int:
        return self._index[contig].length

    def fetch(self, contig: str, start: int, end: int) -> str:
        """Sequence for ``[start, end)``, 0-based half-open, clamped to the contig."""
        entry = self._index.get(contig)
        if entry is None:
            raise KeyError(
                f"{contig!r} is not in {self.path}. Contig naming must match the VCF "
                f"(known: {sorted(self._index)[:5]}...)"
            )
        start = max(0, start)
        end = min(entry.length, end)
        if start >= end:
            return ""

        def byte_offset(pos: int) -> int:
            # `line_width` includes the newline, `line_bases` does not; the gap is
            # what turns a base coordinate into a file offset.
            return entry.offset + pos // entry.line_bases * entry.line_width + pos % entry.line_bases

        self._fh.seek(byte_offset(start))
        raw = self._fh.read(byte_offset(end) - byte_offset(start))
        return raw.decode("ascii").replace("\n", "").replace("\r", "").upper()

    def close(self) -> None:
        self._fh.close()


# -- VCF reading --------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int  # 1-based, as in the VCF
    ref: str
    alt: str

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


def read_vcf(path: str) -> Iterator[Variant]:
    """Yield biallelic records from a (possibly bgzipped) VCF.

    Multi-allelic sites are split into one record per ALT rather than skipped,
    because the pipeline normalises with ``bcftools norm -m -any`` upstream and a
    silent skip here would make the two counts disagree with no signal.
    """
    opener = open
    mode = "rt"
    if path.endswith(".gz"):
        import gzip

        opener = gzip.open  # type: ignore[assignment]

    with opener(path, mode) as handle:  # type: ignore[operator]
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            chrom, pos, _id, ref, alts = fields[:5]
            for alt in alts.split(","):
                if alt in (".", "*") or not alt:
                    continue
                yield Variant(chrom, int(pos), ref.upper(), alt.upper())


# -- scoring ------------------------------------------------------------------


class VariantScorer:
    """Embed the reference and alternate windows and report their distance.

    Callable on a dict of columns so it can be handed straight to Ray Data's
    ``map_batches`` with no adapter:

        ds.map_batches(VariantScorer, fn_constructor_kwargs={"reference": ref},
                       batch_size=64, num_gpus=1, concurrency=2)

    and used directly from the CLI below for the Nextflow process. The model is
    loaded once per instance -- per actor under Ray, per invocation under
    Nextflow -- which is why the image bakes the weights in rather than letting
    every task in a fan-out pull them from Hugging Face at once.
    """

    def __init__(
        self,
        reference: str,
        model_name: str = DEFAULT_MODEL,
        context: int = DEFAULT_CONTEXT,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.context = context
        self.fasta = IndexedFasta(reference)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self._torch = torch

    def _windows(self, variant: Variant) -> tuple[str, str]:
        """The reference and alternate sequence around one variant."""
        start = variant.pos - 1 - self.context
        end = variant.pos - 1 + len(variant.ref) + self.context
        window = self.fasta.fetch(variant.chrom, start, end)

        # Where the REF allele begins inside the window, after clamping at a
        # contig edge trimmed the left flank.
        left = min(self.context, variant.pos - 1)
        observed = window[left : left + len(variant.ref)]
        alt_window = window[:left] + variant.alt + window[left + len(variant.ref) :]
        return window, alt_window if observed == variant.ref else ""

    def score_variants(self, variants: list[Variant]) -> list[dict]:
        torch = self._torch
        refs, alts, kept = [], [], []
        mismatched = 0
        for variant in variants:
            ref_win, alt_win = self._windows(variant)
            if not ref_win or not alt_win:
                # Either the contig edge truncated the window, or the VCF's REF
                # does not match the reference at that position -- which means the
                # VCF and the FASTA disagree, and a score computed anyway would be
                # meaningless rather than merely noisy.
                mismatched += 1
                continue
            refs.append(ref_win)
            alts.append(alt_win)
            kept.append(variant)

        if not kept:
            return []

        with torch.inference_mode():
            ref_emb = self._embed(refs)
            alt_emb = self._embed(alts)
            cosine = torch.nn.functional.cosine_similarity(ref_emb, alt_emb, dim=-1)
            l2 = torch.linalg.vector_norm(ref_emb - alt_emb, dim=-1)

        return [
            {
                "chrom": v.chrom,
                "pos": v.pos,
                "ref": v.ref,
                "alt": v.alt,
                "embedding_distance": float(l2[i]),
                "cosine_similarity": float(cosine[i]),
            }
            for i, v in enumerate(kept)
        ]

    def _embed(self, sequences: list[str]):
        torch = self._torch
        encoded = self.tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        output = self.model(**encoded)
        hidden = output.last_hidden_state
        # Mean-pool over real tokens only. Including padding would make the score
        # depend on the longest sequence in the batch, which would make the same
        # variant score differently depending on what it was batched with -- and
        # that is exactly the kind of bug a batch-size change hides.
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # -- Ray Data interface ------------------------------------------------

    def __call__(self, batch: dict) -> dict:
        """``map_batches`` entry point. Takes and returns dicts of columns."""
        variants = [
            Variant(str(c), int(p), str(r), str(a))
            for c, p, r, a in zip(
                batch["chrom"], batch["pos"], batch["ref"], batch["alt"], strict=True
            )
        ]
        scored = self.score_variants(variants)
        if not scored:
            return {k: [] for k in
                    ("chrom", "pos", "ref", "alt", "embedding_distance", "cosine_similarity")}
        return {key: [row[key] for row in scored] for key in scored[0]}


# -- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    scorer = VariantScorer(args.reference, model_name=args.model, context=args.context)
    print(f"score_variants: model={args.model} device={scorer.device}", file=sys.stderr)

    written = 0
    with open(args.output, "w") as out:
        out.write("chrom\tpos\tref\talt\tembedding_distance\tcosine_similarity\n")
        batch: list[Variant] = []
        for variant in read_vcf(args.vcf):
            batch.append(variant)
            if len(batch) >= args.batch_size:
                written += _flush(scorer, batch, out)
                batch = []
        written += _flush(scorer, batch, out)

    print(f"score_variants: wrote {written} rows to {args.output}", file=sys.stderr)
    return 0


def _flush(scorer: VariantScorer, batch: list[Variant], out) -> int:
    if not batch:
        return 0
    rows = scorer.score_variants(batch)
    for row in rows:
        out.write(
            f"{row['chrom']}\t{row['pos']}\t{row['ref']}\t{row['alt']}\t"
            f"{row['embedding_distance']:.6f}\t{row['cosine_similarity']:.6f}\n"
        )
    return len(rows)


if __name__ == "__main__":
    sys.exit(main())
