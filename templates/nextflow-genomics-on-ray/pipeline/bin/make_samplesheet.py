#!/usr/bin/env python3
"""Build the pipeline's samplesheet from a staged MANIFEST.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

COLUMNS = ["sample", "fastq_1", "fastq_2", "truth_vcf", "truth_bed"]


class ManifestError(RuntimeError):
    """The manifest and the staged files disagree."""


def load_manifest(data_dir: str) -> dict:
    path = os.path.join(data_dir, "MANIFEST.json")
    if not os.path.exists(path):
        raise ManifestError(
            f"{path} not found.\n"
            "Stage the demo data first -- the notebook's 'Stage the reads' step, or\n"
            "  aws s3 cp --no-sign-request --recursive "
            "s3://anyscale-public-materials/genomics/giab-trio-illumina-chr20/<scale>/ "
            f"{data_dir}/"
        )
    with open(path) as handle:
        return json.load(handle)


def sha256(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def build_rows(manifest: dict, data_dir: str, verify: bool) -> list[dict[str, str]]:
    rows = []
    for sample in manifest.get("samples", []):
        row = {"sample": sample["id"]}
        for column, key in (("fastq_1", "fastq_1"), ("fastq_2", "fastq_2")):
            path = os.path.join(data_dir, sample[key])
            if not os.path.exists(path):
                raise ManifestError(
                    f"{sample['id']}: manifest lists {sample[key]} but {path} is missing.\n"
                    "The staged copy is incomplete; re-run the download."
                )
            if verify:
                want = sample.get(f"sha256_{column[-1]}")
                if want:
                    got = sha256(path)
                    if got != want:
                        # A truncated FASTQ reads fine and yields a quietly worse callset.
                        raise ManifestError(
                            f"{sample['id']} {os.path.basename(path)}: checksum mismatch\n"
                            f"  expected {want}\n  actual   {got}"
                        )
            row[column] = os.path.abspath(path)

        truth_vcf, truth_bed = sample.get("truth_vcf"), sample.get("truth_bed")
        row["truth_vcf"] = row["truth_bed"] = ""
        if truth_vcf or truth_bed:
            if not (truth_vcf and truth_bed):
                raise ManifestError(
                    f"{sample['id']}: manifest lists one of truth_vcf/truth_bed but not both"
                )
            for rel in (truth_vcf, f"{truth_vcf}.tbi", truth_bed):
                if not os.path.exists(os.path.join(data_dir, rel)):
                    raise ManifestError(
                        f"{sample['id']}: manifest lists {rel} but "
                        f"{os.path.join(data_dir, rel)} is missing."
                    )
            row["truth_vcf"] = os.path.abspath(os.path.join(data_dir, truth_vcf))
            row["truth_bed"] = os.path.abspath(os.path.join(data_dir, truth_bed))
        rows.append(row)

    if not rows:
        raise ManifestError("manifest lists no samples")
    return rows


def check_tranche_resources(manifest: dict, data_dir: str, verify: bool) -> list[str]:
    """Absolute paths of the manifest's FilterVariantTranches resources, checked."""
    paths = []
    for resource in manifest.get("tranche_resources", []):
        path = os.path.join(data_dir, resource["vcf"])
        for needed in (path, f"{path}.tbi"):
            if not os.path.exists(needed):
                raise ManifestError(
                    f"tranche resource {resource.get('name', resource['vcf'])}: "
                    f"manifest lists {resource['vcf']} but {needed} is missing."
                )
        want = resource.get("sha256")
        if verify and want:
            got = sha256(path)
            if got != want:
                raise ManifestError(
                    f"tranche resource {os.path.basename(path)}: checksum mismatch\n"
                    f"  expected {want}\n  actual   {got}"
                )
        paths.append(os.path.abspath(path))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resources-output",
        help="also write the tranche resources, comma-separated, for main.nf's "
        "--tranche_resources (an empty file when the manifest lists none)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip checksum verification (faster; only for a re-run you already trust)",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.data_dir)
    rows = build_rows(manifest, args.data_dir, verify=not args.no_verify)
    resources = check_tranche_resources(manifest, args.data_dir, verify=not args.no_verify)

    with open(args.output, "w") as out:
        out.write(",".join(COLUMNS) + "\n")
        for row in rows:
            out.write(",".join(row[c] for c in COLUMNS) + "\n")
    if args.resources_output:
        with open(args.resources_output, "w") as out:
            out.write(",".join(resources) + "\n")

    print(
        f"make_samplesheet: {len(rows)} sample(s) and {len(resources)} tranche "
        f"resource(s) from {manifest.get('scale', '?')} ({manifest.get('region', '?')}) "
        f"-> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
