"""
FASTA file reader for Ray Data.

Parses standard FASTA files into a Ray Dataset with columns:
  sequence_id, organism_id, sequence, length

FASTA format recap (for non-biotech audience):
  - Each record starts with a ">" header line containing the sequence ID and metadata
  - Followed by one or more lines of amino acid sequence (A-Z letters)
  - Records are separated by the next ">" line

This module supports two loading paths:
  1. read_fasta_as_ray_dataset() — parse raw .fasta files on the fly
  2. read_parquet_corpus() — load pre-generated .parquet (faster, preferred)
"""
import os
from typing import Optional

import pandas as pd
import ray.data


def parse_fasta_file(fasta_path: str) -> list[dict]:
    """Parse a single FASTA file into a list of record dicts.

    Header format expected: >SEQUENCE_ID|ORGANISM_ID description
    """
    records = []
    current_id = None
    current_org = None
    current_seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous record
                if current_id is not None:
                    seq = "".join(current_seq_parts)
                    records.append({
                        "sequence_id": current_id,
                        "organism_id": current_org or "UNKNOWN",
                        "sequence": seq,
                        "length": len(seq),
                    })
                # Parse new header: >SEQ_ID|ORG_ID description
                header = line[1:].strip()
                parts = header.split("|", 1)
                current_id = parts[0].strip()
                if len(parts) > 1:
                    # organism_id is everything before the first space in the second part
                    org_rest = parts[1].strip()
                    current_org = org_rest.split()[0] if org_rest else "UNKNOWN"
                else:
                    current_org = "UNKNOWN"
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Don't forget the last record
    if current_id is not None:
        seq = "".join(current_seq_parts)
        records.append({
            "sequence_id": current_id,
            "organism_id": current_org or "UNKNOWN",
            "sequence": seq,
            "length": len(seq),
        })

    return records


def read_fasta_as_ray_dataset(fasta_path: str) -> "ray.data.Dataset":
    """Stream-parse a FASTA file into a Ray Dataset.

    Returns a Dataset with schema:
        sequence_id: str
        organism_id: str
        sequence: str
        length: int
    """
    records = parse_fasta_file(fasta_path)
    df = pd.DataFrame(records)
    return ray.data.from_pandas(df)


def read_parquet_corpus(parquet_path: str, num_blocks: Optional[int] = None) -> "ray.data.Dataset":
    """Load a pre-generated Parquet corpus as a Ray Dataset.

    This is the preferred path for production — Parquet supports columnar reads,
    predicate pushdown, and parallel loading across cluster nodes.
    """
    kwargs = {}
    if num_blocks is not None:
        kwargs["override_num_blocks"] = num_blocks
    return ray.data.read_parquet(parquet_path, **kwargs)
