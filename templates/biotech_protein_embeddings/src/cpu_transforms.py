"""
CPU-bound preprocessing transforms for the protein embedding pipeline.
Runs on CPU worker nodes via Ray Data map_batches.

Two stages:
  1. validate_and_filter — drop sequences with non-canonical amino acids or bad lengths
  2. assign_length_bucket — bucket by length for GPU-efficient batching

Biology context:
  - The 20 canonical amino acids are: A C D E F G H I K L M N P Q R S T V W Y
  - Non-canonical codes (X, B, Z, U, O, J) appear in databases as ambiguous or rare AAs.
    More than ~1% non-canonical AAs in a sequence suggests a quality problem.
  - Protein lengths in nature range from ~20 aa (peptides) to >30,000 aa (titin).
    ESM-2 has a practical limit of 1024 tokens for efficient inference.
"""
import numpy as np

# The 20 standard amino acids — the alphabet of protein sequences
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Length bucket boundaries for GPU-efficient batching.
# Why bucket: ESM-2's transformer attention is O(L^2) in sequence length.
# Without bucketing, one long sequence forces the entire batch to pad to that length,
# wasting >70% of GPU compute. Bucketing groups similar lengths together.
LENGTH_BUCKETS = [
    (0,   128,  0),   # bucket 0: short peptides and small proteins
    (129, 256,  1),   # bucket 1: typical single-domain proteins
    (257, 512,  2),   # bucket 2: multi-domain proteins
    (513, 1024, 3),   # bucket 3: large proteins (ESM-2 max input)
]


def validate_and_filter(batch: dict) -> dict:
    """Validate protein sequences and filter out bad ones.

    Checks:
      - Non-canonical amino acid fraction (>1% threshold -> drop)
      - Length bounds (< 20 aa or > 1024 aa -> drop)

    Input batch fields: sequence_id, organism_id, sequence, length
    Output adds: passed_validation (bool)
    Drops rows that fail validation.
    """
    sequences = batch["sequence"]
    lengths = batch["length"]
    sequence_ids = batch["sequence_id"]
    organism_ids = batch["organism_id"]

    out = {
        "sequence_id": [],
        "organism_id": [],
        "sequence": [],
        "length": [],
        "passed_validation": [],
    }
    dropped = 0

    for i in range(len(sequences)):
        seq = sequences[i]
        if isinstance(seq, (bytes, np.bytes_)):
            seq = seq.decode("utf-8")
        seq_len = int(lengths[i])

        # Check length bounds
        if seq_len < 20 or seq_len > 1024:
            dropped += 1
            continue

        # Check non-canonical amino acid fraction
        non_canonical_count = sum(1 for aa in seq if aa not in VALID_AA)
        non_canonical_frac = non_canonical_count / max(len(seq), 1)

        if non_canonical_frac > 0.01:
            dropped += 1
            continue

        sid = sequence_ids[i]
        oid = organism_ids[i]
        if isinstance(sid, (bytes, np.bytes_)):
            sid = sid.decode("utf-8")
        if isinstance(oid, (bytes, np.bytes_)):
            oid = oid.decode("utf-8")
        out["sequence_id"].append(sid)
        out["organism_id"].append(oid)
        out["sequence"].append(seq)
        out["length"].append(seq_len)
        out["passed_validation"].append(True)

    if dropped > 0:
        print(f"  [validate] Dropped {dropped} sequences in this batch "
              f"(non-canonical AAs or length out of bounds)")

    return out


def assign_length_bucket(batch: dict) -> dict:
    """Assign each sequence to a length bucket for GPU-efficient batching.

    Buckets:
      0: 20-128 aa   (short peptides, small proteins)
      1: 129-256 aa  (typical single-domain proteins)
      2: 257-512 aa  (multi-domain proteins)
      3: 513-1024 aa (large proteins, ESM-2 max)

    Adding a bucket column lets us sort by it before the GPU stage,
    so each GPU batch contains sequences of similar length. This minimizes
    padding waste and can boost throughput 2-3x.
    """
    lengths = batch["length"]
    buckets = []

    for seq_length in lengths:
        seq_length = int(seq_length)
        assigned = 0
        for lo, hi, bucket_id in LENGTH_BUCKETS:
            if lo <= seq_length <= hi:
                assigned = bucket_id
                break
        buckets.append(assigned)

    batch["length_bucket"] = buckets
    return batch
