"""
Synthetic protein FASTA corpus generator.

Generates realistic protein sequences by starting from seed motifs representing
~20 protein families, then applying random point mutations to simulate natural
sequence diversity. Also produces a taxonomy lookup table and labeled homolog
test pairs for downstream validation.

Writes:
  - FASTA file (one sequence per record)
  - Parquet version of the same corpus (for Ray Data)
  - taxonomy_lookup.parquet (organism_id -> taxonomy fields)
  - homolog_test_pairs.csv (labeled pairs for cosine sim validation)

Biology context for non-biotech audience:
  - Proteins are chains of amino acids (AAs), written as strings over a 20-letter alphabet.
  - Proteins in the same "family" share a common ancestor and tend to have similar
    3D structure and function, even if their sequences have diverged 40-70%.
  - A "homolog" is a protein related by common ancestry. Homolog pairs should have
    higher cosine similarity in embedding space than random pairs.
  - FASTA is the standard text format: >header line, then sequence lines.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd

SCALE_MAP = {"small": 10_000, "medium": 100_000, "large": 500_000}

# The 20 canonical amino acids (single-letter codes)
CANONICAL_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Non-canonical AAs that occasionally appear in real datasets (e.g., selenocysteine U,
# pyrrolysine O, ambiguous X/B/Z/J). We inject a small fraction to test filtering.
NON_CANONICAL_AAS = list("XBZUOJ")

# ~20 seed motifs representing protein family conserved regions.
# These are realistic subsequences inspired by well-known protein families
# (kinase domains, zinc fingers, immunoglobulin folds, etc.).
SEED_FAMILIES = [
    ("kinase_domain",       "DLKPENIVLQRGELGHVHGKIYHRDLKAANFLTSEDKNVLISDFGLATVK"),
    ("zinc_finger_C2H2",    "YKCPECGKSFSQKSNLQKHQRTHTGEKP"),
    ("immunoglobulin_fold",  "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYDMHWVRQAPGKGLEWVSAI"),
    ("helicase_motif",      "DECHQSIDAGQKRFAPTLTITKGEQQNFVVTEDQYRKVINALMNPIKELIS"),
    ("p53_dbd",             "SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEV"),
    ("globin_fold",         "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"),
    ("ras_gtpase",          "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYNRKQVVIDGETC"),
    ("sh3_domain",          "ALYDFVASGDNTLSITKGEKLRVLGYNHNGENNRDQTAVVNLGIKKHKLQ"),
    ("egf_like",            "CRVYGPEVSEECLQCPECQKFARDCLQFYICPPHTQFCFHGECRDICQHPG"),
    ("dehydrogenase",       "MGCKAIALITGTASQYGQATAIGDALILQGDIALYTMKERHAAYQVDPDQPV"),
    ("protease_serine",     "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGE"),
    ("dna_binding_hth",     "MQRTKLQAFADALEQHPDLAQEIGVSRAALKQARERHGITLQNLA"),
    ("atp_synthase",        "MQLNSTEISELIKQRIAQFNVVSEAHNEGTIVSVSDGVIRIHGLADCMQGER"),
    ("cytochrome_c",        "GDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFSYTDAN"),
    ("rhodopsin_tm",        "MNGTEGPNFYVPFSNKTGVVRSPFEAPQYYLAEPWQFSMLAAYMFLLIMLGF"),
    ("collagen_repeat",     "GPPGPPGPPGPPGPPGFPGAVGAKGEAGPQGPRGSEGPQGVRGEPGPPGPA"),
    ("ferredoxin",          "AYVINDSCIACGACKPECPVNIIQGSIYAIDADSCIDCGSCASVCPVGAPNPE"),
    ("ubiquitin",           "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDG"),
    ("insulin_chain",       "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
    ("lysozyme",            "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDY"),
]

# Organisms for taxonomy lookup
ORGANISMS = [
    ("ORG_001", "Homo sapiens",         "Eukaryota", "Mammalia",   "Primates"),
    ("ORG_002", "Mus musculus",          "Eukaryota", "Mammalia",   "Rodentia"),
    ("ORG_003", "Escherichia coli",      "Bacteria",  "Proteobacteria", "Enterobacterales"),
    ("ORG_004", "Saccharomyces cerevisiae", "Eukaryota", "Fungi",   "Saccharomycetales"),
    ("ORG_005", "Drosophila melanogaster", "Eukaryota", "Insecta",  "Diptera"),
    ("ORG_006", "Arabidopsis thaliana",  "Eukaryota", "Viridiplantae", "Brassicales"),
    ("ORG_007", "Caenorhabditis elegans", "Eukaryota", "Chromadorea", "Rhabditida"),
    ("ORG_008", "Danio rerio",           "Eukaryota", "Actinopterygii", "Cypriniformes"),
    ("ORG_009", "Bacillus subtilis",     "Bacteria",  "Bacillota",  "Bacillales"),
    ("ORG_010", "Staphylococcus aureus", "Bacteria",  "Bacillota",  "Staphylococcales"),
    ("ORG_011", "Mycobacterium tuberculosis", "Bacteria", "Actinomycetota", "Mycobacteriales"),
    ("ORG_012", "Pseudomonas aeruginosa", "Bacteria", "Proteobacteria", "Pseudomonadales"),
    ("ORG_013", "Rattus norvegicus",     "Eukaryota", "Mammalia",   "Rodentia"),
    ("ORG_014", "Gallus gallus",         "Eukaryota", "Aves",       "Galliformes"),
    ("ORG_015", "Xenopus laevis",        "Eukaryota", "Amphibia",   "Anura"),
]


def _mutate_sequence(seq: str, mutation_rate: float, rng: np.random.Generator) -> str:
    """Apply random point mutations to a protein sequence.

    Each position has a `mutation_rate` chance of being replaced with a random
    canonical amino acid. This simulates natural sequence divergence within a
    protein family (homologs typically share 30-90% identity).
    """
    aa_array = list(seq)
    for i in range(len(aa_array)):
        if rng.random() < mutation_rate:
            aa_array[i] = rng.choice(CANONICAL_AAS)
    return "".join(aa_array)


def _extend_sequence(seed: str, target_length: int, rng: np.random.Generator) -> str:
    """Extend or trim a seed motif to the target length.

    Real proteins have variable-length flanking regions around conserved domains.
    We pad with random AAs to reach the target length, or truncate if shorter.
    """
    if len(seed) >= target_length:
        return seed[:target_length]

    # Add random flanking residues before and after the seed
    remaining = target_length - len(seed)
    prefix_len = rng.integers(0, remaining + 1)
    suffix_len = remaining - prefix_len
    prefix = "".join(rng.choice(CANONICAL_AAS, size=prefix_len))
    suffix = "".join(rng.choice(CANONICAL_AAS, size=suffix_len))
    return prefix + seed + suffix


def _inject_non_canonical(seq: str, rng: np.random.Generator, rate: float = 0.03) -> str:
    """Inject non-canonical amino acids at a low rate to test validation filtering.

    Real datasets contain ambiguous codes (X = unknown, B = Asp/Asn, Z = Glu/Gln)
    and rare AAs (U = selenocysteine, O = pyrrolysine).
    """
    aa_array = list(seq)
    for i in range(len(aa_array)):
        if rng.random() < rate:
            aa_array[i] = rng.choice(NON_CANONICAL_AAS)
    return "".join(aa_array)


def generate_corpus(
    num_sequences: int = 100_000,
    seed: int = 42,
    inject_bad_fraction: float = 0.08,
) -> tuple[list[dict], pd.DataFrame]:
    """Generate a synthetic protein corpus with realistic properties.

    Returns:
        records: list of dicts with sequence_id, organism_id, sequence, family
        taxonomy_df: DataFrame mapping organism_id -> taxonomy fields
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Length distribution: bimodal, mimicking real protein databases
    # Most proteins are 100-400 aa, with a tail extending to 1024+
    records = []

    # Pre-compute which sequences will be "bad" (non-canonical AA injection)
    bad_indices = set(rng.choice(num_sequences, size=int(num_sequences * inject_bad_fraction), replace=False))

    organism_ids = [org[0] for org in ORGANISMS]

    for i in range(num_sequences):
        # Pick a random seed family
        family_name, seed_motif = SEED_FAMILIES[rng.integers(0, len(SEED_FAMILIES))]

        # Sample target length from a realistic distribution
        # Bimodal: 60% short-medium (20-400), 30% medium (200-700), 10% long (500-1024)
        r = rng.random()
        if r < 0.60:
            target_length = int(rng.integers(20, 400))
        elif r < 0.90:
            target_length = int(rng.integers(200, 700))
        else:
            target_length = int(rng.integers(500, 1024))

        # Build the sequence: extend seed to target length, then mutate
        mutation_rate = rng.uniform(0.10, 0.30)
        seq = _extend_sequence(seed_motif, target_length, rng)
        seq = _mutate_sequence(seq, mutation_rate, rng)

        # Inject non-canonical AAs for a fraction of sequences (to test filtering)
        if i in bad_indices:
            seq = _inject_non_canonical(seq, rng, rate=0.03)

        organism_id = rng.choice(organism_ids)
        sequence_id = f"PROT_{i:07d}"

        records.append({
            "sequence_id": sequence_id,
            "organism_id": organism_id,
            "sequence": seq,
            "family": family_name,
            "length": len(seq),
        })

    # Build taxonomy lookup
    taxonomy_df = pd.DataFrame(ORGANISMS, columns=[
        "organism_id", "species", "domain", "class_name", "order"
    ])

    print(f"Generated {len(records):,} protein sequences")
    print(f"  Families: {len(SEED_FAMILIES)}")
    print(f"  Length range: {min(r['length'] for r in records)}-{max(r['length'] for r in records)} aa")
    print(f"  Organisms: {len(ORGANISMS)}")
    print(f"  Bad sequences (non-canonical AAs): {len(bad_indices):,} ({inject_bad_fraction*100:.0f}%)")

    return records, taxonomy_df


def _build_homolog_pairs(records: list[dict], rng: np.random.Generator, num_pairs: int = 50) -> pd.DataFrame:
    """Build labeled test pairs for embedding validation.

    Homolog pairs: two sequences from the SAME protein family (should have high cosine sim).
    Random pairs: two sequences from DIFFERENT families (should have low cosine sim).
    """
    # Group sequences by family
    family_to_ids = {}
    for r in records:
        family_to_ids.setdefault(r["family"], []).append(r["sequence_id"])

    families = list(family_to_ids.keys())
    pairs = []

    # Generate homolog pairs (same family)
    for _ in range(num_pairs // 2):
        fam = rng.choice(families)
        ids = family_to_ids[fam]
        if len(ids) < 2:
            continue
        idx = rng.choice(len(ids), size=2, replace=False)
        pairs.append({
            "seq_id_a": ids[idx[0]],
            "seq_id_b": ids[idx[1]],
            "relationship": "homolog",
            "family": fam,
        })

    # Generate random pairs (different families)
    for _ in range(num_pairs - len(pairs)):
        fam_a, fam_b = rng.choice(families, size=2, replace=False)
        id_a = rng.choice(family_to_ids[fam_a])
        id_b = rng.choice(family_to_ids[fam_b])
        pairs.append({
            "seq_id_a": id_a,
            "seq_id_b": id_b,
            "relationship": "random",
            "family": f"{fam_a}/{fam_b}",
        })

    return pd.DataFrame(pairs)


def write_fasta(records: list[dict], fasta_path: str):
    """Write records to a standard FASTA file.

    FASTA format:
      >sequence_id|organism_id description
      MKTLLILAVF...
    """
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, "w") as f:
        for r in records:
            header = f">{r['sequence_id']}|{r['organism_id']} family={r['family']}"
            f.write(header + "\n")
            # Write sequence in 80-character lines (FASTA convention)
            seq = r["sequence"]
            for j in range(0, len(seq), 80):
                f.write(seq[j:j+80] + "\n")


def save_corpus(
    output_dir: str,
    num_sequences: int = 100_000,
    seed: int = 42,
) -> dict:
    """Generate and save all corpus artifacts.

    Writes:
      - corpus.fasta (standard FASTA format)
      - corpus.parquet (for Ray Data read_parquet)
      - taxonomy_lookup.parquet
      - homolog_test_pairs.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    records, taxonomy_df = generate_corpus(num_sequences=num_sequences, seed=seed)
    rng = np.random.default_rng(seed + 1)

    # Write FASTA
    fasta_path = os.path.join(output_dir, "corpus.fasta")
    write_fasta(records, fasta_path)
    print(f"  Saved FASTA           -> {fasta_path}")

    # Write Parquet (for efficient Ray Data loading)
    corpus_df = pd.DataFrame(records)
    parquet_path = os.path.join(output_dir, "corpus.parquet")
    corpus_df.to_parquet(parquet_path, index=False)
    print(f"  Saved Parquet         -> {parquet_path}")

    # Write taxonomy lookup
    taxonomy_path = os.path.join(output_dir, "taxonomy_lookup.parquet")
    taxonomy_df.to_parquet(taxonomy_path, index=False)
    print(f"  Saved taxonomy        -> {taxonomy_path}")

    # Write homolog test pairs
    pairs_df = _build_homolog_pairs(records, rng, num_pairs=50)
    pairs_path = os.path.join(output_dir, "homolog_test_pairs.csv")
    pairs_df.to_csv(pairs_path, index=False)
    print(f"  Saved homolog pairs   -> {pairs_path} ({len(pairs_df)} pairs)")

    return {
        "fasta_path": fasta_path,
        "parquet_path": parquet_path,
        "taxonomy_path": taxonomy_path,
        "pairs_path": pairs_path,
        "num_sequences": len(records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic protein corpus")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--output-dir", default="/mnt/cluster_storage/protein-embeddings/raw")
    args = parser.parse_args()
    save_corpus(args.output_dir, num_sequences=SCALE_MAP[args.scale])
