"""
Generate synthetic protein-protein and protein-ligand candidate tables for screening.

Produces realistic amino acid distributions and drug-like SMILES for demo purposes.
No external databases required -- everything is self-contained.
"""
import os
import random

import numpy as np
import pandas as pd

# ── Scale map: number of candidate complexes per scale tier ──────────────
SCALE_MAP = {"small": 50, "medium": 500, "large": 2000}

# ── Amino acid distribution (roughly matching natural protein frequencies) ──
# These frequencies are based on observed amino acid composition in the PDB.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_WEIGHTS = [
    0.074, 0.025, 0.054, 0.054, 0.047,  # A, C, D, E, F
    0.074, 0.026, 0.068, 0.058, 0.099,  # G, H, I, K, L
    0.025, 0.045, 0.039, 0.034, 0.052,  # M, N, P, Q, R
    0.057, 0.051, 0.073, 0.013, 0.032,  # S, T, V, W, Y
]

# ── Fixed target protein sequence (~150 amino acids) ─────────────────────
# Inspired by a short receptor extracellular domain. Fixed across all screens
# so that MSA can be pre-computed once and reused.
TARGET_SEQUENCE = (
    "MKTLLPVLVMSLAISGAYAAQPARVVWAQEGAPAQLPCSPTIPLQDLSLLRRAGVTWQHQ"
    "PDSGPPAAAPGHPLAPGPHPAAPSSWGPRPRRYTVLSVGPGGLRSGRLPLQPRVQLDERR"
    "PQAGLATRGKFAAATGATPGSPFG"
)

# ── Seed scaffold for binder candidates (~65 amino acids) ────────────────
# A small designed protein scaffold. Mutations are introduced to create
# a realistic distribution of binder variants with varying binding potential.
BINDER_SCAFFOLD = (
    "MTEYKLVVVGAVGVGKSALTIQLIQNHFVDEYDPTIEDAYRKQVVIDGETCLL"
    "DILDTAYSSYRKQV"
)

# ── Hardcoded drug-like SMILES (~100 compounds) ─────────────────────────
# These are representative drug-like molecules covering diverse scaffolds.
# Used for protein-ligand screening without requiring ChEMBL/ZINC downloads.
DRUG_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",                                       # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                                   # Ibuprofen
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",                         # Testosterone
    "OC(=O)c1ccccc1O",                                               # Salicylic acid
    "CC(=O)Nc1ccc(O)cc1",                                            # Acetaminophen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                                 # Caffeine
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",                     # Glucose
    "CC(C)NCC(O)c1ccc(O)c(O)c1",                                     # Isoproterenol
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",                              # Pyrene
    "CC(=O)OC1=CC=CC=C1C(=O)O",                                     # Aspirin alt
    "C1CCCCC1N",                                                      # Cyclohexylamine
    "c1ccncc1",                                                       # Pyridine
    "CC1=CC=C(C=C1)S(=O)(=O)N",                                     # Toluenesulfonamide
    "OC(=O)C(F)(F)F",                                                # Trifluoroacetic acid
    "CC(C)(C)c1ccc(O)cc1",                                           # 4-tert-butylphenol
    "c1ccc(cc1)C(=O)O",                                              # Benzoic acid
    "NC(=O)c1cccnc1",                                                # Nicotinamide
    "Oc1cccc2ccccc12",                                               # 1-Naphthol
    "CC(=O)c1ccccc1",                                                # Acetophenone
    "OC(=O)c1cc(O)c(O)c(O)c1",                                      # Gallic acid
    "CC(C)CC(=O)O",                                                   # Isovaleric acid
    "c1ccc(cc1)N",                                                    # Aniline
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",                                   # Citric acid
    "CC(=O)Oc1ccccc1",                                                # Phenyl acetate
    "c1cc[nH]c1",                                                     # Pyrrole
    "NC(=O)c1ccc(N)cc1",                                              # 4-aminobenzamide
    "OC(=O)/C=C/c1ccccc1",                                           # Cinnamic acid
    "CC(O)CC(=O)O",                                                   # 3-hydroxybutyric acid
    "Oc1ccc(cc1)C(=O)O",                                             # 4-hydroxybenzoic acid
    "c1ccc(cc1)CO",                                                   # Benzyl alcohol
    "NC(CC(=O)O)C(=O)O",                                             # Aspartic acid
    "CC(=O)NCCC1=CNc2ccc(O)cc21",                                   # Melatonin analog
    "OC(=O)c1cccnc1",                                                # Nicotinic acid
    "c1ccoc1",                                                        # Furan
    "CC(=O)OCC",                                                      # Ethyl acetate
    "OC(=O)c1cccc(O)c1",                                             # 3-hydroxybenzoic acid
    "NC(Cc1ccc(O)cc1)C(=O)O",                                        # Tyrosine
    "NC(Cc1c[nH]c2ccccc12)C(=O)O",                                  # Tryptophan
    "NC(CCCCN)C(=O)O",                                               # Lysine
    "NC(Cc1ccccc1)C(=O)O",                                           # Phenylalanine
    "NC(CS)C(=O)O",                                                   # Cysteine
    "NC(CCCNC(=N)N)C(=O)O",                                         # Arginine
    "NC(CC(=O)N)C(=O)O",                                             # Asparagine
    "NC(CCC(=O)O)C(=O)O",                                            # Glutamic acid
    "NC(CCC(=O)N)C(=O)O",                                            # Glutamine
    "NC(CO)C(=O)O",                                                   # Serine
    "NC([C@@H](C)O)C(=O)O",                                         # Threonine
    "CC(C)C[C@@H](N)C(=O)O",                                        # Leucine
    "CC[C@H](C)[C@@H](N)C(=O)O",                                    # Isoleucine
    "NC(CC(=O)O)C(=O)O",                                             # Aspartic acid alt
    "OC(=O)CNC(=O)C(=O)O",                                          # Oxalylglycine
    "c1ccc(-c2ccccc2)cc1",                                            # Biphenyl
    "CC(=O)Nc1ccc(F)cc1",                                            # Fluoroacetanilide
    "O=C1CCCN1",                                                      # 2-pyrrolidinone
    "COc1ccc(cc1)C(=O)O",                                            # Anisic acid
    "CC(=O)c1ccc(O)cc1",                                              # 4-hydroxyacetophenone
    "OC(=O)c1cc(F)cc(F)c1",                                          # 3,5-difluorobenzoic acid
    "CC1=CC(=O)c2ccccc2C1=O",                                        # Menadione
    "c1ccc2[nH]ccc2c1",                                              # Indole
    "CC(=O)NC(CSCC(=O)O)C(=O)O",                                    # N-acetylcysteine deriv
    "OC(=O)c1ccc(Cl)cc1",                                            # 4-chlorobenzoic acid
    "Nc1ccc(Cl)cc1",                                                  # 4-chloroaniline
    "OC(=O)c1cccc(Cl)c1",                                            # 3-chlorobenzoic acid
    "CC(=O)Nc1ccc(Cl)cc1",                                           # Chloroacetanilide
    "c1ccc(cc1)C#N",                                                  # Benzonitrile
    "CC1CC(=O)CC(C)C1",                                               # Dimedone
    "O=C1CCC(=O)N1",                                                  # Succinimide
    "OC(=O)C=Cc1ccccc1",                                             # Cinnamic acid alt
    "Oc1ccc(F)cc1",                                                   # 4-fluorophenol
    "COc1ccccc1O",                                                    # Guaiacol
    "OC(=O)c1cc(Br)ccc1O",                                          # 5-bromosalicylic acid
    "CC(C)(O)C(=O)O",                                                # 2-hydroxyisobutyric acid
    "CC(=O)c1ccccn1",                                                # 2-acetylpyridine
    "OC(=O)c1ccc(Br)cc1",                                           # 4-bromobenzoic acid
    "NC(=O)c1cccc(O)c1",                                             # 3-hydroxybenzamide
    "Oc1ccc(Br)cc1",                                                  # 4-bromophenol
    "OC(=O)c1cccc2ccccc12",                                          # 1-naphthoic acid
    "c1cc(O)c(O)cc1",                                                # Catechol
    "CC(=O)OC1CC(C)(C)C(=O)C(C)C1OC(C)=O",                         # Dihydroartemisinin analog
    "OC(=O)c1ccc(O)c(O)c1",                                         # Protocatechuic acid
    "c1ccc(cc1)S(=O)(=O)O",                                          # Benzenesulfonic acid
    "NC(=S)N",                                                        # Thiourea
    "OC(=O)c1cc(I)ccc1O",                                           # 5-iodosalicylic acid
    "COc1cc(C=O)ccc1O",                                              # Vanillin
    "CC(=O)NC1CCC(O)CC1",                                            # Acetamidocyclohexanol
    "OC(=O)c1ccc2OCOc2c1",                                          # Piperonylic acid
    "c1cc(O)ccc1N",                                                   # 3-aminophenol
    "OC(=O)c1ccncc1C(=O)O",                                         # 2,3-pyridinedicarboxylic acid
    "CC(CC(=O)O)C(=O)O",                                            # Methylsuccinic acid
    "Oc1cccc(O)c1",                                                  # Resorcinol
    "c1ccc(cc1)C(=O)Cl",                                             # Benzoyl chloride
    "CC(=O)c1ccccc1O",                                               # 2-hydroxyacetophenone
    "OC(=O)c1ccccc1Cl",                                              # 2-chlorobenzoic acid
    "NC(=O)c1cccc(Cl)c1",                                            # 3-chlorobenzamide
    "Oc1ccc(cc1)N(=O)=O",                                           # 4-nitrophenol
    "OC(=O)c1cccc(N)c1",                                             # 3-aminobenzoic acid
    "c1ccc(O)c(N)c1",                                                # 2-aminophenol
    "OC(=O)c1ccc(N)cc1",                                             # 4-aminobenzoic acid
    "CC(=O)Nc1cccc(C(=O)O)c1",                                      # 3-acetamidobenzoic acid
    "OC(=O)CC(O)C(=O)O",                                            # Malic acid
]


def _random_sequence(length: int, rng: random.Random) -> str:
    """Generate a random amino acid sequence with realistic residue frequencies."""
    return "".join(rng.choices(AMINO_ACIDS, weights=AA_WEIGHTS, k=length))


def _mutate_scaffold(scaffold: str, mutation_rate: float, rng: random.Random) -> str:
    """Introduce point mutations into a scaffold sequence.

    This simulates the process of computational protein design, where a binder
    scaffold is diversified by mutating surface residues to create a library
    of candidate binders with varying binding potential.
    """
    seq = list(scaffold)
    for i in range(len(seq)):
        if rng.random() < mutation_rate:
            seq[i] = rng.choices(AMINO_ACIDS, weights=AA_WEIGHTS, k=1)[0]
    return "".join(seq)


def build_protein_protein_candidates(
    num_candidates: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a screening table of target + candidate binder complexes.

    Each row represents one protein-protein complex to fold:
    - target_seq: the fixed receptor protein (~150 aa)
    - binder_seq: a mutated variant of the seed scaffold (~50-80 aa)
    - complex_type: 'pp' (protein-protein)

    Returns DataFrame with columns:
        complex_id, target_seq, binder_seq, complex_type
    """
    rng = random.Random(seed)
    rows = []

    for i in range(num_candidates):
        # Vary mutation rate to create spread in binding potential
        # Lower mutation rates = closer to scaffold = more likely to bind
        mutation_rate = rng.uniform(0.05, 0.45)

        # Vary binder length slightly (50-80 aa) by truncating or extending scaffold
        length = rng.randint(50, 80)
        if length <= len(BINDER_SCAFFOLD):
            base = BINDER_SCAFFOLD[:length]
        else:
            base = BINDER_SCAFFOLD + _random_sequence(length - len(BINDER_SCAFFOLD), rng)

        binder_seq = _mutate_scaffold(base, mutation_rate, rng)

        rows.append({
            "complex_id": f"pp_{i:05d}",
            "target_seq": TARGET_SEQUENCE,
            "binder_seq": binder_seq,
            "ligand_smiles": "",
            "complex_type": "pp",
        })

    return pd.DataFrame(rows)


def build_protein_ligand_candidates(
    num_candidates: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a screening table of target + candidate ligand complexes.

    Each row represents one protein-ligand complex to fold:
    - target_seq: the fixed receptor protein (~150 aa)
    - ligand_smiles: a drug-like SMILES string
    - complex_type: 'pl' (protein-ligand)

    Returns DataFrame with columns:
        complex_id, target_seq, binder_seq, ligand_smiles, complex_type
    """
    rng = random.Random(seed)
    rows = []

    for i in range(num_candidates):
        smiles = rng.choice(DRUG_SMILES)
        rows.append({
            "complex_id": f"pl_{i:05d}",
            "target_seq": TARGET_SEQUENCE,
            "binder_seq": "",
            "ligand_smiles": smiles,
            "complex_type": "pl",
        })

    return pd.DataFrame(rows)


def generate_candidates(
    output_path: str,
    num_candidates: int = 500,
    complex_type: str = "pp",
    seed: int = 42,
) -> str:
    """Generate candidate complexes and save as Parquet.

    Args:
        output_path: Path to write the Parquet file.
        num_candidates: Number of candidate complexes.
        complex_type: 'pp' (protein-protein) or 'pl' (protein-ligand).
        seed: Random seed for reproducibility.

    Returns:
        Path to the written Parquet file.
    """
    if complex_type == "pp":
        df = build_protein_protein_candidates(num_candidates, seed)
    elif complex_type == "pl":
        df = build_protein_ligand_candidates(num_candidates, seed)
    else:
        raise ValueError(f"Unknown complex_type: {complex_type}. Use 'pp' or 'pl'.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Generated {len(df):,} {complex_type} candidates → {output_path}")
    return output_path
