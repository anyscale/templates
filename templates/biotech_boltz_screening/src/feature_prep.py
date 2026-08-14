"""
CPU stage: parse sequences, validate amino acids, and build Boltz input dicts.

Converts raw candidate rows (target_seq + binder_seq or ligand_smiles) into the
YAML schema `boltz predict` consumes.

MSAs default to `empty`, Boltz's single-sequence mode. It costs accuracy, and it
is what lets this template run anywhere: the alternative is `--use_msa_server`
(network round-trip to an MMseqs2 server, rate-limited) or shipping a sequence
database. Pass `target_msa_path` an `.a3m` to screen for real -- screens fix the
target and vary the binder, so one alignment covers every complex.
"""
import json

import numpy as np

# ── Valid amino acid alphabet ────────────────────────────────────────────
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Boltz's literal for single-sequence mode. Override with a path to an .a3m.
DEFAULT_TARGET_MSA = "empty"


def _validate_sequence(seq: str) -> bool:
    """Check that a protein sequence contains only standard amino acids."""
    return all(c in VALID_AA for c in seq.upper())


def _build_protein_protein_input(
    complex_id: str,
    target_seq: str,
    binder_seq: str,
    target_msa_path: str,
) -> dict:
    """Build Boltz input dict for a protein-protein complex.

    Boltz input schema (YAML):
      sequences:
        - protein:
            id: A
            sequence: <target_seq>
            msa: empty                  # or a path to an .a3m
        - protein:
            id: B
            sequence: <binder_seq>
            msa: empty                  # single-sequence mode
    """
    return {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": target_seq.upper(),
                    "msa": target_msa_path,
                }
            },
            {
                "protein": {
                    "id": "B",
                    "sequence": binder_seq.upper(),
                    # Designed binders have no meaningful alignment anyway.
                    "msa": "empty",
                }
            },
        ]
    }


def _build_protein_ligand_input(
    complex_id: str,
    target_seq: str,
    ligand_smiles: str,
    target_msa_path: str,
) -> dict:
    """Build Boltz input dict for a protein-ligand complex.

    Boltz input schema (YAML):
      sequences:
        - protein:
            id: A
            sequence: <target_seq>
            msa: empty                  # or a path to an .a3m
        - ligand:
            id: B
            smiles: <SMILES>
    """
    return {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": target_seq.upper(),
                    "msa": target_msa_path,
                }
            },
            {
                "ligand": {
                    "id": "B",
                    "smiles": ligand_smiles,
                }
            },
        ]
    }


def build_boltz_input_batch(
    batch: dict,
    target_msa_path: str = DEFAULT_TARGET_MSA,
) -> dict:
    """CPU map_batches function: convert a batch of candidate rows into Boltz inputs.

    For each row, validates sequences and builds the appropriate input dict
    (protein-protein or protein-ligand) based on the complex_type column.

    Input columns: complex_id, target_seq, binder_seq, ligand_smiles, complex_type
    Output columns: complex_id, complex_type, boltz_input (JSON string),
                    target_len, binder_len, is_valid
    """
    n = len(batch["complex_id"])

    out = {
        "complex_id": [],
        "complex_type": [],
        "boltz_input": [],       # JSON-serialized Boltz input dict
        "target_len": [],        # number of residues in target
        "binder_len": [],        # number of residues in binder (0 for ligand complexes)
        "is_valid": [],          # whether input passed validation
    }

    for i in range(n):
        complex_id = batch["complex_id"][i]
        complex_type = batch["complex_type"][i]
        target_seq = batch["target_seq"][i]

        # Handle numpy bytes/strings
        if isinstance(complex_id, bytes):
            complex_id = complex_id.decode("utf-8")
        if isinstance(complex_type, bytes):
            complex_type = complex_type.decode("utf-8")
        if isinstance(target_seq, bytes):
            target_seq = target_seq.decode("utf-8")

        target_len = len(target_seq)
        is_valid = _validate_sequence(target_seq)

        if complex_type == "pp":
            binder_seq = batch["binder_seq"][i]
            if isinstance(binder_seq, bytes):
                binder_seq = binder_seq.decode("utf-8")
            is_valid = is_valid and _validate_sequence(binder_seq) and len(binder_seq) > 0
            binder_len = len(binder_seq)
            boltz_input = _build_protein_protein_input(
                complex_id, target_seq, binder_seq, target_msa_path,
            )
        elif complex_type == "pl":
            ligand_smiles = batch["ligand_smiles"][i]
            if isinstance(ligand_smiles, bytes):
                ligand_smiles = ligand_smiles.decode("utf-8")
            is_valid = is_valid and len(ligand_smiles) > 0
            binder_len = 0
            boltz_input = _build_protein_ligand_input(
                complex_id, target_seq, ligand_smiles, target_msa_path,
            )
        else:
            is_valid = False
            binder_len = 0
            boltz_input = {}

        out["complex_id"].append(complex_id)
        out["complex_type"].append(complex_type)
        out["boltz_input"].append(json.dumps(boltz_input))
        out["target_len"].append(target_len)
        out["binder_len"].append(binder_len)
        out["is_valid"].append(is_valid)

    return out
