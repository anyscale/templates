"""Minimal graph featurizer — the molecule path only.

Reproduces kMoL's GraphFeaturizer + RdkitDescriptorComputer +
AbstractTorchGeometricFeaturizer._process exactly (same atom-feature order, same
edge-index construction/sort, same 17 RDKit descriptors), using the same vendored
dgllife atom featurizers kMoL uses. This is the piece that must be byte-identical
for parity; it deliberately avoids featurizers.py (openbabel/prody/openfold).

Note: for this GCN config edge_attr is NOT consumed by the model (LEConv with
propagate_edge_features=False, edge_features=0). Bonds only define connectivity
(edge_index). Bond features are still produced for faithfulness but do not affect
the output.
"""
import itertools
from functools import partial
from typing import Callable, List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.data import Data as TorchGeometricData

from .dgllife_featurizers import (
    atom_degree_one_hot,
    atom_formal_charge,
    atom_hybridization_one_hot,
    atom_implicit_valence_one_hot,
    atom_is_aromatic,
    atom_num_radical_electrons,
    atom_total_num_H_one_hot,
    atom_type_one_hot,
    bond_is_conjugated,
    bond_is_in_ring,
    bond_stereo_one_hot,
    bond_type_one_hot,
)


class RdkitDescriptorComputer:
    """17 molecule-level RDKit descriptors, in kMoL's exact order."""

    def _get_descriptor_calculators(self) -> List[Callable]:
        from rdkit.Chem import (
            Crippen, Descriptors, GraphDescriptors, Lipinski, MolSurf, QED, rdMolDescriptors,
        )
        return [
            Descriptors.MolWt,
            Descriptors.NumRadicalElectrons,
            Descriptors.NumValenceElectrons,
            rdMolDescriptors.CalcTPSA,
            MolSurf.LabuteASA,
            GraphDescriptors.BalabanJ,
            Lipinski.RingCount,
            Lipinski.NumAliphaticRings,
            Lipinski.NumSaturatedRings,
            Lipinski.NumRotatableBonds,
            Lipinski.NumHeteroatoms,
            Lipinski.HeavyAtomCount,
            Lipinski.NumHDonors,
            Lipinski.NumHAcceptors,
            Lipinski.NumAromaticRings,
            Crippen.MolLogP,
            QED.qed,
        ]

    def run(self, mol: Chem.Mol) -> List[float]:
        return [featurizer(mol) for featurizer in self._get_descriptor_calculators()]


class GraphFeaturizer:
    DEFAULT_ATOM_TYPES = ["B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Br", "I"]

    def __init__(self, allowed_atom_types: Optional[List[str]] = None):
        self._allowed_atom_types = allowed_atom_types or self.DEFAULT_ATOM_TYPES
        self._descriptor_calculator = RdkitDescriptorComputer()

    # ---- atom/bond featurizers (kMoL's exact lists) ----
    def _list_atom_featurizers(self) -> List[Callable]:  # 45 features
        return [
            partial(atom_type_one_hot, allowable_set=self._allowed_atom_types, encode_unknown=True),
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot,
        ]

    def _list_bond_featurizers(self) -> List[Callable]:  # 12 features
        return [bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot]

    def _featurize_atom(self, atom: Chem.Atom) -> List[float]:
        return list(itertools.chain.from_iterable(f(atom) for f in self._list_atom_featurizers()))

    def _featurize_bond(self, bond: Chem.Bond) -> List[float]:
        return list(itertools.chain.from_iterable(f(bond) for f in self._list_bond_featurizers()))

    # ---- graph construction (matches AbstractTorchGeometricFeaturizer._process) ----
    def featurize(self, smiles: str) -> TorchGeometricData:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not featurize: [{smiles}]")

        atom_features = [self._featurize_atom(a) for a in mol.GetAtoms()]
        atom_features = torch.FloatTensor(atom_features).view(-1, len(atom_features[0]))

        edge_indices, edge_attributes = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
            bf = self._featurize_bond(bond)
            edge_attributes += [bf, bf]

        edge_indices = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
        edge_attributes = torch.FloatTensor(edge_attributes)
        if edge_indices.numel() > 0:
            perm = (edge_indices[0] * atom_features.size(0) + edge_indices[1]).argsort()
            edge_indices, edge_attributes = edge_indices[:, perm], edge_attributes[perm]

        data = TorchGeometricData(
            x=atom_features, edge_index=edge_indices, edge_attr=edge_attributes, smiles=smiles
        )
        mf = self._descriptor_calculator.run(mol)
        data.molecule_features = torch.FloatTensor(mf).view(-1, len(mf))
        return data


def collate(data_list: List[TorchGeometricData]) -> Batch:
    """Batch a list of PyG Data into one Batch — equivalent to kMoL's
    GeneralCollater (which wraps torch_geometric's Collater / from_data_list)."""
    return Batch.from_data_list(data_list)
