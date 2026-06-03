
from __future__ import annotations

from collections import defaultdict
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

def build_mpnn(
	molecule: Dict[str, Sequence[str]],
	med_voc: Dict[int, str],
	radius: int = 1,
	device: torch.device = torch.device("cpu"),
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, int]], int, torch.Tensor]:
	try:
		from rdkit import Chem
	except ImportError as exc:
		raise ImportError("RDKit is required to build MPNN inputs.") from exc

	atom_dict: Dict[Any, int] = defaultdict(lambda: len(atom_dict))
	bond_dict: Dict[str, int] = defaultdict(lambda: len(bond_dict))
	fingerprint_dict: Dict[Any, int] = defaultdict(lambda: len(fingerprint_dict))
	edge_dict: Dict[Any, int] = defaultdict(lambda: len(edge_dict))

	def create_atoms(mol, atom_dict: Dict[Any, int]) -> np.ndarray:
		atoms = [a.GetSymbol() for a in mol.GetAtoms()]
		for a in mol.GetAromaticAtoms():
			idx = a.GetIdx()
			atoms[idx] = (atoms[idx], "aromatic")
		atoms = [atom_dict[a] for a in atoms]
		return np.array(atoms)

	def create_ijbonddict(mol, bond_dict: Dict[str, int]) -> Dict[int, List[Tuple[int, int]]]:
		i_jbond_dict: Dict[int, List[Tuple[int, int]]] = defaultdict(lambda: [])
		for bond in mol.GetBonds():
			i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
			bond_id = bond_dict[str(bond.GetBondType())]
			i_jbond_dict[i].append((j, bond_id))
			i_jbond_dict[j].append((i, bond_id))
		return i_jbond_dict

	def extract_fingerprints(
		radius: int,
		atoms: np.ndarray,
		i_jbond_dict: Dict[int, List[Tuple[int, int]]],
		fingerprint_dict: Dict[Any, int],
		edge_dict: Dict[Any, int],
	) -> np.ndarray:
		if len(atoms) == 1 or radius == 0:
			return np.array([fingerprint_dict[a] for a in atoms])

		nodes = atoms
		i_jedge_dict = i_jbond_dict

		for _ in range(radius):
			nodes_next = []
			for i, j_edge in i_jedge_dict.items():
				neighbors = [(nodes[j], edge) for j, edge in j_edge]
				fingerprint = (nodes[i], tuple(sorted(neighbors)))
				nodes_next.append(fingerprint_dict[fingerprint])

			i_jedge_dict_next: Dict[int, List[Tuple[int, int]]] = defaultdict(lambda: [])
			for i, j_edge in i_jedge_dict.items():
				for j, edge in j_edge:
					both_side = tuple(sorted((nodes[i], nodes[j])))
					edge_id = edge_dict[(both_side, edge)]
					i_jedge_dict_next[i].append((j, edge_id))

			nodes = np.array(nodes_next)
			i_jedge_dict = i_jedge_dict_next

		return nodes

	mpnn_set: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
	average_index: List[int] = []

	for _, atc3 in med_voc.items():
		smiles_list = list(molecule[atc3])
		counter = 0
		for smiles in smiles_list:
			try:
				mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
				atoms = create_atoms(mol, atom_dict)
				molecular_size = len(atoms)
				i_jbond_dict = create_ijbonddict(mol, bond_dict)
				fingerprints = extract_fingerprints(
					radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
				)
				adjacency = Chem.GetAdjacencyMatrix(mol)
				for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
					fingerprints = np.append(fingerprints, 1)

				fingerprints_tensor = torch.LongTensor(fingerprints).to(device)
				adjacency_tensor = torch.FloatTensor(adjacency).to(device)
				mpnn_set.append((fingerprints_tensor, adjacency_tensor, molecular_size))
				counter += 1
			except Exception:
				continue

		average_index.append(counter)

	n_fingerprint = len(fingerprint_dict)
	n_col = sum(average_index)
	n_row = len(average_index)

	average_projection = np.zeros((n_row, n_col))
	col_counter = 0
	for i, item in enumerate(average_index):
		if item > 0:
			average_projection[i, col_counter : col_counter + item] = 1 / item
		col_counter += item

	return mpnn_set, n_fingerprint, torch.FloatTensor(average_projection)