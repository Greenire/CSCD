from typing import List, Tuple, Union

import pandas as pd
from rdkit import Chem

def get_mol_indices(smiles_list: Union[str,List[str]], mol_registry: pd.DataFrame) -> Tuple[pd.DataFrame, List[int], List[str]]:
    indices = []
    new_smiles = []
    smiles_list = [smiles_list] if isinstance(smiles_list, str) else smiles_list
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in mol_registry['smiles'].values:
            idx = mol_registry[mol_registry['smiles'] == smiles]['index'].iloc[0]
            indices.append(idx)
        else:
            # Add new molecule to registry
            new_idx = len(mol_registry)
            mol_registry = pd.concat([
                mol_registry,
                pd.DataFrame({'index': [new_idx], 'smiles': [smiles]})
            ], ignore_index=True)
            indices.append(new_idx)
            new_smiles.append(smiles)
    return mol_registry, indices, new_smiles