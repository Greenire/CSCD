from rdkit import Chem

from typing import Union, Optional

from .logger import setup_logger

_logger = setup_logger(__name__)

def get_sulfur_idx(mol: Chem.Mol, symbol: str, neighbor_count: int) -> Union[int, None]:
    """
    Get the index of a sulfur or selenium atom that meets given conditions.
    
    Args:
        mol: RDKit molecule object
        symbol: Atom symbol ('S' or 'Se')
        neighbor_count: Number of neighbor atoms
        
    Returns:
        Index of the atom that meets conditions, or None if not found
    """
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == symbol and len(atom.GetNeighbors()) == neighbor_count:
            if neighbor_count == 1:
                return atom.GetIdx()
            elif neighbor_count == 2:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'N':
                        return atom.GetIdx()
    return None

def get_nitrogen_idx(mol: Chem.Mol, sulfur_idx: int) -> Union[int, None]:
    """
    Get the index of a nitrogen atom connected to the given sulfur atom.
    
    Args:
        mol: RDKit molecule object
        sulfur_idx: Index of the sulfur atom
        
    Returns:
        Index of the nitrogen atom, or None if not found
    """
    sulfur_atom = mol.GetAtomWithIdx(sulfur_idx)
    for neighbor in sulfur_atom.GetNeighbors():
        if neighbor.GetSymbol() == 'N':
            return neighbor.GetIdx()
    return None

def get_atom_idx(mol: Chem.Mol, symbol: str, neighbor_symbol: Optional[str] = None) -> Optional[int]:
    """Get the index of an atom with specific symbol and optional neighbor atom.
    
    Args:
        mol: RDKit molecule object
        symbol: Target atom symbol
        neighbor_symbol: Optional symbol of neighboring atom
        
    Returns:
        Index of the matching atom or None if not found
    """
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != symbol:
            continue
            
        if neighbor_symbol is None:
            return atom.GetIdx()
            
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == neighbor_symbol:
                return atom.GetIdx()
    return None

def merge_intermediates(smi1: str, smi2: str) -> str:
    """
    Merge two molecules by adding chemical bonds and adjusting charges.
    
    This function is used to combine reaction intermediates by:
    1. Combining the molecules
    2. Finding and connecting S-S bonds
    3. Adjusting formal charges
    4. Removing leaving groups
    5. Extracting the largest fragment
    
    Args:
        smi1: SMILES string of first molecule, maybe S reagent
        smi2: SMILES string of second molecule, maybe S catlyst
        
    Returns:
        SMILES string of merged molecule
        
    Raises:
        ValueError: If input SMILES strings are invalid
    """
    # Create RDKit molecules
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES strings provided")
    
    try:
        # Combine molecules
        merged_mol = Chem.CombineMols(mol1, mol2)
        
        # Get indices of S-S bond and charged sulfur
        s_s_idx = get_sulfur_idx(merged_mol, 'S', 2) or get_sulfur_idx(merged_mol, 'Se', 2)
        s_cat_idx = get_sulfur_idx(merged_mol, 'S', 1) or get_sulfur_idx(merged_mol, 'Se', 1)
        
        if s_cat_idx is None or s_s_idx is None:
            raise ValueError(f"Could not find required sulfur atoms, smi1: {smi1} and smi2:{smi2}")
            # return Chem.MolToSmiles(merged_mol)
            
        # Add positive charge
        atom = merged_mol.GetAtomWithIdx(s_cat_idx)
        atom.SetFormalCharge(1)
        
        # Create editable molecule and connect molecules
        ed_merged_mol = Chem.EditableMol(merged_mol)
        ed_merged_mol.AddBond(s_s_idx, s_cat_idx, order=Chem.rdchem.BondType.SINGLE)
        
        # Remove leaving group
        s_n_idx = get_nitrogen_idx(merged_mol, s_s_idx)
        if s_n_idx is not None:
            ed_merged_mol.RemoveBond(s_s_idx, s_n_idx)
            
        # Get final molecule and extract largest fragment
        merged_mol = ed_merged_mol.GetMol()
        frags = Chem.GetMolFrags(merged_mol, asMols=True)
        largest_frag = max(frags, key=lambda frag: frag.GetNumAtoms())
        
        # Convert to SMILES
        final_smi = Chem.MolToSmiles(largest_frag)
        return final_smi
        
    except Exception as e:
        _logger.error(f"Error merging molecules: {str(e)}")
        raise

def merge_products(smi1: str, smi2: str) -> str:
    """Merge two product molecules by connecting carbon and sulfur atoms.
    
    This function:
    1. Combines the molecules
    2. Finds Si, S, and C atoms
    3. Creates a new C-S bond
    4. Removes Si atom and S-N bond
    5. Returns the largest fragment
    
    Args:
        smi1: SMILES string of first molecule
        smi2: SMILES string of second molecule
        
    Returns:
        SMILES string of merged molecule
        
    Raises:
        ValueError: If input SMILES are invalid or required atoms not found
    """
    try:
        # Create RDKit molecules
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        
        if mol1 is None or mol2 is None:
            raise ValueError("Invalid SMILES strings provided")
            
        # Combine molecules
        combined_mol = Chem.CombineMols(mol1, mol2)
        
        # Find required atoms
        si_atom_idx = get_atom_idx(combined_mol, '*')
        if si_atom_idx is None:
            raise ValueError("Label atom not found")
            
        s_atom_idx = get_sulfur_idx(combined_mol, 'S', 2) or get_sulfur_idx(combined_mol, 'Se', 2)
        if s_atom_idx is None:
            raise ValueError("Sulfur atom not found")
            
        n_atom_idx = get_nitrogen_idx(combined_mol, s_atom_idx)
        if n_atom_idx is None:
            raise ValueError("Nitrogen atom not found")
            
        # Find carbon atom connected to silicon
        si_atom = combined_mol.GetAtomWithIdx(si_atom_idx)
        c_atom_idx = None
        for atom in si_atom.GetNeighbors():
            if atom.GetSymbol() == 'C':
                c_atom_idx = atom.GetIdx()
                break
                
        if c_atom_idx is None:
            raise ValueError("Carbon atom not found")
            
        # Edit molecule
        editable_mol = Chem.EditableMol(combined_mol)
        editable_mol.AddBond(c_atom_idx, s_atom_idx, order=Chem.rdchem.BondType.SINGLE)
        editable_mol.RemoveBond(s_atom_idx, n_atom_idx)
        editable_mol.RemoveAtom(si_atom_idx)
        
        # Get final molecule
        merged_mol = editable_mol.GetMol()
        
        # Extract largest fragment
        frags = Chem.GetMolFrags(merged_mol, asMols=True)
        largest_frag = max(frags, key=lambda frag: frag.GetNumAtoms())
        
        # Convert to SMILES
        return Chem.MolToSmiles(largest_frag)
        
    except Exception as e:
        _logger.error(f"Error merging product molecules: {str(e)}")
        raise