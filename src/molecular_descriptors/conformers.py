from ast import Tuple
from typing import List, Optional, Union, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
import heapq

from ..utils.config import Config, get_config

class ConformerGenerator:
    """Generator for molecular conformers"""
    
    # Default settings
    DEFAULT_SETTINGS = {
        'num_confs': 100,
        'max_attempts': 1000,
        'prune_rms_thresh': 0.1,
        'random_seed': 42,
        'energy_window': 10.0,  # kcal/mol
        'force_field': 'MMFF94',  # or 'UFF'
        'optimize_confs': True
    }
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize conformer generator
        
        Args:
            config: Configuration with settings:
                num_confs: Number of conformers to generate
                max_attempts: Maximum attempts for generation
                prune_rms_thresh: RMS threshold for pruning
                random_seed: Random seed for reproducibility
                energy_window: Energy window for keeping conformers
                force_field: Force field for optimization
                optimize_confs: Whether to optimize conformers
        """
        self.config = config or get_config()
        
        # Set parameters
        for key, default in self.DEFAULT_SETTINGS.items():
            setattr(self, f"_{key}", self.config.get(key, default))
            
    def _optimize_conformer(self, mol: Chem.Mol, conf_id: int) -> float:
        """
        Optimize a single conformer
        
        Args:
            mol: Molecule to optimize
            conf_id: Conformer ID
            
        Returns:
            Energy of optimized conformer
        """
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if self._force_field == 'MMFF94':
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, props, confId=conf_id
            )
        else:  # UFF
            ff = AllChem.UFFGetMoleculeForceField(mol, props, confId=conf_id)
            
        if ff is None:
            return float('inf')
            
        ff.Minimize(maxIts=200)
        return ff.CalcEnergy()
        
    def generate(self, smiles: str, num_top: int = 1) -> Tuple[Chem.Mol, List[int]]:
        """
        Generate conformers for a molecule
        
        Args:
            smiles: SMILES string of the molecule
            num_top: Return only top N conformers
            
        Returns:
            Tuple of molecule with conformer ids
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        mol = Chem.AddHs(mol)
            
        # Generate conformers
        if not AllChem.EmbedMultipleConfs(
            mol,
            numConfs=self._num_confs,
            randomSeed=self._random_seed,
            numThreads=0  # Use all available
        ):
            raise RuntimeError("Failed to generate conformers")
        
        # Optimize conformers if requested
        energies = []
        for conf_id in range(mol.GetNumConformers()):
            energy = self._optimize_conformer(mol, conf_id)
            energies.append((conf_id, energy))

        # Filter conformers by energy
        top_k_confs = heapq.nsmallest(num_top, energies, key=lambda x: x[1])
        top_k_confs = [tmp[0] for tmp in top_k_confs]
                
        return mol, top_k_confs
