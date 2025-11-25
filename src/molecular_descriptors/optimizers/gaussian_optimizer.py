from typing import Dict, List, Optional, Union, Any
import pandas as pd
from pathlib import Path
import json
from rdkit import Chem

from ..base import BaseDescriptorCalculator
from ..utils.logger import setup_logger, log_operation
from ..conformers import ConformerGenerator
from ..jobs import SlurmJobManager
from ..post_processing import MultiwfnProcessor, MorfeusProcessor

class GaussianOptimizer(BaseDescriptorCalculator):
    """Calculator for DFT-based molecular descriptors using Gaussian"""
    
    # Default calculation parameters
    DEFAULT_PARAMS = {
        'method': 'B3LYP',
        'basis': '6-31G(d)',
        'memory': '8GB',
        'nproc': 8,
        'charge': 0,
        'multiplicity': 1
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gaussian calculator
        
        Args:
            config: Configuration dictionary with:
                gaussian_params: Gaussian calculation parameters
                post_processors: List of post-processing tools
                conformer_params: Parameters for conformer generation
                job_params: Parameters for job submission
        """
        self.config = config
        self._logger = setup_logger(__name__)
        
        if not self.validate_config(config):
            raise ValueError("Invalid configuration")
            
        # Setup parameters
        self._setup_params()
        
        # Initialize job manager and conformer generator
        self.job_manager = SlurmJobManager(config.get('job_params', {}))
        self.conformer_generator = ConformerGenerator(config.get('conformer_params', {}))
        
        # Initialize post-processors
        self._setup_post_processors()
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        if 'gaussian_params' not in config:
            return False
            
        if 'post_processors' in config:
            valid_processors = ['multiwfn', 'morfeus']
            if not all(p in valid_processors for p in config['post_processors']):
                return False
                
        return True
        
    def supports_feature(self, feature: str) -> bool:
        """Check if feature is supported"""
        supported = ['conformers', 'multiwfn', 'morfeus']
        return feature in supported
        
    def _setup_params(self):
        """Setup calculation parameters"""
        params = self.config.get('gaussian_params', {})
        self._params = self.DEFAULT_PARAMS.copy()
        self._params.update(params)
        
    def _setup_post_processors(self):
        """Setup post-processing tools"""
        self._post_processors = []
        for proc in self.config.get('post_processors', []):
            if proc == 'multiwfn':
                self._post_processors.append(
                    MultiwfnProcessor(self.config.get('multiwfn_params', {}))
                )
            elif proc == 'morfeus':
                self._post_processors.append(
                    MorfeusProcessor(self.config.get('morfeus_params', {}))
                )
                
    def _generate_input(self, mol: Chem.Mol, job_dir: Path) -> None:
        """Generate Gaussian input file"""
        with open(job_dir / 'input.gjf', 'w') as f:
            # Write header
            f.write(f"%mem={self._params['memory']}\n")
            f.write(f"%nproc={self._params['nproc']}\n")
            f.write(f"# {self._params['method']}/{self._params['basis']} opt freq\n\n")
            f.write("Title\n\n")
            
            # Write charge and multiplicity
            f.write(f"{self._params['charge']} {self._params['multiplicity']}\n")
            
            # Write coordinates
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                f.write(f"{atom.GetSymbol():2} {pos.x:10.6f} {pos.y:10.6f} {pos.z:10.6f}\n")
            f.write("\n")
            
    @log_operation("Preparing Gaussian calculations")
    def prepare_calculation(self, smiles: Union[str, List[str]], 
                          workdir: Path, **kwargs) -> Dict[str, Any]:
        """
        Prepare Gaussian calculation jobs
        
        Args:
            smiles: SMILES string(s)
            workdir: Working directory
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with job specifications
        """
        # Convert input to list
        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            smiles_list = smiles
            
        # Prepare jobs
        jobs = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
                
            # Generate conformers if requested
            if self.config.get('conformers', False):
                conformers = self.conformer_generator.generate(mol)
            else:
                mol = Chem.AddHs(mol)
                Chem.EmbedMolecule(mol, randomSeed=42)
                conformers = [mol]
                
            # Generate job for each conformer
            for j, conf in enumerate(conformers):
                job_dir = workdir / f"mol_{i}" / f"conf_{j}"
                job_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate input files
                self._generate_input(conf, job_dir)
                
                # Save job metadata
                job_meta = {
                    'smiles': smi,
                    'mol_id': i,
                    'conf_id': j,
                    'job_dir': str(job_dir),
                    'status': 'pending',
                    'post_process': self.config.get('post_processors', [])
                }
                with open(job_dir / 'job_meta.json', 'w') as f:
                    json.dump(job_meta, f, indent=2)
                    
                jobs.append(job_meta)
                
        return {'jobs': jobs}
        
    def calculate(self, smiles_list: List[str], n_conformers: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Calculate descriptors for a list of SMILES
        
        Args:
            smiles_list: List of SMILES strings
            n_conformers: Number of conformers to generate per molecule
            **kwargs: Additional arguments passed to job manager
            
        Returns:
            DataFrame with calculated descriptors
        """
        results = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self._logger.warning(f"Failed to parse SMILES: {smiles}")
                continue
                
            # Generate conformers if requested
            if n_conformers is not None:
                conformers = self.conformer_generator.generate(mol, n_conformers)
                if not conformers:
                    self._logger.warning(f"Failed to generate conformers for: {smiles}")
                    continue
                    
                # Calculate descriptors for each conformer
                for i, conf in enumerate(conformers):
                    descriptors = self._calculate_descriptors(conf, **kwargs)
                    descriptors.update({
                        'SMILES': smiles,
                        'conformer_id': i
                    })
                    results.append(descriptors)
            else:
                # Calculate descriptors for single 3D conformation
                mol = self.conformer_generator.generate(mol, 1)[0]
                descriptors = self._calculate_descriptors(mol, **kwargs)
                descriptors['SMILES'] = smiles
                results.append(descriptors)
                
        return pd.DataFrame(results)
