from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

from ...utils.config import Config
from ...utils.logger import log_info
from ..base import BaseOptimizer

class RDKitOptimizer(BaseOptimizer):
    """RDKit-based molecular descriptor calculator.
    
    Calculates molecular descriptors using RDKit's capabilities. Supports multiple
    descriptor types and ensures efficient calculation with result caching.
    
    Attributes:
        STORAGE_METHOD (str): Always 'single' as RDKit calculations are stored in CSV
        SUPPORTED_TYPES (dict): Mapping of supported descriptor types to their implementations
    """
    
    STORAGE_METHOD = 'single'
    
    SUPPORTED_TYPES = {
        '2d': [name[0] for name in Descriptors._descList],
        'morgan': lambda m, r, n: AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n),
        'maccs': MACCSkeys.GenMACCSKeys,
        'topological': Chem.RDKFingerprint
    }
    
    def __init__(
        self,
        raw_data_name: str,
        species_name: str,
        descriptor_type: str = 'morgan',
        config: Optional[Config] = None
    ) -> None:
        """Initialize RDKit calculator.
        
        Args:
            raw_data_name: Name of the raw data file
            species_name: Name of the molecular species
            descriptor_type: Type of descriptors to calculate, one of:
                - '2d': 2D molecular descriptors
                - 'morgan': Morgan fingerprints
                - 'maccs': MACCS keys
                - 'topological': Topological fingerprints
            config: Configuration dictionary
            
        Raises:
            ValueError: If descriptor_type is not supported
        """
        if descriptor_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported descriptor type: {descriptor_type}. "
                f"Must be one of: {list(self.SUPPORTED_TYPES.keys())}"
            )
            
        super().__init__(raw_data_name, species_name, descriptor_type, config)
        self._setup_descriptor_calculator()
        
    def _setup_descriptor_calculator(self) -> None:
        """Setup the appropriate descriptor calculation function."""
        if self.descriptor_type == '2d':
            self._setup_2d_descriptors()
        elif self.descriptor_type == 'morgan':
            self._setup_morgan_fingerprint()
        else:
            # For MACCS and topological, use directly from SUPPORTED_TYPES
            self._descriptor_calculator = self.SUPPORTED_TYPES[self.descriptor_type]
            
    def _setup_2d_descriptors(self) -> None:
        """Setup calculators for 2D descriptors."""
        self._descriptor_names = self.SUPPORTED_TYPES['2d']
        self._calculators = {}
        
        for name in self._descriptor_names:
            if hasattr(Descriptors, name):
                self._calculators[name] = getattr(Descriptors, name)
                
    def _setup_morgan_fingerprint(self) -> None:
        """Setup Morgan fingerprint calculator with configured parameters."""
        radius = self.config.get('molecular_descriptors.rdkit.morgan.radius', 3)
        nbits = self.config.get('molecular_descriptors.rdkit.morgan.nbits', 1024)
        
        self._descriptor_calculator = lambda mol: self.SUPPORTED_TYPES['morgan'](
            mol, radius, nbits
        )
        
    def calculate(
        self,
        smiles: Union[str, List[str]],
        is_conformer: bool = False,
        save: bool = True,
        overwrite: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, None]:
        """Calculate RDKit descriptors for molecules.
        
        Args:
            smiles: SMILES string or list of SMILES strings
            is_conformer: Whether conformer generation is required
            save: Whether to save results to file
            overwrite: If True, force recalculation of all molecules
            **kwargs: Additional calculation parameters
            
        Returns:
            DataFrame with calculated descriptors or None if all molecules have
            existing results (and overwrite is False)
        """
        smiles_list = [smiles] if isinstance(smiles, str) else smiles
        self._logger.info(
            f"Calculating {self.descriptor_type} descriptors for {len(set(smiles_list))} molecules"
            f"{' (overwriting existing results)' if overwrite else ''}"
        )
        
        # Check which molecules need calculation
        to_calculate, existing_results = self.check_calculations(
            smiles_list, is_conformer, overwrite
        )
        
        if not to_calculate and not overwrite:
            self._logger.info("All molecules have existing results")
            return None
            
        # Calculate descriptors for new molecules
        raw_results = []
        mol_indices = []
        failed_smiles = []
        
        for idx, smiles in to_calculate.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self._logger.warning(f"Failed to parse SMILES: {smiles}")
                failed_smiles.append(smiles)
                continue
                
            try:
                if self.descriptor_type == '2d':
                    descriptors = self._calculate_2d_descriptors(mol)
                else:
                    descriptors = np.array(self._descriptor_calculator(mol))
                raw_results.append(descriptors)
                mol_indices.append(idx)
                
            except Exception as e:
                self._logger.error(
                    f"Failed to calculate {self.descriptor_type} descriptors "
                    f"for {smiles}: {str(e)}"
                )
                failed_smiles.append(smiles)
                continue
                
        # Process results into DataFrame
        if not raw_results:
            self._logger.warning("No successful calculations")
            return pd.DataFrame()
            
        self._logger.info(
            f"Successfully calculated descriptors for {len(raw_results)} molecules "
            f"({len(failed_smiles)} failed)"
        )
            
        new_df = self._process_results(raw_results, mol_indices)
        log_info(self._logger, "Generated descriptor DataFrame", new_df)
        
        if save:
            self.save_results(new_df, mol_indices)
            
        return new_df
        
    def _process_results(
        self,
        raw_results: List[Union[List[float], np.ndarray]],
        mol_indices: List[int]
    ) -> pd.DataFrame:
        """Process raw calculation results into a DataFrame.
        
        Args:
            raw_results: List of raw descriptor values
            mol_indices: List of molecule indices corresponding to results
            
        Returns:
            DataFrame containing processed results with appropriate column names
        """
        if self.descriptor_type == '2d':
            df_cols = self._descriptor_names
        else:
            # For fingerprints, create numbered columns
            n_bits = len(raw_results[0])
            df_cols = [f'{self.descriptor_type}_{i}' for i in range(n_bits)]
            
        return pd.DataFrame(raw_results, columns=df_cols, index=mol_indices)
        
    def save_results(self, results: pd.DataFrame, mol_indices: List[int]) -> None:
        """Save calculation results to file.
        
        Args:
            results: DataFrame containing calculation results
            mol_indices: List of molecule indices corresponding to results
        """
        self._logger.info(f"Saving results for {len(mol_indices)} molecules")
        
        results['mol_index'] = mol_indices
        results = results[['mol_index'] + results.columns[:-1].tolist()]

        if not self.results_file.exists():
            results.to_csv(self.results_file, index=False)
            self._logger.info(f"Created new results file: {self.results_file}")
            return
            
        # Update existing results
        self._logger.info("Updating existing results")
        existing_df = pd.read_csv(self.results_file)
        existing_df = existing_df[~existing_df.index.isin(mol_indices)]
        results = pd.concat([existing_df, results], ignore_index=False)
        results.to_csv(self.results_file, index=False)
        self._logger.info("Results saved successfully")
        
    def _check_optimization_finished(self, calc_dir: Path) -> bool:
        """Check if optimization calculation is finished.
        
        For RDKit calculations, this is always True as they are synchronous.
        
        Args:
            calc_dir: Directory containing calculation files
            
        Returns:
            Always True for RDKit calculations
        """
        return True
        
    def _calculate_2d_descriptors(self, mol: Chem.Mol) -> List[float]:
        """Calculate all 2D descriptors for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of descriptor values in the order of self._descriptor_names
        """
        descriptors = []
        failed_descriptors = []
        
        for name in self._descriptor_names:
            try:
                calculator = self._calculators[name]
                value = calculator(mol)
                descriptors.append(value)
            except Exception as e:
                self._logger.warning(f"Failed to calculate {name}: {str(e)}")
                descriptors.append(None)
                failed_descriptors.append(name)
                
        if failed_descriptors:
            self._logger.warning(
                f"Failed to calculate {len(failed_descriptors)} descriptors: "
                f"{', '.join(failed_descriptors)}"
            )
                
        return descriptors