from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

from numpy import indices
import pandas as pd
from rdkit import Chem

from ..utils.config import Config, get_config
from ..utils.logger import setup_logger
from ..utils.functool import get_mol_indices

class MoleculeCalculation(NamedTuple):
    mol_index: int
    smiles: str
    conformer_id: int

class BaseCalculator(ABC):
    """Abstract base class for molecular calculations.
    
    This class provides the foundation for all molecular calculations, including
    both geometry optimization and descriptor calculations. It manages a central
    registry of molecules and provides common functionality for file operations
    and molecule management.
    
    Attributes:
        raw_data_name (str): Name of the raw data file or dataset
        species_name (str): Name of the molecular species being processed
        descriptor_type (str): Type of descriptors to calculate
        config (Dict[str, Any]): Configuration dictionary for the calculator
        mol_registry_file (Path): Path to the molecule registry file
        _logger: Logger instance for this calculator
        
    Args:
        raw_data_name: Name of raw data file or dataset
        species_name: Name of molecular species
        descriptor_type: Type of descriptors to calculate
        config: Optional configuration dictionary
    """
    
    def __init__(
        self,
        raw_data_name: str,
        species_name: str,
        descriptor_type: Optional[str] = None,
        config: Optional[Config] = None
    ) -> None:            
        self.config = config or get_config()
        self.raw_data_name = raw_data_name
        self.species_name = species_name
        self.descriptor_type = descriptor_type
        self._logger = setup_logger(__name__)
        
        self._logger.info(
            f"Initializing {self.__class__.__name__} for {species_name} "
            f"using {descriptor_type if descriptor_type else 'default'} descriptors"
        )
        
        # Setup storage paths and create directories
        self.data_root = Path(self.config.get('data_processing.calculator_path', 'data/calculators'))
        self.mol_registry_file = self.data_root / 'mol_list.csv'
        self._setup_storage_paths()

    @abstractmethod
    def _setup_storage_paths(self) -> None:
        """Setup storage paths for calculator.
        
        This method should be implemented by subclasses to setup their specific
        storage paths and directories.
        """
        pass

    def _get_mol_indices(self, smiles_list: List[str]) -> Tuple[List[int], List[str]]:
        """Get or create molecule indices from registry.
        
        Manages a central registry of all molecules and their indices, creating
        new entries for previously unseen molecules.
        
        Args:
            smiles_list: List of SMILES strings to process
            
        Returns:
            Tuple of:
            - List of indices for all molecules
            - List of new SMILES strings that were added to registry
        """
        # Hook for pre-processing
        self._before_get_indices()
        
        # Load registry
        mol_registry = self._load_registry()
        
        # Process each SMILES string
        mol_registry, indices, new_smiles = get_mol_indices(smiles_list, mol_registry)
        
        # Hook for post-processing
        self._after_get_indices(mol_registry, new_smiles)
        
        return indices, new_smiles
        
    def _before_get_indices(self):
        """Hook method called before getting indices.
        
        Subclasses can override this to add pre-processing steps.
        """
        pass
        
    def _load_registry(self) -> pd.DataFrame:
        """Hook method for loading the molecule registry.
        
        Returns:
            DataFrame containing the molecule registry
        """
        return pd.read_csv(self.mol_registry_file)
        
    def _after_get_indices(self, mol_registry: pd.DataFrame, new_smiles: List[str]):
        """Hook method called after getting indices.
        
        Args:
            mol_registry: Updated molecule registry
            new_smiles: List of new SMILES strings added to registry
        """
        pass

class BaseOptimizer(BaseCalculator, ABC):
    """Base class for molecular geometry optimizers.
    
    This class provides the foundation for all molecular geometry optimizers,
    handling:
    - Molecule registry management
    - Calculation results storage and retrieval
    - Directory structure setup for different storage methods
    
    Attributes:
        STORAGE_METHOD (str): Must be defined in subclasses. Options: ['single', 'multi']
            - 'single': Store all results in a single CSV file (e.g., RDKit descriptors)
            - 'multi': Store results in separate directories (e.g., Gaussian calculations)
    
    Args:
        raw_data_name: Name of the raw data file
        species_name: Name of the molecular species
        descriptor_type: Type of descriptors to calculate
        config: Configuration dictionary
    """
    
    STORAGE_METHOD: str  # Must be defined in subclasses: 'single' or 'multi'
    
    def __init__(
        self,
        raw_data_name: str,
        species_name: str,
        descriptor_type: Optional[str] = None,
        config: Optional[Config] = None
    ) -> None:
        """Initialize base optimizer
        
        Args:
            raw_data_name: Name of the raw data file
            species_name: Name of the molecular species
            descriptor_type: Type of descriptors to calculate
            config: Configuration dictionary
        """
        if not hasattr(self, 'STORAGE_METHOD') or self.STORAGE_METHOD not in ['single', 'multi']:
            raise ValueError("Optimizer must define STORAGE_METHOD as either 'single' or 'multi'")
            
        super().__init__(raw_data_name, species_name, descriptor_type, config)
        
    def _setup_storage_paths(self) -> None:
        """Setup directory structure for calculation results.
        
        Creates necessary directories and sets up paths for:
        - Molecule registry file
        - Calculation results (file or directory based on STORAGE_METHOD)
        - Working directory for multi-file storage methods
        """
        # Get optimizer type from class name
        class_name = self.__class__.__name__.lower()
        optimizer_type = class_name[:-9] if class_name.endswith('optimizer') else class_name
        
        # Setup base directories
        self.opt_dir = self.data_root / 'optimizers' / optimizer_type
        self.opt_dir.mkdir(parents=True, exist_ok=True)

        if self.STORAGE_METHOD == 'single':
            if self.descriptor_type is not None:
                self.results_file = Path(self.opt_dir) / f'{self.raw_data_name}_{self.species_name}_{self.descriptor_type}.csv'
            else:
                self.results_file = Path(self.opt_dir) / f'{self.raw_data_name}_{self.species_name}.csv'
            self._logger.info(f"Results will be stored in: {self.results_file}")
        elif self.STORAGE_METHOD == 'multi':
            self.results_file = Path(self.opt_dir) / f'{self.raw_data_name}_{self.species_name}'
            self._logger.info(f"Results will be stored in directory: {self.results_file}")
        
    def _before_get_indices(self):
        if not self.mol_registry_file.exists():
            self._logger.info("Creating new molecule registry")
            pd.DataFrame(columns=['index', 'smiles']).to_csv(self.mol_registry_file, index=False)

    def _after_get_indices(self, mol_registry: pd.DataFrame, new_smiles: List[str]):
        if new_smiles:
            self._logger.info(f"Added {len(new_smiles)} new molecules to registry")
            mol_registry.to_csv(self.mol_registry_file, index=False)
        
    def check_calculations(
        self,
        smiles_list: List[str],
        is_conformer: bool = False,
        overwrite: bool = False
    ) -> Tuple[Dict[int, str], Dict[int, Any]]:
        """Check which molecules need calculation and retrieve existing results.

        Args:
            smiles_list: List of SMILES strings to check
            is_conformer: Whether conformer generation is required
            overwrite: If True, force recalculation of all molecules

        Returns:
            Tuple of:
            - Dictionary mapping molecule indices to SMILES that need calculation
            - Dictionary mapping molecule indices to existing results
        """
        # If overwrite is True, calculate all molecules
        if overwrite:
            mol_indices, _ = self._get_mol_indices(smiles_list)
            return {idx: smiles for idx, smiles in zip(mol_indices, smiles_list)}, {}
            
        # Get molecule indices and initialize results
        mol_indices, new_smiles = self._get_mol_indices(smiles_list)
        to_calculate = {}
        existing_results = {}

        # Get number of required conformers
        n_required_conformers = self.config.get('molecular_descriptors.conformer.num_conformers', 5)
        
        # Helper function to check conformer requirements
        def check_conformer_count(n_conf: int) -> bool:
            return n_conf >= n_required_conformers
            
        # Helper function to process molecule
        def process_molecule(smiles: str, idx: int) -> None:
            should_skip = False

            if smiles not in new_smiles:
                if self.STORAGE_METHOD == 'single':
                    # Check molecule in results file
                    mol_results = results_df[results_df['mol_index'] == idx]
                    should_skip = not mol_results.empty and (not is_conformer or check_conformer_count(len(mol_results)))
                        
                elif self.STORAGE_METHOD == 'multi':
                    if not is_conformer:
                        # Check single optimization
                        mol_dir = self.results_file / str(idx)
                        should_skip = mol_dir.exists() and self._check_optimization_finished(mol_dir)
                    else:
                        # Check conformer calculations
                        conf_dirs = [d for d in self.results_file.glob(f"{idx}_*") if d.is_dir()]
                        finished_dirs = [d for d in conf_dirs if self._check_optimization_finished(d)]
                        should_skip = finished_dirs and check_conformer_count(len(finished_dirs))
            
            if should_skip:
                existing_results[idx] = smiles
            else:
                to_calculate[idx] = smiles
            
        # Process all molecules
        if self.STORAGE_METHOD == 'single':
            if not self.results_file.exists():
                return {idx: smiles for idx, smiles in zip(mol_indices, smiles_list)}, {}
            results_df = pd.read_csv(self.results_file)
        
        for smiles, idx in zip(smiles_list, mol_indices):
            process_molecule(smiles, idx)
            
        return to_calculate, existing_results
    
    @abstractmethod
    def calculate(self, smiles: Union[str, List[str]], **kwargs) -> Any:
        """Calculate molecular descriptors"""
        pass

class BaseDescriptor(BaseCalculator, ABC):
    """Base class for molecular descriptor calculators.
    
    This class extends BaseCalculator to provide functionality specific to
    descriptor calculations. It inherits molecule registry management and
    file operations from parent classes, while adding methods for
    descriptor calculation and result management.
    
    The class is designed to work with optimized molecular geometries to
    calculate various molecular descriptors. It provides a framework for
    implementing specific descriptor calculators (e.g., Morfeus, RDKit).
    
    Attributes:
        Inherits all attributes from BaseCalculator
        DESCRIPTOR_COLUMNS (List[str]): List of descriptor names to calculate
        
    Args:
        raw_data_name: Name of raw data file or dataset
        species_name: Name of molecular species
        optimizer_type: Type of optimizer to use
        descriptor_function: Function to use for descriptor calculation
        config: Optional configuration dictionary
        
    Note:
        Subclasses should:
        1. Define DESCRIPTOR_COLUMNS to specify which descriptors to calculate
        2. Implement calculate_descriptors() method for actual calculations
        3. Handle any descriptor-specific configuration in __init__
    """
    
    def __init__(
        self,
        raw_data_name: str,
        species_name: str,
        optimizer_type: str,
        descriptor_function: Optional[str] = None,
        config: Optional[Config] = None
    ) -> None:
        """Initialize BaseDescriptor.
        
        Args:
            raw_data_name: Name of the raw data file
            species_name: Name of the molecular species
            optimizer_type: Type of optimizer to use (e.g., 'xtb', 'rdkit')
            descriptor_function: Function to use for descriptor calculation
            config: Optional configuration dictionary
            
        Note:
            The optimizer_type and descriptor_function parameters allow for
            flexible configuration of the calculation pipeline.
        """
        self.optimizer_type = optimizer_type
        self.descriptor_function = descriptor_function
        super().__init__(raw_data_name, species_name, config=config)
        
    def _setup_storage_paths(self) -> None:
        class_name = self.__class__.__name__.lower()
        descriptor_type = class_name[:-10] if class_name.endswith('descriptor') else class_name

        self.des_dir = self.data_root / 'descriptors' / descriptor_type
        self.des_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = Path(self.des_dir) / f'{self.raw_data_name}_{self.species_name}_{self.optimizer_type}_{self.descriptor_function}.csv'

        self.opt_dir = self.data_root / 'optimizers' / self.optimizer_type / f'{self.raw_data_name}_{self.species_name}'

    def _before_get_indices(self):
        if not self.mol_registry_file.exists():
            raise FileNotFoundError(
                f"Molecule registry file not found: {self.mol_registry_file}"
            )

    def _after_get_indices(self, mol_registry: pd.DataFrame, new_smiles: List[str]):
        if any(new_smiles):
            raise ValueError(
                f"New molecules found in submit smiles: {new_smiles}"
            )
        
    def check_calculations(
        self,
        smiles_list: List[str],
        n_conformers: int = 1,
        overwrite: bool = False
    ) -> Tuple[List[MoleculeCalculation], List[MoleculeCalculation]]:
        """Check which molecules need calculation.
        
        Determines which molecules from the input list need to have their
        descriptors calculated, taking into account existing results and
        the overwrite flag.
        
        Args:
            smiles_list: List of SMILES strings to check
            n_conformers: Number of conformers for each molecule
            overwrite: Whether to overwrite existing results
            
        Returns:
            Tuple of:
            - List of MoleculeCalculation objects for molecules to calculate
            - List of MoleculeCalculation objects for existing results
        """

        
        if overwrite or not self.results_file.exists():
            mol_indices, _ = self._get_mol_indices(smiles_list)
            return [MoleculeCalculation(mol_indices[i], smiles_list[i], j) 
                    for i in range(len(mol_indices)) for j in range(n_conformers)], []

        results_df = pd.read_csv(self.results_file)    
        mol_indices, _ = self._get_mol_indices(smiles_list)
        to_calculate = []
        existing_results = []

        if n_conformers <= 1:
            exist_idx = results_df['index'][results_df['index'].isin(mol_indices)].values
            calculate_idx = [idx for idx, val in enumerate(mol_indices) if val not in exist_idx]
            exist_idx = [idx for idx, val in enumerate(mol_indices) if val in exist_idx]
            to_calculate = [MoleculeCalculation(mol_indices[i], smiles_list[i], 0) for i in calculate_idx]
            existing_results = [MoleculeCalculation(mol_indices[i], smiles_list[i], 0) for i in exist_idx]
        else:
            # Handle conformer case
            raise NotImplementedError(f'{self.__class__.__name__} does not support conformer')
            # TODO: implement conformer support


        return to_calculate, existing_results

class BaseJobManager(ABC):
    """Base class for managing calculation jobs"""
    
    @abstractmethod
    def create_job_script(self, job_spec: Dict[str, Any], template: str) -> str:
        """
        Create job submission script
        
        Args:
            job_spec: Job specifications
            template: Template name for job script
            
        Returns:
            Generated job script content
        """
        pass
        
    @abstractmethod
    def submit_jobs(self, job_specs: List[Dict[str, Any]], 
                   batch_size: int = 100) -> List[str]:
        """
        Submit jobs to computation system
        
        Args:
            job_specs: List of job specifications
            batch_size: Number of calculations per job
            
        Returns:
            List of job IDs
        """
        pass
        
    @abstractmethod
    def check_status(self, job_ids: List[str]) -> Dict[str, str]:
        """
        Check status of submitted jobs
        
        Args:
            job_ids: List of job IDs to check
            
        Returns:
            Dictionary mapping job IDs to their status
        """
        pass

class BasePostProcessor(ABC):
    """Base class for post-processing calculation results"""
    
    @abstractmethod
    def process(self, calc_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Process calculation results
        
        Args:
            calc_dir: Directory containing calculation results
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary containing processed results
        """
        pass
