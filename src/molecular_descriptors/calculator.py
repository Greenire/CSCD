from typing import Dict, List, Optional, Union, Any
import importlib
import inspect

from .base import BaseDescriptorCalculator
from .optimizers.rdkit_optimizer import RDKitOptimizer
from .optimizers.gaussian_optimizer import GaussianOptimizer
from .optimizers.xtb_optimizer import XTBOptimizer
from .descriptors.multiwfn_descriptor import MultiwfnDescriptor
from .descriptors.morfeus_descriptor import MorfeusDescriptor
from ..utils.logger import setup_logger

class DescriptorCalculator:
    """
    Unified interface for molecular descriptor calculations.
    
    This class serves as a factory and dispatcher for molecular calculations,
    supporting both single-step (combined) and two-step (separate optimization
    and descriptor) calculations.
    
    Calculation Modes:
    1. Combined (e.g., RDKit): Single calculator handles both optimization and descriptors
    2. Two-step:
       - Structure optimization (e.g., XTB, Gaussian)
       - Descriptor calculation (e.g., Morfeus, Multiwfn)
    """
    
    BUILTIN_CALCULATORS = {
        # Combined calculators
        'rdkit': {
            'type': 'combined',
            'optimizer': RDKitOptimizer,
            'descriptor': None
        },
        
        # Structure optimizers
        'gaussian': {
            'type': 'optimizer',
            'optimizer': GaussianOptimizer,
            'descriptor': None
        },
        'xtb': {
            'type': 'optimizer',
            'optimizer': XTBOptimizer,
            'descriptor': None
        },
        
        # Descriptor calculators
        'multiwfn': {
            'type': 'descriptor',
            'optimizer': None,
            'descriptor': MultiwfnDescriptor
        },
        'morfeus': {
            'type': 'descriptor',
            'optimizer': None,
            'descriptor': MorfeusDescriptor
        }
    }
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 optimizer_type: Optional[str] = None,
                 descriptor_type: Optional[str] = None):
        """
        Initialize the calculator dispatcher.
        
        Args:
            config: Configuration dictionary
            optimizer_type: Type of optimizer/calculator to use
            descriptor_type: Type of descriptor to calculate
        
        Raises:
            ValueError: If invalid calculator configuration is provided
        """
        self.config = config or {}
        self._logger = setup_logger(__name__)
        
        # Determine calculator types
        self.optimizer_type = optimizer_type or self._get_default_optimizer()
        self.descriptor_type = descriptor_type or self._get_default_descriptor()
        
        # Initialize calculator components
        self._optimizer = None
        self._descriptor = None
        self._setup_calculator()
        
    def _get_default_optimizer(self) -> str:
        """Get default optimizer type from config or fallback to rdkit"""
        calculators = self.config.get('molecular_descriptors.species_calculators', [])
        return calculators[0]['optimizer'] if calculators else 'rdkit'
    
    def _get_default_descriptor(self) -> str:
        """Get default descriptor type"""
        if self.optimizer_type == 'rdkit':
            return 'morgan'  # Default RDKit descriptor
        raise ValueError("Descriptor type must be specified for non-RDKit calculators")
    
    def _setup_calculator(self):
        """
        Initialize calculator components based on specified types.
        
        Raises:
            ValueError: If invalid calculator configuration is provided
        """
        # Get optimizer info
        opt_info = self.BUILTIN_CALCULATORS.get(self.optimizer_type)
        if opt_info is None:
            self._optimizer = self._load_custom_calculator(self.optimizer_type)
            return
            
        # Handle combined calculator case
        if opt_info['type'] == 'combined':
            self._optimizer = opt_info['optimizer'](self.config)
            self._optimizer.descriptor_type = self.descriptor_type
            return
            
        # Handle two-step calculation case
        self._optimizer = opt_info['optimizer'](self.config)
        
        # Setup descriptor calculator if needed
        if opt_info['type'] == 'optimizer':
            desc_info = self.BUILTIN_CALCULATORS.get(self.descriptor_type)
            if desc_info is None:
                self._descriptor = self._load_custom_calculator(self.descriptor_type)
            elif desc_info['type'] == 'descriptor':
                self._descriptor = desc_info['descriptor'](self.config)
            else:
                raise ValueError(f"Invalid descriptor type: {self.descriptor_type}")

    def calculate(self, 
                 smiles: Union[str, List[str]], 
                 raw_data_name: str,
                 species_name: str, 
                 is_conformer: bool = False, 
                 **kwargs) -> Any:
        """
        Dispatch calculation to appropriate calculator(s).
        
        Args:
            smiles: SMILES string or list of SMILES
            raw_data_name: Name of raw data file
            species_name: Name of species
            is_conformer: Whether conformers are involved
            **kwargs: Additional arguments for specific calculators
        
        Returns:
            Calculation results from the calculator
        
        Raises:
            ValueError: If calculator configuration is invalid
        """
        smiles_list = smiles if isinstance(smiles, list) else [smiles]
        calc_info = self.BUILTIN_CALCULATORS.get(self.optimizer_type)
        
        # Handle custom calculator case
        if calc_info is None:
            return self._optimizer.calculate(smiles_list, raw_data_name, 
                                          species_name, is_conformer,  self.config, **kwargs)
        
        # Handle different calculator types
        if calc_info['type'] == 'combined':
            return self._optimizer.calculate(smiles_list, raw_data_name, species_name, 
                    is_conformer,  self.descriptor_type, self.config, **kwargs)
            
        elif calc_info['type'] == 'optimizer':
            if self._descriptor is None:
                raise ValueError("No descriptor calculator configured")
            # Run two-step calculation
            self._optimizer.calculate(smiles_list, raw_data_name, 
                                   species_name, is_conformer, self.config, **kwargs)
            return self._descriptor.calculate(smiles_list, raw_data_name,
                                           species_name, is_conformer, self.config, **kwargs)
            
        elif calc_info['type'] == 'descriptor':
            raise ValueError(f"{self.optimizer_type} is a descriptor type, not an optimizer")
            
        else:
            raise ValueError(f"Invalid calculator type: {calc_info['type']}")
            
    def _load_custom_calculator(self, calculator_type: str) -> BaseDescriptorCalculator:
        """
        Load a custom calculator class.
        
        Args:
            calculator_type: Name of the calculator type
            
        Returns:
            Instance of custom calculator
            
        Raises:
            ValueError: If custom calculator cannot be loaded
        """
        try:
            module = importlib.import_module(
                f".optimizers.{calculator_type}_optimizer",
                package="molecular_descriptors"
            )
            
            # Find calculator class that inherits from BaseDescriptorCalculator
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseDescriptorCalculator) and 
                    obj != BaseDescriptorCalculator):
                    return obj(self.config)
                    
            raise ValueError(f"No valid calculator class found in {calculator_type}_optimizer.py")
                
        except ImportError as e:
            raise ValueError(f"Calculator type '{calculator_type}' not found: {e}")
