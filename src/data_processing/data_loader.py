from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import functools
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.logger import setup_logger
from ..utils.config import Config

def require_data_loaded(func: Callable) -> Callable:
    """Decorator to ensure data is loaded before method execution.
    
    Args:
        func: Method to decorate
        
    Returns:
        Decorated method that checks if data is loaded
        
    Raises:
        RuntimeError: If data is not loaded when method is called
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._raw_data is None:
            raise RuntimeError(
                f"Data must be loaded before calling {func.__name__}. "
                "Please call load_data() first."
            )
        return func(self, *args, **kwargs)
    return wrapper

class DataLoader:
    """
    Data loader for reaction data.
    
    Handles loading and categorizing data columns into four groups:
    1. Metadata: Optional auxiliary data (e.g., ID, type).
    2. Species: Chemical species SMILES.
    3. Conditions: Reaction conditions (e.g., temperature, time).
    4. Labels: Target variables to predict.
    
    Column groups must be arranged in this order in the raw data file.
    All groups except Labels are optional.
    """

    # Column groups in order of appearance
    COLUMN_GROUPS = ['metadata', 'species', 'conditions', 'labels']

    def __init__(self, config: Config, path: Optional[str] = None) -> None:
        """
        Initialize data loader.
        
        Args:
            config: Configuration object with:
                data_processing:
                    raw_data_path: Path to raw data CSV.
                data_structure:
                    column_groups: Dictionary mapping group names to [start, end] indices.
                molecular_descriptors:
                    species_calculators: List of calculator configurations.
            path: Optional path to the data file.
        """
        self.config = config
        self._logger = setup_logger(__name__)
        
        # Get data path from config
        self.data_path = path or Path(config.get('data_processing.raw_data_path', 'data/raw/reactions.csv'))
        self._raw_data = None
        self.column_indices = {}
        
    def load_data(self) -> None:
        """Load raw data from CSV file."""
        try:
            self._raw_data = pd.read_csv(self.data_path)
            if self._raw_data.empty:
                self._logger.warning(f"Empty data file: {self.data_path}")
                raise ValueError("Empty data file")
        except FileNotFoundError as e:
            self._logger.error(f"Data file not found: {self.data_path}")
            raise
        except Exception as e:
            self._logger.error(f"Failed to load data from {self.data_path}: {e}")
            raise ValueError(f"Failed to load data: {e}")
            
        self.column_indices = self._get_column_indices()
        
    @property
    def data(self) -> Union[pd.DataFrame, None]:
        """Get raw data."""
        return self._raw_data
    
    @require_data_loaded
    def get_metadata(self) -> Optional[pd.DataFrame]:
        """Get metadata group data."""
        return self._get_data_by_group('metadata')
        
    @require_data_loaded
    def get_species_data(self) -> Tuple[List[str], List[List[str]], List[Dict[str, Union[str, bool]]]]:
        """
        Get species data, including column names, SMILES strings, and calculators.
        
        Returns:
            A tuple containing:
                - A list of species column names.
                - A list of lists of SMILES strings for each species, where each list
                  contains the SMILES strings for a particular species.
                - A list of dictionaries, where each dictionary contains the
                  configuration for a particular species calculator. The dictionary
                  should contain the following keys:
                    - `optimizer`: The name of the optimizer to use for this species.
                    - `descriptor`: The type of descriptor to calculate for this species.
                    - `is_conformer`: A boolean indicating whether conformer descriptors
                      should be calculated for this species.
        """
        species_range = self.column_indices.get('species')
        if not species_range:
            return [], [], []
        
        start, end = species_range
        species_cols = self._raw_data.columns[start:end].tolist()
        species_smiles = [self._raw_data[col].fillna('').tolist() for col in species_cols]
        
        calculators = self.config.get('molecular_descriptors.species_calculators', [])
        default_calc = {'optimizer': 'rdkit', 'descriptor': 'morgan', 'is_conformer': False}
        calculators = (calculators + [default_calc] * len(species_cols))[:len(species_cols)]
        
        return species_cols, species_smiles, calculators
        
    @require_data_loaded
    def get_condition_data(self) -> Optional[pd.DataFrame]:
        """Get reaction condition data."""
        return self._get_data_by_group('conditions')
        
    @require_data_loaded
    def get_label_data(self) -> pd.DataFrame:
        """Get label data."""
        label_data = self._get_data_by_group('labels')
        if label_data is None:
            raise ValueError("No label columns found in data")
        return label_data
    
    def _get_data_by_group(self, group: str) -> Optional[pd.DataFrame]:
        """
        Get data for a specific group.
        
        Args:
            group: The name of the column group to retrieve data for.
        
        Returns:
            DataFrame containing the data for the specified group or None if not found.
        """
        indices = self.column_indices.get(group)
        return self._raw_data.iloc[:, indices[0]:indices[1]] if indices else None
        
    def _get_column_indices(self) -> Dict[str, Tuple[int, int]]:
        """
        Get column indices for each group from the configuration.
        
        Returns:
            Dictionary mapping group names to their start and end indices.
        
        Raises:
            ValueError: If the configuration is invalid or no label columns are specified.
        """
        if self._raw_data is None or self._raw_data.empty:
            return {}
            
        column_groups = self.config.get('data_structure.column_groups', {})
        if not column_groups:
            self._logger.warning("No column group indices specified in config")
            return {}
            
        indices = {}
        last_end = 0
        
        for group in self.COLUMN_GROUPS:
            start, end = column_groups.get(group, (None, None))
            if None in (start, end) or start < last_end or start < 0 or end > len(self._raw_data.columns):
                self._logger.error(f"Invalid indices for group {group}: {[start, end]}")
                continue
                
            indices[group] = (start, end)
            last_end = end
                
        if 'labels' not in indices:
            self._logger.error("No label columns specified in config")
            raise ValueError("No label columns specified in config")
            
        return indices

    def drop_duplicate_rows(self, dup_cols: List[str]) -> None:
        """
        Drop duplicate rows in the raw data.
        
        Args:
            dup_cols: List of column names to check for duplicates.
        """
        self._raw_data = self._raw_data.groupby(dup_cols, as_index=False)['ddg'].max()