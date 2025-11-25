from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from morfeus import (BuriedVolume, Dispersion, Pyramidalization, SASA,
                    Sterimol, read_xyz)
from rdkit import Chem

from ..base import BaseDescriptor
from ...utils.logger import setup_logger

class MorfeusDescriptor(BaseDescriptor):
    """Morfeus-based molecular descriptor calculator.
    
    Calculates molecular descriptors using Morfeus package.
    Supports various geometric and electronic descriptors.
    Results are stored in a structured format.
    """
    
    INTER_DESCRIPTOR_COLUMNS = [
        'area', 'volume',  # Surface area and volume
        'P_bv', 'S_cat_bv', 'S_S_bv',  # Buried volumes
        'S_cat_darea', 'S_cat_pint', 'S_S_darea', 'S_S_pint',  # Dispersion
        'S_cat_P', 'S_S_P', 'S_cat_Pangle', 'S_S_Pangle',  # Pyramidalization
        'S_cat_sarea', 'S_cat_svol', 'S_S_sarea', 'S_S_svol',  # SASA
        'S_toS_Lvalue', 'S_toS_B1value', 'S_toS_B5value',  # Sterimol S-S
        'S_toC_Lvalue', 'S_toC_B1value', 'S_toC_B5value',  # Sterimol S-C
        'P_charge', 'S_cat_charge', 'S_S_charge'  # Atomic charges
    ]

    PRODUCT_S_DESCRIPTOR_COLUMNS = [
        'area', 'volume',  # Surface area and volume
        'S_bv',  # Buried volumes
        'S_darea', 'S_pint',  # Dispersion
        'S_P', 'S_Pangle',  # Pyramidalization
        'S_sarea', 'S_svol',  # SASA
        'S_toC1_Lvalue', 'S_toC1_B1value', 'S_toC1_B5value',  # Sterimol S-C1
        'S_toC2_Lvalue', 'S_toC2_B1value', 'S_toC2_B5value',  # Sterimol S-C2
        'S_charge'  # Atomic charges
    ]

    def __init__(
        self, 
        raw_data_name: str, 
        species_name: str, 
        optimizer_type: str,
        descriptor_function: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Morfeus descriptor calculator."""
        super().__init__(raw_data_name, species_name, optimizer_type, 
            descriptor_function=descriptor_function, config=config)

    def calculate(
        self,
        smiles: Union[str, List[str]],
        n_conformers: int = 1,
        overwrite: bool = False,
        save: bool = True,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Calculate descriptors for molecules."""
        # Convert single SMILES to list
        smiles_list = [smiles] if isinstance(smiles, str) else list(set(smiles))
        
        # Check which molecules need calculation
        to_calculate, existing_results = self.check_calculations(
            smiles_list, n_conformers, overwrite
        )

        if not to_calculate:
            self._logger.info("No new molecules to calculate")
            return pd.DataFrame()
            
        self._logger.info(f"Calculating descriptors for {len(to_calculate)} molecules")
        
        # Calculate descriptors
        results = []
        for idx, smiles, conf_id in to_calculate:
            try:
                self._logger.debug(f"Processing molecule {idx}: {smiles}")
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    self._logger.error(f"Invalid SMILES string: {smiles}")
                    continue
                
                # Get XYZ directory
                xyz_dir = Path(self.opt_dir) / f"{idx}_{conf_id}"
                if not xyz_dir.exists():
                    self._logger.error(f"XYZ directory not found: {xyz_dir}")
                    continue
                    
                # Calculate descriptors based on function type
                if self.descriptor_function in ['intermediate', 'product_s']:
                    self._logger.debug(f"Calculating {self.descriptor_function} descriptors for molecule {idx}")
                    descriptors = self.calculate_from_xyz(xyz_dir, self.descriptor_function)
                else:
                    self._logger.error(f"Unknown descriptor function: {self.descriptor_function}")
                    continue
                    
                descriptors['smiles'] = smiles
                descriptors['index'] = idx
                descriptors['conf_id'] = conf_id
                results.append(descriptors)
                self._logger.debug(f"Successfully processed molecule {idx} with conf_id {conf_id}")
                
            except Exception as e:
                self._logger.error(f"Error calculating descriptors for molecule {idx} with conf_id {conf_id}: {str(e)}")
                continue
                
        # Convert results to DataFrame
        if results:
            results_df = pd.DataFrame(results)
            if self.descriptor_function == 'intermediate':
                results_df = results_df[['index', 'conf_id', 'smiles'] + self.INTER_DESCRIPTOR_COLUMNS]
            elif self.descriptor_function == 'product_s':
                results_df = results_df[['index', 'conf_id', 'smiles'] + self.PRODUCT_S_DESCRIPTOR_COLUMNS]
            
            # Save results
            if save:
                self._logger.info("Saving results to file")
                self.save_results(results_df, overwrite)
                self._logger.info(f"Successfully processed and saved results for {len(results)} molecules")
            
            return results_df
        
        self._logger.warning("No results were generated")
        return pd.DataFrame()

    def save_results(self, results_df: pd.DataFrame, overwrite: bool = False):
        """Save calculation results to CSV file.
        
        Args:
            results_df: DataFrame containing results
            overwrite: Whether to overwrite existing results
        """
        if self.results_file.exists() and not overwrite:
            des_df = pd.read_csv(self.results_file)
            des_df = pd.concat([des_df, results_df], ignore_index=True)
            des_df.to_csv(self.results_file, index=False)
        else:
            results_df.to_csv(self.results_file, index=False)
        self._logger.info(f"Saved results to {self.results_file}")

    @staticmethod
    def calculate_from_xyz(
        xyz_dir: Union[str, Path],
        descriptor_type: str
    ) -> Dict[str, float]:
        """Calculate molecular descriptors directly from XYZ file.

        Args:
            xyz_dir: Directory containing the XYZ file
            descriptor_type: Type of descriptors to calculate
                           'intermediate' or 'product_s'

        Returns:
            Dictionary containing calculated descriptors
        """
        return calculate_descriptors(xyz_dir, descriptor_type)

def get_intermediate_atom_indices(elements: np.ndarray, coordinates: np.ndarray) -> Dict[str, int]:
    """Get indices of important atoms for intermediate molecules.

    Args:
        elements: Array of element symbols
        coordinates: Array of atomic coordinates

    Returns:
        Dictionary mapping atom types to their indices
    """
    indices = {
        'P': np.where(elements == 'P')[0][0],
        'S': np.where(np.isin(elements, ['S', 'Se']))[0] 
    }
    
    for atom_idx in indices['S']:
        if np.linalg.norm(coordinates[atom_idx] - coordinates[indices['P']]) < 3:
            s_cat_index = atom_idx
            continue
        if np.linalg.norm(coordinates[atom_idx] - coordinates[indices['P']]) < 4:
            s_s_index = atom_idx

    return {
        'P': indices['P'] + 1,
        'S_cat': s_cat_index + 1,
        'S_S': s_s_index + 1,
        'C_O': s_s_index + 3,
        'C_Ar': s_s_index + 4,
        'C': s_s_index + 2  # Carbon is typically next to sulfur
    }


def get_product_s_indices(elements: np.ndarray, coordinates: np.ndarray) -> Dict[str, int]:
    """Get indices of important atoms in product molecules.

    Args:
        elements: Array of element symbols
        coordinates: Array of atomic coordinates

    Returns:
        Dictionary mapping atom types to their indices
    """
    indices = {
        'S': np.where(np.isin(elements, ['S', 'Se']))[0][1]
    }
    s_index = indices['S'] + 1
    if elements[0] == 'O':
        return {
            'S': s_index,
            'C_p': s_index + 1,
            'C_Ar': 26,
            'C_O': s_index - 2,
            'C1': s_index + 1,
            'C2': s_index - 1
        }
    else:
        return {
            'S': s_index,
            'C_p': s_index + 1,
            'C_Ar': s_index - 4,
            'C_O': s_index - 3,
            'C1': s_index + 1,
            'C2': s_index - 1
        }
    # if 'O' in elements:
    #     indices = {
    #         'S': np.where(np.isin(elements, ['S', 'Se']))[0],
    #         'O': np.where(elements == 'O')[0]
    #     }

    #     s_index = indices['S'] + 1
    #     min_S_O_dist = float('inf')
    #     for s_idx in indices['S']:
    #         for o_idx in indices['O']:
    #             dist = np.linalg.norm(coordinates[s_idx] - coordinates[o_idx])
    #             if dist < min_S_O_dist:
    #                 min_S_O_dist = dist
    #                 s_index = s_idx + 1

    # else:
    #     s_index = np.where(np.isin(elements, ['S', 'Se']))[0] + 1

    # if np.linalg.norm(coordinates[s_index - 1] - coordinates[s_index - 2]) < \
    #     np.linalg.norm(coordinates[s_index - 1] - coordinates[s_index]):
    #     return {
    #         'S': s_index,
    #         'C1': s_index - 1,
    #         'C2': s_index + 1
    #     } 
    # else:   
    #     return {
    #         'S': s_index,
    #         'C1': s_index + 1,
    #         'C2': s_index - 1
    #     }


def calculate_intermediate_descriptors(xyz_dir: Path) -> Dict[str, float]:
    """Calculate Morfeus descriptors for intermediate molecules.

    Args:
        xyz_dir: Directory containing the XYZ and charges files

    Returns:
        Dictionary of calculated descriptors
    """
    elements, coordinates = read_xyz(str(xyz_dir / 'xtbopt.xyz'))
    indices = get_intermediate_atom_indices(elements, coordinates)
    
    # Initialize calculators
    disp = Dispersion(elements, coordinates)
    sasa = SASA(elements, coordinates)
    sterimol_to_s = Sterimol(elements, coordinates, indices['S_S'], indices['S_cat'])
    sterimol_to_c = Sterimol(elements, coordinates, indices['C_O'], indices['C_Ar'])
    
    descriptors = {
        # Surface area and volume
        'area': disp.area,
        'volume': disp.volume,
        
        # Buried volumes
        'P_bv': BuriedVolume(elements, coordinates, indices['P']).fraction_buried_volume,
        'S_cat_bv': BuriedVolume(elements, coordinates, indices['S_cat']).fraction_buried_volume,
        'S_S_bv': BuriedVolume(elements, coordinates, indices['C_O']).fraction_buried_volume,
        
        # Dispersion interactions
        'S_cat_darea': disp.atom_areas[indices['S_cat']],
        'S_cat_pint': disp.atom_p_int[indices['S_cat']],
        'S_S_darea': disp.atom_areas[indices['C_O']],
        'S_S_pint': disp.atom_p_int[indices['C_O']],
        
        # Pyramidalization
        'S_cat_P': Pyramidalization(coordinates, indices['S_cat']).P,
        'S_S_P': Pyramidalization(coordinates, indices['C_O']).P,
        'S_cat_Pangle': Pyramidalization(coordinates, indices['S_cat']).P_angle,
        'S_S_Pangle': Pyramidalization(coordinates, indices['C_O']).P_angle,
        
        # SASA for specific atoms
        'S_cat_sarea': sasa.atom_areas[indices['S_cat']],
        'S_cat_svol': sasa.atom_volumes[indices['S_cat']],
        'S_S_sarea': sasa.atom_areas[indices['C_O']],
        'S_S_svol': sasa.atom_volumes[indices['C_O']],
        
        # Sterimol parameters
        'S_toS_Lvalue': sterimol_to_s.L_value,
        'S_toS_B1value': sterimol_to_s.B_1_value,
        'S_toS_B5value': sterimol_to_s.B_5_value,
        'S_toC_Lvalue': sterimol_to_c.L_value,
        'S_toC_B1value': sterimol_to_c.B_1_value,
        'S_toC_B5value': sterimol_to_c.B_5_value,
        
        # Initialize charges
        'P_charge': 0.0,
        'S_cat_charge': 0.0,
        'S_S_charge': 0.0
    }

    # Read charges if available
    charges_file = xyz_dir / 'charges'
    if charges_file.exists():
        with open(charges_file, 'r') as file:
            charge_lines = [float(line.strip()) for line in file]
            descriptors['P_charge'] = charge_lines[indices['P'] - 1]
            descriptors['S_cat_charge'] = charge_lines[indices['S_cat'] - 1]
            descriptors['S_S_charge'] = charge_lines[indices['S_S'] - 1]

    return descriptors


def calculate_product_s_descriptors(xyz_dir: Path) -> Dict[str, float]:
    """Calculate Morfeus descriptors for product molecules.

    Args:
        xyz_dir: Directory containing the XYZ and charges files

    Returns:
        Dictionary of calculated descriptors
    """
    elements, coordinates = read_xyz(str(xyz_dir / 'xtbopt.xyz'))
    indices = get_product_s_indices(elements, coordinates)
    
    # Initialize calculators
    disp = Dispersion(elements, coordinates)
    sasa = SASA(elements, coordinates)
    sterimol_to_c1 = Sterimol(elements, coordinates, indices['S'], indices['C_p'])
    sterimol_to_c2 = Sterimol(elements, coordinates, indices['C_O'], indices['C_Ar'])
    
    descriptors = {
        # Surface area and volume
        'area': disp.area,
        'volume': disp.volume,
        
        # Buried volumes
        'S_bv': BuriedVolume(elements, coordinates, indices['C_O']).fraction_buried_volume,
        
        # Dispersion interactions
        'S_darea': disp.atom_areas[indices['C_O']],
        'S_pint': disp.atom_p_int[indices['C_O']],
        
        # Pyramidalization
        'S_P': Pyramidalization(coordinates, indices['C_O']).P,
        'S_Pangle': Pyramidalization(coordinates, indices['C_O']).P_angle,
        
        # SASA for specific atoms
        'S_sarea': sasa.atom_areas[indices['C_O']],
        'S_svol': sasa.atom_volumes[indices['C_O']],
        
        # Sterimol parameters
        'S_toC1_Lvalue': sterimol_to_c1.L_value,
        'S_toC1_B1value': sterimol_to_c1.B_1_value,
        'S_toC1_B5value': sterimol_to_c1.B_5_value,
        'S_toC2_Lvalue': sterimol_to_c2.L_value,
        'S_toC2_B1value': sterimol_to_c2.B_1_value,
        'S_toC2_B5value': sterimol_to_c2.B_5_value,
        
        # Initialize charge
        'S_charge': 0.0
    }

    # Read charges if available
    charges_file = xyz_dir / 'charges'
    if charges_file.exists():
        with open(charges_file, 'r') as file:
            charge_lines = [float(line.strip()) for line in file]
            descriptors['S_charge'] = charge_lines[indices['S'] - 1]

    return descriptors


def calculate_descriptors(
    xyz_dir: Union[str, Path],
    descriptor_type: str = "intermediate"
) -> Dict[str, float]:
    """Calculate molecular descriptors from XYZ file.

    Args:
        xyz_dir: Directory containing the XYZ file
        descriptor_type: Type of descriptors to calculate
                       'intermediate' or 'product_s'

    Returns:
        Dictionary containing calculated descriptors
    """
    xyz_dir = Path(xyz_dir)
    if not xyz_dir.exists():
        raise ValueError(f"Directory not found: {xyz_dir}")

    if descriptor_type == "intermediate":
        return calculate_intermediate_descriptors(xyz_dir)
    elif descriptor_type == "product_s":
        return calculate_product_s_descriptors(xyz_dir)
    else:
        raise ValueError(f"Unknown descriptor type: {descriptor_type}")


