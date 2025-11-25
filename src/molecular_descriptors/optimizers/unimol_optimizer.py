from typing import List, Optional, Union

from unimol_tools import UniMolRepr
import pandas as pd

from ...utils.config import Config
from ...utils.logger import log_info
from ..base import BaseOptimizer

class UniMolOptimizer(BaseOptimizer):
    STORAGE_METHOD = 'single'

    def __init__(self, raw_data_name: str, species_name: str, config: Optional[Config] = None):
        super().__init__(raw_data_name, species_name, config)


    def calculate(self, smiles: Union[str, List[str]],
                  save: bool = True) -> pd.DataFrame:
        smiles_list = [smiles] if isinstance(smiles, str) else smiles

        to_calculate, existing_results = self.check_calculations(smiles_list)
        self._logger.info(f"Calculating UniMol descriptors for {len(to_calculate)} molecules")

        unimol_repr = self.quick_calculate(list(to_calculate.values()))
        mol_indices = list(to_calculate.keys())

        if save:
            self._save_results(unimol_repr, mol_indices)
        return unimol_repr


    def _save_results(self, results: pd.DataFrame, mol_indices: List[int]) -> None:
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

    @staticmethod
    def quick_calculate(smiles: Union[str, List[str]]) -> pd.DataFrame:
        smiles_list = smiles if isinstance(smiles, list) else [smiles]
        clf = UniMolRepr(data_type='molecule',
                         remove_hs=False,
                         model_name='unimolv2',
                         model_size='164m',
                         use_gpu=False)
        
        unimol_repr = clf.get_repr(smiles_list)
        unimol_repr = pd.DataFrame(unimol_repr['cls_repr'])

        return unimol_repr
        