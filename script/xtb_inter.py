
from src.data_processing.data_loader import DataLoader
from src.molecular_descriptors.optimizers.xtb_optimizer import XTBOptimizer
from src.utils.config import get_config
from src.utils.molecule_operator import merge_intermediates

def main():
    config = get_config()
    loader = DataLoader(config)
    loader.load_data()
    _, smiles_list, species_calculators = loader.get_species_data()
    xtb_opt = XTBOptimizer(raw_data_name='raw_data', species_name='inter', config=config)
    intermeiate_list = [merge_intermediates(i, j) for i, j in zip(smiles_list[1], smiles_list[2])]
    xtb_opt.calculate(intermeiate_list, is_conformer=True, overwrite=False)

if __name__ == '__main__':
    main()