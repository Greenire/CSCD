from src.data_processing.data_loader import DataLoader
from src.molecular_descriptors.optimizers.unimol_optimizer import UniMolOptimizer
from src.utils.config import get_config
from src.utils.molecule_operator import merge_intermediates, merge_products


def main():
    config = get_config()
    loader = DataLoader(config)
    loader.load_data()
    _, smiles_list, species_calculators = loader.get_species_data()

    product_s_list = [merge_products(smi1, smi2) for smi1 in set(smiles_list[3]) for smi2 in set(smiles_list[1])]
    morfeus_des = UniMolOptimizer(raw_data_name='raw_data', species_name='product_s', config=config)
    morfeus_des.calculate(product_s_list)

    morfeus_des = UniMolOptimizer(raw_data_name='raw_data', species_name='inter_v', config=config)
    intermeiate_list = [merge_intermediates(i, j) for i in set(smiles_list[1]) for j in set(smiles_list[2])]
    morfeus_des.calculate(intermeiate_list)

if __name__ == '__main__':
    main()