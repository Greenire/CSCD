from src.data_processing.data_loader import DataLoader
from src.molecular_descriptors.descriptors.morfeus_descriptor import MorfeusDescriptor
from src.utils.config import get_config
from src.utils.molecule_operator import merge_intermediates

def main():
    config = get_config()
    loader = DataLoader(config)
    loader.load_data()
    _, smiles_list, species_calculators = loader.get_species_data()

    intermeiate_list = [merge_intermediates(i, j) for i, j in zip(smiles_list[1], smiles_list[2])]
    morfeus_des = MorfeusDescriptor(raw_data_name='raw_data', species_name='inter_v',
                                    optimizer_type='xtb', descriptor_function='intermediate_conf', config=config)
    morfeus_des.calculate(intermeiate_list, n_conformers=5, overwrite=False)

if __name__ == '__main__':
    main()