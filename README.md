# CSCD

This repository contains code for the paper:  
"Machine Learning-Guided Design of Chiral Catalysts and Reagents for Enantioselective Carbon–Sulfenylation and Carbon–Disulfuration Reactions of Alkynes"

# Project Structure
- `src/`: Source code
  - `molecules/`: Molecule handling and descriptor calculation
  - `data_processing/`: Data transformation and cleaning
  - `utils/`: Utility functions and helpers
- `data/`: Data and morfeus descriptors storage
- `notebooks/`: Jupyter notebooks for analysis
- `config/`: Configuration files

# System requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
rdkit>=2022.9.3
tensorflow>=2.8.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
morfeus-ml>=0.5.0  # Optional: for descriptor calculation using morfeus
pytest==7.3.1
pytest-cov==4.0.0
```
