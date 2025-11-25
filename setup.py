from setuptools import setup, find_packages

setup(
    name="molflux",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "rdkit",
        "pandas",
        "numpy",
        "pyyaml"
    ]
)
