from setuptools import setup, find_packages

setup(
    name='nexa-nexa_preprocessing',
    version='0.9',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
)
