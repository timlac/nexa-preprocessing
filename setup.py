from setuptools import setup, find_packages

setup(
    name='nexa-nexa_preprocessing',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
)
