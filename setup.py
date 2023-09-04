from setuptools import setup, find_packages

setup(
    name='preprocessing',
    version='0.8',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
)
