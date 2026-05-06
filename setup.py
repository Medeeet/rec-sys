from setuptools import setup, find_packages

setup(
    name="adaptive-recsys",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
)
