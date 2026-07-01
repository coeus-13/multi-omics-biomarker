from setuptools import find_packages, setup

setup(
    name="multi-omics-biomarker",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.11",
)
