from setuptools import find_packages, setup

setup(
    name="HeraSys",
    version="0.1",
    packages=find_packages(include=["HeraSys", "HeraSys.*"]),
    python_requires=">=3.10",
)
