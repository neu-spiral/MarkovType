from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    required = f.readlines()

setup(
    name="markovtype",
    version="0.0.1",
    install_requires=required,
    packages=find_packages(),
    package_dir={"": "src"},
)
