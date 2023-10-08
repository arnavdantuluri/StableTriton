import pathlib

import pkg_resources
from setuptools import find_packages, setup


with pathlib.Path("requirements.txt").open() as f:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]


setup(
    name="flashfuse",
    version="0.1",
    license="Apache License 2.0",
    license_files=("LICENSE",),
    description="Accelerate diffusion model inference with custom GPU kernels written in Triton",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    keywords=["Deep Learning"],
    long_description_content_type="text/markdown",
    url="https://github.com/arnavdantuluri/StableTriton/tree/main",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    python_requires="==3.10.*",
)