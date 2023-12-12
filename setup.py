from distutils.core import setup
from setuptools import find_packages

kwards = {"py-modules": []}
setup(
    name="analysis_toolbox",
    version="0.0.1",
    description="Simple tools for analysing distrometer data.",
    **kwards
)
