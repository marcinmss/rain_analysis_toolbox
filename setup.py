from distutils.core import setup
from setuptools import find_packages

setup(
    name="analysis_toolbox",
    version="0.0.1",
    description="Simple tools for analysing distrometer data.",
    url="https://github.com/marcinmss/rain_analysis_toolbox",
    author="Marcio Matheus",
    author_email="marciomsantoss@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
    ],
)
