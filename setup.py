from setuptools import find_packages, setup

setup(
    name="gsmini",
    version="1.0",
    python_requires=">=3.8",
    packages=find_packages(),
    license="",
    author="Yipai Du",
    author_email="yipai.du@outlook.com",
    install_requires=[
        "numpy>=1.17.4",
        "open3d>=0.12.0",
        "opencv-python>=4.7.0.68",
        "scikit-image>=0.18.3",
        "scipy>=1.10.1",
        "setuptools>=45.2.0",
        "torch>=1.11.0",
    ],
    package_data={"": ["nnmini.pt"]},
    description="A reduced interface for GelSight Mini tactile sensor, adapted from the official GelSight SDK.",
)
