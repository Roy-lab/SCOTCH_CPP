from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# Collect all relevant C++ source files
module_sources = [
    os.path.join("modules", f)
    for f in os.listdir("modules")
    if f.endswith(".cpp")
]

cpp_backend_file = "SCOTCH_cpp_backend.cpp"
sources = module_sources + [cpp_backend_file]


ext_modules = [
    Pybind11Extension(
        name="SCOTCH.SCOTCH_cpp_backend",  # Importable as SCOTCH.SCOTCH_cpp_backend
        sources=sources,
        include_dirs=[
            'modules',
            '.'
        ],
        libraries=["gsl", "gslcblas"],  # Link to GSL libraries
        language="c++",
        cxx_std=14,
    )
]

# Define the package
setup(
    name="SCOTCH_CPP",
    version="1.0.0",
    description="A Python package for SCOTCH with C++ backend for NMTF.",
    #long_description=open("README.md").read(),
    #long_description_content_type="text/markdown",
    author="Spencer Halberg-Spencer",
    author_email="shalberg@wisc.edu",
    url="https://github.com/your-repo/scotch",  # Update this with your repo link
    packages=find_packages(),  # Automatically find Python packages
    ext_modules=ext_modules,  # Add compiled extensions
    cmdclass={"build_ext": build_ext},  # Custom build class for Pybind11
    install_requires=[
        "numpy",  # Declare Python dependencies
        "pandas",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.6",  # Define Python version compatibility
    classifiers=[
        "Programming Language :: C++ :: 14",
        "Operating System :: OS Independent",
    ],
)
