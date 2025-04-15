from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os
import glob

enrich_analyzer_sources = glob.glob('SCOTCH_CPP/PyEnrichAnalyzer/*.C')


cpp_backend_file = "SCOTCH_CPP/SCOTCH_cpp_backend.cpp"
#sources = module_sources + [cpp_backend_file]

# Define the object files to link with
object_files = [
    'SCOTCH_CPP/modules/random_svd/matrix_funcs.o',
    'SCOTCH_CPP/modules/random_svd/rsvd.o',
]

cpp_backend_files = ['SCOTCH_CPP/SCOTCH_cpp_backend.cpp',
                    'SCOTCH_CPP/modules/initialization.cpp',
                    'SCOTCH_CPP/modules/io.cpp',
                    'SCOTCH_CPP/modules/nmtf.cpp',
                    'SCOTCH_CPP/modules/utils.cpp']

cpp_backend_ext = [
    Extension(
        name="SCOTCH_cpp_backend",  # Importable as SCOTCH_CPP.SCOTCH_cpp_backend
        sources=cpp_backend_files,  # Only include the main source file for the extension
        include_dirs=[
            'SCOTCH_CPP/modules',  # Ensure this points to the correct directory
            'SCOTCH_CPP/',          # The root directory for any other headers
            pybind11.get_include(), # Include pybind11 headers
        ],
        libraries=["gsl", "gslcblas"],  # Link to GSL libraries
        library_dirs=["/opt/homebrew/opt/gsl"],  # Specify where the GSL libraries are located
        extra_objects=object_files,  # Link the object files directly without recompiling
        language="c++"
    )
]

enrich_analyzer_sources = glob.glob('SCOTCH_CPP/PyEnrichAnalyzer/*.C')
pyEnrichAnalyzer_ext = [
    Pybind11Extension(
        'pyEnrichAnalyzer',  # name of output module
        enrich_analyzer_sources,
        include_dirs=[  # locations of your includes here
            '/usr/local/include',
            '/usr/local/include/python3.11m',
            '/usr/local/include/gsl/',
            pybind11.get_include(),
        ],
        libraries=['gsl'],  # add needed C/C++ libraries here
        library_dirs=['/usr/local', '/usr/local/lib'],
        language='c++',
        extra_compile_args=['-std=c++14']
    ),
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
    url="https://github.com/Roy-lab/SCOTCH_CPP.git",  # Update this with your repo link
    packages=find_packages(),  # Automatically find Python packages
    ext_modules=cpp_backend_ext + pyEnrichAnalyzer_ext,  # Add compiled extensions
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
