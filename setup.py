# Module imports
from setuptools import setup

# Arguments
version = "0.0.2"

# Long description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

# Run stup function
setup(
    name = 'nexport',
    version = version,
    description = 'A Python package for exporting the weights and biases of neural networks.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Jordan Welsman',
    author_email = 'welsman@lbl.gov',
    url = 'https://github.com/JordanWelsman/nexport/',
    py_modules = ["__init__"],
    classifiers = ["Development Status :: 2 - Pre-Alpha", "Intended Audience :: Other Audience", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent", "Programming Language :: Python :: 3.9", "Topic :: Scientific/Engineering :: Mathematics"],
    package_dir = {'': 'nexport'},
    extras_require = {
        "dev": [
            "pytest >= 7.1",
        ],
    },
)