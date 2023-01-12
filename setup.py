# Module imports
from setuptools import setup

# Arguments
version = "0.2.2"

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
    py_modules = ["__init__", "colors", "utils"],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    package_dir = {'': 'nexport'},
    extras_require = {
        "dev": [
            "pytest >= 7.1",
        ],
    },
)