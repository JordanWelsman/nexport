# Module imports
from setuptools import setup

# Arguments
version = "0.4.0"
python_version = ">=3.10"

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
    py_modules = ["__init__", "calculators", "colors", "models", "utils"],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
    ],
    package_dir = {'': 'nexport'},
    install_requires = [
        "jutils",
        "numpy",
    ],
    extras_require = {
        "dev": [
            "pytest"
        ],
        "pytorch": [
            "torch"
        ],
        "tensorflow": [
            "tensorflow"
        ]
    },
)
