# Dunder attributes
__version__ = "v0.4.2" # update setup.py
__author__ = "Jordan Welsman"

# Import submodules so submodule functions
# are usable at 'import nexport' level
from .calculators import *
from .generic import *
from .models import *
from .utils import *
from .pytorch import __all__
from .tensorflow import __all__

# Initialize super-attribute with framework detection
__framework__ = detect_framework()

# Only show functions specified in
# submodules' __all__ to the outside world
__all__ = calculators.__all__, generic.__all__, models.__all__, utils.__all__, pytorch.__all__, tensorflow.__all__, __framework__
