# COPYRIGHT NOTICE

# “Neural Network Export Package (nexport) v0.4.6” Copyright (c) 2023,
# The Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.

# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights. As
# such, the U.S. Government has been granted for itself and others acting on
# its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
# Software to reproduce, distribute copies to the public, prepare derivative
# works, and perform publicly and display publicly, and to permit others to do so.


# Module imports
import json


# External function visibility
__all__ = ['append_extension', 'import_from_json']


# Module functions

def append_extension(filename: str, extension: str) -> str:
    """
    Function which constructs the filename
    and extension so the user doesn't have to.
    """
    filename = filename.replace(' ', '_')
    match extension:
        case "txt" | "json" | "csv" | "xml":
            return filename + "." + extension
        case other:
            raise RuntimeError(f"This filetype ({other}) is unrecognized and will not be supported in the near future.")


def import_from_json(filename: str):
    """
    Function which imports a model from a
    JSON file and returns it as an object.
    """
    with open(filename, 'r') as f:
        json_data = json.load(f)
    print(json_data)
    
    python_object = json.loads(json_data)
    print(python_object)
