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


from torch import nn

# File imports
from nexport.colors import Color as c

# External function visibility
__all__ = ['calculate_params', 'calculate_layers', 'calculate_neurons']


# Module functions

def calculate_params(model: object, param_types: str = "wbt") -> list:
    """
    Calculte & return model's parameters.
    """

    model_dictionary = model.state_dict()
    weights = 0
    biases = 0
    total = 0

    for x, item in enumerate(model_dictionary):
        if x % 2 == 0: # if even (weight)
            for y in model_dictionary[item]:
                weights += len(y)
        else: # if odd (bias)
            biases += len(model_dictionary[item])
    total = weights + biases

    param_list = []
    for param_type in param_types:
        match param_type:
            case "w":
                param_list.append(weights)
            case "b":
                param_list.append(biases)
            case "t":
                param_list.append(total)
    
    return param_list


def calculate_layers(model: object, include_io: bool = True) -> int:
    """
    Calculate & return model's layer count.
    """

    model_dictionary = model.state_dict()

    if include_io:
        return int(len(model_dictionary.keys()) / 2) + 1
    else:
        return int(len(model_dictionary.keys()) / 2) - 1


def calculate_neurons(model: object, include_io: bool = True) -> int:
    """
    Calculate & return model's neuron count.
    """

    model_dictionary = model.state_dict()
    input_neurons = len(list(model_dictionary.values())[0][0])
    hidden_neurons = 0
    output_neurons = len(list(model_dictionary.values())[-1])

    for x, item in enumerate(model_dictionary):
        if x % 2 == 0: # if even (weight)
            pass
        else: # if odd (bias)
            hidden_neurons += len(model_dictionary[item])
    
    if include_io:
        hidden_neurons -= output_neurons
        return input_neurons + hidden_neurons + output_neurons
    else:
        return hidden_neurons - output_neurons
