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


def calculate_layers() -> None:
    pass


def calculate_neurons() -> None:
    pass
