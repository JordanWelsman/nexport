# Module imports
from torch import nn
import numpy as np
import datetime as dt
import torch as torch
from torch import nn
import os
import json
import datetime as dt
import time as t

# File imports
from nexport.colors import Color as c

# External function visibility
__all__ = ['calculate_params', 'export_to_file', 'import_from_file', 'export_to_json', 'FFNetwork', 'BFNetwork', 'ICARNetwork']

# Module functions

def generate_filename(param_type: str = "wb"):
    """
    Function which generates a filename.
    """
    now = dt.datetime.now()
    match param_type:
        case "wb":
            return f"weights_bias_{now.microsecond}"


def append_extension(filename: str, extension: str) -> str:
    """
    Function which constructs the filename
    and extension so the user doesn't have to.
    """
    filename = filename.replace(' ', '_')
    match extension:
        case "txt":
            return filename + ".txt"
        case "json":
            return filename + ".json"
        case "csv":
            return filename + ".csv"
        case "xml":
            return filename + ".xml"
        case other:
            raise Exception(f"Unsupported filetype: {other}. Please enter valid filetype.")


def calculate_params(model: object, param_type: str = "t") -> list:
    """
    Function which calculates the number
    of trainable parameters of a passed model.
    """
    weights = 0
    biases = 0

    for x, y in enumerate(model.parameters()):
        if (x % 2 != 1):
            print("Weight")
            for i, z in enumerate(y):
                print(i)
        else:
            print("Bias")

    total = weights + biases

    match param_type:
        case "w":
            return [weights]
        case "b":
            return [biases]
        case "t":
            return [total]
        case "wb":
            return [weights, biases]
        case "wt":
            return [weights, total]
        case "bt":
            return [biases, total]
        case "wbt":
            return [weights, biases, total]
        case other:
            raise Exception(f"{c.MAGENTA}Unsupported parameter type:{c.DEFAULT} {other}. {c.LIGHTRED}Please enter valid parameter types.{c.DEFAULT}")


def export_to_file(model: object, filename: str) -> None:
    """
    Function which exports all weight
    and bias arrays to a file.
    """
    filename = append_extension(filename=filename, extension='txt')
    print(f"Creating file: {c.YELLOW}{filename}{c.DEFAULT}")
    f = open(filename, "w")
    for x, y in enumerate(model.parameters()): # access parameter array
        if (x % 2 != 1): # if array is not odd (even index = weights)
            print(f"Extracting {c.GREEN}weights{c.DEFAULT} from layer {c.RED}{(int(x/2))+1}{c.DEFAULT}")
            for i, z in enumerate(y): # access weight array x in layer y
                f.write("[") if i == 0 else f.write(" ")
                weights = []
                for w in z: # access weights for neuron w in from neuron z in layer y
                    weights.append(float(w))
                f.write(str(weights))
                f.write("]") if i == len(y) - 1 else f.write("")
                f.write("\n")
            f.write("\n")
        else: # if array is odd (odd index = biases)
            print(f"Extracting {c.MAGENTA}biases{c.DEFAULT} from layer  {c.RED}{(int(x/2))+1}{c.DEFAULT}")
            biases = []
            for b in y:
                biases.append(float(b))
            f.write(str(biases))
            f.write("\n\n\n")
    print(f"Saving file: {c.YELLOW}{filename}{c.DEFAULT}")
    f.close()
    print(f"{c.CYAN}Done!{c.DEFAULT}")


def import_from_file(filepath: str, framework: str = "PyTorch", architecture: str = "linear") -> object:
    """
    Function which imports weight and bias
    arrays and instantiates a model.
    """
    with open("qr_WnB_noBypass") as file:
        lines = file.read().splitlines() # read file and store in lines

    model_weights = []
    model_biases = []
    current_weights = []
    layer = 1
    neuron = 1

    for x, line in enumerate(lines):
        if line.startswith("[[") or line.startswith(" ["): # if row of weight array
            print(f"Line {x+1}: Weights for neuron {neuron} of layer {layer}") # print line and key
            current_weights.append(line.replace("[", "").replace("]", "").split(" ")) # store line in current weight array
            while "" in current_weights[-1]:
                current_weights[-1].remove("")
            if len(lines[x+1]) == 0: # if end of weights
                model_weights.append(current_weights.copy()) # write current array to weight array
                current_weights.clear() # reset current weights
                
            neuron += 1 # increment neuron number
            
        elif line.startswith("[") and len(lines[x+1]) == 0: # if bias array
            print(f"Line {x+1}: Biases for layer {layer}") # 
            model_biases.append(line.replace("[", "").replace("]", "").split(" ")) # write line to dictionary using key
            while "" in model_biases[-1]:
                model_biases[-1].remove("")
            layer += 1 # increment layer number
            neuron = 1 # reset neuron number
            
        else:
            print("-")

    for x, layer in enumerate(model_weights):
        for y, neuron in enumerate(layer):
            layer[y] = np.array(neuron)
            layer[y] = layer[y].astype(np.float)

    for x, layer in enumerate(model_biases):
        model_biases[x] = np.array(layer)
        model_biases[x] = model_biases[x].astype(np.float)


def create_paramater_arrays(model:object, verbose: int = 1) -> tuple:
    """
    Function which splits a model's state_dict into weight
    and bias arrays and returns them for indexing elsewhere.
    """
    model_dictionary = model.state_dict()
    weights = []
    biases = []

    # Loop which creates parameter arrays from model's state_dict
    for x, item in enumerate(model_dictionary):
        if x % 2 == 0: # if even (weight)
            weights.append(model_dictionary[item])
        else: # if odd (bias)
            biases.append(model_dictionary[item])

    if verbose >= 2: # if verbose set to at least 1
        print(f"{c.RED}Successfully extracted {c.LIGHTRED}parameters.{c.DEFAULT}")

    return weights, biases # return weights & biases as tuple


def create_layer_object(weights: list, biases: list, verbose: int = 1) -> list:
    """
    Function which constructs a single layer from
    parameter arrays and returns it as a list of neurons.
    """
    neuron_list = []
    temp_weights = []
    temp_bias = 0
    temp_dict = {}
    
    # Loop which creates a layer as a list from parameter arrays
    for i in range(len(weights)):
        for j in weights[i]:
            temp_weights.append(j.item())
        temp_bias = biases[i].item()
        temp_dict["weights"] = temp_weights.copy()
        temp_dict["bias"] = temp_bias
        neuron_list.append(temp_dict.copy())
        temp_weights.clear()
        temp_dict.clear()

    if verbose >= 3: # if verbose set to at least 2
        print(f"{c.LIGHTYELLOW}    Layer created.{c.DEFAULT}")

    return neuron_list # return constructed layer


def create_model_metadata(model_name: str, model_author: str = None) -> dict:
    model_metadata = {
        "modelName": model_name,
        "modelAuthor": model_author,
        "compilationDate": str(dt.datetime.now())
    }

    return model_metadata # return model metadata object


def create_model_object(model: object, verbose: int = 1, include_metadata: bool = None, model_name: str = None, model_author: str = None) -> object:
    """
    Function which creates a model object from a
    collection of layers instantiated with layer
    objects (neuron lists).
    """
    hidden_layers = []
    output_layer = []
    model_object = {}
    weights, biases = create_paramater_arrays(model=model, verbose=verbose)

    if include_metadata: # insert model metadata into model object
        model_object["model"] = create_model_metadata(model_name=model_name, model_author=model_author)

    if verbose >= 3: # if verbose set to at least 2
        print(f"{c.YELLOW}Creating layers...{c.DEFAULT}")

    # Loop which creates a network object from a series of single-layer lists
    for x, layer in enumerate(weights):
        if x != len(weights)-1:
            hidden_layers.append(create_layer_object(weights=weights[x], biases=biases[x], verbose=verbose))
        else:
            output_layer = create_layer_object(weights=weights[x], biases=biases[x], verbose=verbose)
    
    model_object["hidden_layers"] = hidden_layers
    model_object["output_layer"] = output_layer

    if verbose >= 2: # if verbose set to at least 1
        print(f"{c.GREEN}Successfully created {c.LIGHTGREEN}model object.{c.DEFAULT}")

    return model_object # return constructed network


def export_to_json(model: object, filename: str = "model", indent: int = 4, verbose: int = 1, include_metadata: bool = False, model_name: str = "My Model", model_author: str = os.getlogin()):
    """
    Function which exports a passed model
    object to a JSON file.
    """
    t1 = t.time()
    model_object = {}
    if include_metadata:
        model_object = create_model_object(model=model, verbose=verbose, include_metadata=include_metadata, model_name=model_name, model_author=model_author)
    else:
        model_object = create_model_object(model=model)
    json_object = json.dumps(obj=model_object, indent=indent)
    with open(append_extension(filename=filename, extension="json"), "w") as outfile:
        outfile.write(json_object)

    t2 = t.time()
    time = t2 - t1

    if verbose >= 1: # if verbose set to at least 1
        print(f"{c.CYAN}Exported model to {c.LIGHTCYAN}'{append_extension(filename=filename, extension='json')}'{c.CYAN}!{c.DEFAULT}")
    if verbose >= 2: # if verbose set to at least 2
        print(f"{c.MAGENTA}    Time taken: {c.LIGHTMAGENTA}{round(time, 2)}{c.MAGENTA}s{c.DEFAULT}")



# Model classes

class FFNetwork(nn.Module):
    def __init__(self):
        super(FFNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # 25 parameters
            nn.Linear(2, 3), # 6 weights, 3 biases
            nn.ReLU(),
            nn.Linear(3, 3), # 9 weights, 3 biases
            nn.ReLU(),
            nn.Linear(3, 1) # 3 weights, 1 bias
        )


class BFNetwork(nn.Module):
    def __init__(self):
        super(BFNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )


class ICARNetwork(nn.Module):
    def __init__(self):
        super(ICARNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
