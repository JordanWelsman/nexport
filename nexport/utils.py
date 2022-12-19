# Module imports
from torch import nn
import numpy as np
import datetime as dt
import torch as torch
from torch import nn

# File imports
from colors import color as c


# Module functions

def append_extension(filename: str, extension: str) -> str:
    """Function which constructs the filename and extension so the user doesn't have to"""
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
    """Function which calculates the number of trainable parameters of a passed model."""
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
    """Function which exports all weight and bias arrays to a file."""
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
    """Function which imports weight and bias arrays and instantiates a model."""
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


def export_to_json(model: object, filename: str):
    """Function which exports all weight and bias arrays to JSON."""


def generate_filename(param_type: str = "wb"):
    now = dt.datetime.now()
    match param_type:
        case "wb":
            return f"weights_bias_{now.microsecond}"


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


# Runtime environment

model = FFNetwork()


# print(append_extension("weights model", "txt"))
print(f"Number of parameters: {calculate_params(model=model, param_type='wbt')}")

# generate_filename()