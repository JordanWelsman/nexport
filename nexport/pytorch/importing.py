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
import numpy as np
import torch
from torch import nn
from nexport import colors as c
from nexport import models

# External function visibility
__all__ = ['import_from_file', 'import_from_json', "discover_architecture", 'get_activation']


# Module functions

def discover_architecture(weight_array: list) -> list:
    """
    Function which iterates through and reports
    passed networks' internal architectures.
    """
    architecture: list[list] = []
    hidden_architecture: list[list] = []
    binary_activation = "binary_step"
    sigmoid_activation = "sigmoid"

    # Discover each hidden layer's input & output count
    for x, layer in enumerate(weight_array):
        if x == 0:
            # Discover input/output layer's input & output connection count
            architecture.append([len(layer[0]), len(layer), sigmoid_activation])
        elif x == len(weight_array) - 1:
            # Append hidden weights & discover output layer's input & output connection count
            architecture.append(hidden_architecture)
            architecture.append([len(layer[0]), len(layer), sigmoid_activation])
        else:
            # Discover output layer's input & output connection count
            hidden_architecture.append([len(layer[0]), len(layer), sigmoid_activation])

    return architecture


def get_activation(activation: str) -> object:
    """
    Function which retrieves activation
    function from model framework or nexport
    itself depending on passed string.
    """
    match activation:
        case "sigmoid":
            return nn.Sigmoid()
        case "relu":
            return nn.ReLU()
        case "linear":
            return nn.Linear()
        case "binary_step":
            return models.XORNetwork.StepHS()


def import_from_file(filepath: str, verbose: int = None) -> object:
    """
    Function which imports weight and bias
    arrays and instantiates a model.
    """
    import numpy as np

    with open(filepath) as file:
        lines = file.read().splitlines() # read file and store in lines

    model_weights = []
    model_biases = []
    current_weights = []
    layer = 1
    neuron = 1

    for x, line in enumerate(lines):
        if line.startswith("[[") or line.startswith(" ["): # if row of weight array
            # print(f"Line {x+1}: Weights for neuron {neuron} of layer {layer}") # broadcast line and key
            current_weights.append(line.replace("[", "").replace("]", "").replace(",", "").split(" ")) # store line in current weight array
            while "" in current_weights[-1]:
                current_weights[-1].remove("")
            if len(lines[x+1]) == 0: # if end of weights
                model_weights.append(current_weights.copy()) # write current array to weight array
                current_weights.clear() # reset current weights
                
            neuron += 1 # increment neuron number
            
        elif line.startswith("[") and len(lines[x+1]) == 0: # if bias array
            # print(f"Line {x+1}: Biases for layer {layer}") # broadcast line and key 
            model_biases.append(line.replace("[", "").replace("]", "").replace(",", "").split(" ")) # write line to dictionary using key
            while "" in model_biases[-1]:
                model_biases[-1].remove("")
            layer += 1 # increment layer number
            neuron = 1 # reset neuron number
            
        else:
            # print("-")
            pass

    for x, layer in enumerate(model_weights):
        for y, neuron in enumerate(layer):
            layer[y] = np.array(neuron)
            layer[y] = layer[y].astype(np.float)

    for x, layer in enumerate(model_biases):
        model_biases[x] = np.array(layer)
        model_biases[x] = model_biases[x].astype(np.float)
    
    architecture = discover_architecture(weight_array=model_weights)
    print(f"Model architecture:\n{c.Color.BLUE}{architecture}{c.Color.DEFAULT}")

    class InputBlock(nn.Module):
        def __init__(self, architecture: list, weights: list, biases: list):
            super(InputBlock, self).__init__()
            
            self.input_layer = nn.Linear(architecture[0], architecture[1])
            self.activation = get_activation(architecture[-1])
            
            self.input_layer.weight.data = torch.Tensor(np.transpose(np.array(weights)))
            self.input_layer.bias.data = torch.Tensor(biases)
            
        def forward(self, input):
            return self.activation(self.input_layer(input))

    class ComputeBlock(nn.Module):
        def __init__(self, architecture: list[list], weights: list, biases: list):
            super(ComputeBlock, self).__init__()
            
            self.comp_layer = nn.Linear(architecture[0], architecture[1])
            self.activation = get_activation(architecture[-1])
            
            self.comp_layer.weight.data = torch.Tensor(weights)
            self.comp_layer.bias.data = torch.Tensor(biases)
            
        def forward(self, input):
            return self.activation(self.comp_layer(input))

    class OutputBlock(nn.Module): 
        def __init__(self, architecture: list, weights: list, biases: list):
            super(OutputBlock, self).__init__()
            
            self.output_layer = nn.Linear(architecture[0], architecture[1])
            self.activation = get_activation(architecture[-1])
            
            self.output_layer.weight.data = torch.Tensor(np.transpose(np.array(weights)))
            self.output_layer.bias.data = torch.Tensor(biases)
            
        def forward(self, input):
            return self.activation(self.output_layer(input))

    class Model(nn.Module):
        def __init__(self, model_architecture: list[list]):
            super(Model, self).__init__()
            self.input_architecture = model_architecture[0]
            self.hidden_architecture = model_architecture[1]
            self.output_architecture = model_architecture[-1]
            self.num_blocks = len(self.hidden_architecture)

            print(f"Total blocks:\n{c.Color.YELLOW}Input:      {1}\nHidden:     {len(self.hidden_architecture)}\nOutput:     {1}{c.Color.DEFAULT}")
            
            # Manufacture input block
            self.input_block = InputBlock(architecture=self.input_architecture, weights=np.array(model_weights[0]), biases=np.array(model_biases[0]))

            # Manufacture hidden (compute) blocks
            self.compute_block = nn.ModuleList([
                ComputeBlock(architecture=self.hidden_architecture[x], weights=np.array(model_weights[x+1]), biases=np.array(model_biases[x+1]))
                for x in range(self.num_blocks)
            ])

            # Manufacture output block
            self.output_block = OutputBlock(architecture=self.output_architecture, weights=np.array(model_weights[-1]), biases=np.array(model_biases[-1]))
        
        
        def forward(self, input):
            output = self.input_block(input)
            for x in range(50):  
                output = self.compute_block[x](output)
            output = self.output_block(output)
            return output

    network = Model(model_architecture=architecture)
    return network


def import_from_json(filepath: str) -> object:
    pass
