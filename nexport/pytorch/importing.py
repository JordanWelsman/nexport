# Module imports
import numpy as np
import torch
from torch import nn

# External function visibility
__all__ = ['import_from_file']


# Module functions

def import_from_file(filepath: str) -> object:
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
            print(f"Line {x+1}: Weights for neuron {neuron} of layer {layer}") # broadcast line and key
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

    class InputBlock(nn.Module):
        def __init__(self, weights: list, biases: list):
            super(InputBlock, self).__init__()
            
            self.input_layer = nn.Linear(10, 64)
            self.activation = nn.Sigmoid()
            
            self.input_layer.weight.data = torch.Tensor(np.transpose(np.array(weights)))
            self.input_layer.bias.data = torch.Tensor(biases)
            
            
        def forward(self, input):
            return self.activation(self.input_layer(input))

    class ComputeBlock(nn.Module):
        def __init__(self, weights: list, biases: list):
            super(ComputeBlock, self).__init__()
            
            self.comp_layer = nn.Linear(64, 64)
            self.activation = nn.Sigmoid()
            
            self.comp_layer.weight.data = torch.Tensor(weights)
            self.comp_layer.bias.data = torch.Tensor(biases)
            
        def forward(self, input):
            return self.activation(self.comp_layer(input))

    class OutputBlock(nn.Module):
        def __init__(self, weights: list, biases: list):
            super(OutputBlock, self).__init__()
            
            self.output_layer = nn.Linear(64, 1)
            self.activation = nn.Sigmoid()
            
            self.output_layer.weight.data = torch.Tensor(np.transpose(np.array(weights)))
            self.output_layer.bias.data = torch.Tensor(biases)
            
            
        def forward(self, input):
            return self.activation(self.output_layer(input))

    class Model(nn.Module):
        def __init__(self, num_blocks):
            super(Model, self).__init__()
            self.num_blocks = num_blocks
            
            self.input_block = InputBlock(weights = model_weights[0], biases = model_biases[0])
            
            self.compute_block = nn.ModuleList([
                ComputeBlock(weights = model_weights[x+1], biases = model_biases[x+1])
                for x in range(num_blocks)
            ])
            self.output_block = OutputBlock(weights = np.array(model_weights[-1]), biases = np.array(model_biases[-1]))
        
        
        def forward(self, input):
            output = self.input_block(input)
            for x in range(50):  
                output = self.compute_block[x](output)
            output = self.output_block(output)
            return output

    network = Model(50)
    return network