# Module imports
import numpy as np
import torch.nn

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
