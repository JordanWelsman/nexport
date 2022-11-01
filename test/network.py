import torch
from torch import nn
from torch import tensor

def print_one(model: object):
    """Function which prints the first weight of the first layer to the terminal."""
    print(list(model.parameters())[0])


def print_wb(model: object):
    """Function which prints all weights and biases to the terminal."""
    for x, y in enumerate(model.parameters()): # access parameter array
        if (x % 2 != 1): # if array is not odd (even index = weights)
            print(f"\n----------[ Layer {(int(x/2))+1} weights: ]----------")
            for z in y: # access weight array x in layer y
                for w in z: # access weights for neuron w in from neuron z in layer y
                    print(float(w))
        else: # if array is odd (odd index = biases)
            print(f"\n----------[ Layer {(int(x/2))+1} biases:  ]----------")
            for b in y:
                print(float(b))


def print_wb_array(model: object):
    """Function which inserts all weights and biases into an array."""
    for x, y in enumerate(model.parameters()): # access parameter array
        if (x % 2 != 1): # if array is not odd (even index = weights)
            weight_array = []
            for z in y: # access weight array x in layer y
                weights = []
                for w in z: # access weights for neuron w in from neuron z in layer y
                    weights.append(float(w))
                weight_array.append(weights)
            print(weight_array)
        else: # if array is odd (odd index = biases)
            biases = []
            for b in y:
                biases.append(float(b))
            print(biases)


def print_wb_file(model: object, filename: str):
    """Function which exports all weight and bias arrays to a file."""
    f = open(filename, "w")
    for x, y in enumerate(model.parameters()): # access parameter array
        if (x % 2 != 1): # if array is not odd (even index = weights)
            for i, z in enumerate(y): # access weight array x in layer y
                if i == 0:
                    f.write("[")
                else:
                    f.write(" ")
                weights = []
                for w in z: # access weights for neuron w in from neuron z in layer y
                    weights.append(float(w))
                # weight_array.append(weights)
                f.write(str(weights))
                if i == len(y) - 1:
                    f.write("]")
                f.write("\n")
            f.write("\n")
        else: # if array is odd (odd index = biases)
            biases = []
            for b in y:
                biases.append(float(b))
            f.write(str(biases))
            f.write("\n\n\n")
    f.close()


def test_w_b(model: object):
    assert len(list(model.parameters)) % 2 == 0


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


model = BFNetwork()

print("")

# print_one(model)
# print_wb(model)
# print_wb_array(model)
print_wb_file(model, "weights_and_biases.txt")