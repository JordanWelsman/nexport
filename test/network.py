import torch
from torch import nn
from torch import tensor

def print_one(model: object):
    """Function which prints the first weight of the first layer to the terminal."""
    print(list(model.parameters())[0])


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


model = FFNetwork()


print("")

print_one(model)