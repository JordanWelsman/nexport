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
import torch
from torch import nn

# External function visibility
__all__ = ['FFNetwork', 'BFNetwork', 'ICARNetwork', 'XORNetwork']


# Model classes

class FFNetwork(nn.Module): # Feed Forward Network
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

class MONetwork(nn.Module): # Multiple Output Network
    def __init__(self):
        super(MONetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )


class BFNetwork(nn.Module): # Large test Network
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


class ICARNetwork(nn.Module): # ICAR Network
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
            nn.Linear(64, 1)
        )


class XORNetwork(nn.Module): # XOR logic Network
    def __init__(self):
        super(XORNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        firstlayer = nn.Linear(2, 3)
        firstlayer.weight.data = XORNetwork.first_weights
        firstlayer.bias.data = XORNetwork.first_biases
        
        secondlayer = nn.Linear(3, 3)
        secondlayer.weight.data = XORNetwork.second_weights
        secondlayer.bias.data = XORNetwork.second_biases
        
        thirdlayer = nn.Linear(3, 1)
        thirdlayer.weight.data = XORNetwork.third_weights
        thirdlayer.bias.data = XORNetwork.third_biases
        
        self.linear_step_stack = nn.Sequential(
            firstlayer,
            self.StepHS(),
            secondlayer,
            self.StepHS(),
            thirdlayer,
            self.StepHS()
        )

    # Network parameters
    ## Hidden layer 1
    first_weights = torch.tensor([[1.0, 0.0],
                                [1.0, 1.0],
                                [0.0, 1.0]])
    first_biases = torch.tensor([0.0, -1.99, 0.0])

    ## Hidden layer 2
    second_weights = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
    second_biases = torch.tensor([0.0, 0.0, 0.0])

    ## Output layer
    third_weights = torch.tensor([[1.0, -2.0, 1.0]])
    third_biases = torch.tensor([0.0])

    # Binary step activation function
    class StepHS(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, input):
            output = torch.heaviside(input, torch.zeros(input.shape[0]))
            return torch.Tensor(output)
        
    # Forward function (matmul with activation function)
    def forward(self, input):
        return self.linear_step_stack(input)


class AltXORNetwork(nn.Module): # Alternative XOR logic Network
    def __init__(self):
        super(AltXORNetwork, self).__init__()
        self.flatten = nn.Flatten()

        firstlayer = nn.Linear(2, 3)
        firstlayer.weight.data = self.first_weights
        firstlayer.bias.data = self.first_biases
        
        secondlayer = nn.Linear(3, 3)
        secondlayer.weight.data = self.second_weights
        secondlayer.bias.data = self.second_biases
        
        thirdlayer = nn.Linear(3, 1)
        thirdlayer.weight.data = self.third_weights
        thirdlayer.bias.data = self.third_biases
        
        self.linear_step_stack = nn.Sequential(
            firstlayer,
            self.StepHS(),
            secondlayer,
            self.StepHS(),
            thirdlayer,
            self.StepHS()
        )
    
    # Network parameters
    # Hidden layer 1
    first_weights = torch.tensor([[1.0, 0.0],
                                [1.0, 1.0],
                                [0.0, 1.0]])
    first_biases = torch.tensor([0.0, -1.99, 0.0])

    # Hidden layer 2
    second_weights = torch.tensor([[0.0, 0.0, 0.0],
                                [1.0, -2.0, 1.0],
                                [0.0, 0.0, 0.0]])
    second_biases = torch.tensor([0.0, 0.0, 0.0])

    # Output layer
    third_weights = torch.tensor([[0.0, 1.0, 0.0]])
    third_biases = torch.tensor([0.0])

    # Binary step activation function
    class StepHS(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, input):
            output = torch.heaviside(input, torch.zeros(input.shape[0]))
            return torch.Tensor(output)
    
    # Forward function (matmul with activation function)
    def forward(self, input):
        return self.linear_step_stack(input)
