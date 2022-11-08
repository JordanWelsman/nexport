# Module imports
from torch import nn
import datetime as dt

# File imports
from colors import color as c


# Module functions

def export_to_file(model: object, filename: str):
    """Function which exports all weight and bias arrays to a file."""
    print(f"Creating file: {c.YELLOW}{filename}{c.DEFAULT}")
    f = open(filename, "w")
    for x, y in enumerate(model.parameters()): # access parameter array
        if (x % 2 != 1): # if array is not odd (even index = weights)
            print(f"Extracting {c.GREEN}weights{c.DEFAULT} from layer {c.RED}{(int(x/2))+1}{c.DEFAULT}")
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
            print(f"Extracting {c.MAGENTA}biases{c.DEFAULT} from layer  {c.RED}{(int(x/2))+1}{c.DEFAULT}")
            biases = []
            for b in y:
                biases.append(float(b))
            f.write(str(biases))
            f.write("\n\n\n")
    print(f"Saving file: {c.YELLOW}{filename}{c.DEFAULT}")
    f.close()
    print(f"{c.CYAN}Done!{c.DEFAULT}")


def calculate_params(model: object, param_type: str = "wb"):
    """Function which calculates the number of trainable parameters of a passed model."""
    print(f"Param type: {param_type}")


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


# Runtime environment

model = FFNetwork()

print("")

export_to_file(model, "weights_and_biases.txt")

# calculate_params(model)

# generate_filename()