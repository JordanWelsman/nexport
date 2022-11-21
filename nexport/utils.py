# Module imports
from torch import nn
import datetime as dt

# File imports
from colors import color as c


# Module functions

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


def export_to_json(model: object, filename: str):
    """Function which exports all weight and bias arrays to JSON."""


def append_extension(filename: str, extension: str) -> str:
    """Function which constructs the filename and extension so the user doesn't have to"""
    filename = filename.replace(' ', '_')
    match extension:
        case "txt":
            return filename + ".txt"
        case "json":
            return filename + ".json"
        case "xml":
            return filename + ".xml"
        case other:
            raise Exception(f"Unsupported filetype: {other}.Please provide a supported filetype.")


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
export_to_file(model=model, filename="weights and biases")
# print(append_extension("w and b", "json"))

# calculate_params(model)
# generate_filename()