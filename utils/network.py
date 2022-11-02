from torch import nn


def print_wb_file(model: object, filename: str):
    """Function which exports all weight and bias arrays to a file."""
    print(f"Creating file: {filename}")
    f = open(filename, "w")
    for x, y in enumerate(model.parameters()): # access parameter array
        if (x % 2 != 1): # if array is not odd (even index = weights)
            print(f"Extracting weights from layer {(int(x/2))+1}")
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
            print(f"Extracting biases from layer  {(int(x/2))+1}")
            biases = []
            for b in y:
                biases.append(float(b))
            f.write(str(biases))
            f.write("\n\n\n")
    print(f"Saving file: {filename}")
    f.close()
    print("Done!")


def calculate_params(model: object, param_type: str = "wb"):
    """Function which calculates the number of trainable parameters of a passed model."""
    print(f"Param type: {param_type}")


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
print_wb_file(model, "weights_and_biases.txt")
calculate_params(model)