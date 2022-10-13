```
 __   _ _______ _     _  _____   _____   ______ _______
 | \  | |______  \___/  |_____] |     | |_____/    |   
 |  \_| |______ _/   \_ |       |_____| |    \_    |   
                                                       
```

# Overview

`nexport` is a lightweight Python `3.10+` package whic enables neural network developers to export the weights and biases of trained networks to a human-readable file.

# Table of contents

- [Overview](#overview)
- [Table of contents](#table-of-contents)
- [Objectives](#objectives)
- [History](#history)

# Objectives

- Export weights and biases to human-readable file
- Ensure compatability with all popular neural network development software

# History

This package is intended to be used in conjunction with [inference-engine](https://github.com/BerkeleyLab/inference-engine). As such, `nexport` was developed by the `inference-engine` developers to enable compatability between the two softwares. `nexport` does this by exporting the weights and biases from networks compiled in `PyTorch`, `Keras`, and `TensorFlow` into standardized human-readable files. These files can be read by `inference-engine` to instantiate the netwoks in Fortran 2018 for inference.
