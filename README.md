# nexport

```ASCII
 __   _ _______ _     _  _____   _____   ______ _______
 | \  | |______  \___/  |_____] |     | |_____/    |   
 |  \_| |______ _/   \_ |       |_____| |    \_    |   
```

# Overview

`nexport` is a lightweight Python `3.10+` package whic enables neural network developers to export the weights and biases of trained networks to a human-readable file.

# Table of contents

- [nexport](#nexport)
- [Overview](#overview)
- [Table of contents](#table-of-contents)
- [Objectives](#objectives)
  - [Personal objectives](#personal-objectives)
- [History](#history)
  - [Changelog](#changelog)
    - [v0.0.0](#v000)
    - [v0.0.1](#v001)
    - [v0.0.2](#v002)

# Objectives

- Export weights and biases to human-readable file
- Ensure compatability with all popular neural network development software

## Personal objectives

- Learn `PyTorch`
- Understand how `PyTorch` and `Keras` construct neural networks
- Learn how to publish a python package
- Create a pseudo-pipeline between `Python` and `Fortran` projects (`PyTorch`-`nexport`-`inference-engine`)
- Write a paper on this software

# History

This package is intended to be used in conjunction with [inference-engine](https://github.com/BerkeleyLab/inference-engine). As such, `nexport` was developed by the `inference-engine` developers to enable compatability between the two softwares. `nexport` does this by exporting the weights and biases from networks compiled in `PyTorch`, `Keras`, and `TensorFlow` into standardized human-readable files. These files can be read by `inference-engine` to instantiate the netwoks in Fortran 2018 for inference.

## Changelog

### v0.0.0

- Created project

### v0.0.1

- Developed `README`
- Added `LICENSE`
- Created package files

### v0.0.2

- Added command line tools
