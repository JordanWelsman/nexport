<!-- Old ASCII logo
```ascii
 __   _ _______ _     _  _____   _____   ______ _______
 | \  | |______  \___/  |_____] |     | |_____/    |   
 |  \_| |______ _/   \_ |       |_____| |    \_    |   
```
-->

<!-- Logo -->
<a href="https://pypi.org/project/nexport"><img src="https://user-images.githubusercontent.com/61209125/216233823-976b0cb4-e53a-464f-b72f-e25cfbd165e1.svg" width=100%></a>

------------------------------------------------------

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/JordanWelsman/nexport)
![GitHub Release Date](https://img.shields.io/github/release-date/JordanWelsman/nexport)
![GitHub repo size](https://img.shields.io/github/repo-size/JordanWelsman/nexport)
![GitHub](https://img.shields.io/github/license/JordanWelsman/nexport)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/JordanWelsman/nexport)

![PyPI](https://img.shields.io/pypi/v/nexport)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nexport)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/nexport)
![PyPI - Status](https://img.shields.io/pypi/status/nexport)
![PyPI - Downloads](https://img.shields.io/pypi/dm/nexport)

# Overview

`nexport` is a lightweight `python 3` package which enables neural network developers to export the weights and biases of trained networks to useful file types.

# Table of contents

- [Overview](#overview)
- [Table of contents](#table-of-contents)
- [Current support](#current-support)
- [Install \& use](#install--use)
  - [Test](#test)
- [Objectives](#objectives)
  - [Personal objectives](#personal-objectives)
- [History](#history)
- [Credits](#credits)
- [License](#license)
- [Links](#links)

# Current support

<!--
| Filetype       | PyTorch            | Keras/TensorFlow |
| -------------: | :----------------: | :--------------: |
| Text (`.txt`)  | :white_check_mark: | :construction:   |
| JSON (`.json`) | :white_check_mark: | :x:              |
| CSV (`.csv`)   | :x:                | :x:              |
| XML (`.xml`)   | :x:                | :x:              |
-->

<!-- Ugly HTML table -->
<table>
  <tr>
    <th rowspan=2>Filetype</th>
    <th colspan=2>PyTorch</th>
    <th colspan=2>Keras/TensorFlow</th>
  </tr>
  <tr>
    <th>Export</th>
    <th>Import</th>
    <th>Export</th>
    <th>Import</th>
  </tr>
  <tr>
    <td>Text (.txt)</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
    <td>❌</td>
  </tr>
  <tr>
    <td>JSON (.json)</td>
    <td>✅</td>
    <td>🚧</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
  <tr>
    <td>CSV (.csv)</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
  <tr>
    <td>XML (.xml)</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
</table>

# Install & use

1. From terminal:
`pip install nexport`
2. From python environment:
`import nexport`

## Test

1. Clone repository:
`git clone https://github.com/JordanWelsman/nexport.git`
2. Build module for testing:
`python3 setup.py bdist_wheel`
3. Install module locally:
`pip install -e .`
4. Run tests with `PyTest`:
`pytest`

# Objectives

- Export weights and biases to human-readable file
- Ensure compatability with all popular neural network development software

## Personal objectives

- Learn `PyTorch`
- Understand how `PyTorch` and `Keras` construct neural networks
- Learn how to publish a python package
- Create a pseudo-pipeline between `Python` frameworks and `Fortran` projects (`PyTorch/Keras`-`nexport`-`inference-engine`)
- Write a paper on this software

# History

This package is intended to be used in conjunction with [inference-engine](https://github.com/BerkeleyLab/inference-engine). As such, `nexport` was developed by the `inference-engine` developers to enable compatability between the two softwares. `nexport` does this by exporting the weights and biases from networks compiled in `PyTorch`, `Keras`, and `TensorFlow` into standardized human-readable files. These files can be read by `inference-engine` to instantiate the netwoks in Fortran 2018 for inference.

# Credits

`nexport` was created and is currently maintained by **Jordan Welsman**.
Parts of this project were based on prior work by **[Tan Nguyen](mailto:TanNguyen@lbl.gov)**.

# License

`nexport` is developed and distributed under a modified `BSD` license.
> See `LICENSE` for more details.

# Links

:file_folder: [See this project on GitHub](https://github.com/JordanWelsman/nexport/)

:gift: [See this project on PyPI](https://pypi.org/project/nexport/)

:cat: [Follow me on GitHub](https://github.com/JordanWelsman/)

:briefcase: [Connect with me on Linkedin](https://linkedin.com/in/JordanWelsman/)

:email: [Send me an email](mailto:welsman@lbl.gov)

:thought_balloon: [Based on this project](https://github.com/tannguyen153/icar_pt)
