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

<div align="center">

[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/JordanWelsman/nexport?style=for-the-badge)](https://github.com/JordanWelsman/nexport/tags)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/JordanWelsman/nexport?style=for-the-badge)](https://github.com/JordanWelsman/nexport/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/JordanWelsman/nexport?style=for-the-badge)](https://github.com/JordanWelsman/nexport/wiki/Version-History)
[![GitHub license](https://img.shields.io/badge/license-LBNL%20BSD-blue?style=for-the-badge)](https://github.com/JordanWelsman/nexport/blob/main/LICENSE.md)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/JordanWelsman/nexport?style=for-the-badge)](https://github.com/JordanWelsman/nexport/commits/main)
[![GitHub wiki](https://img.shields.io/badge/wiki-nexport-blueviolet?style=for-the-badge)](https://github.com/JordanWelsman/nexport/wiki)

</div>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/nexport?style=for-the-badge)](https://pypi.org/project/nexport)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nexport?style=for-the-badge)](https://docs.python.org/3/whatsnew/3.10.html)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/nexport?style=for-the-badge)](https://pypi.org/project/nexport/#files)
[![PyPI - Status](https://img.shields.io/pypi/status/nexport?style=for-the-badge)](https://pypi.org/project/nexport/#data)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/nexport?style=for-the-badge)](https://pypi.org/project/nexport/#history)

</div>
<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/JordanWelsman/nexport?style=for-the-badge)](https://github.com/JordanWelsman/nexport/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/JordanWelsman/nexport?style=for-the-badge)](https://github.com/JordanWelsman/nexport/watchers)
[![GitHub forks](https://img.shields.io/github/forks/JordanWelsman/nexport?style=for-the-badge)](https://github.com/JordanWelsman/nexport/network/members)
![Lines of code](https://img.shields.io/tokei/lines/github/JordanWelsman/nexport?style=for-the-badge)
![GitHub repo file count](https://img.shields.io/github/directory-file-count/JordanWelsman/nexport?style=for-the-badge)
![GitHub repo size](https://img.shields.io/github/repo-size/JordanWelsman/nexport?style=for-the-badge)

</div>

# Overview

nexport is a lightweight `Python 3.10+` package which empowers deep learning developers in exporting the trainable parameters of deep neural networks to human-readable and transrerable file types.

# Table of contents

- [Overview](#overview)
- [Table of contents](#table-of-contents)
- [Current support](#current-support)
- [Install \& use](#install--use)
- [Objectives](#objectives)
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

<!--

## Test

1. Clone repository:
`git clone https://github.com/JordanWelsman/nexport.git`
2. Build module for testing:
`python3 setup.py bdist_wheel`
3. Install module locally:
`pip install -e .`
4. Run tests with `PyTest`:
`pytest`

-->

# Objectives

- Export weights and biases to human-readable file
- Ensure compatability with all popular neural network development software

# History

This package is intended to be used in conjunction with [inference-engine](https://github.com/BerkeleyLab/inference-engine). As such, `nexport` was developed by the `inference-engine` developers to enable compatability between the two softwares. `nexport` does this by exporting the weights and biases from networks compiled in `PyTorch`, `Keras`, and `TensorFlow` into standardized human-readable files. These files can be read by `inference-engine` to instantiate the netwoks in Fortran 2018 for inference.

# Credits

nexport was created and is currently maintained by **Jordan Welsman**.
Parts of this project were based on prior work by **[Tan Nguyen](mailto:TanNguyen@lbl.gov)**.

# License

nexport is developed and distributed under a Berkeley Laboratory modified `BSD` license.
> **Note**
> See `LICENSE` for more details.

# Links

:file_folder: [See this project on GitHub](https://github.com/JordanWelsman/nexport/)

:gift: [See this project on PyPI](https://pypi.org/project/nexport/)

:cat: [Follow me on GitHub](https://github.com/JordanWelsman/)

:briefcase: [Connect with me on Linkedin](https://linkedin.com/in/JordanWelsman/)

:email: [Send me an email](mailto:welsman@lbl.gov)

:thought_balloon: [Based on this project](https://github.com/tannguyen153/icar_pt)
