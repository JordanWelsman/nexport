---
title: 'nexport: A Python package for exporting the weights and biases of neural networks'
tags:
  - Python
  - neural networks
  - parameters
  - human readable
  - export
  - import
  - utilities
authors:
  - name: Jordan A. Welsman
    orcid: 0000-0002-2882-594X
    corresponding: true
    equal-contrib: false
    affiliation: "1, 2"
  - name: Damian W. I. Rouson
    orcid: 0000-0002-2344-868X
    equal-contrib: true
    affiliation: "1, 3"
  - name: Tan Thanh Nhat Nguyen
    orcid: 0000-0003-3748-403X
    equal-contrib: true
    affiliation: "1, 4"
affiliations:
 - name: Lawrence Berkeley National Laboratory, USA
   index: 1
 - name: Bournemouth University, UK
   index: 2
 - name: Stanford University, USA
   index: 3
 - name: University of California, San Diego, USA
   index: 4
date: March 2023
bibliography: paper.bib
---

# Summary

Neural networks trained in Python with deep learning frameworks such as `PyTorch` [@pytorch] and `TensorFlow` [@tensorflow] only exist while they are loaded in memory. The trainable parameters of such models are created through random initialization and depend on volatile memory for persistance. While there exists ways of exporting the parameters of these models so they can be loaded into memory later, they often compile them into binary files which are not human-readable or readable by other programming laguages due to language-specific loading methods. This poses an issue to inter-language research. `nexport` proposes a solution to this problem by allowing the user to export their model parameters to various human-readable and language-agnostic file formats commonly used by many programming languages.

# Statement of need

<!--
`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.
-->

<!--
# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.
-->

<!--
# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"
-->

<!--
# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }
-->

# Acknowledgements

nexport was developed at Lawrence Berkeley National Laboratory, and was wholly funded
under contract with the U.S. Department of Energy (DOE). It is actively being developed as
an open-source project.

# References