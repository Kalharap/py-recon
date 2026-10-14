# py-recon  

A Python toolkit for **dendrochronological reconstruction** — reconstructing hydroclimate (or other) variables from tree-ring chronologies.

## Overview

`pyrecon` provides Python implementations of the classic calibration–verification
reconstruction methods used in dendrochronology. It demonstrates how to reconstruct
a target series from tree-ring chronologies using three regression approaches:

- **Stepwise regression** (`pyrecon/stepwise.py`)
- **Principle Component regression** (`pyrecon/pca.py`)
- **k-nearest neighbors/ Non-parametric regression** (`pyrecon/knn.py`)

## Documentation

Full documentation of the inputs and outputs for each method lives in the docstrings of the corresponding module — see `pyrecon/stepwise.py`, `pyrecon/pca.py`, and `pyrecon/knn.py`.

## Example

The included `example.ipynb` reproduces an Upper Colorado River Basin streamflow
reconstruction at the Lees Ferry gauge, using the residual chronologies of Woodhouse et al.
(2006):

> https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2005WR004455

Although the example reconstructs streamflow, the method is general: by swapping in
a different observed target (the `y` series), you can reconstruct any other parameter — precipitation, temperature, PDSI, etc.

## Installation

```bash
git clone https://github.com/Kalharap/py-recon.git
cd py-recon
pip install -r requirements.txt
```

## Repository structure

```
py-recon/
├── pyrecon/          # reconstruction methods + plotting
│   ├── stepwise.py
│   ├── pca.py
│   ├── knn.py
│   └── plotting.py
├── example.ipynb     # end-to-end example
├── requirements.txt
└── LICENSE
```


## Contact

perern1@unlv.nevada.edu
Kalhara Perera, Department of Physics and Astronomy, University of Nevada, Las Vegas

## Issues

Please open an [issue](https://github.com/Kalharap/py-recon/issues) if you have any questions, suggestions, or encounter any problems with the resources in this repository.

