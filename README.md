[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23424323.svg)](https://doi.org/10.5281/zenodo.23424323)

# py-recon  

A Python toolkit for **dendrochronological reconstructions** 

## Overview

`pyrecon` provides Python implementations of the classic calibration–verification methods used in dendroclimatology to reconstruct a target variable from tree-ring chronologies. It brings three established regression approaches under a single, consistent interface:

- **Stepwise regression** — forward/backward selection of the most informative chronologies.
- **Principal component regression (PCR)** — regression on the leading principal components of the chronology network.
- **Weighted k-nearest-neighbors (KNN)** — Non-parametric reconstruction in principal-component space.

## Documentation

Full documentation of the inputs and outputs for each method lives in the docstrings of the corresponding module — see `pyrecon/stepwise.py`, `pyrecon/pca.py`, and `pyrecon/knn.py`.


## Example

The included `example.ipynb` reproduces an Upper Colorado River Basin streamflow reconstruction at the Lees Ferry gauge, using the residual and standard  chronologies of Woodhouse et al.(2006):

> https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2005WR004455

Although the example reconstructs streamflow, the method is general: by swapping in a different observed target (the `y` series), you can reconstruct any other parameter — precipitation, temperature, PDSI, etc.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/Kalharap/py-recon.git
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/Kalharap/py-recon.git
cd py-recon
pip install -e .
```

## Repository structure

```
py-recon/
├── pyrecon/          # reconstruction methods + plotting
│   ├── stepwise.py
│   ├── pca.py
│   ├── knn.py
│   └── plotting.py
├── example_residual/standard_chrons.ipynb     # end-to-end example
├── requirements.txt
└── LICENSE
```

## Citation

If you use py-recon in your research, please cite it:

> Perera, K. (2026). *py-recon: A Python toolkit for tree-ring based climate reconstruction* (Version v0.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23424323
```bibtex
@software{perera_pyrecon_2026,
  author    = {Perera, Kalhara},
  title     = {py-recon: A Python toolkit for tree-ring based climate reconstruction},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.1.0},
  doi       = {10.5281/zenodo.23424323},
  url       = {https://doi.org/10.5281/zenodo.23424323}
}
```

## Contact

perern1@unlv.nevada.edu
Kalhara Perera, Department of Physics and Astronomy, University of Nevada, Las Vegas

## Issues

Please open an [issue](https://github.com/Kalharap/py-recon/issues) if you have any questions, suggestions, or encounter any problems with the resources in this repository.

