# The effect of spherical projection on spin tests for brain maps

This repository contains scripts and functions to reproduce the results presented in [The effect of spherical projection on spin tests for brain maps](https://doi.org/10.1162/IMAG.a.118).

## The main script

[main_script.py](main_script.py) contains a script that allows anyone to replicate the analysis described in the paper by simply running it. This script uses functions stored in [functions.py](functions.py) to run the analyses and plot the figures presented in the paper.

## The results

The [results](results) folder contains the main results of the experiments presented in the paper. Some results were too big to be pre-computed and included in this repository (random maps, permutations and correlations between original and rotated maps). These results can however be easily re-computed using [main_script.py](main_script.py).

## The figures

The [figures](figures) folder contains the main elements of the figures presented in the manuscript. As mentionned above, the code to re-generate each element is stored in [main_script.py](main_script.py).

## The requirements

The experiments presented in this repository make use of a certain number of python packages that will be necessary to run the main script. These packages are:

- [pyvista](<https://docs.pyvista.org/>)
- [numpy](<https://numpy.org/doc/stable/reference/>)
- [scipy](<https://docs.scipy.org/doc/scipy/reference/>)
- [tqdm](<https://github.com/tqdm/tqdm>)
- [gstools](<https://github.com/GeoStat-Framework/GSTools>)
- [neuromaps](<https://github.com/netneurolab/neuromaps>)
- [nibabel](<https://github.com/nipy/nibabel>)
- [meshio](<https://github.com/nschloe/meshio>)
- [matplotlib](<https://matplotlib.org/>)
- [seaborn](<https://seaborn.pydata.org/>)
- [palettable](<https://jiffyclub.github.io/palettable/>)
