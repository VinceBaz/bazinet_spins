# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:03:17 2025

This is the main script of this repository. It can be used to reproduce the
results presented in "The effect of spherical projection on spin tests for
brain maps".

The computations of some of the results presented in the paper takes a long
time. Pre-computed results are therefore saved in the `results/` directory
of this repository.

The script is split into different cells (separated by `#%%`). Each cell
represent lines of codes used to load the data, compute a specific result, or
plot the elements of a specific figure:

DATA: Load data used for the experiments

RESULT 1: Spin test on the sphere

@author: Vincent Bazinet
"""

import os

# CHANGE THIS TO THE PATH TO THE GIT-HUB REPOSITORY
os.chdir((os.path.expanduser("~") + "/OneDrive - McGill University/"
          "projects (current)/variance/git-hub repository/bazinet_spins"))

# IMPORT STATEMENTS
import functions as fn
import gstools as gs
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import trange
from neuromaps.images import construct_shape_gii

# MATPLOTLIB RCPARAMS
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.family': 'Calibri'})
plt.rcParams.update({'font.weight': 'light'})
plt.rcParams.update({'axes.titlesize': 12})
plt.rcParams.update({'svg.fonttype': 'none'})
plt.rcParams.update({'axes.spines.top': False})
plt.rcParams.update({'axes.spines.right': False})

#%% DATA: Load data used for the experiments



#%% RESULTS 1: Spin test on the sphere

# Load surface mesh
sphere_mesh_L = fn.load_mesh('fsaverage', '10k', 'sphere', 'L')

# Setup parameters
n_maps = 1000
rng = np.random.default_rng(3194)
lengths = [1, 10, 20, 30, 40, 50]
save_path = 'results/random_maps/sphere'

for i in trange(n_maps):
    seed = rng.integers(0, 2**32-1)  # Get the same random seed across lengths
    for l in lengths:
        model = gs.Gaussian(dim=3, len_scale=l, var=1.0)
        srf = gs.SRF(model)
        random_map = srf.mesh(sphere_mesh_L, points='points', seed=seed)
        gii_image = construct_shape_gii(random_map)
        nib.save(gii_image, f'{save_path}/length_{l}/{i}.shape.gii')