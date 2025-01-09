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

RESULT 1: Random maps on the spherical mesh
RESULT 2: Random maps on the pial mesh
RESULT 3: Generate permutations
RESULT 4: Quantify the quality of spins
RESULT 5: Compute delta Moran's I for random maps on spherical mesh
RESULT 6: Compute delta Moran's I for random maps on pial mesh
RESULT 7: Compute false positive rates for random maps on spherical mesh
RESULT 8: Compute false positive rates for random maps on pial mesh

@author: Vincent Bazinet
"""

import os

# CHANGE THIS TO THE PATH TO THE GIT REPOSITORY
os.chdir((os.path.expanduser("~") + "/OneDrive - McGill University/"
          "projects (current)/variance/git-hub repository/bazinet_spins"))

# IMPORT STATEMENTS
import functions as fn
import gstools as gs
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.stats import zscore
from scipy.spatial import cKDTree
from neuromaps.images import construct_shape_gii, load_data
from neuromaps.nulls.spins import _gen_rotation

# MATPLOTLIB RCPARAMS
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.family': 'Calibri'})
plt.rcParams.update({'font.weight': 'light'})
plt.rcParams.update({'axes.titlesize': 12})
plt.rcParams.update({'svg.fonttype': 'none'})
plt.rcParams.update({'axes.spines.top': False})
plt.rcParams.update({'axes.spines.right': False})

#%% RESULTS 1: Random maps on the spherical mesh

# Load surface mesh
sphere_mesh_L = fn.load_mesh('fsaverage', '10k', 'sphere', 'L')

# Setup parameters
n_maps = 1000
rng = np.random.default_rng(3194)
lengths = [1, 10, 20, 30, 40, 50]
save_path = 'results/random_maps/sphere'

# Generate random maps
for i in trange(n_maps):
    seed = rng.integers(0, 2**32-1)  # Get the same random seed across lengths
    for l in lengths:
        model = gs.Gaussian(dim=3, len_scale=l, var=1.0)
        srf = gs.SRF(model)
        random_map = srf.mesh(sphere_mesh_L, points='points', seed=seed)
        gii_image = construct_shape_gii(random_map)
        nib.save(gii_image, f'{save_path}/length_{l}/{i}.shape.gii')

#%% RESULTS 2: Random maps on the pial mesh

# Load surface mesh
pial_mesh_L = fn.load_mesh("fsaverage", '10k', 'pial', 'L')

# Setup parameters
n_maps = 1000
rng = np.random.default_rng(3194)
lengths = [1, 10, 20, 30, 40, 50]
save_path = 'results/random_maps/pial'

# Generate random maps
for i in trange(n_maps):
    seed = rng.integers(0, 2**32-1)  # Get the same random seed across lengths
    for l in lengths:
        model = gs.Gaussian(dim=3, len_scale=l, var=1.0)
        srf = gs.SRF(model)
        random_map = srf.mesh(pial_mesh_L, points='points', seed=seed)
        gii_image = construct_shape_gii(random_map)
        nib.save(gii_image, f'{save_path}/length_{l}/{i}.shape.gii')

#%% RESULT 3: Generate permutations

# Load coordinates of vertices on surface mesh
vertices_sphere = fn.get_vertices('fsaverage', '10k', 'sphere', 'L')

# Generate permutations
n_perm = 10000
permutations = np.zeros((n_perm, len(vertices_sphere)))
for i in trange(n_perm):
    rotation, _ = _gen_rotation()
    rotated_vertices = vertices_sphere @ rotation
    _, permutations[i,:] = cKDTree(rotated_vertices).query(vertices_sphere, 1)

# Save permutations
np.save("results/permutations.npy", permutations)

#%% RESULT 4: Quantify the quality of spins

# Compute distance matrix (between vertices on pial surface)
dist = fn.compute_euclidean_distance('fsaverage', '10k', 'pial', 'L')

# Retrieve upper-triangular indices (and standardize distance values)
triu_id = np.triu_indices(len(dist), 1)
z_dist_triu = zscore(dist[triu_id])

# Load permutations
permutations = np.load("results/permutations.npy")
n_perm = len(permutations)

# Compute correlation between original and permuted distance matrices
spin_qualities = np.zeros((n_perm))
for i in trange(n_perm):
    spin_qualities[i] = fn.evaluate_spin_quality(
        dist, z_dist_triu, triu_id, permutations[i])

# Save results
np.save("results/spin_qualities.npy", spin_qualities)

#%% RESULT 5: Compute delta Moran's I for random maps on spherical mesh

# Compute weight matrix (for spherical mesh)
dist = fn.compute_euclidean_distance('fsaverage', '10k', 'sphere', 'L')
W = fn.inverse(dist, normalize=True)

# Setup parameters
n_perm = 1000
n_maps = 1000
map_path = 'results/random_maps/sphere/length_50/'
permutations = np.load("results/permutations.npy")[:n_perm,:]

# Compute Moran's I for empirical and permuted maps
delta_I = np.zeros((n_maps))
for i in trange(n_maps):
    X = load_data(f"{map_path}/{i}.shape.gii")
    X_perm = X[permutations]
    I = fn.morans_i(W, X)
    I_perm = [fn.morans_i(W, X_perm[k,:]) for k in trange(n_perm)]
    delta_I[i] = I - np.mean(I_perm)

# Save results
np.save("results/delta_I/sphere.npy", delta_I)

#%% RESULT 6: Compute delta Moran's I for random maps on pial mesh

# Compute weight matrix (for spherical mesh)
dist = fn.compute_euclidean_distance('fsaverage', '10k', 'pial', 'L')
W = fn.inverse(dist, normalize=True)

# Setup parameters
n_perm = 1000
n_maps = 1000
map_path = 'results/random_maps/pial/length_50/'
permutations = np.load("results/permutations.npy")[:n_perm,:]

# Compute Moran's I for empirical and permuted maps
delta_I = np.zeros((n_maps))
for i in trange(n_maps):
    X = load_data(f"{map_path}/{i}.shape.gii")
    X_perm = X[permutations]
    I = fn.morans_i(W, X)
    I_perm = [fn.morans_i(W, X_perm[k,:]) for k in trange(n_perm)]
    delta_I[i] = I - np.mean(I_perm)

# Save results
np.save("results/delta_I/pial.npy", delta_I)

#%% RESULT 7: Compute false positive rates for random maps on spherical mesh

# Setup parameters and load permutations
n_maps = 1000
map_path = 'results/random_maps/sphere/'
lengths = [1, 10, 20, 30, 40, 50]
n_perm = 1000
perm = np.load("results/permutations.npy")[:n_perm,:]

for l in lengths:

    # load maps
    random_maps = np.array(
        [load_data(f"{map_path}/length_{l}/{i}.shape.gii")
         for i in range(n_maps)])

    # Compute false positive rates for each map individually
    FPRs = np.zeros((n_maps))
    for i in trange(n_maps):
        p_all = np.zeros((n_maps-1))
        zX = zscore(random_maps[i,:])[:, np.newaxis]
        zX_perm = zscore(random_maps[i,:][perm].T, axis=0)
        for j, jj in enumerate(np.delete(np.arange(n_maps), i)):
            zY = zscore(random_maps[jj, :])[:, np.newaxis]
            r = (zX * zY).mean()
            r_spun = (zY * zX_perm).mean(axis=0)
            p_all[j] = fn.get_p_value(r_spun, r)
        FPRs[i] = np.count_nonzero(p_all < 0.05) / (n_maps-1)

    # Save results
    np.save(f"results/FPRs/sphere/length_{l}.npy", FPRs)

#%% RESULT 8: Compute false positive rates for random maps on pial mesh

# Setup parameters and load permutations
n_maps = 1000
map_path = 'results/random_maps/pial/'
n_perm = 10000
perm = np.load("results/permutations.npy")
perm_quality = np.load("results/spin_qualities.npy")

# Sample subset of permutations (based on the quality of the spins)
quantiles_all = np.arange(0, 91, 2.5)
n_q = len(quantiles_all)
quantiles_spins, quantiles_nb = fn.sample_subset_permutations(
    quantiles_all, perm_quality, n_perm=1000)
spin_subsets = perm[quantiles_spins]

for l in [1, 10, 20, 30, 40, 50]:

    # Load random maps
    random_maps = np.array(
        [load_data(f"{map_path}/length_{l}/{i}.shape.gii")
         for i in range(n_maps)])

    # Compute p-values for each threshold (and each map)
    FPRs = np.zeros((n_maps, n_q))
    for i in trange(n_maps):
        p_all = np.zeros((n_maps-1, n_q))
        zX = zscore(random_maps[i,:])[:, np.newaxis]
        zX_perm = zscore(random_maps[i,:][spin_subsets].T, axis=0)
        for j, jj in enumerate(np.delete(np.arange(n_maps), i)):
            zY = zscore(random_maps[jj, :])[:, np.newaxis]
            r = (zX * zY).mean()
            r_spun = (zY * zX_perm).mean(axis=0)
            for k, q in enumerate(quantiles_all):
                p_all[j, k] = fn.get_p_value(r_spun[quantiles_nb >= q][:1000], r)
        FPRs[i,:] = np.count_nonzero(p_all < 0.05, axis=0) / (n_maps-1)

    # Save results
    np.save(f"results/FPRs/pial/length_{l}.npy", FPRs)
