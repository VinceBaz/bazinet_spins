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
RESULT 9: Original vs. rotated map similarity across levels of removal

FIGURE 2: Spherical projections inflate false positive rates 
FIGURE 3: Targeted removal of poor nulls improves performance

@author: Vincent Bazinet
"""

import os

# CHANGE THIS TO THE PATH TO THE GIT REPOSITORY
os.chdir((os.path.expanduser("~") + "/OneDrive - McGill University/"
          "projects (current)/variance/git-hub repository/bazinet_spins"))

# IMPORT STATEMENTS
import functions as fn
import pyvista as pv
import gstools as gs
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.stats import zscore
from scipy.spatial import cKDTree
from neuromaps.images import construct_shape_gii, load_data
from neuromaps.nulls.spins import _gen_rotation
from palettable.colorbrewer.diverging import Spectral_11_r
from palettable.cartocolors.sequential import SunsetDark_7
    
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
                p_all[j, k] = fn.get_p_value(r_spun[quantiles_nb >= q][:1000],
                                             r)
        FPRs[i,:] = np.count_nonzero(p_all < 0.05, axis=0) / (n_maps-1)

    # Save results
    np.save(f"results/FPRs/pial/length_{l}.npy", FPRs)

#%% RESULT 9: Original vs. rotated map similarity across levels of removal

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

    # Load random map + compute correlations for each threshold
    r_all = np.zeros((n_q, n_maps, 1000))
    for i in trange(n_maps):
        zX = zscore(random_maps[i,:])[:, np.newaxis]
        zX_perm = zscore(random_maps[i,:][spin_subsets].T, axis=0)
        r_spun = (zX * zX_perm).mean(axis=0)
        for k, q in enumerate(quantiles_all):
            r_all[k, i, :] = r_spun[quantiles_nb >= q][:1000]
    
    # Save results
    np.save(f"results/original_vs_rotated_r/length_{l}.npy",
            r_all)

#%% FIGURE 2: Spherical projections inflate false positive rates 

'''
Plot maps on surface meshes
'''

lengths = [1, 10, 20, 30, 40, 50]
cmap = Spectral_11_r.mpl_colormap

# spherical mesh
map_path = 'results/random_maps/sphere/'
sphere_mesh = fn.load_mesh('fsaverage', '10k', 'sphere', 'L', 'polydata')
for l in lengths:
    save_path = f'figures/figure_2/random_maps/sphere/length_{l}.png'
    random_map = load_data(f'{map_path}/length_{l}/0.shape.gii')
    fn.plot_surface_map(random_map, sphere_mesh, cmap=cmap, save=True,
                        save_path=save_path)

# pial mesh
map_path = 'results/random_maps/pial/'
pial_mesh = fn.load_mesh('fsaverage', '10k', 'pial', 'L', 'polydata')
for l in lengths:
    save_path = f'figures/figure_2/random_maps/pial/length_{l}.png'
    random_map = load_data(f'{map_path}/length_{l}/0.shape.gii')
    fn.plot_surface_map(random_map, pial_mesh, cmap=cmap, view='yz_negative',
                        save=True, save_path=save_path)

'''
Plot FPR across levels of autocorrelation (length)
'''

lengths = [1, 10, 20, 30, 40, 50]

# spherical mesh
FPRs = [np.load(f'results/FPRs/sphere/length_{l}.npy') for l in lengths]
fn.boxplot(FPRs, xticks=lengths, xlabel='length', ylabel='FPR',
           figsize=(2.5, 3), tight=True)
plt.savefig("figures/figure_2/FPR_across_lengths_sphere.svg")

# pial mesh
FPRs = [np.load(f'results/FPRs/pial/length_{l}.npy')[:,0] for l in lengths]
fn.boxplot(FPRs, xticks=lengths, xlabel='length', ylabel='FPR',
           figsize=(2.5, 3), tight=True)
plt.savefig("figures/figure_2/FPR_across_lengths_pial.svg")

'''
Plot scatterplot of the relationship between delta I and FPR
'''

# spherical mesh
FPRs = np.load('results/FPRs/sphere/length_50.npy')
delta_I = np.load("results/delta_I/sphere.npy")
fn.scatterplot(delta_I, FPRs, xlabel='$\Delta$I', ylabel='FPR', figsize=(3, 3),
                 s=2, compute_r=True, rasterized=True, c='black', tight=True)
plt.savefig("figures/figure_2/scatterplot_deltaI_FPR_sphere.svg", dpi=300)

# pial mesh
FPRs = np.load('results/FPRs/pial/length_50.npy')[:,0]
delta_I = np.load("results/delta_I/pial.npy")
fn.scatterplot(delta_I, FPRs, xlabel='$\Delta$I', ylabel='FPR', figsize=(3, 3),
                 s=2, compute_r=True, rasterized=True, c='black', tight=True)
plt.savefig("figures/figure_2/scatterplot_deltaI_FPR_pial.svg", dpi=300)

#%% FIGURE 3: Targeted removal of poor nulls improves performance

'''
Lineplot of FPR across lengths, for different levels of targeted removal.
'''

quantiles_all = np.arange(0, 91, 2.5)
lengths = [1, 10, 20, 30, 40, 50]
cmap = Spectral_11_r.mpl_colormap

FPR_means = np.array(
    [np.load(f'results/FPRs/pial/length_{l}.npy').mean(axis=0) 
     for l in lengths])

colors_all = fn.get_color_distribution(quantiles_all,
                                       cmap=SunsetDark_7.mpl_colormap)
fn.lineplot(lengths, FPR_means.T, colors=colors_all, xlabel='lengths',
            ylabel='FPR', figsize=(4, 3), tight=True)
plt.savefig("figures/figure_3/lineplots_FPR_across_removal_levels.svg")

'''
Boxplot of FPR across lengths at a 77.5 removal
'''

# pial mesh
FPRs = [np.load(f'results/FPRs/pial/length_{l}.npy')[:,31] for l in lengths]
fn.boxplot(FPRs, xticks=lengths, xlabel='length', ylabel='FPR',
           figsize=(2.5, 3), tight=True)
plt.savefig("figures/figure_3/FPR_across_lengths_77.5_removal.svg")

'''
Lineplot of average similarity across length for different levels of targeted
removal
'''

quantiles_all = np.arange(0, 91, 2.5)
lengths = [1, 10, 20, 30, 40, 50]
cmap = Spectral_11_r.mpl_colormap

r_mean = np.array(
    [np.abs(np.load(f"results/original_vs_rotated_r/length_{l}.npy"
            )).mean(axis=2).mean(axis=1)
     for l in lengths]
    )

colors = fn.get_color_distribution(quantiles_all,
                                   cmap=SunsetDark_7.mpl_colormap)
fn.lineplot(lengths, r_mean.T, colors=colors, figsize=(4, 3),
              xlabel='length', ylabel='average similarity', tight=True)
plt.savefig("figures/figure_3/lineplots_average_similarity.svg")