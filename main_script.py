# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:03:17 2025

This is the main script of this repository. It can be used to reproduce the
results presented in "The effect of spherical projection on spin tests for
brain maps".

The computation of some of the results presented in the paper takes a long
time. Pre-computed results are therefore saved in the `results/` directory
of this repository.

The script is split into different cells (separated by `#%%`). Each cell
represent lines of codes used to load the data, compute a specific result, or
plot the elements of a specific figure:

RESULT 1: Random maps on the spherical and pial meshes
RESULT 2: Generate permutations
RESULT 3: Quantify the quality of spins
RESULT 4: Compute delta Moran's I for random maps on spherical mesh
RESULT 5: Compute delta Moran's I for random maps on pial mesh
RESULT 6: Compute false positive rates for random maps on spherical mesh
RESULT 7: Compute false positive rates for random maps on pial mesh
RESULT 8: Original vs. rotated map similarity across levels of removal
RESULT 9: Compute standardized Moran's I of empirical brain maps

FIGURE 2: Spherical projections inflate false positive rates
FIGURE 3: Targeted removal of poor nulls improves performance
FIGURE S1: Impact of distortions on local spatial autocorrelation
FIGURE S2: Impact of distortions on autocorrelation of empirical brain maps
FIGURE S3: Quality of spins
FIGURE S4: Rotations quality associated with deviations in Moran’s I
FIGURE S5: Distribution of rotation quality correlations
FIGURE S6: Gaussian variogram models

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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from tqdm import trange, tqdm
from scipy.stats import zscore, pearsonr
from scipy.spatial import cKDTree
from neuromaps.images import construct_shape_gii, load_data
from neuromaps.nulls.spins import _gen_rotation
from neuromaps.transforms import (
    civet_to_fsaverage, fslr_to_fsaverage, fsaverage_to_fsaverage)
from palettable.colorbrewer.diverging import Spectral_11_r
from palettable.cartocolors.sequential import SunsetDark_7, agSunset_7_r
from palettable.cartocolors.qualitative import Pastel_7
from palettable.colorbrewer.diverging import RdBu_11_r

# MATPLOTLIB RCPARAMS
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.family': 'Calibri'})
plt.rcParams.update({'font.weight': 'light'})
plt.rcParams.update({'axes.titlesize': 12})
plt.rcParams.update({'svg.fonttype': 'none'})
plt.rcParams.update({'axes.spines.top': False})
plt.rcParams.update({'axes.spines.right': False})

#%% RESULT 1: Random maps on the spherical and pial meshes

# Setup parameters
n_maps = 1000
lengths = [1, 10, 20, 30, 40, 50]
surfaces = ['sphere', 'pial']

for surface in surfaces:

    # Setup RNG + path
    rng = np.random.default_rng(3194)
    save_path = f'results/random_maps/{surface}'

    # Load surface mesh
    mesh_L = fn.load_mesh('fsaverage', '10k', surface, 'L')

    # Generate (and save) random maps
    for i in trange(n_maps):
        seed = rng.integers(0, 2**32-1)  # get the same seeds across lengths
        for l in lengths:
            model = gs.Gaussian(dim=3, len_scale=l, var=1.0)
            srf = gs.SRF(model)
            random_map = srf.mesh(mesh_L, points='points', seed=seed)
            gii_image = construct_shape_gii(random_map)
            nib.save(gii_image, f'{save_path}/length_{l}/{i}.shape.gii')

#%% RESULT 2: Generate permutations

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

#%% RESULT 3: Quantify the quality of spins

# Compute distance matrix (between vertices on pial surface)
dist = fn.compute_euclidean_distance('fsaverage', '10k', 'pial', 'L')

# Load permutations
permutations = np.load("results/permutations.npy")
n_perm = len(permutations)

# Retrieve upper-triangular indices (and standardize distance values)
triu_id = np.triu_indices(len(dist), 1)
z_dist_triu = zscore(dist[triu_id])

# Compute correlation between original and permuted distance matrices
spin_qualities = np.zeros((n_perm))
for i in trange(n_perm):
    spin_qualities[i] = fn.evaluate_spin_quality(
        dist, z_dist_triu, triu_id, permutations[i])

# Save results
np.save("results/spin_qualities.npy", spin_qualities)

#%% RESULT 4: Compute delta Moran's I for random maps on spherical mesh

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

#%% RESULT 5: Compute delta Moran's I for random maps on pial mesh

# Compute weight matrix (for spherical mesh)
dist = fn.compute_euclidean_distance('fsaverage', '10k', 'pial', 'L')
W = fn.inverse(dist, normalize=True)

# Setup parameters
n_perm = 1000
n_maps = 1000
map_path = 'results/random_maps/pial/length_50/'
permutations = np.load("results/permutations.npy")[:n_perm,:]

# Instantiate results arrays
I_all = np.zeros((n_maps))
I_perm_all = np.zeros((n_maps, n_perm))
delta_I = np.zeros((n_maps))

# Compute Moran's I for empirical and permuted maps
for i in trange(n_maps):
    X = load_data(f"{map_path}/{i}.shape.gii")
    X_perm = X[permutations]
    I_all[i] = fn.morans_i(W, X)
    I_perm_all[i] = [fn.morans_i(W, X_perm[k,:]) for k in trange(n_perm)]
    delta_I[i] = I_all[i] - np.mean(I_perm_all[i])

# Save results
np.save("results/delta_I/pial.npy", delta_I)
np.save("results/I/pial.npy", I_all)
np.save("results/I_perm/pial.npy", I_perm_all)

#%% RESULT 6: Compute false positive rates for random maps on spherical mesh

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

#%% RESULT 7: Compute false positive rates for random maps on pial mesh

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

#%% RESULT 8: Original vs. rotated map similarity across levels of removal

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

#%% RESULT 9: Compute standardized Moran's I of empirical brain maps

'''
Fetch brain maps
'''

brain_maps = fn.fetch_neuromaps_maps()

'''
Transform maps (surface) to fsaverage_10k
'''

for (source, desc, space, den), map_file in tqdm(brain_maps.items(), total=28):
    if space == 'fsaverage':
        map_data = fsaverage_to_fsaverage(map_file, '10k')[0].agg_data()
    elif space == 'fsLR':
        map_data = fslr_to_fsaverage(map_file, '10k')[0].agg_data()
    elif space == 'civet':
        map_data = civet_to_fsaverage(map_file, '10k')[0].agg_data()
    np.save(f"results/neuromaps/fsaverage_10k/{source}_{desc}.npy", map_data)

'''
Evaluate standardized I (relative to spun maps)
'''

# Compute spatial weights (from geodesic distance)
geo_dist = fn.compute_geodesic_distance('fsaverage', '10k', 'pial')
W_geo = fn.inverse(geo_dist, normalize=True)

# Load permutations
n_nulls = 1000
permutations = np.load("results/permutations.npy")[:n_nulls]

# For each map:
for (source, desc, _, _), map_file in tqdm(brain_maps.items(), total=28):

    # Load data
    save_path = f"results/neuromaps/standardized_I/{source}_{desc}.pickle"
    map_data = np.load(f"results/neuromaps/fsaverage_10k/"
                       f"{source}_{desc}.npy")

    # Instantiate results dictionary
    R = {}

    # Mask brain data (ignore parcels with values of 0 or NaN)
    mw_mask = (map_data!=0) & (~np.isnan(map_data))
    data_masked = map_data[mw_mask]
    w_masked = W_geo[mw_mask, :][:, mw_mask]

    # Compute empirical Moran's I
    R['I'] = fn.morans_i(w_masked, data_masked)

    # Compute Moran's I for spun maps
    R['I_spin'] = np.zeros((n_nulls))
    for i in trange(n_nulls):
        data_spin = map_data[permutations[i]]
        spin_mask = (data_spin!=0) & (~np.isnan(data_spin))
        data_spin_masked = data_spin[spin_mask]
        w_spin_masked = W_geo[spin_mask, :][:, spin_mask]
        R['I_spin'][i] = fn.morans_i(w_spin_masked, data_spin_masked)

    # Compute standardized Moran's I
    R['zI'] = fn.standardize_scores(R['I_spin'], R['I'])

    # Save results in a picked dictionary
    fn.save_pickle(R, save_path)

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

#%% FIGURE S1: Impact of distortions on local spatial autocorrelation

# Load a random map and a random spin to run the analyses
X = load_data("results/random_maps/pial/length_50/0.shape.gii")
spin = np.load("results/permutations.npy")[1]

# Transform the original map
X_spin = X[spin]

'''
Plot the random maps
'''

# Load surface meshes
pial_mesh = fn.load_mesh('fsaverage', '10k', 'pial', 'L', 'polydata')
sphere_mesh = fn.load_mesh('fsaverage', '10k', 'sphere', 'L', 'polydata')

# Plot original map
for surface, mesh in zip(['pial', 'sphere'], [pial_mesh, sphere_mesh]):
    save_path = f'figures/figure_s1/random_map/{surface}.png'
    fn.plot_surface_map(X, mesh, cmap=Spectral_11_r.mpl_colormap, save=True,
                        save_path=save_path, view='yz_negative')

# Plot spun map
for surface, mesh in zip(['pial', 'sphere'], [pial_mesh, sphere_mesh]):
    save_path = f'figures/figure_s1/random_map/{surface}_spin.png'
    fn.plot_surface_map(X_spin, mesh, cmap=Spectral_11_r.mpl_colormap, save=True,
                        save_path=save_path, view='yz_negative')

'''
Plot the local Moran's I of the random map
'''

# Compute weight matrices
dist_sphere = fn.compute_euclidean_distance('fsaverage', '10k', 'sphere', 'L')
dist_pial = fn.compute_euclidean_distance('fsaverage', '10k', 'pial', 'L')
W_sphere = fn.inverse(dist_sphere, normalize=True)
W_pial = fn.inverse(dist_pial, normalize=True)

# Compute local Moran's I
lI_pial = fn.morans_i(W_pial, X, local=True)
lI_sphere = fn.morans_i(W_sphere, X, local=True)
lI_pial_spin = fn.morans_i(W_pial, X_spin, local=True)
lI_sphere_spin = fn.morans_i(W_sphere, X_spin, local=True)

# Setup the parameters
params = {'pial': [lI_pial, pial_mesh],
          'sphere': [lI_sphere, sphere_mesh],
          'sphere_spin': [lI_sphere_spin, sphere_mesh],
          'pial_spin': [lI_pial_spin, pial_mesh]}

# Plot maps
for name, (lI, mesh) in params.items():
    save_path = f'figures/figure_s1/local_I/{name}.png'
    m = max(abs(lI.min()), lI.max())
    fn.plot_surface_map(lI, mesh, cmap=RdBu_11_r.mpl_colormap, save=True,
                        save_path=save_path, view='yz_negative', clim=[-m, m])

'''
Plot the WP-WP' matrix
'''

# Compute WP-WP' matrix
W_pial_spin = W_pial[spin,:][:,spin]
W_delta = W_pial - W_pial_spin

# Plot matrix
vmin = 0 - 2 * np.std(W_delta)
vmax = 0 + 2 * np.std(W_delta)
fn.plot_matrix(W_delta, cmap=RdBu_11_r.mpl_colormap, vmin=vmin, vmax=vmax)
plt.savefig("figures/figure_s1/W_delta.svg", dpi=600)

'''
Plot the Moran scatterplot
'''

# Standardized values of X
zX = zscore(X)

# Compute weighted average in neighborhoods of X
wX = np.average(np.repeat(zX[np.newaxis], 10242, axis=0),
                weights=W_pial, axis=1)

# Compute weighted average in new neighborhoods of X (after spin)
wX_spin = np.average(np.repeat(zX[np.newaxis], 10242, axis=0),
                     weights=W_pial_spin, axis=1)

# Compute difference in weighted average in neighborhoods
wX_diff = wX - wX_spin

# Compute Moran's I after spin (X is fixed: weights are spun)
lI_spin_fixed = fn.morans_i(W_pial_spin, X, local=True)

# Compute differences in Moran's I
lI_diff = lI_pial - lI_spin_fixed

# Plot scatterplot
m = max(abs(lI_diff.min()), lI_diff.max())
fn.scatterplot(zX, wX_diff, c=lI_diff, cmap=RdBu_11_r.mpl_colormap,
               vmin=-m, vmax=m, plot_x_0=True, plot_y_0=True, s=2,
               figsize=(3, 3), ylabel=r'$\Delta$WX', xlabel='x', rasterized=True)

# Plot local differences in Moran's I
m = max(abs(lI_diff.min()), lI_diff.max())
save_path = 'figures/figure_s1/delta_local_I.png'
fn.plot_surface_map(lI_diff, pial_mesh, cmap=RdBu_11_r.mpl_colormap, save=True,
                    save_path=save_path, view='yz_negative', clim=[-m, m])

#%% FIGURE S2: Impact of distortions on autocorrelation of empirical brain maps

'''
Fetch brain maps
'''

brain_maps = fn.fetch_neuromaps_maps()

'''
Setup map names and categories
'''

map_names = ['gene PC1', '5HT1b', '5HT2a', '5HT1a', '5HTT', '5HT4',
             'MEG-alpha', 'MEG-beta','MEG-delta', 'MEG-gamma1', 'MEG-gamma2',
             'MEG-theta', 'MEG-timescale', 'myelin', 'thickness', 'FC-grad1',
             'intersubjcvar', 'GABAa', 'cbf', 'cbv', 'cmr02', 'cmrglc',
             'scaling-hcp', 'scaling-nih', 'scaling-pnc', 'SAaxis',
             'FC-homology', 'evoexp']

categories = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 1,
                       5, 5, 5, 5, 6, 6, 6, 4, 4, 6])

categories_labels = ['genetic', 'receptors', 'MEG', 'structural', 'functional',
                     'metabolism', 'expansion']

'''
Load standardized Moran's I
'''

n_maps = len(map_names)

I_all = np.zeros((n_maps))
I_spin_all = np.zeros((n_maps, 1000))
zI_all = np.zeros((n_maps))

for i, ((source, desc, _, _), map_file) in enumerate(brain_maps.items()):
    R = fn.load_pickle("results/neuromaps/standardized_I/"
                       f"{source}_{desc}.pickle")
    I_all[i] = R['I']
    I_spin_all[i] = R['I_spin']
    zI_all[i] = R['zI']

'''
Setup plotting parameters (category colors + ordering)
'''

# Setup colors
colors_fill = [to_rgba('#e1f4f5ff'), to_rgba('#fff3d7ff'), to_rgba('#fde4d9ff'),
               to_rgba('#f0ddf9ff'), to_rgba('#e3f1dbff'), to_rgba('#dbe5fbff'),
               to_rgba('#ffd7e5ff')]
categories_colors_fill = []
categories_colors = []
for i in range(n_maps):
    categories_colors.append(to_rgba(Pastel_7.hex_colors[categories[i]]))
    categories_colors_fill.append(colors_fill[categories[i]])

# Setup order (based on categories and Moran's I)
map_order = np.lexsort((I_all, categories))
I_o = I_all[map_order]
I_spin_o = I_spin_all[map_order]
zI_o = zI_all[map_order]
colors_fill_o = np.array(categories_colors_fill)[map_order]
colors_o = np.array(categories_colors)[map_order]

'''
Plot boxplot of Moran's I values
'''

fn.boxplot(I_spin_o, positions=np.arange(n_maps), figsize=(10.6, 2), showfliers=False,
           edge_colors=colors_o, face_colors=colors_fill_o)
plt.scatter(np.arange(n_maps), I_o, c=colors_o)
plt.xticks(np.arange(n_maps), np.array(map_names)[map_order], rotation=90)
plt.ylabel("Moran's I")
plt.savefig("figures/figure_s2/boxplot_I.svg")

'''
Plot barplot of standardized Moran's I values
'''

plt.figure(figsize=(10.6, 2))
plt.bar(np.arange(n_maps), zI_o, color=list(colors_fill_o), edgecolor=colors_o)
plt.xticks(np.arange(n_maps), np.array(map_names)[map_order], rotation=90)
plt.ylabel("standardized I")
plt.savefig("figures/figure_s2/barplot_z_I.svg")

#%% FIGURE S3: Quality of spins

'''
Load spins and compute distances
'''

# Load spin
spin = np.load("results/permutations.npy")[0]

# Compute distances
dist_pial = fn.compute_euclidean_distance('fsaverage', '10k', 'pial', 'L')
dist_sphere = fn.compute_euclidean_distance('fsaverage', '10k', 'sphere', 'L')
dist_sphere_spin = dist_sphere[spin,:][:,spin]
dist_pial_spin = dist_pial[spin,:][:,spin]

'''
Plot matrices
'''

params = {'dist_pial': dist_pial,
          'dist_sphere': dist_sphere,
          'dist_sphere_spin': dist_sphere_spin,
          'dist_pial_spin': dist_pial_spin}

for name, dist in params.items():
    plt.figure()
    plt.imshow(dist, cmap=SunsetDark_7.mpl_colormap)
    plt.savefig(f"figures/figure_s3/{name}.svg", dpi=300)

'''
Compute quality of spins
'''

triu_id = np.triu_indices(10242, 1)

print(pearsonr(dist_pial[triu_id], dist_sphere[triu_id]))
print(pearsonr(dist_sphere[triu_id], dist_sphere_spin[triu_id]))
print(pearsonr(dist_pial[triu_id], dist_pial_spin[triu_id]))

#%% FIGURE S4: Rotations quality associated with deviations in Moran’s I

# Load spin qualities
spin_qualities = np.load("results/spin_qualities.npy")[:1000]

# Compute average deviation in Moran's I for each spin
I_perm = np.load("results/I_perm/pial.npy")
I_emp = np.load("results/I/pial.npy")
dI_spin = np.abs(I_perm - I_emp[:, np.newaxis]).mean(axis=0)

# Plot figure
fn.scatterplot(spin_qualities, dI_spin, compute_r=True, s=1, figsize=(3, 3),
               compute_rho=True, r_round=4,
               ylabel=r"average deviation in Morans I", xlabel=r"$r^{PP'}$",
               rasterized=True, tight=True)
plt.savefig("figures/figure_s4/quality_I_scatterplot.svg", dpi=600)

#%% FIGURE S5: Distribution of rotation quality correlations

# Load spin qualities
spin_qualities = np.load("results/spin_qualities.npy")

# Find correlation values for each threshold
thr_all = np.arange(0, 91, 2.5)
nt = len(thr_all)
thr_r = [np.percentile(spin_qualities, t) for t in thr_all]

# Plot distribution
fig, ax = plt.subplots(figsize=(6, 3))
sns.distplot(spin_qualities, color='darkgray', ax=ax, kde=False)
colors = fn.get_color_distribution(thr_all, cmap=SunsetDark_7.mpl_colormap)
ymin, ymax = ax.get_ylim()
for i in range(nt):
    plt.plot([thr_r[i], thr_r[i]], [ymin, ymax],
             color=colors[i], linestyle='dashed', linewidth=1)
plt.xlabel("alignment")
plt.savefig("figures/figure_s5/quality_distribution.svg")

#%% FIGURE S6: Gaussian variogram models

# Load distance matrix
dist = fn.compute_euclidean_distance('fsaverage', '10k', 'pial', 'L')
lengths = [1, 10, 20, 30, 40, 50]

# Compute theoretical variograms
variogram_values = []
for l in tqdm(lengths, total=len(lengths)):
    var = fn.variogram_function(dist, length=l)
    mean_bins, bins, bin_centers = fn.get_mean_in_bins(
        dist, var, n_bins=1000, triu=True)
    variogram_values.append(mean_bins)

colors = fn.get_color_distribution(lengths, cmap=agSunset_7_r.mpl_colormap)
fn.lineplot(bin_centers, variogram_values, colors=colors, labels=lengths,
            figsize=(4, 3), xlabel='distance', ylabel='semivariance', tight=True)

plt.savefig("figures/figure_s6/variograms.svg")
