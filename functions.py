# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:16:54 2025

This file contains function used to load the required data, run the analyses
and plot the figures presented in 'The effect of spherical projection on spin
tests for brain maps'.

@author: Vincent Bazinet
"""

import meshio
import nibabel as nib
import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from neuromaps.datasets import fetch_atlas
from tqdm import trange


'''
RESULTS FUNCTIONS
'''

def evaluate_spin_quality(dist, z_dist_triu, triu_ids, perm):
    '''
    Function to evalute the quality of a spin but computing the correlation
    between the original and permuted distance matrix.

    Parameters
    ----------
    dist: (n, n) ndarray
        Original distance matrix between vertices on the surface mesh.
    z_dist_triu: (k,) ndarray
        Standardized upper-triangular values in the orginal distance matrix
    triu_ids: (n, n) boolean array
        Upper triangular indices.
    perm: (n) ndarray
        Permutation of the vertices

    Returns
    -------
    rotated_r: float
        Pearson correlation between the original and permuted distance matrix
    '''

    dist_perm = dist[perm, :][:, perm]
    z_dist_triu_perm = zscore(dist_perm[triu_ids])
    rotated_r = (z_dist_triu_perm * z_dist_triu).mean()

    return rotated_r


def compute_euclidean_distance(atlas, density, surface, hemi='L'):
    '''
    Function to compute the euclidean distance matrix for a given brain
    surface mesh.

    Parameters
    ----------

    Returns
    -------
    '''

    vertices = get_vertices(atlas, density, surface, 'L')
    dist = cdist(vertices, vertices)

    return dist


def morans_i(dist, y, normalize=False, local=False, invert_dist=False):
    """
    Calculates Moran's I from distance matrix `dist` and brain map `y`

    Parameters
    ----------
    dist : (N, N) array_like
        Distance matrix between `N` regions / vertices / voxels / whatever
    y : (N,) array_like
        Brain map variable of interest
    normalize : bool, optional
        Whether to normalize rows of distance matrix prior to calculation.
        Default: False
    local : bool, optional
        Whether to calculate local Moran's I instead of global. Default: False
    invert_dist : bool, optional
        Whether to invert the distance matrix to generate a weight matrix.
        Default: True

    Returns
    -------
    i : float
        Moran's I, measure of spatial autocorrelation
    """

    # convert distance matrix to weights
    if invert_dist:
        with np.errstate(divide='ignore'):
            dist = 1 / dist
    np.fill_diagonal(dist, 0)

    # normalize rows, if desired
    if normalize:
        dist /= dist.sum(axis=-1, keepdims=True)

    # calculate Moran's I
    z = y - y.mean()
    if local:
        with np.errstate(all='ignore'):
            z /= y.std()

    zl = np.squeeze(dist @ z[:, None])
    den = (z * z).sum()

    if local:
        return (len(y) - 1) * z * zl / den

    return len(y) / dist.sum() * (z * zl).sum() / den


'''
UTILITY FUNCTIONS
'''


def load_mesh(atlas, density, surface, hemi='L'):
    '''
    Function to load a surface mesh of the brain, in the format defined by
    `meshio`.

    Parameters
    ----------

    Returns
    -------
    mesh: meshio.Mesh object
        Surface mesh of the brain.

    '''

    if hemi == 'L':
        hemiid = 0
    elif hemi == 'R':
        hemiid = 1
    gii_mesh = nib.load(fetch_atlas(atlas, density)[surface][hemiid])
    points, triangles = gii_mesh.agg_data()
    mesh = meshio.Mesh(points, {'triangle': triangles})

    return mesh


def get_vertices(atlas, density, surface, hemi='L'):
    '''
    Function to load the coordinates of vertices on a surface mesh.

    Parameters
    ----------

    Returns
    -------
    vertices: (n,) ndarray
        Coordinate for each of the `n` vertices on the mesh.

    '''

    if hemi == 'L':
        hemiid = 0
    elif hemi == 'R':
        hemiid = 1
    gii_mesh = nib.load(fetch_atlas(atlas, density)[surface][hemiid])
    vertices, _ = gii_mesh.agg_data()

    return vertices


def inverse(A, k=1, normalize=False):
    '''
    Function that returns the original matrix with the inverse values at
    non-zero indices.

    Parameters
    ----------

    Returns
    -------
    '''

    w = A.copy()
    w[A > 0] = 1/(A[A > 0])
    w = w**k
    if normalize:
        w /= w.sum(axis=-1, keepdims=True)

    return w


def sample_subset_permutations(quantiles, quality, n_perm=1000):
    '''
    Function to sample a subset of permutations (based on the quality of the
    spins)

    Parameters
    ----------

    Returns
    -------
    '''

    # Find IDs of spins (based on thresholds)
    n_q = len(quantiles)
    quantiles_ids = [quality > np.percentile(quality, q) for q in quantiles]
    quantiles_spins = np.unique(
        [np.where(quantiles_ids[i])[0][:n_perm] for i in range(n_q)])

    # Assign threshold to each ID of spins
    n_perm_quantiles = len(quantiles_spins)
    quantiles_nb = np.zeros((n_perm_quantiles))
    quantiles_nb[:] = np.nan
    for i in trange(n_q):
        for j in range(n_perm_quantiles):
            if quantiles_spins[j] in np.where(quantiles_ids[i])[0][:n_perm]:
                quantiles_nb[j] = quantiles[i]

    return quantiles_spins, quantiles_nb


def get_p_value(perm, emp, axis=0):
    '''
    Utility function to compute the p-value (two-tailed) of a score, relative
    to a null distribution.

    Parameters
    ----------
    perm: array-like
        Null distribution of (permuted) scores.
    emp: float or array-like
        Empirical score.
    axis: float
        Axis of the `perm` array associated with the null scores.
    '''

    k = perm.shape[axis]
    perm_moved = np.moveaxis(perm, axis, 0)
    perm_mean = np.mean(perm_moved, axis=0)

    # Compute p-value
    pval = np.count_nonzero(abs(perm_moved-perm_mean) > abs(emp-perm_mean),
                            axis=0)/k

    return pval
