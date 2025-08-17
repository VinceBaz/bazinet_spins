# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:16:54 2025

This file contains function used to load the required data, run the analyses
and plot the figures presented in 'The effect of spherical projection on spin
tests for brain maps'.

@author: Vincent Bazinet
"""

import pickle
import meshio
import nibabel as nib
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
from matplotlib import cm
from scipy.stats import zscore, pearsonr, rankdata, mstats
from scipy.spatial.distance import cdist
from matplotlib.colors import is_color_like, rgb2hex
from neuromaps.datasets import fetch_atlas
from neuromaps.points import get_surface_distance
from neuromaps.datasets import fetch_annotation


'''
RESULTS FUNCTIONS
'''


def evaluate_spin_quality(dist, z_dist_triu, triu_ids, perm):
    '''
    Evaluate the quality of a spin.

    This function efficiently evaluates the quality of a spin by computing the
    correlation between the upper triangular values of the original and
    permuted distance matrix.

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
    atlas: str
        Atlas to fetch. Available options are: 'civet', 'fsaverage', 'fsLR' or
        'MNI152'.
    density: str
        Density of the fetched atlas.
    surface: str
        Surface of the fetched atlas.
    hemi: str
        Hemisphere of the fetched atlas

    Returns
    -------
    dist: (n, n) ndarray
        Distance matrix capturing the euclidean distance between each vertices
        of a brain surface mesh.
    '''

    vertices = get_vertices(atlas, density, surface, hemi)
    dist = cdist(vertices, vertices)

    return dist


def compute_geodesic_distance(atlas, density, surface, hemi='L'):
    '''
    Function to compute the geodesic distance matrix for a given brain surface
    mesh.

    Parameters
    ----------
    atlas: str
        Atlas to fetch. Available options are: 'civet', 'fsaverage', 'fsLR' or
        'MNI152'.
    density: str
        Density of the fetched atlas.
    surface: str
        Surface of the fetched atlas.
    hemi: str
        Hemisphere of the fetched atlas.

    Returns
    -------
    dist: (n, n) ndarray
        Distance matrix capturing the geodesic distance between each vertices
        of a brain surface mesh.
    '''

    if hemi == 'L':
        hemiid = 0
    elif hemi == 'R':
        hemiid = 1

    mesh = fetch_atlas(atlas, density)[surface][hemiid]
    dist = get_surface_distance(mesh)

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


def variogram_function(dist, model_name='gaussian', length=50):
    '''
    Compute the variogram (for a specified model) of a distance matrix.

    Parameters
    ----------
    dist: (n, n) ndarray
        Matrix capturing the distance relationship between individual vertices
        of a surface mesh
    model_name: str
        Name of the variogram model
    length: int
        Length parameter of the variogram model

    Returns
    -------
    var: (n, n) ndarray
        Matrix capturing the variogram value of each distance values in the
        `dist` matrix.
    '''
    dist = np.abs(dist)

    if model_name == 'gaussian':
        cov = covariance_function(dist, model_name, length)
        var = 1-cov

    return var


def covariance_function(dist, model_name='gaussian', length=50):
    '''
    Compute the covariance (for a specific model) of a distance matrix.

    Parameters
    ----------
    dist: (n, n) ndarray
        Matrix capturing the distance relationship between individual vertices
        of a surface mesh
    model_name: str
        Name of the variogram model
    length: int
        Length parameter of the variogram model

    Returns
    -------
    cov: (n, n) ndarray
        Matrix capturing the covariance value of each distance values in the
        `dist` matrix.
    '''
    dist = np.abs(dist)

    if model_name == 'gaussian':
        s = np.sqrt(np.pi)/2
        cov = np.exp(-((s * dist)/length)**2)

    return cov


'''
UTILITY FUNCTIONS
'''


def load_mesh(atlas, density, surface, hemi='L', data_format='meshio'):
    '''
    Function to load a surface mesh of the brain, either in the format defined
    by `meshio` or in the format defined by pyvista (`PolyData`).

    Parameters
    ----------
    atlas: str
        Atlas to fetch. Available options are: 'civet', 'fsaverage', 'fsLR' or
        'MNI152'.
    density: str
        Density of the fetched atlas.
    surface: str
        Surface of the fetched atlas.
    hemi: str
        Hemisphere of the fetched atlas
    data_format: str
        Format of the mesh data returned.

    Returns
    -------
    mesh: meshio.Mesh or pv.PolyData object
        Surface mesh of the brain.

    '''
    if hemi == 'L':
        hemiid = 0
    elif hemi == 'R':
        hemiid = 1
    gii_mesh = nib.load(fetch_atlas(atlas, density)[surface][hemiid])
    points, triangles = gii_mesh.agg_data()

    if data_format == 'meshio':
        mesh = meshio.Mesh(points, {'triangle': triangles})
    elif data_format == 'polydata':
        mesh = pv.PolyData(
            points,
            np.c_[np.ones((triangles.shape[0],), dtype=int)*3, triangles]
            )

    return mesh


def fetch_neuromaps_maps():
    '''
    Fetch the maps from neuromaps used in the manuscript.

    The function fetches maps from neuromaps that were originally published
    in a surface space. We also ignore the maps from hill2010 since they are
    only available on the right hemisphere. We also ignore the fc gradient
    maps other than the first one.

    Returns:
    -------
    brain_maps: dict
        Dictionary of brain maps. The keys of the dictionaries are tuples
        of the parameters of each map (source, desc, space, den) and the
        values correspond to the path to each downloaded files.
    '''

    # Fetch brain maps originally published in a surface space
    brain_maps = fetch_annotation(space=['fsaverage', 'fsLR', 'civet'])

    # Remove fc gradient maps + hill2010 (only available on right hemisphere)
    del brain_maps[('hill2010', 'devexp', 'fsLR', '164k')]
    del brain_maps[('hill2010', 'evoexp', 'fsLR', '164k')]
    del brain_maps[('margulies2016', 'fcgradient10', 'fsLR', '32k')]
    del brain_maps[('margulies2016', 'fcgradient09', 'fsLR', '32k')]
    del brain_maps[('margulies2016', 'fcgradient08', 'fsLR', '32k')]
    del brain_maps[('margulies2016', 'fcgradient07', 'fsLR', '32k')]
    del brain_maps[('margulies2016', 'fcgradient06', 'fsLR', '32k')]
    del brain_maps[('margulies2016', 'fcgradient05', 'fsLR', '32k')]
    del brain_maps[('margulies2016', 'fcgradient04', 'fsLR', '32k')]
    del brain_maps[('margulies2016', 'fcgradient03', 'fsLR', '32k')]
    del brain_maps[('margulies2016', 'fcgradient02', 'fsLR', '32k')]

    return brain_maps


def get_vertices(atlas, density, surface, hemi='L'):
    '''
    Function to load the coordinates of vertices on a surface mesh.

    Parameters
    ----------
    atlas: str
        Atlas to fetch. Available options are: 'civet', 'fsaverage', 'fsLR' or
        'MNI152'.
    density: str
        Density of the fetched atlas.
    surface: str
        Surface of the fetched atlas.
    hemi: str
        Hemisphere of the fetched atlas

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
    A : (n, n) ndarray
        Matrix for which we want to compute inverse values.
    k : float
        Exponent to be applied to the matrix values before computing the
        inverse.
    normalize: bool
        If `True`, then each row of the inverse matrix will be normalize such
        that the values in the row sum to 1.

    Returns
    -------
    w: (n, n) ndarray
        Matrix where each non-zero entries correspond to the inverse of the
        non-zero entries in matrix `A`
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
    quantiles: (n_thres,) array-like
        Quantile values (thresholds) at which we want to sample a subset of
        permutations
    quality: (n_perm,) array-like
        Array of quality values quantifying the quality of each spin
    n_perm: int
        Number of samples to be sampled from each thresholded distributions.

    Returns
    -------
    quantiles_spins: array-like
        Array of indices associated with a permutation (from the `quality`
        array) that have been sampled in one of the subsets.
    quantiles_nb: array-like
        Quantile "bin" for which the specific spin has been sampled.
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

    Returns
    -------
    p-value: float or array-like
        p-values of each empirical score.
    '''

    k = perm.shape[axis]
    perm_moved = np.moveaxis(perm, axis, 0)
    perm_mean = np.mean(perm_moved, axis=0)

    # Compute p-value
    pval = np.count_nonzero(abs(perm_moved-perm_mean) > abs(emp-perm_mean),
                            axis=0)/k

    return pval


def standardize_scores(surr, emp, axis=None, ignore_nan=False):
    '''
    Utility function to standardize scores relative to a null distribution.

    Parameters
    ----------
    perm: array-like
        Null distribution of scores.
    emp: float or array-like
        Empirical scores.
    axis: int
        Axis along which perm scores are distributed.
    ignore_nan: bool
        If True, then nan values are ignored when standardizing scores

    Returns
    -------
    z_emp: float or array-like
        Empirical scores standardized relative to null distribution of scores.
    '''

    if ignore_nan:
        return (emp - np.nanmean(surr, axis=axis)) / np.nanstd(surr, axis=axis)
    else:
        return (emp - np.mean(surr, axis=axis)) / np.std(surr, axis=axis)


def load_pickle(path):
    '''
    Load pickled data.

    Parameters
    ----------
    path: path-like
        Path to the pickled file to be loaded

    Returns
    -------
    data: object
        Loaded data
    '''

    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    return data


def save_pickle(data, path):
    '''
    Save data as a `.pickle` file

    Parameters
    ----------
    data: object
        Data to be pickled
    file: path-like
        Filename where the pickled data will be saved
    '''
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_mean_in_bins(X, Y, bins=None, n_bins=10, bin_type='quantiles',
                     triu=False, ignore_empty_bins=True):
    '''
    Function to create bins using an independent variable `X` and compute the
    mean scores of a variable `Y` within those bins.

    Parameters
    ----------
    X: (n,) or (n, n) array-like
        Values of the independent variable used to compute the bins
    Y: (n,) or (n, n) array-like
        Values of the dependent variables for which we want binned means.
    bins: (n_bins+1,)
        Array specifying the bin edges. If set to `None` (default), the bin
        edges are automatically calculated.
    n_bins: int
        Number of bins to use.
    bin_type: str
        The types of bins used in the function. Available options are
        `quantiles` (equal number of edges in each bin) or `histogram` (equal
        width between each bin).
    triu: Bool
        If `True`, compute the mean in each bin for the upper triangular values
        of a 2-dimensional matrix.
    ignore_empty_bins: bool
        If `True`, empty bins are ignored

    Returns
    -------
    mean_bins: (n_bins) ndarray
        Mean values of X in each bin
    bins: (n_bins+1) ndarray
        Bin edges
    bin_centers: (n_bins) ndarray
        Center of each bin.
    '''

    if triu:
        triu_indices = np.triu_indices(len(X), 1)
        X = X[triu_indices]
        Y = Y[triu_indices]

    X_labels, bin_centers, bins = get_bins_labels(X, n_bins=n_bins, triu=False,
                                                  bin_type=bin_type, bins=bins)
    mean_bins = mean_in_group(X_labels, Y, ignore_0=ignore_empty_bins)

    return mean_bins, bins, bin_centers


def get_bins_labels(X, n_bins=50, triu=True, bin_type='histogram', bins=None):
    '''
    Get the bin index (label) associated with each value in an array `X`.
    '''
    if triu:
        triu_indices = np.triu_indices(len(X), 1)
        X = X[triu_indices]

    if bins is None:
        if bin_type == "quantiles":
            bins = mstats.mquantiles(X, np.linspace(0, 1, n_bins+1))
        elif bin_type == 'histogram':
            bins = np.linspace(X.min(), X.max(), n_bins+1)

    bin_labels = np.digitize(X, bins) - 1
    bin_labels[X == bins[n_bins]] = n_bins - 1

    bin_centers = (bins[:-1] + bins[1:]) / 2

    return bin_labels, bin_centers, bins


def mean_in_group(X, Y, ignore_0=True):
    '''
    Get the mean of X, grouped by integer values indexed in Y.
    '''

    sums = np.bincount(X, weights=Y)
    counts = np.bincount(X)

    if ignore_0:
        return sums[sums > 0] / counts[counts > 0]
    else:
        return sums / counts


'''
VISUALIZATION FUNCTIONS
'''


def plot_surface_map(map, mesh, cmap='viridis', save=False, save_path=None,
                     view='default', clim=None):
    '''
    Plot a brain map (values) on a brain surface mesh using Pyvista.s

    Parameters
    ----------
    map: (n_vertices) array-like
        Brain map values to plot on the surface mesh
    mesh: pv.PolyData object
        Surface mesh in `pv.PolyData` format
    '''
    pl = pv.Plotter(window_size=(1000, 1000), lighting="none", off_screen=True)
    mesh.point_data['map'] = map
    pl.add_mesh(mesh, scalars='map', cmap=cmap, clim=clim)
    pl.remove_scalar_bar()
    if view == 'yz_negative':
        pl.view_yz(negative=True)
    pl.show(auto_close=False)

    if save:
        plt.ioff()
        plt.figure()
        plt.imshow(pl.image)
        plt.axis('off')
        plt.savefig(save_path, dpi=600)
        plt.close('all')
        plt.ion()


def boxplot(results, figsize=(2, 3), widths=0.8, showfliers=True,
            edge_colors='black', face_colors='lightgray',
            median_color=None, significants=None, positions=None, vert=True,
            ax=None, xlabel=None, ylabel=None, xticks=None, yticks=None,
            tight=False):
    '''
    Function to plot results in a boxplot

    Parameters
    ----------
    results: (n_boxes, n_observations) ndarray
        Results to be plotted in the boxplot

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure associated with the boxplot
    '''

    # Setup the flierprops dictionary
    flierprops = dict(marker='+',
                      markerfacecolor='lightgray',
                      markeredgecolor='lightgray')

    # Initialize the figure (if no `ax` provided)
    if ax is None:
        fig = plt.figure(figsize=figsize, frameon=False)
        ax = plt.gca()
    else:
        fig = plt.gcf()

    n_boxes = len(results)

    # Add axis labels (optional)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Setup default positions
    if positions is None:
        positions = np.arange(1, n_boxes + 1)

    if is_color_like(edge_colors):
        edge_colors = [edge_colors] * n_boxes
    if is_color_like(face_colors):
        face_colors = [face_colors] * n_boxes

    # Plot each box individually
    for i in range(n_boxes):

        bplot = ax.boxplot(results[i],
                           widths=widths,
                           showfliers=showfliers,
                           patch_artist=True,
                           zorder=0,
                           flierprops=flierprops,
                           showcaps=False,
                           vert=vert,
                           positions=[positions[i]])

        for element in ['boxes', 'whiskers', 'fliers',
                        'means', 'medians', 'caps']:
            if element == 'medians' and median_color is not None:
                plt.setp(bplot[element], color=median_color)
            else:
                plt.setp(bplot[element], color=edge_colors[i])

        for patch in bplot['boxes']:

            if significants is not None:
                if significants[i]:
                    patch.set(facecolor=face_colors[i])
                else:
                    patch.set(facecolor='white')
            else:
                patch.set(facecolor=face_colors[i])

    # Add axis ticks (optional)
    if xticks is not None:
        plt.xticks(np.arange(1, len(xticks)+1), xticks)
    if yticks is not None:
        plt.yticks(np.arange(1, len(yticks)+1), yticks)

    if tight:
        plt.tight_layout()

    return fig


def scatterplot(X, Y, triu=False, tight=False, figsize=None, c='black',
                xlabel=None, ylabel=None, xscale='linear',
                plot_y_mean=False, plot_x_mean=False, plot_identity=False,
                compute_r=False, compute_rho=False, r_round=None,
                r_title="r: ", rho_title='rho: ', plot_cbar=False,
                cbar_label='', ma_width=9, plot_x_0=False,
                plot_y_0=False, **kwargs):
    ''' Wrapper function to plot a scatterplot (using matplotlib).'''

    # Only look at upper triangular indices
    if triu:
        X = X[np.triu_indices(len(X), 1)]
        Y = Y[np.triu_indices(len(Y), 1)]
        if isinstance(c, np.ndarray):
            c = c[np.triu_indices(len(c), 1)]

    plt.figure(figsize=figsize)
    plt.scatter(X, Y, c=c, **kwargs)

    if compute_r and compute_rho:
        r, _ = pearsonr(X, Y)
        rho, _ = pearsonr(rankdata(X), rankdata(Y))
        if r_round is None:
            plt.title(f"{r_title}{r} | {rho_title}{rho}")
        else:
            plt.title(f"{r_title}{round(r, r_round)} | "
                      f"{rho_title}{round(rho, r_round)}")
    elif compute_r:
        r, _ = pearsonr(X, Y)
        if r_round is None:
            plt.title(f"{r_title}{r}")
        else:
            plt.title(f"{r_title}{round(r, r_round)}")
    elif compute_rho:
        rho, _ = pearsonr(rankdata(X), rankdata(Y))
        if r_round is None:
            plt.title(f"{rho_title}{rho}")
        else:
            plt.title(f"{rho_title}{round(rho, r_round)}")

    if plot_y_mean:
        plt.plot([X.min(), X.max()], [Y.mean(), Y.mean()],
                 color='lightgray',
                 linestyle='dashed')
    if plot_x_mean:
        plt.plot([X.mean(), X.mean()], [Y.min(), Y.max()],
                 color='lightgray',
                 linestyle='dashed')
    if plot_identity:
        plt.plot([X.min(), X.max()], [X.min(), X.max()],
                 color='lightgray',
                 linestyle='dashed')
    if plot_y_0:
        plt.plot([X.min(), X.max()], [0, 0],
                 color='lightgray',
                 linestyle='dashed')
    if plot_x_0:
        plt.plot([0, 0], [Y.min(), Y.max()],
                 color='lightgray',
                 linestyle='dashed')

    # Change x/y labels if not None (if None, leave as is)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.xscale(xscale)

    if plot_cbar:
        cbar = plt.colorbar()
        cbar.set_label(cbar_label)

    if tight:
        plt.tight_layout()


def lineplot(X, Y, figsize=None, xlabel=None, ylabel=None, colors=None,
             labels=None, xscale='linear', tight=False, **kwargs):
    '''
    Wrapper function to plot a line plot (using matplotlib)

    Parameters
    ----------
    Y: (n_lines, n_observation) array-like
        Lines to plot in the figure. Each row correspond to a specific line.
    '''

    Y = np.atleast_2d(Y)

    plt.figure(figsize=figsize)
    for i, line in enumerate(Y):
        if colors is not None:
            kwargs['color'] = colors[i]
        if labels is not None:
            label = labels[i]
        else:
            label = None
        plt.plot(X, line, label=label, **kwargs)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.xscale(xscale)

    if labels is not None:
        plt.legend()

    if tight:
        plt.tight_layout()


def get_color_distribution(values, cmap="viridis", vmin=None, vmax=None,
                           default_color='black', color_format='rgba'):
    '''
    Function to get a color for individual values of a distribution.
    '''

    values = np.asarray(values)

    if values.min() == values.max():
        c = np.full((len(values)), default_color, dtype="<U10")
    else:
        c = cm.get_cmap(cmap)(mpl.colors.Normalize(vmin, vmax)(values))
        if color_format == 'hex':
            c_hex = []
            for i in range(len(c)):
                c_hex.append(rgb2hex(c[i, :], keep_alpha=True))
            c = c_hex

    return c


def plot_matrix(X, figsize=None, save_path=None, dpi=300, colorbar=False,
                xticks=None, yticks=None, auto_locator=True, round_ticks=True,
                n_decimals=2, cbar_label='', vmin=None, vmax=None, xlabel=None,
                ylabel=None, **kwargs):
    ''' Wrapper function to plot a matrix (using matplotlib).'''

    if vmin is None:
        vmin = np.nanmin(X)
    if vmax is None:
        vmax = np.nanmax(X)

    plt.figure(figsize=figsize)
    plt.imshow(X, vmin=vmin, vmax=vmax, **kwargs)

    # Set colorbar (optional)
    if colorbar:
        cbar = plt.colorbar()
        cbar.set_label(cbar_label, rotation=90)
        cbar.set_ticks([vmin, vmax])

    # Set xticks (optional)
    if xticks is not None:
        if round_ticks:
            xticks = np.round(xticks, n_decimals)
        plt.xticks(np.arange(len(xticks)), xticks)
        if auto_locator:
            plt.gca().xaxis.set_major_locator(plt.AutoLocator())

    # Set yticks (optional)
    if yticks is not None:
        if round_ticks:
            yticks = np.round(yticks, n_decimals)
        plt.yticks(np.arange(len(yticks)), yticks)
        if auto_locator:
            plt.gca().yaxis.set_major_locator(plt.AutoLocator())

    # Add axis labels (optional)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Save figure (optional)
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
