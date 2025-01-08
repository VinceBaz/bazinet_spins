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
from neuromaps.datasets import fetch_atlas

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