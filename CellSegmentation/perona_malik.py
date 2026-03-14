#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:46:42 2024

@author: lucas
"""

import numpy as np
from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
from TissueSegmentation.functionPercNorm import functionPercNorm
from skimage import exposure

def isotropic_diffusion_3d(volume, volume_sample, factor_conduction = 100, num_iter=10, delta_t=1/7, kappa=1, option=2):
    """
    isotropic_diffusion_3d.
    
    Parameters:
    image (ndarray): 3D input image.
    num_iter (int): Number of iterations.
    delta_t (float): Integration constant (usually <= 1/7).
    kappa (float): Conduction coefficient, controls sensitivity to edges.
    option (int): 1 for the first diffusion equation, 2 for the second.
    
    Returns:
    ndarray: Diffused image.
    """
    # Initialize output array
    diffused_image = np.copy(volume).astype(np.float32)
    
    # # Do not conduct through background
    # conduction_north = np.pad(volume_sample[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode='edge') - volume_sample
    # conduction_south = np.pad(volume_sample[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode='edge') - volume_sample
    # conduction_east = np.pad(volume_sample[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='edge') - volume_sample
    # conduction_west = np.pad(volume_sample[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='edge') - volume_sample
    # conduction_up = np.pad(volume_sample[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='edge') - volume_sample
    # conduction_down = np.pad(volume_sample[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='edge') - volume_sample
    
    conduction = np.ones_like(volume)
    
    conduction_north    = conduction * factor_conduction
    conduction_south    = conduction * factor_conduction
    conduction_east     = conduction * factor_conduction
    conduction_west     = conduction * factor_conduction
    conduction_up       = conduction * factor_conduction
    conduction_down     = conduction * factor_conduction
    
    del conduction
    
    for _ in range(num_iter):
        # Compute gradients in x, y, z directions
        grad_north = np.pad(diffused_image[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode='edge') - diffused_image
        grad_south = np.pad(diffused_image[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode='edge') - diffused_image
        grad_east = np.pad(diffused_image[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='edge') - diffused_image
        grad_west = np.pad(diffused_image[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='edge') - diffused_image
        grad_up = np.pad(diffused_image[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='edge') - diffused_image
        grad_down = np.pad(diffused_image[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='edge') - diffused_image
    
        # Compute conductance
        if option == 1:
            c_north = np.exp(-(conduction_north / kappa) ** 2)
            c_south = np.exp(-(conduction_south / kappa) ** 2)
            c_east = np.exp(-(conduction_east / kappa) ** 2)
            c_west = np.exp(-(conduction_west / kappa) ** 2)
            c_up = np.exp(-(conduction_up / kappa) ** 2)
            c_down = np.exp(-(conduction_down / kappa) ** 2)
        elif option == 2:
            c_north = 1. / (1. + (conduction_north / kappa) ** 2)
            c_south = 1. / (1. + (conduction_south / kappa) ** 2)
            c_east = 1. / (1. + (conduction_east / kappa) ** 2)
            c_west = 1. / (1. + (conduction_west / kappa) ** 2)
            c_up = 1. / (1. + (conduction_up / kappa) ** 2)
            c_down = 1. / (1. + (conduction_down / kappa) ** 2)
            
        c_north = 1.
        c_south = 1.
        c_east = 1.
        c_west = 1.
        c_up = 1.
        c_down = 1.
        # Update the image
        diffused_image += delta_t * (
            c_north * grad_north + c_south * grad_south +
            c_east * grad_east + c_west * grad_west +
            c_up * grad_up + c_down * grad_down
        )
        del grad_north, grad_south, grad_east, grad_west, grad_up, grad_down
        del c_north, c_south, c_east, c_west, c_up, c_down
    
    return diffused_image

def perona_malik_3d_no_edge(volume, volume_sample, factor_conduction = 100, num_iter=10, delta_t=1/7, kappa=5, option=2):
    """
    Perform Perona-Malik anisotropic diffusion on a 3D image.
    
    Parameters:
    image (ndarray): 3D input image.
    num_iter (int): Number of iterations.
    delta_t (float): Integration constant (usually <= 1/7).
    kappa (float): Conduction coefficient, controls sensitivity to edges.
    option (int): 1 for the first diffusion equation, 2 for the second.
    
    Returns:
    ndarray: Diffused image.
    """
    # Initialize output array
    diffused_image = np.copy(volume).astype(np.float32)
    
    conduction = (volume_sample == 0) * factor_conduction
    conduction_north = np.pad(conduction[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode='edge')
    conduction_south = np.pad(conduction[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode='edge')
    conduction_east = np.pad(conduction[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='edge')
    conduction_west = np.pad(conduction[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='edge')
    conduction_up = np.pad(conduction[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='edge')
    conduction_down = np.pad(conduction[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='edge')
    
    del conduction
    
    for _ in range(num_iter):
        # Compute gradients in x, y, z directions
        grad_north = np.pad(diffused_image[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode='edge') - diffused_image
        grad_south = np.pad(diffused_image[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode='edge') - diffused_image
        grad_east = np.pad(diffused_image[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='edge') - diffused_image
        grad_west = np.pad(diffused_image[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='edge') - diffused_image
        grad_up = np.pad(diffused_image[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='edge') - diffused_image
        grad_down = np.pad(diffused_image[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='edge') - diffused_image
    
        # Compute conductance
        if option == 1:
            c_north = np.exp(-(conduction_north / kappa) ** 2)
            c_south = np.exp(-(conduction_south / kappa) ** 2)
            c_east = np.exp(-(conduction_east / kappa) ** 2)
            c_west = np.exp(-(conduction_west / kappa) ** 2)
            c_up = np.exp(-(conduction_up / kappa) ** 2)
            c_down = np.exp(-(conduction_down / kappa) ** 2)
        elif option == 2:
            c_north = 1. / (1. + (conduction_north / kappa) ** 2)
            c_south = 1. / (1. + (conduction_south / kappa) ** 2)
            c_east = 1. / (1. + (conduction_east / kappa) ** 2)
            c_west = 1. / (1. + (conduction_west / kappa) ** 2)
            c_up = 1. / (1. + (conduction_up / kappa) ** 2)
            c_down = 1. / (1. + (conduction_down / kappa) ** 2)
    
        # Update the image
        diffused_image += delta_t * (
            c_north * grad_north + c_south * grad_south +
            c_east * grad_east + c_west * grad_west +
            c_up * grad_up + c_down * grad_down
        )
        del grad_north, grad_south, grad_east, grad_west, grad_up, grad_down
        del c_north, c_south, c_east, c_west, c_up, c_down
    
    return diffused_image

def perona_malik_3d(volume, volume_sample, factor_conduction = 100, num_iter=10, delta_t=1/7, kappa=5, option=2):
    """
    Perform Perona-Malik anisotropic diffusion on a 3D image.
    
    Parameters:
    image (ndarray): 3D input image.
    num_iter (int): Number of iterations.
    delta_t (float): Integration constant (usually <= 1/7).
    kappa (float): Conduction coefficient, controls sensitivity to edges.
    option (int): 1 for the first diffusion equation, 2 for the second.
    
    Returns:
    ndarray: Diffused image.
    """
    # Initialize output array
    diffused_image = np.copy(volume).astype(np.float32)
    
    # Do not conduct through background
    conduction_north = np.pad(volume_sample[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode='edge') - volume_sample
    conduction_south = np.pad(volume_sample[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode='edge') - volume_sample
    conduction_east = np.pad(volume_sample[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='edge') - volume_sample
    conduction_west = np.pad(volume_sample[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='edge') - volume_sample
    conduction_up = np.pad(volume_sample[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='edge') - volume_sample
    conduction_down = np.pad(volume_sample[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='edge') - volume_sample
    
    conduction_north = conduction_north * factor_conduction
    conduction_south = conduction_south * factor_conduction
    conduction_east = conduction_east * factor_conduction
    conduction_west = conduction_west * factor_conduction
    conduction_up = conduction_up * factor_conduction
    conduction_down = conduction_down * factor_conduction
    
    for _ in range(num_iter):
        # Compute gradients in x, y, z directions
        grad_north = np.pad(diffused_image[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode='edge') - diffused_image
        grad_south = np.pad(diffused_image[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode='edge') - diffused_image
        grad_east = np.pad(diffused_image[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='edge') - diffused_image
        grad_west = np.pad(diffused_image[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='edge') - diffused_image
        grad_up = np.pad(diffused_image[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='edge') - diffused_image
        grad_down = np.pad(diffused_image[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='edge') - diffused_image
    
        # Compute conductance
        if option == 1:
            c_north = np.exp(-(conduction_north / kappa) ** 2)
            c_south = np.exp(-(conduction_south / kappa) ** 2)
            c_east = np.exp(-(conduction_east / kappa) ** 2)
            c_west = np.exp(-(conduction_west / kappa) ** 2)
            c_up = np.exp(-(conduction_up / kappa) ** 2)
            c_down = np.exp(-(conduction_down / kappa) ** 2)
        elif option == 2:
            c_north = 1. / (1. + (conduction_north / kappa) ** 2)
            c_south = 1. / (1. + (conduction_south / kappa) ** 2)
            c_east = 1. / (1. + (conduction_east / kappa) ** 2)
            c_west = 1. / (1. + (conduction_west / kappa) ** 2)
            c_up = 1. / (1. + (conduction_up / kappa) ** 2)
            c_down = 1. / (1. + (conduction_down / kappa) ** 2)
    
        # Update the image
        diffused_image += delta_t * (
            c_north * grad_north + c_south * grad_south +
            c_east * grad_east + c_west * grad_west +
            c_up * grad_up + c_down * grad_down
        )
        
        del grad_north, grad_south, grad_east, grad_west, grad_up, grad_down
        del c_north, c_south, c_east, c_west, c_up, c_down
    
    return diffused_image

# def perona_malik_3d_original(image, num_iter=10, delta_t=1/7, kappa=50, option=2):
#     """
#     Perform Perona-Malik anisotropic diffusion on a 3D image.

#     Parameters:
#     image (ndarray): 3D input image.
#     num_iter (int): Number of iterations.
#     delta_t (float): Integration constant (usually <= 1/7).
#     kappa (float): Conduction coefficient, controls sensitivity to edges.
#     option (int): 1 for the first diffusion equation, 2 for the second.

#     Returns:
#     ndarray: Diffused image.
#     """
#     # Initialize output array
#     diffused_image = image.astype(np.float32)
    

#     for _ in range(num_iter):
#         # Compute gradients in x, y, z directions
#         grad_north = np.pad(diffused_image[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode='edge') - diffused_image
#         grad_south = np.pad(diffused_image[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode='edge') - diffused_image
#         grad_east = np.pad(diffused_image[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='edge') - diffused_image
#         grad_west = np.pad(diffused_image[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='edge') - diffused_image
#         grad_up = np.pad(diffused_image[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='edge') - diffused_image
#         grad_down = np.pad(diffused_image[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='edge') - diffused_image

#         # Compute conductance
#         if option == 1:
#             c_north = np.exp(-(grad_north / kappa) ** 2)
#             c_south = np.exp(-(grad_south / kappa) ** 2)
#             c_east = np.exp(-(grad_east / kappa) ** 2)
#             c_west = np.exp(-(grad_west / kappa) ** 2)
#             c_up = np.exp(-(grad_up / kappa) ** 2)
#             c_down = np.exp(-(grad_down / kappa) ** 2)
#         elif option == 2:
#             c_north = 1. / (1. + (grad_north / kappa) ** 2)
#             c_south = 1. / (1. + (grad_south / kappa) ** 2)
#             c_east = 1. / (1. + (grad_east / kappa) ** 2)
#             c_west = 1. / (1. + (grad_west / kappa) ** 2)
#             c_up = 1. / (1. + (grad_up / kappa) ** 2)
#             c_down = 1. / (1. + (grad_down / kappa) ** 2)

#         # Update the image
#         diffused_image += delta_t * (
#             c_north * grad_north + c_south * grad_south +
#             c_east * grad_east + c_west * grad_west +
#             c_up * grad_up + c_down * grad_down
#         )
    
#     return diffused_image

def apply_isotropic_diffusion_3d(path_volume_cells, path_volume_mask, path_volume_cells_diffused, num_iter = 500, bitdepth = 8):
    volume_cells = functionReadTIFFMultipage(path_volume_cells, bitdepth)
    volume_cells_mask = functionReadTIFFMultipage(path_volume_mask, bitdepth)
    volume_cells_mask = np.float32(volume_cells_mask > 0)
    volume_cells_diffused = isotropic_diffusion_3d(volume_cells, volume_cells_mask, num_iter=num_iter)
    
    functionSaveTIFFMultipage(volume_cells_diffused, path_volume_cells_diffused, bitdepth = bitdepth)
    

def apply_perona_malik_3d(path_volume_cells, path_volume_mask, path_volume_cells_diffused, num_iter=2, bitdepth = 8):
    volume_cells = functionReadTIFFMultipage(path_volume_cells, bitdepth)
    volume_cells_mask = functionReadTIFFMultipage(path_volume_mask, bitdepth)
    volume_cells_mask = np.float32(volume_cells_mask > 0)
    volume_cells_diffused = perona_malik_3d(volume_cells, volume_cells_mask, num_iter=num_iter)
    
    functionSaveTIFFMultipage(volume_cells_diffused, path_volume_cells_diffused, bitdepth = bitdepth)
    
def apply_perona_malik_3d_no_edge(path_volume_cells, path_volume_mask, path_volume_cells_diffused, num_iter=2, bitdepth = 8):
    volume_cells = functionReadTIFFMultipage(path_volume_cells, bitdepth)
    volume_cells_mask = functionReadTIFFMultipage(path_volume_mask, bitdepth)
    volume_cells_mask = np.float32(volume_cells_mask > 0)
    volume_cells_diffused = perona_malik_3d_no_edge(volume_cells, volume_cells_mask, num_iter=num_iter)
    
    
    functionSaveTIFFMultipage(volume_cells_diffused, path_volume_cells_diffused, bitdepth = bitdepth)    
    
def normalize_perona_malik(output_file_prolif, output_file_prolif_hist_norm, bitdepth = 8, out_range = (0, 255)):
    volume_diffused = functionReadTIFFMultipage(output_file_prolif, bitdepth)
    shape_volume = volume_diffused.shape
    vector_diffused_norm = functionPercNorm(volume_diffused.flatten())
    volume_diffused_norm = vector_diffused_norm.reshape(shape_volume)
        
    imgNorm = exposure.rescale_intensity(volume_diffused_norm, out_range=out_range)
    del volume_diffused_norm
    functionSaveTIFFMultipage(imgNorm, output_file_prolif_hist_norm, bitdepth = bitdepth)
