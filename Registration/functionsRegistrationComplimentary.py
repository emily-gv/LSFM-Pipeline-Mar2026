#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:54:25 2025

@author: lucas
"""

import os
import numpy as np
from ManualCorrection.TIFFMultipage import functionSaveTIFFMultipage, functionReadTIFFMultipage
from scipy.ndimage import center_of_mass
import nibabel as nib
from scipy.stats import pearsonr, zscore
import pandas as pd

import numpy as np
import scipy.ndimage as ndi

def get_volume_edge(volume_np, kernel_size = 5):
    # Edge detection using Sobel filters
    sx = ndi.sobel(volume_np>0, axis=0)
    sy = ndi.sobel(volume_np>0, axis=1)
    sz = ndi.sobel(volume_np>0, axis=2)

    # Magnitude of the gradient (edge strength)
    edge_magnitude = np.sqrt(sx**2 + sy**2 + sz**2)

    # Thresholding to create a binary edge map
    threshold = np.percentile(edge_magnitude, 0.85) # Adjust threshold as needed. 95th percentile is often good.
    volume_edge = edge_magnitude > threshold
    
    # Dilation to thicken the edges (3x3 structuring element)
    structure = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
    volume_edge_dilated = ndi.binary_dilation(volume_edge, structure=structure)
    
    return volume_edge_dilated

def percentage_lms_in_surface(volume_edge, lms):

    total_lms = len(lms)
    if total_lms == 0:
        return 0.0  # Avoid division by zero if no landmarks are provided.

    hints_xyz = 0
    hints_yxz = 0
    hints_zyx = 0
    for x, y, z in lms:
        x = np.int32(x)
        y = np.int32(y)
        z = np.int32(z)
        # Check if the landmark is within the volume bounds
        if 0 <= x < volume_edge.shape[0] and 0 <= y < volume_edge.shape[1] and 0 <= z < volume_edge.shape[2]:
            if volume_edge[x, y, z]:
                hints_xyz += 1
            if volume_edge[y, x, z]:
                hints_yxz += 1
            if volume_edge[z, y, x]:
                hints_zyx += 1

    xyz = hints_xyz / total_lms if total_lms>0 else 0.0
    yxz = hints_yxz / total_lms if total_lms>0 else 0.0    
    zyx = hints_zyx / total_lms if total_lms>0 else 0.0    
    
    return xyz, yxz, zyx

def compute_IoU_samples(sampleNames, ending_moved_volumes, regTypeFolder, ending_folder_output):
    nSamples = len(sampleNames)
    nVolumes = len(ending_moved_volumes)

    for i in range(nVolumes):
        ending_volume = ending_moved_volumes[i]
        # print('--------------')
        # print('--- ending_volume: ' + ending_volume)
        
        mat_iou = np.ones((nSamples, nSamples)) * (-1)
        mat_corr = np.ones((nSamples, nSamples)) * (-1)    
        
        vector_iou = []
        vector_corr = []
        vector_sample_names = []
        
        for it1 in range(len(sampleNames)):
            sampleName1 = sampleNames[it1]
            # print('--- ---'+str(it1)+' sampleName1: ' + sampleName1)
            # print(f"Processing: {path1}")
            folderOutput1 =  os.path.join(regTypeFolder, sampleName1 + ending_folder_output)
            pathVolume1 = os.path.join(folderOutput1, sampleName1 + ending_volume)
            volume1 = functionReadTIFFMultipage(pathVolume1,8)
            for it2 in range(len(sampleNames)):
                if it1 == it2:
                    mat_corr[it1, it2] = 1 # Diagonal is 1
                    mat_iou[it1, it2] = 1 # Diagonal is 1
                elif mat_iou[it1, it2] == -1: # To not repeat analysis
                
                    sampleName2 = sampleNames[it2]
                    # print('--- --- ---'+str(it2)+' sampleName2: ' + sampleName2)
                    folderOutput2 =  os.path.join(regTypeFolder, sampleName2 + ending_folder_output)
                    pathVolume2 = os.path.join(folderOutput2, sampleName2 + ending_volume)
                    volume2 = functionReadTIFFMultipage(pathVolume2,8)
                    
                    # Calculate union and intersection
                    volume_union = np.logical_or(volume1 > 0, volume2 > 0)
                    volume_interc = np.logical_and(volume1 > 0, volume2 > 0)
                    
                    # Extract values for correlation
                    values1 = volume1[volume_union].astype(np.float64)
                    values2 = volume2[volume_union].astype(np.float64)
                    
                    # Compute IoU
                    iou = np.sum(volume_interc) / np.sum(volume_union)
                    mat_iou[it1, it2] = iou
                    mat_iou[it2, it1] = iou
                    
                    # Compute correlationiou
                    coef_corr, _ = pearsonr(values1, values2)
                    mat_corr[it1, it2] = coef_corr
                    mat_corr[it2, it1] = coef_corr
                    
                    # print(f"IoU: {iou}, Corr: {coef_corr}")
                    
                    vector_iou.append(iou)
                    vector_iou.append(iou)
                    vector_corr.append(coef_corr)
                    vector_corr.append(coef_corr)
                    vector_sample_names.append(sampleName2)
                    vector_sample_names.append(sampleName1)
                    
                    # print('IoU: ' + sampleName1 + ' - ' + sampleName2 + ': ' + str(iou))
                    
                    del volume2, volume_union, volume_interc, values1, values2
            
            del volume1
        
        # Create a DataFrame
        df = pd.DataFrame({"Sample": vector_sample_names, "IoU": vector_iou})
        
        mean_iou = np.mean(vector_iou)
        std_iou = np.std(vector_iou)
        mean_corr = np.mean(coef_corr)
        std_corr = np.std(coef_corr)
        n_ious = len(vector_iou)
        
        
        path_iou_stats = os.path.join(regTypeFolder,'IoU_stats.txt')
        with open(path_iou_stats, "a") as file:
            file.write("---------------------------\n")
            for s_name in sampleNames:
                file.write(f"{s_name}\n")
            file.write("---------------------------\n")
            file.write(f"From folder: {regTypeFolder}\n")
            file.write(ending_volume+": \n")
            file.write("---------------------------\n")
            file.write(f"n IoUs:\n{n_ious},\nMean IoU: {mean_iou:.5f} ({std_iou:.5f}),\n")
            file.write(f"Mean Corr:\n{mean_corr:.5f} ({std_corr:.5f})\n")
            file.write("---------------------------\n")
            file.write(f"mat_iou: {mat_iou}")
            file.write("\n---------------------------\n")
            result = df.groupby("Sample")["IoU"].mean()
            file.write(f"{result}")
            file.write("\n---------------------------\n")
            result = df.groupby("Sample")["IoU"].agg(["mean", "std"])
            # Compute z-score and add it as a new column
            # result["z-score"] = abs(zscore(result["mean"]))

            file.write(f"{result}")
            file.write("\n"+ending_volume+"\n\n")
            file.write("---------------------------\n")

def extract_affine_transform_info(transform, fixed_image=None):
    """
    Extracts all relevant information about an ANTs affine transform and returns it as a list.

    Args:
        transform: An ants.ANTsTransform object.
        fixed_image: (Optional) An ants.ANTsImage object. If provided, includes image spacing and origin.

    Returns:
        A list containing the transform information.
    """

    if transform.dimension != 3:
        raise ValueError("This function is designed for 3D affine transforms.")

    parameters = transform.parameters
    matrix = np.array(parameters[:9]).reshape(3, 3)
    displacement_vector = np.array(parameters[9:])

    info = []
    info.extend(matrix.flatten().tolist())  # Flatten matrix and add to list
    info.extend(displacement_vector.tolist())

    if fixed_image:
        info.extend(fixed_image.spacing)
        info.extend(fixed_image.origin)
        info.extend(fixed_image.direction.flatten().tolist())
    info.append(transform.type)
    info.append(transform.dimension)

    return info

def save_affine_transform_info_to_txt(transform, filename, fixed_image=None):
    """
    Extracts and saves affine transform information to a .txt file.

    Args:
        transform: An ants.ANTsTransform object.
        filename: The path to the output .txt file.
        fixed_image: (Optional) An ants.ANTsImage object.
    """

    info = extract_affine_transform_info(transform, fixed_image)

    with open(filename, "w") as f:
        for item in info:
            f.write(str(item) + "\n")

def create_magnitude_displacement_volume(syn_path_nii_gz, str_ending_tiff = '_magnitude.tiff', str_ending_nii = '_signed_magnitude_traspose_NoArg.nii'):
    nifti_img = nib.load(syn_path_nii_gz)

    image_data = nifti_img.get_fdata()

    # Remove singleton dimension: (Z, Y, X, 3)
    displacement_vectors = np.squeeze(image_data)  # Shape: (325, 325, 325, 3)

    # Compute the magnitude volume
    magnitude_volume = np.sqrt(np.sum(displacement_vectors**2, axis=-1))  # Shape: (325, 325, 325)

    # Compute the center of mass of the magnitude volume
    com = np.array(center_of_mass(magnitude_volume))  # (z, y, x)

    # Create coordinate grid
    z, y, x = np.indices(magnitude_volume.shape)

    # Compute vectors from the center of mass to each voxel
    position_vectors = np.stack([z, y, x], axis=-1) - com  # Shape: (325, 325, 325, 3)

    # Normalize position vectors to unit vectors
    norms = np.linalg.norm(position_vectors, axis=-1, keepdims=True)
    unit_position_vectors = np.divide(position_vectors, norms, out=np.zeros_like(position_vectors), where=(norms > 0))

    # Compute dot product between displacement vectors and unit position vectors
    dot_product = np.sum(displacement_vectors * unit_position_vectors, axis=-1)

    # Signed magnitude: positive if outward, negative if inward
    signed_magnitude_volume = np.sign(dot_product) * magnitude_volume
    
    # signed_magnitude_volume = np.transpose(signed_magnitude_volume, (2, 1, 0))
    signed_magnitude_volume = np.transpose(signed_magnitude_volume)
    # signed_magnitude_volume = np.transpose(signed_magnitude_volume, (0, -1, -2))

    # Print results
    print("Center of mass (z, y, x):", com)
    print("Signed Magnitude - Min:", np.min(signed_magnitude_volume))
    print("Signed Magnitude - Max:", np.max(signed_magnitude_volume))

    functionSaveTIFFMultipage(np.abs(magnitude_volume), syn_path_nii_gz + str_ending_tiff, 8)

    output_path = syn_path_nii_gz + str_ending_nii
    # Force qform/sform to match the affine transformation
    # Get the original affine matrix
    affine = nifti_img.affine.copy()
    
    # Flip the axes back by multiplying by -1 in the (0,0) and (1,1) positions
    affine[0, 0] *= -1
    affine[1, 1] *= -1
    affine[0, 3] = 0
    affine[1, 3] = 0
    affine[2, 3] = 0
    
    signed_magnitude_nifti = nib.Nifti1Image(signed_magnitude_volume, affine=affine, header=nifti_img.header)
    # signed_magnitude_nifti.set_qform(nifti_img.affine)
    # signed_magnitude_nifti.set_sform(nifti_img.affine)
    print("Affine Matrix:\n", nifti_img.affine)
    signed_magnitude_nifti.header.set_data_dtype(np.float32)
    nib.save(signed_magnitude_nifti, output_path)