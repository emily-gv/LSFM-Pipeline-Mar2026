#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:48:40 2024

@author: lucas
"""
import os
from TissueSegmentation.functionCellposeSegmentation import functionCellposeSegmentation_tiling #functionCellposeSegmentation_tiling_noNB #  #functionCellposeSegmentation_label 
from TissueSegmentation.functionMergeProcessedTilesNB import functionMergeProcessedTilesNB
from TissueSegmentation.functionCreateVolume import functionCreateVolume
from TissueSegmentation.data_loader import get_files_from_path, get_images_from_path
# from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
from TissueSegmentation.functionRemoveFolderContent import functionRemoveFolderContent
import cv2
import shutil

import numpy as np
def applyDiffGenPerc(folder_input, folder_input_npy, folder_output):
    formats = [".npz"]
    list_1 = get_files_from_path(folder_input_npy, ACCEPTABLE_FORMATS = formats)
    n_list_1 = len(list_1)
    
    for i in range(n_list_1):
        
        #Get NPZ file
        file_path = list_1[i]
        file_filename = os.path.basename(file_path)
        img1_filename = file_filename[:-len(formats[0])]
        inputName = os.path.join(folder_input_npy, file_filename)
        # print(inputName)
        loaded = np.load(inputName)
        diff_perc = np.float32(loaded['diff_perc']) #Remember precision was lost here
        
        #Get Original Image
        img1_path = os.path.join(folder_input, img1_filename)
        original_image = cv2.imread(img1_path, 0)
        if original_image is None:
            print('Error reading: ' + img1_filename)
        original_image = original_image.astype(np.float32)
        original_image = np.array(original_image)
        # print(original_image.shape)
        # print(diff_perc.shape)
        to_sum = np.multiply(diff_perc, original_image)
        # print(to_sum.shape)
        image_generated = to_sum + original_image
        
        outputName = os.path.join(folder_output, img1_filename)
        cv2.imwrite(outputName, image_generated)
    
def functionApplyGenMultiplication(folder_input, folder_input_npy, folder_output):
    # Iterate over all files in the source directory
    for filename in os.listdir(folder_input):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            # Construct full file paths
            source_file = os.path.join(folder_input, filename)
            destination_file = os.path.join(folder_output, filename)
            
            # Copy the file to the destination directory
            shutil.copy2(source_file, destination_file)
    
    applyDiffGenPerc(folder_input, folder_input_npy, folder_output)

def segmentCells(folder_sample, folder_tiles, folder_cell_segmented_label, folder_cell_segmented_binary,\
                 folder_cell_slices_label, folder_cell_slices_binary, \
                     dest_file_cell_label_tiff, dest_file_cell_binary_tiff, path_cell_unet, window_cellpose_tile = 131):

    if not os.path.exists(folder_cell_segmented_label):     os.makedirs(folder_cell_segmented_label)
    if not os.path.exists(folder_cell_segmented_binary):    os.makedirs(folder_cell_segmented_binary)
    functionCellposeSegmentation_tiling(folder_tiles, folder_cell_segmented_label, folder_cell_segmented_binary, path_cell_unet, tile_window = window_cellpose_tile) #b are the generated images
    
    if not os.path.exists(folder_cell_slices_label):    os.makedirs(folder_cell_slices_label)
    if not os.path.exists(folder_cell_slices_binary):   os.makedirs(folder_cell_slices_binary)
    functionMergeProcessedTilesNB(folder_cell_segmented_label, folder_cell_slices_label, folder_tiles)
    functionMergeProcessedTilesNB(folder_cell_segmented_binary, folder_cell_slices_binary, folder_tiles)
    
def maskOutFolder(folder_slices_cells, folder_slices_tissue, folder_cell_masked):

    if not os.path.exists(folder_cell_masked):     os.makedirs(folder_cell_masked)
    
    list_slices_cells = get_images_from_path(folder_slices_cells)
    list_slices_tissues = get_images_from_path(folder_slices_tissue)
    
    n_list_slices_cells = len(list_slices_cells)
    n_list_slices_tissues = len(list_slices_tissues)
    
    if n_list_slices_cells != n_list_slices_tissues:
        print("DIFFERENT NUMBER OF IMAGES")
    else:
        for i in range(n_list_slices_cells):
            img_cells_path = list_slices_cells[i]
            img_cells_filename = os.path.basename(img_cells_path)
            
            img_cells = cv2.imread(img_cells_path, 0)
            if img_cells is None:
                print('Error reading: ' + img_cells_filename)
            img_cells = img_cells.astype(np.float32)
            img_cells = np.array(img_cells)
            
            img_tissue_path = os.path.join(folder_slices_tissue, img_cells_filename)
            img_tissue = cv2.imread(img_tissue_path, 0)
            if img_tissue is None:
                print('Error reading: ' + img_cells_filename)
            img_tissue = img_tissue.astype(np.float32)
            img_tissue = np.array(img_tissue)
            
            img_cells[img_tissue==0] = 0 #Where no tissue, no cells
            
            output_path = os.path.join(folder_cell_masked, img_cells_filename)
            cv2.imwrite(output_path, img_cells)
    