#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:02:49 2023

@author: lucas
"""

from skimage import measure
import os
import matplotlib.pyplot as plt
from cellpose.io import imread
import numpy as np
import csv
import pandas
import cv2
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import disk

# TODO: implement for tiffs
# NOT FINISHED OR CLEANED

###################################   PARAMETERS   #########################

folder_original = ''
folder_segmentations = ''
ending_segmentation = '_cellpose.png'
flag_show = True
th_size = 10000
from scipy.ndimage import label
from scipy import ndimage

##############################################################################

def matching_label_pairs_perc(matrix1, matrix2, min_perc = 0.5):
    # Convert matrices to NumPy arrays
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)

    unique_labels_matrix1 = np.unique(array1)
    
    # Initialize a list to store matching label pairs
    matching_pairs = []

    # Iterate over unique labels in matrix1
    for label1 in unique_labels_matrix1:
        # Find indices where the label appears in matrix1
        indices_matrix1 = np.where(array1 == label1)
        number_pixels_obj_1 = np.count_nonzero(array1 == label1)

        # Extract corresponding labels from matrix2
        corresponding_labels_matrix2 = array2[indices_matrix1]

        # Iterate over unique labels in matrix2 corresponding to label1 in matrix1
        for label2 in np.unique(corresponding_labels_matrix2):
            # second object size
            number_pixels_obj_2 = np.count_nonzero(array2 == label2)
            
            n_pixels_and= np.count_nonzero(corresponding_labels_matrix2 == label2)
            perc_matching = n_pixels_and/np.min([number_pixels_obj_1,number_pixels_obj_2])
            
            if perc_matching>min_perc:
                matching_pairs.append((label1, label2))
            
    matching_pairs_non_zero_left = [elem for elem in matching_pairs if elem[0] != 0]
    matching_pairs_non_zero = [elem for elem in matching_pairs if (elem[0] != 0 and elem[1] != 0)]

    return matching_pairs, matching_pairs_non_zero_left, matching_pairs_non_zero

from TissueSegmentation.data_loader import get_images_from_path

# Where the first path (img_cc3_binary) is the images you want to REMOVE cells from
def binary_to_label(img_cc3_binary, img_phh3_binary):
    img_cc3_binary_shifted = cv2.imread(img_cc3_binary, cv2.IMREAD_UNCHANGED)
    img_phh3_binary = cv2.imread(img_phh3_binary, cv2.IMREAD_UNCHANGED)

    cc3_binaryToLabel, nA = label(img_cc3_binary_shifted)
    phh3_binaryToLabel, nB = label(img_phh3_binary)

# Where the first path (folder_cc3_binary) is the images you want to REMOVE cells from
def remove_cells_folder_vs_folder(folder_cc3_binary, folder_phh3_binary, output_folder):
    list_slices_1 = get_images_from_path(folder_cc3_binary)
    list_slices_2 = get_images_from_path(folder_phh3_binary)
    n_list_slices_1 = len(list_slices_1)
    n_list_slices_2 = len(list_slices_2)
    if n_list_slices_1 != n_list_slices_2:
        print('Folders with different number of slices')
        print('folder 1:' + str(n_list_slices_1))
        print('folder 2:' + str(n_list_slices_2))
        return

    for i in range(n_list_slices_1):
        img_slice_path_1 = list_slices_1[i]
        img_slice_path_2 = list_slices_2[i]

        cc3_label, phh3_label = binary_to_label(img_slice_path_1, img_slice_path_2)
        _, _, matching_pairs = matching_label_pairs_perc(cc3_label, phh3_label, min_perc = 0.5)

        img_cc3_filtered = img_cc3_binary_shifted.copy()
        labels_to_remove = [label1 for label1, _ in matching_pairs]
        remove_mask = np.isin(cc3_binaryToLabel, labels_to_remove)
        img_cc3_filtered[remove_mask] = 0

        base = os.path.splitext(os.path.basename(img_slice_path_1))[0]
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, base + ".png") 
        cv2.imwrite(output_path, img_cc3_filtered)

def main():
    # img_phh3_binary = cv2.imread('/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/Step25b_pHH3_slices_binary/Aug2025_27_slice_0500.png', cv2.IMREAD_UNCHANGED)
    # img_cc3_binary = cv2.imread('/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_binary/Aug2025_27_slice_0500.png', cv2.IMREAD_UNCHANGED)
    # img_cc3_binary_shifted = cv2.imread('/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_binary_shifted/Aug2025_27_slice_0500_shifted.png', cv2.IMREAD_UNCHANGED)

    # cc3_binaryToLabel, nA = label(img_cc3_binary_shifted)
    # phh3_binaryToLabel, nB = label(img_phh3_binary)

    # _, _, matching_pairs = matching_label_pairs_perc(cc3_binaryToLabel, phh3_binaryToLabel, min_perc = 0.5)

    # img_cc3_filtered = img_cc3_binary_shifted.copy()
    # labels_to_remove = [label1 for label1, _ in matching_pairs]
    # remove_mask = np.isin(cc3_binaryToLabel, labels_to_remove)
    # img_cc3_filtered[remove_mask] = 0

    # print("phh3 cell count:", phh3_binaryToLabel.max())
    # print("CC3 cell count:", cc3_binaryToLabel.max())

    # cv2.imwrite('/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/Aug2025_27_slice_0500_cc3_filtered.png', img_cc3_filtered)

if __name__ == "__main__":
    main()
