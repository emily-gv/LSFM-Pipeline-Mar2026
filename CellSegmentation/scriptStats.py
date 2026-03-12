#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:25:20 2024

@author: lucas
"""
from TissueSegmentation.data_loader import get_files_from_path, get_images_from_path
#from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
    # NOT USED IN main_01 OR main_02
import cv2
import numpy as np
from matplotlib.pylab import plt
import os

import numpy as np
# from scipy.ndimage import zoom

def plot_tissue_perc(vector_mesen_perc, vector_ne_perc):
    n_list = len(vector_mesen_perc)
    x = range(1, n_list + 1) # start x axis at 1 not 0

    fig = plt.figure(figsize=(12,6))
    plt.plot(x, vector_mesen_perc, label='Mes')
    plt.plot(x, vector_ne_perc, label='NE')
    
    # Add in a title and axes labels
    plt.title('Percentage of each tissue in the sample')
    plt.xlabel('Slice')
    plt.ylabel('Percentage of tissue')
    plt.ylim([0.0, 1])
    plt.legend()
    
    return fig
    

# def plots_tissue_perc_tiff(path_tissue, folder_sample, sample_name, str_description, LABEL_MES = 50, LABEL_NE = 100):
#     volume = functionReadTIFFMultipage(path_tissue, 8)
#     [h,w,n_slices] = volume.shape
#     vector_mesen_perc = []
#     vector_ne_perc = []
#     vector_area = []
    
#     for i in range(n_slices):
#         slice_tissue = volume[:,:,i]
#         area = np.float32(np.sum(slice_tissue > 0)) + 0.000001
        
#         total_mes = np.sum(slice_tissue == LABEL_MES)
#         vector_mesen_perc.append(total_mes/area)
        
#         total_ne = np.float32(np.sum(slice_tissue == LABEL_NE))
#         vector_ne_perc.append(total_ne/area)
        
#         vector_area.append(area)
    
#     fig_handle = plot_tissue_perc(vector_mesen_perc, vector_ne_perc)
#     str_fig_loss = os.path.join(folder_sample, sample_name  + '_' + str_description + '_perc.png')
#     fig_handle.savefig(str_fig_loss, dpi=300)
    
#     return vector_area, vector_mesen_perc, vector_ne_perc

def plots_tissue_perc_folder(folder_tissues_slices, folder_sample, sample_name, LABEL_MES = 50, LABEL_NE = 100):
    list_slices_tissues = get_images_from_path(folder_tissues_slices)
    n_list_slices_tissues = len(list_slices_tissues)
    
    vector_mesen_perc = []
    vector_ne_perc = []
    vector_area = []
    
    for i in range(n_list_slices_tissues):
        img_tissues_path = list_slices_tissues[i]
        img_tissue = cv2.imread(img_tissues_path, 0)
        if img_tissue is None:
            print('Error reading: ' + os.path.basename(img_tissues_path))        
        img_tissue = np.array(img_tissue)
        
        area = np.float32(np.sum(img_tissue > 0)) + 0.000001
        
        total_mes = np.sum(img_tissue == LABEL_MES)
        vector_mesen_perc.append(total_mes/area)
        
        total_ne = np.float32(np.sum(img_tissue == LABEL_NE))
        vector_ne_perc.append(total_ne/area)
        
        vector_area.append(area)
        
    fig_handle = plot_tissue_perc(vector_mesen_perc, vector_ne_perc)
    str_fig_loss = os.path.join(folder_sample, sample_name  + '_tissue_perc.png')
    fig_handle.savefig(str_fig_loss, dpi=300)
    
    return vector_area, vector_mesen_perc, vector_ne_perc

def plot_IoU(vector_IoU, str_description):
    n_list = len(vector_IoU)
    fig = plt.figure(figsize=(12,6))
    plt.plot(range(n_list), vector_IoU, label='IoU')
    # Add in a title and axes labels
    plt.title('IoU through slices: ' + str_description)
    plt.xlabel('Slice')
    plt.ylabel('IoU')
    plt.ylim([0.0, 1.1])
    
    return fig

def IoU(slice_1, slice_2, threshold):
    img_union = np.bitwise_or(slice_1>=threshold, slice_2>=threshold)
    img_inter = np.bitwise_and(slice_1>=threshold, slice_2>=threshold)
    
    IoU = float(np.sum(img_inter)) / (float(np.sum(img_union) + 0.000001 ))
    
    return IoU

def plot_IoU_folder(folder_slices, folder_sample, sample_name, str_description, threshold=1):
    list_slices = get_images_from_path(folder_slices)
    n_list_slices = len(list_slices)
    vector_IoU = []
    for i in range(n_list_slices-1):
        # Read image
        img_slice_path = list_slices[i]
        img_slice = cv2.imread(img_slice_path, 0)
        if img_slice is None:
            print('Error reading: ' + os.path.basename(img_slice_path), flush=True)
        img_slice = np.array(img_slice)
        
        # Read next image
        img_slice_path = list_slices[i+1]
        img_slice_next = cv2.imread(img_slice_path, 0)
        if img_slice_next is None:
            print('Error reading: ' + os.path.basename(img_slice_path), flush=True)
        img_slice_next = np.array(img_slice_next)
        
        iou = IoU(img_slice, img_slice_next, threshold)
        vector_IoU.append(iou)
    
    fig_handle = plot_IoU(vector_IoU, str_description)
    str_fig_loss = os.path.join(folder_sample, sample_name  + '_IoU_'+str_description+'.png')
    fig_handle.savefig(str_fig_loss, dpi=300)
    
    return vector_IoU
        
def plot_IoU_folder_vs_folder(folder_slices_1, folder_slices_2, folder_sample, sample_name, str_description, threshold=1):
    list_slices_1 = get_images_from_path(folder_slices_1)
    list_slices_2 = get_images_from_path(folder_slices_2)
    n_list_slices_1 = len(list_slices_1)
    n_list_slices_2 = len(list_slices_2)
    if n_list_slices_1 != n_list_slices_2:
        print('Cannot compute IoU between folder with different number of slices')
        print('folder 1:' + str(n_list_slices_1))
        print('folder 2:' + str(n_list_slices_2))
        return 
    vector_IoU = []
    for i in range(n_list_slices_1):
        # Read image
        img_slice_path_1 = list_slices_1[i]
        img_slice = cv2.imread(img_slice_path_1, 0)
        if img_slice is None:
            print('Error reading: ' + os.path.basename(img_slice_path_1))
        img_slice_1 = np.array(img_slice)
        
        # Read other slice
        img_slice_path_2 = list_slices_2[i]
        img_slice_2 = cv2.imread(img_slice_path_2, 0)
        if img_slice_2 is None:
            print('Error reading: ' + os.path.basename(img_slice_path_2))
        img_slice_2 = np.array(img_slice_2)
        
        iou = IoU(img_slice_1, img_slice_2, threshold)
        vector_IoU.append(iou)
    
    fig_handle = plot_IoU(vector_IoU, str_description)
    str_fig_loss = os.path.join(folder_sample, sample_name  + '_IoU_'+str_description+'.png')
    fig_handle.savefig(str_fig_loss, dpi=300)
    
    return vector_IoU

# def plot_IoU_tissue(path_tissue, folder_sample, sample_name, str_description, threshold=1):
#     volume = functionReadTIFFMultipage(path_tissue, 8)
#     [h,w,n_slices] = volume.shape
#     vector_IoU = []
#     for i in range(n_slices-1): 
#         img_slice = volume[:,:,i]
#         img_slice_next = volume[:,:,i+1]
        
#         iou = IoU(img_slice, img_slice_next, threshold)
#         vector_IoU.append(iou)
    
#     fig_handle = plot_IoU(vector_IoU, str_description)
#     str_fig_loss = os.path.join(folder_sample, sample_name  + '_IoU_'+str_description+'.png')
#     fig_handle.savefig(str_fig_loss, dpi=300)
    
#     return vector_IoU

# def get_IoU_tissue_window(path_tissue, path_iou_window, window_size = 20, bitdepth = 8):
        
#     volume = functionReadTIFFMultipage(path_tissue, bitdepth)
#     [h,w,n_slices] = volume.shape
    
#     # Initialize an empty array B with the same shape as AAA
#     B = np.zeros_like(volume)
    
#     # half_window = window_size // 2
    
#     # # Iterate through the third dimension
#     # for z in range(n_slices-1):
#     #     slice_2d        = volume[:, :, z]
#     #     slice_2d_next   = volume[:, :, z+1]
        
#     #     # Initialize an empty 2D array to hold the results for this slice
#     #     result_slice = np.zeros((h, w))
    
#     #     # Iterate over the slice with a moving window
#     #     # for i in range(half_window, h - half_window, window_size):
#     #     #     for j in range(half_window, w - half_window, window_size):
#     #     for i in range(half_window, h - half_window, 1):
#     #         for j in range(half_window, w - half_window, 1):
#     #             # Extract the window
#     #             window1 =       slice_2d[i-half_window:i+half_window, j-half_window:j+half_window]
#     #             window2 =  slice_2d_next[i-half_window:i+half_window, j-half_window:j+half_window]
                
#     #             # Compute the mean of the window
#     #             iou = IoU(window1, window2, threshold = 1)
                
#     #             # Assign the mean value to the center of the window in the result slice
#     #             result_slice[i, j] = iou * 255.
    
#     #     # Store the result slice in the corresponding slice of B
#     #     B[:, :, z] = result_slice
    
#     # Iterate through the third dimension
#     for z in range(n_slices-1):
#         # Extract the 2D slice for the current Z index
#         slice_2d        = volume[:, :, z]
#         slice_2d_next   = volume[:, :, z+1]
        
#         # Compute the number of windows in X and Y directions
#         n_windows_x = h // window_size
#         n_windows_y = w // window_size
    
#         # # Initialize an array to store the averaged windows
#         # iou_windows = np.zeros((n_windows_x, n_windows_y))
    
#         # Loop over the windows
#         for i in range(n_windows_x):
#             for j in range(n_windows_y):
#                 # Extract the 20x20 window
#                 window1 = slice_2d[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
#                 window2 = slice_2d_next[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
#                 # Compute the IoU
#                 iou = IoU(window1, window2, threshold = 1)
#                 #iou_windows[i, j] = iou
                
#                 B[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size, z] = iou * 255
    
#         # # Upscale the averaged windows back to the original size of the slice
#         # # Use zoom to match the original X,Y dimensions
#         # upscaled_avg = zoom(iou_windows, (window_size, window_size), order=3)
    
    
#         # # Initialize a new array of the original size (X, Y) filled with zeros
#         # padded_upscaled_avg = np.zeros((h, w))
#         # # Determine the size of the upscaled average result
#         # upscaled_x, upscaled_y = upscaled_avg.shape
#         # # Copy the upscaled result into the padded array
#         # padded_upscaled_avg[:upscaled_x, :upscaled_y] = upscaled_avg * 255.
#         # # Store the result in the corresponding slice of B
#         # B[:, :, z] = padded_upscaled_avg
    
#     #Copy for the last slice
#     B[:, :, -1] = B[:, :, -2]
    
#     del volume
    
#     functionSaveTIFFMultipage(B,path_iou_window,bitdepth)


def plots_cells_perc_folder(folder_cells_slices, vector_area, vector_mesen_perc, folder_sample, sample_name, str_cells = 'nuclei_perc_raw', LABEL_MES = 50):
    list_cell_tissues = get_images_from_path(folder_cells_slices)
    n_list_cell_tissues = len(list_cell_tissues)
    
    
    vector_cell_area = []
    vector_cell_perc = []
    
    for i in range(n_list_cell_tissues):
        # Read image
        img_tissues_path = list_cell_tissues[i]
        img_tissue = cv2.imread(img_tissues_path, 0)
        if img_tissue is None:
            print('Error reading: ' + os.path.basename(img_tissues_path))
        img_tissue = np.array(img_tissue)
        
        #cells area
        cells_area = np.float32(np.sum(img_tissue > 0))
        vector_cell_area.append(cells_area)
        
        #cells area over mesen
        area_mesen = np.float32(vector_area[i] * vector_mesen_perc[i])
        vector_cell_perc.append(cells_area / (area_mesen + 0.000001))
        
    x = range(1, n_list_cell_tissues + 1) # start x axis at 1 not 0 

    plt.figure(figsize=(12,6))
    plt.plot(x, vector_cell_perc, label='Cell perc')
    # Add in a title and axes labels
    plt.title('Percentage of Mes occupied with cells')
    plt.xlabel('Slice')
    plt.ylabel('Percentage of tissue')
    
    str_fig_loss = os.path.join(folder_sample, sample_name  + '_'+str_cells+'.png')
    plt.savefig(str_fig_loss, dpi=300)
    
    return vector_cell_area, vector_cell_perc

def main():
    
    folder_sample = '/mnt/DATA/HALLGRIMSSON_LAB/Image-to-Image/Volumes202411_1024/E10.0/Dec2_E10_11'
    sample_name = 'Dec2_E10_11'
    str_description = 'Comparison_slicesPHH3Seg_Step25a_Tiles_256vs1024_nuclei_binary_masked_label'
    folder_slices_1 = os.path.join(folder_sample,'20241208_CellSegmentation_tile_phh3_256_nuclei_256_usingfunctionCellposeSegmentation_tiling_noNB_no_logging/Step27a_Tiles_nuclei_binary_masked_label')
    folder_slices_2 = os.path.join(folder_sample,'20241208_CellSegmentation_tile_phh3_1024_nuclei_512_usingfunctionCellposeSegmentation_tiling_noNB_no_logging/Step27a_Tiles_nuclei_binary_masked_label')
    plot_IoU_folder_vs_folder(folder_slices_1, folder_slices_2, folder_sample, sample_name, str_description, threshold=1)
    
if __name__=="__main__":
    main()

