#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:02:21 2025

@author: lucas
"""

from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
from scipy.stats import pearsonr
from matplotlib.pylab import plt
import os
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
################################# PARAMETERS ##########################


folder = '/home/emily/Desktop/OUTPUT_HIGHLASER/Sim/Sim/Affine/SyN/'
map_name = '_Step53b_cc3_density_histnorm'
# list_iter = [10, 50, 100]

filepath_mask = 'Nosip_10.5_Mes_Mask.tiff'

# To generate data for other analysis
flag_generate_npy_and_masked_values = True
downscale_masked_values = 10 # Int number, bigger than 1s
subfolder_name = 'Nosip_10.5_Face_Mask'

######################################################################

def find_files_with_ending(root_dir, ending):
    matches = []
    basenames = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ending):
                matches.append(os.path.join(dirpath, filename))
                basenames.append(filename)
    return matches, basenames

mask = functionReadTIFFMultipage(os.path.join(folder, filepath_mask), 8)
mask = mask > 0

mask_scaled = mask[::np.int16(downscale_masked_values), ::np.int16(downscale_masked_values), ::np.int16(downscale_masked_values)]

vector_means=[]


subfolder_masked_output = subfolder_name + '_down_' + str(downscale_masked_values)
folder_masked_volumes = os.path.join(folder,subfolder_masked_output)
if flag_generate_npy_and_masked_values and (not os.path.exists(folder_masked_volumes)):
    os.mkdir(folder_masked_volumes)

ending_to_search = map_name + '.tiff' # ending_tiff    
list_path, basenames = find_files_with_ending(folder, ending_to_search)
n_volumes = len(list_path)
print('n_volumes: ' + str(n_volumes))
mat_corr = np.ones((n_volumes, n_volumes)) * (-5)
# print(mat_corr)
list_corrs = []
# print('---------')
for i in range(n_volumes):
    # print(list_path[i])
    volume1 = functionReadTIFFMultipage(list_path[i], 8)
    values_in_mask_1 = volume1[mask]
    
    if flag_generate_npy_and_masked_values:
        volume_scaled = volume1[::np.int16(downscale_masked_values), ::np.int16(downscale_masked_values), ::np.int16(downscale_masked_values)]
        file_output_tiff = os.path.join(folder_masked_volumes, basenames[i] + '_down_' + str(downscale_masked_values) + '.tiff')
        functionSaveTIFFMultipage(volume_scaled, file_output_tiff, 8)
        volume_masked = volume_scaled.copy()
        volume_masked[mask_scaled<1] = 0
        file_output_tiff = os.path.join(folder_masked_volumes, basenames[i] + '_down_' + str(downscale_masked_values) + '_masked.tiff')
        functionSaveTIFFMultipage(volume_masked, file_output_tiff, 8)
        file_output_npy = os.path.join(folder_masked_volumes, basenames[i] + '_down_' + str(downscale_masked_values) + '_masked.npy')
        values_map_downsampled = volume_scaled[mask_scaled>0]
        np.save(file_output_npy, values_map_downsampled)
    
    
    del volume1
    print("n_volumes", n_volumes)
    for j in range(n_volumes):
        if i==j:
            mat_corr[i,j] = 1
        elif mat_corr[i,j]<-2:
            volume2 = functionReadTIFFMultipage(list_path[j], 8)
            values_in_mask_2 = volume2[mask]
            del volume2
            coef_corr, _ = pearsonr(values_in_mask_1, values_in_mask_2)
            mat_corr[i,j] = coef_corr
            mat_corr[j,i] = coef_corr
            list_corrs.append(coef_corr)
            del values_in_mask_2
    del values_in_mask_1

mean_temp = np.mean(list_corrs)
vector_means.append(mean_temp)
print("mean corr: " + str(mean_temp) )
    
fig = plt.figure(figsize=(6,6))
plt.hist(list_corrs, bins='auto')
plt.xlabel('Pearson r')
plt.ylabel('Number of volume pairs')
plt.title('Distribution of pairwise correlations')
plt.tight_layout()

plt.savefig(os.path.join(folder,subfolder_name + '_correlation.png' ))


            
    