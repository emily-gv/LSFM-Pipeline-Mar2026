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
from sklearn.decomposition import PCA
import sys
from mpl_toolkits.mplot3d import Axes3D
os.environ["QT_QPA_PLATFORM"] = "offscreen"
################################# PARAMETERS ##########################


folder = '/home/emily/Desktop/OUTPUT_HIGHLASER/Sim/Sim/Affine/SyN/'
map_name = '_Step53b_cc3_density_histnorm'
# list_iter = [50] #, 100]

# To generate data for other analysis
# flag_generate_npy_and_masked_values = True
downscale_masked_values = 10 # Int number, bigger than 1
subfolder_name = 'Nosip_10.5_Face_Mask'

######################################################################

groups_dict = {
    'nosip_sample5':'WT',
    'Feb2026_6': 'Het',
    'Feb2026_7': 'Het',
    'Feb2026_8': 'Null'
    }

list_names = groups_dict.keys()

def find_files_with_ending(root_dir, ending):
    matches = []
    basenames = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ending):
                matches.append(os.path.join(dirpath, filename))
                basenames.append(filename)
    return matches, basenames

X = [] # Matrix of descriptors
y = [] # groups
subfolder_masked_output = subfolder_name + '_down_' + str(downscale_masked_values)
folder_masked_volumes = os.path.join(folder,subfolder_masked_output)


ending_to_search = map_name + '.tiff' # ending_tiff    
list_path, basenames = find_files_with_ending(folder, ending_to_search)
print(basenames)
n_volumes = len(list_path)
print('n_volumes: ' + str(n_volumes))
for i in range(n_volumes):

    file_npy = os.path.join(folder_masked_volumes, basenames[i] + '_down_' + str(downscale_masked_values) + '_masked.npy')
    print(file_npy)
    prolif = np.load(file_npy)
    print(prolif.shape)
    X.append(prolif)
    print("Prolif", prolif.shape)
    # What sample is?
    sample_name = None
    print(list_names)
    for sample_name_temp in list_names:
        print(sample_name_temp)
        if (sample_name_temp in file_npy):
            sample_name = sample_name_temp
            break

    print(sample_name)
    if sample_name is not None:
        sample_group = groups_dict[sample_name]
        y.append(sample_group)
        print(sample_name, sample_group)
        
    else:
        print("Sample with unknown group")
        sys.exit(1)
        
        
matrix_x = np.vstack([arr.ravel() for arr in X])
print("Shape", matrix_x.shape)
# X = np.array(matrix_x).squeeze(-1)
X = np.array(matrix_x)
print(X)
print(X.shape) # (n_samples, 64)

pca = PCA(n_components=5)  # choose how many components you want
X_pca = pca.fit_transform(X)
print("Hello", np.array(X_pca).shape)
pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]
pc3 = X_pca[:, 2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d') # Create 3D axes


unique_groups = np.unique(y)
y = np.array(y)
print(f"y shape: {y.shape}, X_pca shape: {X_pca.shape}")
print(f"Unique groups: {np.unique(y)}")
for g in np.unique(y):
    print(f"Group '{g}': {np.sum(y == g)} samples")

for g in unique_groups:
    idx = y == g
    # Use ax.scatter for 3D
    ax.scatter(pc1[idx], pc2[idx], pc3[idx], label=g, s=200, edgecolors='black', linewidths=0.5)

# Fix variance ratio indices: PC1 is [0], PC2 is [1], PC3 is [2]
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")

plt.legend()
plt.title("3D PCA: Apoptosis")

plt.savefig(os.path.join(folder, "PCA_3D_apoptosis.png"), dpi=300)


y = np.array(y)
unique_groups = np.unique(y)

plt.figure()

for g in unique_groups:
    idx = y == g
    plt.scatter(pc1[idx], pc2[idx], label=g, s=80)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend()
plt.title("PCA: PC1 vs PC2")

plt.tight_layout()

plt.savefig(os.path.join(folder, "PCA_apoptosis_" + subfolder_masked_output + ".png"), dpi=300, bbox_inches='tight')   # <-- save high resolution
# plt.show()


