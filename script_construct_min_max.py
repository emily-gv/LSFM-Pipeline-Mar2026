from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
from scipy.stats import pearsonr
from matplotlib.pylab import plt
import os
import numpy as np
from PIL import Image

os.environ["QT_QPA_PLATFORM"] = "offscreen"

"""
Generate a 3D TIFF using your max/min PC data within your ROI
"""

################################# PARAMETERS ##########################
folder = '/home/emily/Desktop/SUMMER_2026/Sim/Sim/Affine/SyN'
map_name = '_Step53b_cc3_density_histnorm'
# list_iter = [10, 50, 100]
synthetic_data = "PCA_apoptosis_nosips_PC2_min_wholeHead_down3.csv" # REPEAT USING BOTH MIN AND MAX CSV (output from script_pca_min_max.R)

filepath_mask = 'Nosip_10.5_Tissues_Mask_WholeHead.tiff'  # Must be same mask you used from script_correlation_in_mask
downscale_masked_values = 3 # Int number, bigger than 1s
subfolder_name = 'Nosip_10.5_Mask_WholeHead' # Corresponding to folder output using filepaht_path in script_correlation_in_mask

########################################################################

mask = np.array(Image.open(os.path.join(folder, filepath_mask)))
print(mask.shape)
print(mask.dtype)

synthetic_data = np.loadtxt(os.path.join(folder, synthetic_data), delimiter=",")
print(synthetic_data.shape)

mask = functionReadTIFFMultipage(os.path.join(folder, filepath_mask), 8)
mask = mask > 0
mask_scaled = mask[::np.int16(downscale_masked_values), ::np.int16(downscale_masked_values), ::np.int16(downscale_masked_values)]

output_path = os.path.join(folder, "max_intesity.tiff")
recon_vol = np.zeros(mask_scaled.shape, dtype=np.float32)

print("mask shape:", mask_scaled.shape)
print("masked voxels:", np.count_nonzero(mask_scaled))
print("data values:", len(synthetic_data))

recon_vol[mask_scaled > 0] = synthetic_data # R output CSV
functionSaveTIFFMultipage(recon_vol, output_path, 8) # Check the arguments from the repo