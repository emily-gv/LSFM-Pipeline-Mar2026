#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:56:50 2024

@author: lucas
"""
#from CellSegmentation.scriptStats import plots_tissue_perc_tiff, plot_IoU_tissue, get_IoU_tissue_window
import os
#from CellSegmentation.pipelineCellSegmentation import maskOutTIFF
from CellSegmentation.perona_malik import apply_perona_malik_3d, apply_isotropic_diffusion_3d, apply_perona_malik_3d_no_edge, normalize_perona_malik
import yaml
from AuxFunctions.config_loader import load_config
from AuxFunctions.setup_marker_paths import setup_marker_paths_main_05

# Load YAML config, print sample stats, create sample lookup dictionary
config_filename = "emilygv_config.yml" 
config, sample_dict = load_config(config_filename)
n_samples = len(sample_dict)

print('----- 05 - Cell maps -----', flush = True)

#To compare perona-malik against linear filtering
flag_debug = False
num_iter = config["iter_diffusion"]

for i in range(n_samples):
    sample_name = list(sample_dict.keys())[i] 
    print('Cellular dynamics density: ' + sample_name, flush = True)
    folder_sample = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)

    markers_dict = config["markers"]
    for m in markers_dict:
        if m.get("flag_cell_density"): 
            marker_name = m["name"]
            paths = setup_marker_paths_main_05(marker_name, config, folder_sample, sample_name)
            
            pre_ending_binary = paths["pre_ending_binary"]
            ending_step_tissues = paths["ending_step_tissues"]
            ending_density = paths["ending_density"]
            ending_density_histnorm = paths["ending_density_histnorm"]
            ending_density_isotropic = paths["ending_density_isotropic"]

            # Volumes masked
            # Tissue segmentation NE correction
            input_file_tissues_tiff_17     = ending_step_tissues
            # Marker segmentation corrected
            input_file_gen_tiff_15         = pre_ending_binary
            
            # Proliferation map in Mesenchyme
            output_file_prolif_tiff_gen     = ending_density
            output_file_prolif_tiff_gen_histnorm  = ending_density_histnorm 
            output_file_prolif_tiff_gen_isotropic   = ending_density_isotropic
            
            if os.path.exists(input_file_tissues_tiff_17) and os.path.exists(input_file_gen_tiff_15):
                print ('-- Creating cell maps: ' + sample_name, flush = True)
                apply_perona_malik_3d_no_edge(input_file_gen_tiff_15, input_file_tissues_tiff_17, output_file_prolif_tiff_gen, num_iter = num_iter, bitdepth = 8)
                # Hist norm of cell map
                normalize_perona_malik(output_file_prolif_tiff_gen, output_file_prolif_tiff_gen_histnorm)
                if flag_debug:
                    apply_isotropic_diffusion_3d(input_file_gen_tiff_15, input_file_tissues_tiff_17, output_file_prolif_tiff_gen_isotropic, num_iter = num_iter, bitdepth = 8)
            