import numpy as np
from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
from AuxFunctions.config_loader import load_config
import os
import yaml
import sys
from AuxFunctions.setup_marker_paths import setup_marker_paths_main_03

#---------------------------------------------------------------------------
flag_tissue = True
#---------------------------------------------------------------------------

# MAKE SURE YOU'VE RUN THE NECESSARY CELL MARKER SEGMENTATION FOR EACH OF THESE BEFORE THIS

# Load YAML config, print sample stats, create sample lookup dictionary
config_filename = "config.yml" 
config, sample_dict = load_config(config_filename)
n_samples = len(sample_dict)

print('-----03 - CREATING VOLUME OF TISSUE SEGMENTATION-----', flush = True)
#---------------------------------------------------------------------------

for i in range(n_samples):
    sample_name = list(sample_dict.keys())[i] 
    print('Correcting: ' + sample_name, flush = True)
    folder_sample = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)
    
    # This is the mask that needs to be designed in 3D slicer
    origin_file_volume_mask = os.path.join(folder_sample,sample_name + config["file_tiff_maskout_sedimentation"])
    if not os.path.exists(origin_file_volume_mask):
        print(f"Mask cannot be found: '{origin_file_volume_mask}'", flush = True)
        sys.exit(1)
    volumeMask = functionReadTIFFMultipage(origin_file_volume_mask, 8)

    # Toggle on/off if you've run it on the tissue before
    if flag_tissue==True:    
        # Volumes to mask, tissue segmentation
        origin_file_tiff_tissue_segmentation = config["dest_file_tiff_tissue_segmentation"] #
        origin_file_tiff_ne_segmentation = config["dest_file_tiff_ne_segmentation"] #
        origin_file_tiff_mes_segmentation = config["dest_file_tiff_mes_segmentation"] #
        origin_file_tiff_diff_gen = config["dest_file_tiff_diff_gen"] #
        origin_file_tiff_diff_gen_blur = config["dest_file_tiff_diff_gen_blur"] #
        origin_file_tiff_artifact_prediction = config["dest_file_tiff_artifact_prediction"] #
        dest_file_tiff_tissue_segmentation = config["dest_file_tiff_tissue_segmentation_maskout_sedimentation"] #
        dest_file_tiff_ne_segmentation = config["dest_file_tiff_ne_segmentation_maskout_sedimentation"] #
        dest_file_tiff_mes_segmentation = config["dest_file_tiff_mes_segmentation_maskout_sedimentation"] #
        dest_file_tiff_diff_gen = config["dest_file_tiff_diff_gen_maskout_sedimentation"] #
        dest_file_tiff_diff_gen_blur = config["dest_file_tiff_diff_gen_blur_maskout_sedimentation"] #
        dest_file_tiff_artifact_prediction = config["dest_file_tiff_artifact_prediction_maskout_sedimentation"] #

        volumes_to_mask = [
            origin_file_tiff_tissue_segmentation,
            origin_file_tiff_ne_segmentation,
            origin_file_tiff_mes_segmentation,
            origin_file_tiff_diff_gen,
            origin_file_tiff_diff_gen_blur,
            origin_file_tiff_artifact_prediction,
        ]

        volumes_masked = [
            dest_file_tiff_tissue_segmentation,
            dest_file_tiff_ne_segmentation,
            dest_file_tiff_mes_segmentation,
            dest_file_tiff_diff_gen,
            dest_file_tiff_diff_gen_blur,
            dest_file_tiff_artifact_prediction,
        ]
        n_volumes_to_mask = len(volumes_to_mask)
        for j in range(n_volumes_to_mask):

            input_volume_tiff       = os.path.join(folder_sample,sample_name + volumes_to_mask[j])
            output_volume_Masked    = os.path.join(folder_sample,sample_name + volumes_masked[j])
            if os.path.exists(input_volume_tiff):
                volumeTissue_06 = functionReadTIFFMultipage(input_volume_tiff, 8)
                volumeMasked = np.zeros_like(volumeTissue_06)
                volumeMasked = np.where(volumeMask > 0, volumeTissue_06, volumeMasked)
                functionSaveTIFFMultipage(volumeMasked, output_volume_Masked, 8)
                print (sample_name + volumes_to_mask[j] + ' MASKED', flush = True)
            else:
                print (sample_name + volumes_to_mask[j] + ' not masked', flush = True)

    markers_dict = config["markers"]
    for m in markers_dict:
        if m.get("flag_remove_sedimentation"): 
            marker_name = m["name"]
            paths = setup_marker_paths_main_03(marker_name, config, folder_sample, sample_name)

            volumes_to_mask = [
                paths["origin_file_label_tiff"],
                paths["origin_file_binary_tiff"]
            ]
            volumes_masked = [
                paths["dest_file_label_tiff"],
                paths["dest_file_binary_tiff"]
            ]
            n_volumes_to_mask = len(volumes_to_mask)
            for j in range(n_volumes_to_mask):
                input_volume_tiff = volumes_to_mask[j]
                output_volume_Masked = volumes_masked[j]
                if os.path.exists(input_volume_tiff):
                    volumeTissue_06 = functionReadTIFFMultipage(input_volume_tiff, 8)
                    volumeMasked = np.zeros_like(volumeTissue_06)
                    volumeMasked = np.where(volumeMask > 0, volumeTissue_06, volumeMasked)
                    functionSaveTIFFMultipage(volumeMasked, output_volume_Masked, 8)
                    print (volumes_to_mask[j] + ' MASKED', flush = True)
                else:
                    print (volumes_to_mask[j] + ' not masked', flush = True)