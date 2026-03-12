import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # THIS SUPPRESSES ALL WARNINGS, turn off ('0/1/2') if something isn't working
# os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from TissueSegmentation.functionTileVolumeNB import functionTileVolumeNB
from TissueSegmentation.functionStrategyGen import generateMesenchymeEdge
from TissueSegmentation.functionRemoveFolderContent import functionRemoveFolderContent
from CellSegmentation.pipelineCellSegmentation import segmentCells, maskOutFolder
from CellSegmentation.scriptStats import plots_tissue_perc_folder, plots_cells_perc_folder
from CellSegmentation.scriptStats import plot_IoU_folder
from AuxFunctions.config_loader import load_config
from AuxFunctions.test_cuda import test_cuda
from AuxFunctions.setup_marker_paths import setup_marker_paths_main_02
from TissueSegmentation.functionCreateVolume import functionCreateVolume

import matplotlib.pyplot as plt
import time
import warnings
import sys
import torch
from datetime import datetime

from AuxFunctions.compress_TIFFs import compress_TIFFs_parallel

warnings.filterwarnings("ignore")

#---------------------------------------------------------------------------

# Test if CUDA is available 
test_cuda(True) # TRUE or FALSE

# Load YAML config, print sample stats, create sample lookup dictionary
config_filename = "emilygv_config.yml" 
config, sample_dict = load_config(config_filename)
n_samples = len(sample_dict)

#---------------------------------------------------------------------------

for i in range(n_samples):
    sample_name = list(sample_dict.keys())[i] 
    
    print('----- ANALYZING '+ sample_name +' -----', flush=True)
    start_time_sample = datetime.now()
    print(f"Started at: {start_time_sample}\n\n")


    folder_sample_output = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)
    if not os.path.exists(folder_sample_output):
        os.makedirs(folder_sample_output)
    folder_tissues_slices = os.path.join(folder_sample_output,config["folder_tissues_slices"])

    print('----- GENERATING TISSUE STATS -----', flush=True)
    vector_area_sample, vector_mesen_perc, vector_ne_perc = plots_tissue_perc_folder(folder_tissues_slices, folder_sample_output, sample_name)

    markers_dict = config["markers"]
    resolutionX = sample_dict[sample_name]["resX"]
    resolutionY = sample_dict[sample_name]["resY"]
    resolutionZ = sample_dict[sample_name]["resZ"]
    for m in markers_dict:
        if m.get("flag_segment"): 
            marker_name = m["name"]
            window_cellpose = m["window_cellpose"]
        
            paths = setup_marker_paths_main_02(marker_name, config, sample_dict, sample_name) # can do this outside of loop if the marker-specific paths are need after
            
            if marker_name != "nuclei":
                print(f'----- NOW TILING: {marker_name} -----', flush=True)
                functionTileVolumeNB(folder_input = paths["folder_input_slices"], folder_output = paths["folder_tiles"], sample_name = sample_name,
                patchSize = window_cellpose)
            
            start = time.time()
            start_time = datetime.now()

            print(f'----- NOW SEGMENTING: {marker_name} -----', flush=True) # flush=True prints immediately
            # segmentCells() produces a tiff file that looks like 1 image but is actually a tiff stack of ALL tiffs for sample
            segmentCells(folder_sample_output, paths["folder_tiles"], paths["folder_tiles_segmented_label"], paths["folder_tiles_segmented_binary"], 
                            paths["folder_slices_label"], paths["folder_slices_binary"], paths["dest_file_label_tiff"], paths["dest_file_binary_tiff"],
                                paths["path_cnn"], window_cellpose)
            
            print('----- CREATING VOLUME OF SEGMENTATION-----')
            functionCreateVolume(paths["folder_slices_label"], paths["dest_file_label_tiff"], resX=resolutionX, resY=resolutionY, resZ=resolutionZ, bith_depth = 16)
            functionCreateVolume(paths["folder_slices_binary"], paths["dest_file_binary_tiff"], resX=resolutionX, resY=resolutionY, resZ=resolutionZ, bith_depth = 8)


            end = time.time()
            end_time = datetime.now()
            duration = end - start 
            dest_file_txt = os.path.join(folder_sample_output, sample_name +'_' + marker_name + '_segmentation_time.txt')
            
            with open(dest_file_txt, 'a') as file:
                file.write(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"Duration: {duration:.2f} seconds\n\n")
            
            # THIS BLOCK IS JUST NOTES FOR REFERENCE
            # maskOutFolder(folder_slices_cells, folder_slices_tissue, folder_cell_masked)
            # folder_slices_cells: config[f"folder_{marker}_slices_label"])
                # Segmented cells
            # folder_slices_tissue: folder_tissues_slices: "Step05a_CSGreen_tissues" 
                # These are the tissue segmentations
            # config[f"folder_{marker}_masked_label"])
                # cell segmentation mask
            
            print('---------MASK OUT CELLS IN BACKGROUND--------', flush=True)
            maskOutFolder(paths["folder_slices_label"], folder_tissues_slices, paths["folder_masked_label"])
            maskOutFolder(paths["folder_slices_binary"], folder_tissues_slices, paths["folder_masked_binary"])

            print('---------GENERATE RAW STATS--------', flush=True)
            plots_cells_perc_folder(paths["folder_slices_binary"], vector_area_sample, vector_mesen_perc, folder_sample_output, sample_name, str_cells = f'{marker_name}_perc')
                # Plots percentage of marker cells in mesenchyme across slices
            plot_IoU_folder(paths["folder_masked_binary"], folder_sample_output, sample_name, str_description = marker_name)
                # The IoU here measures how similar one slice is to the next slice in the sequence
                    # Obviously is bad in my sample of 4 slices because they're like 350 slices apart each

            if m.get("flag_mesenchyme"):
                print('----- GENERATING MESENCHYME MASKS -----', flush=True)
                generateMesenchymeEdge(folder_tissues_slices, paths["folder_edge_mesen"])
                maskOutFolder(paths["folder_slices_label"], paths["folder_edge_mesen"], paths["folder_mesen_masked_label"])
                maskOutFolder(paths["folder_slices_binary"], paths["folder_edge_mesen"], paths["folder_mesen_masked_binary"])
                plot_IoU_folder(paths["folder_mesen_masked_binary"], folder_sample_output, sample_name, str_description = f"{marker_name}_mesenchyme")

            functionRemoveFolderContent(paths["folder_tiles"])

    plt.close('all')
    
    final_time_sample = datetime.now()
    print(f"\nFINALIZED {sample_name} Ended at: {final_time_sample}\n", flush=True)