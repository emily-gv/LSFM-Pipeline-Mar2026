import os
import sys
import time
from datetime import datetime
import warnings

from TissueSegmentation.functionTileVolumeNB import functionTileVolumeNB, functionTileVolumeNB_parallel, functionCopyImageAsPNG, functionCopyImageAsPNG_parallel
from TissueSegmentation.functionChangeSize import functionChangeSize
from TissueSegmentation.predict import segment_folder
from TissueSegmentation.functionMergeProcessedTilesNB import functionMergeProcessedTilesNB
from TissueSegmentation.functionCreateVolume import functionCreateVolume
from TissueSegmentation.functionStrategyGen import functionGenerateNuclear, compute_abs_diff_images, compute_perc_diff_images, filter_images, function_overlay_artifacts, create_volume_artifact_prediction
from TissueSegmentation.GAN_FUNCTIONS.functionGenBlur import functionGenBlur
from TissueSegmentation.functionRemoveFolderContent import functionRemoveFolderContent
from CellSegmentation.scriptStats import plot_IoU_folder
import matplotlib.pyplot as plt
from AuxFunctions.compress_TIFFs import compress_TIFFs_parallel
from AuxFunctions.sort_files import sort_files
from AuxFunctions.config_loader import load_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Figure out which flag won't break your computer
# But offscreen if guaranteed because it won't allow pop-ups
os.environ["QT_QPA_PLATFORM"] = "offscreen"


# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/emily/anaconda3/envs/emilygv_thesis' # Tell XLA where CUDA is
# os.environ['PATH'] = '/home/emily/anaconda3/envs/emilygv_thesis/bin:' + os.environ['PATH'] # Add CUDA binaries to PATH
# os.environ['LD_LIBRARY_PATH'] = '/home/emily/anaconda3/envs/emilygv_thesis/lib:' + os.environ.get('LD_LIBRARY_PATH', '') # Add CUDA libraries to PATH

#---------------------------------------------------------------------------

import tensorflow as tf

gg = tf.config.list_physical_devices("GPU")
print("GPUs found: %d" % len(gg))

for g in gg:
    print("\t%s: %s" % (g.device_type, g.name))

config_filename = "config.yml" 
config, sample_dict = load_config(config_filename)
n_samples = len(sample_dict)

#Trained segmenters
size_analysis = 1024
size_tiling = 3000

folder_architectures = config["folder_CNN_architectures"]

# Segmentation architectures
path_tissue_unet    = os.path.join(folder_architectures,config["filename_CNN_tissues"])
path_nuclei_unet    = os.path.join(folder_architectures,config["filename_CNN_nuclei"])
path_phh3_unet      = os.path.join(folder_architectures,config["filename_CNN_phh3"])

# Generators and discriminators for artifacts

path_model_generator_signal_loss        = os.path.join(folder_architectures,config["filename_CNN_signalLoss_gen"])
path_model_discriminator_signal_loss    = os.path.join(folder_architectures,config["filename_CNN_signalLoss_disc"])

# path_model_generator_shadow_weights     = os.path.join(folder_architectures,'SHADOW','shadow-generator_final.h5')        
path_model_generator_shadow             = os.path.join(folder_architectures,config["filename_CNN_shadow_gen"])
path_model_discriminator_shadow         = os.path.join(folder_architectures,config["filename_CNN_shadow_disc"])

# path_model_generator_blur_weights       = os.path.join(folder_architectures,'BLUR','blur-generator_final.h5')        
path_model_generator_blur               = os.path.join(folder_architectures,config["filename_CNN_blur_gen"])
path_model_discriminator_blur           = os.path.join(folder_architectures,config["filename_CNN_blur_disc"])

path_random_forest = os.path.join(folder_architectures,'random_forest')

list_architectures = [path_tissue_unet, path_nuclei_unet, path_phh3_unet, \
                      path_model_generator_signal_loss, path_model_discriminator_signal_loss, \
                          path_model_generator_shadow, path_model_discriminator_shadow, \
                              path_model_generator_blur, path_model_discriminator_blur, path_random_forest]
for file in list_architectures:
    if not os.path.exists(file):
        print(f"File does not exist: {file}", flush=True)
        sys.exit(1)

LABEL_MES = 50
LABEL_NE = 100

if not os.path.exists(config["folder_output"]):
    os.makedirs(config["folder_output"])

#Added to clean GPU memory after using it
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

for i in range(n_samples):
    sample_name = list(sample_dict.keys())[i] 
    resolutionX = sample_dict[sample_name]["resX"]
    resolutionY = sample_dict[sample_name]["resY"]
    resolutionZ = sample_dict[sample_name]["resZ"]
    print('----- Processing: '+ sample_name +' -----', flush=True)
    start_time_sample = datetime.now()
    print(f"Started at: {start_time_sample}\n\n")

    folder_CSGreen = os.path.join(sample_dict[sample_name]["folder_slices"],sample_dict[sample_name]["subfolder_nuclei"])
    if not os.path.exists(folder_CSGreen):
        print(f"Directory does not exist: {folder_CSGreen}", flush=True)
        sys.exit(1)
        
    folder_sample = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)
    if not os.path.exists(config["folder_output"]):
        print("Directory does not exist: " + config["folder_output"], flush=True)
        sys.exit(1)
        
    if not os.path.exists(folder_sample): os.makedirs(folder_sample)
    
    start_time = time.time()
    print('-----01 - TILING CSGREEN-----', flush=True)
    folder_CSGreen_00 = os.path.join(folder_sample,config["folder_CSGreen_copy"])
    if not os.path.exists(folder_CSGreen_00): os.makedirs(folder_CSGreen_00)
    functionCopyImageAsPNG(folder_input = folder_CSGreen, folder_output = folder_CSGreen_00, sample_name = sample_name)
    #functionCopyImageAsPNG_parallel(folder_input = folder_CSGreen, folder_output = folder_CSGreen_00, sample_name = sample_name, n_workers=os.cpu_count()-1)

    folder_CSGreen_tiles = os.path.join(folder_sample,config["folder_nuclei_tiles"])
    if not os.path.exists(folder_CSGreen_tiles): os.makedirs(folder_CSGreen_tiles)
    functionTileVolumeNB(folder_input = folder_CSGreen, folder_output = folder_CSGreen_tiles, sample_name = sample_name, patchSize = size_tiling)
    #functionTileVolumeNB_parallel(folder_input = folder_CSGreen, folder_output = folder_CSGreen_tiles, sample_name = sample_name, \
        #patchSize = size_tiling, n_workers=os.cpu_count()-1)

    end_time = time.time()
    elapsed_time_01 = end_time - start_time
    print((f"01 Tiling COMPLETE: {elapsed_time_01:.2f} seconds.\n"))
        
    start_time = time.time()
    print('-----01b - Generators CSGREEN-----', flush=True)
    folder_CSGreen_tiles_gen = os.path.join(folder_sample,config["folder_CSGreen_tiles_generated"])
    if not os.path.exists(folder_CSGreen_tiles_gen): os.makedirs(folder_CSGreen_tiles_gen)
    dictionary_prediction_artifact = functionGenerateNuclear(folder_CSGreen_tiles, folder_CSGreen_tiles_gen, path_model_generator_signal_loss, path_model_generator_shadow, path_model_generator_blur,\
                          path_model_discriminator_signal_loss, path_model_discriminator_shadow, path_model_discriminator_blur, path_random_forest)
    
    dest_file_tiff_artifact_prediction = os.path.join(folder_sample,sample_name +config["dest_file_tiff_artifact_prediction"])
    create_volume_artifact_prediction(folder_CSGreen_tiles, dest_file_tiff_artifact_prediction, dictionary_prediction_artifact)
        
    folder_01c_diff_gen_img = os.path.join(folder_sample,config["folder_diff_gen_img"])
    folder_01c_diff_gen_perc_npy = os.path.join(folder_sample,config["folder_diff_gen_perc_npy"])
    folder_01c_diff_gen_perc_img = os.path.join(folder_sample,config["folder_diff_gen_perc_img"])
    if not os.path.exists(folder_01c_diff_gen_img): os.makedirs(folder_01c_diff_gen_img)
    if not os.path.exists(folder_01c_diff_gen_perc_npy): os.makedirs(folder_01c_diff_gen_perc_npy)
    if not os.path.exists(folder_01c_diff_gen_perc_img): os.makedirs(folder_01c_diff_gen_perc_img)
    compute_abs_diff_images(folder_CSGreen_tiles, folder_CSGreen_tiles_gen, folder_01c_diff_gen_img)
    compute_perc_diff_images(folder_CSGreen_tiles, folder_CSGreen_tiles_gen, folder_01c_diff_gen_perc_npy, folder_01c_diff_gen_perc_img)
    
    end_time = time.time()
    elapsed_time_01b = end_time - start_time
    print((f"01b Generators COMPLETE: {elapsed_time_01b:.2f} seconds.\n"))
    
    start_time = time.time()
    print('-----01d - Generator Blur CSGREEN-----', flush=True)
    folder_CSGreen_tiles_d = os.path.join(folder_sample,config["folder_CSGreen_tiles_blur_gen"])
    if not os.path.exists(folder_CSGreen_tiles_d): os.makedirs(folder_CSGreen_tiles_d)
    functionGenBlur(folder_CSGreen_tiles, folder_CSGreen_tiles_d, path_model_generator_blur)
    
    folder_01e_diff_gen_blur_img = os.path.join(folder_sample,config["folder_diff_gen_blur_img"])
    if not os.path.exists(folder_01e_diff_gen_blur_img): os.makedirs(folder_01e_diff_gen_blur_img)
    compute_abs_diff_images(folder_CSGreen_tiles, folder_CSGreen_tiles_d, folder_01e_diff_gen_blur_img)
    
    folder_01f_diff_gen_blur_img_filt = os.path.join(folder_sample,config["folder_diff_gen_blur_img_filt"])
    if not os.path.exists(folder_01f_diff_gen_blur_img_filt): os.makedirs(folder_01f_diff_gen_blur_img_filt)
    filter_images(folder_01e_diff_gen_blur_img, folder_01f_diff_gen_blur_img_filt)
    
    end_time = time.time()
    elapsed_time_01d = end_time - start_time
    print(f"01d Blur Gen COMPLETE: {elapsed_time_01d:.2f} seconds.\n")

    start_time = time.time()
    print('-----02 - DOWNSIZING CSGREEN-----', flush=True)
    folder_CSGreen_output_downsample = os.path.join(folder_sample,config["folder_CSGreen_output_downsample"])
    if not os.path.exists(folder_CSGreen_output_downsample):
        os.makedirs(folder_CSGreen_output_downsample)
    functionChangeSize(folder_CSGreen_tiles_gen, folder_CSGreen_output_downsample, dest_size = size_analysis,
                        ending = '.png')

    print('-----03 - TISSUE SEGMENTATION-----', flush=True)
    folder_tissue_segmented_downsample = os.path.join(folder_sample, config["folder_tissue_segmented_downsample"])
    if not os.path.exists(folder_tissue_segmented_downsample):
        os.makedirs(folder_tissue_segmented_downsample)
    segment_folder(path_tissue_unet, folder_CSGreen_output_downsample, folder_tissue_segmented_downsample)
    
    end_time = time.time()
    elapsed_time_03 = end_time - start_time
    print(f"03 Tissue Seg: {elapsed_time_03:.2f} seconds.\n")

    print('-----04 - UPSIZING TISSUE SEGMENTATION-----', flush=True)
    folder_CSGreen_tissues = os.path.join(folder_sample,config["folder_CSGreen_tissues"])
    if not os.path.exists(folder_CSGreen_tissues): os.makedirs(folder_CSGreen_tissues)
    functionChangeSize(folder_tissue_segmented_downsample, folder_CSGreen_tissues, dest_size = size_tiling,
                        ending = '.png')

    print('-----05 - MERGING SLICES OF TISSUE SEGMENTATION-----', flush=True)
    folder_tissues_slices = os.path.join(folder_sample,config["folder_tissues_slices"])
    if not os.path.exists(folder_tissues_slices):
        os.makedirs(folder_tissues_slices)
    functionMergeProcessedTilesNB(folder_CSGreen_tissues, folder_tissues_slices, folder_CSGreen_tiles)

    folder_diff_gen_slices = os.path.join(folder_sample,config["folder_diff_gen_slices"])
    if not os.path.exists(folder_diff_gen_slices): os.makedirs(folder_diff_gen_slices)
    functionMergeProcessedTilesNB(folder_01c_diff_gen_img, folder_diff_gen_slices, folder_CSGreen_tiles)
    
    folder_diff_gen_blur_slices = os.path.join(folder_sample,config["folder_diff_gen_blur_slices"])
    if not os.path.exists(folder_diff_gen_blur_slices): os.makedirs(folder_diff_gen_blur_slices)
    functionMergeProcessedTilesNB(folder_01f_diff_gen_blur_img_filt, folder_diff_gen_blur_slices, folder_CSGreen_tiles)

    print('-----06 - CREATING VOLUME OF TISSUE SEGMENTATION-----', flush=True)
    dest_file_tiff_tissue_segmentation = os.path.join(folder_sample,sample_name + config["dest_file_tiff_tissue_segmentation"])
    functionCreateVolume(folder_tissues_slices, dest_file_tiff_tissue_segmentation, resX=resolutionX, resY=resolutionY, resZ=resolutionZ)
    
    dest_file_tiff_ne_segmentation = os.path.join(folder_sample,sample_name + config["dest_file_tiff_ne_segmentation"])
    functionCreateVolume(folder_tissues_slices, dest_file_tiff_ne_segmentation, resX=resolutionX, resY=resolutionY, resZ=resolutionZ, label = LABEL_NE)
    
    dest_file_tiff_mes_segmentation = os.path.join(folder_sample,sample_name + config["dest_file_tiff_mes_segmentation"])
    functionCreateVolume(folder_tissues_slices, dest_file_tiff_mes_segmentation, resX=resolutionX, resY=resolutionY, resZ=resolutionZ, label = LABEL_MES)
    
    print('-----06b - CREATING VOLUMES OF ARTIFACT DETECTION-----', flush=True)
    dest_file_tiff_diff_gen = os.path.join(folder_sample,sample_name + config["dest_file_tiff_diff_gen"])
    functionCreateVolume(folder_diff_gen_slices, dest_file_tiff_diff_gen)
    
    dest_file_tiff_diff_gen_blur = os.path.join(folder_sample,sample_name + config["dest_file_tiff_diff_gen_blur"])
    functionCreateVolume(folder_diff_gen_blur_slices, dest_file_tiff_diff_gen_blur)
    
    print('-----06d - CREATING IMAGES OF ARTIFACT DETECTION-----', flush=True)
    folder_artifact_overlay = os.path.join(folder_sample,config["folder_artifact_overlay"])
    if not os.path.exists(folder_artifact_overlay): os.makedirs(folder_artifact_overlay)
    function_overlay_artifacts(folder_CSGreen_00, folder_diff_gen_slices, folder_diff_gen_blur_slices, folder_artifact_overlay)
    
    print('-----06e - ANALYSIS-----', flush = True)
    _ = plot_IoU_folder(folder_tissues_slices, folder_sample, sample_name, str_description = 'raw_Sample')
    _ = plot_IoU_folder(folder_tissues_slices, folder_sample, sample_name, str_description = 'raw_NE', threshold = 100)
    
    # # Save the result to a text file
    dest_file_txt = os.path.join(folder_sample,sample_name +'_tissue_segmentation_time.txt')
    with open(dest_file_txt, "w") as file:
        file.write(f"01 Tiling: {elapsed_time_01:.2f} seconds.\n")
        file.write(f"01b Generators: {elapsed_time_01b:.2f} seconds.\n")
        file.write(f"01d Blur Gen: {elapsed_time_01d:.2f} seconds.\n")
        file.write(f"03 Tissue Seg: {elapsed_time_03:.2f} seconds.\n")
        
    plt.close('all')
    
    functionRemoveFolderContent(folder_CSGreen_tissues)
    functionRemoveFolderContent(folder_tissue_segmented_downsample)
    functionRemoveFolderContent(folder_CSGreen_output_downsample)
    
    functionRemoveFolderContent(folder_01c_diff_gen_perc_img)
    functionRemoveFolderContent(folder_01e_diff_gen_blur_img)
    functionRemoveFolderContent(folder_01f_diff_gen_blur_img_filt)
    functionRemoveFolderContent(folder_01c_diff_gen_img)
    
    final_time_sample = datetime.now()
    print(f"\nFINALIZED {sample_name} Ended at: {final_time_sample}\n", flush=True)
