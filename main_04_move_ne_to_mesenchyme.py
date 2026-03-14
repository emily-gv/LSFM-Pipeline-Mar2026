from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
import numpy as np
import os
import yaml
from AuxFunctions.config_loader import load_config

#########  PARAMETERS ########

# Load YAML config, print sample stats, create sample lookup dictionary
config_filename = "config.yml" 
config, sample_dict = load_config(config_filename)
n_samples = len(sample_dict)

print('----- 04 - MOVING NE TO MES -----', flush = True)

#########  END PARAMETERS ########

LABEL_MES = 50
LABEL_NE = 100
#---------------------------------------------------------------------------

for i in range(n_samples):
    sample_name = list(sample_dict.keys())[i] 
    print('Moving NE to Mes: ' + sample_name, flush = True)
    folder_sample = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)
    
    pre_ending_step_tissues = config["dest_file_tiff_tissue_segmentation_maskout_sedimentation"] #
    pre_ending_step_neural = config["dest_file_tiff_ne_segmentation_maskout_sedimentation"] #
    pre_ending_step_mesen = config["dest_file_tiff_mes_segmentation_maskout_sedimentation"] #
    ending_neural_clean = config["file_tiff_ne_correction"] #

    ending_step_tissues = config["dest_file_tiff_tissue_segmentation_ne_correction"] #
    ending_step_mesen = config["dest_file_tiff_mes_segmentation_ne_correction"] #
    ending_step_neural = config["dest_file_tiff_ne_segmentation_ne_correction"] #
    
    input_file_tissues_tiff_15  = os.path.join(folder_sample,sample_name + pre_ending_step_tissues)
    input_file_mesen_tiff_15    = os.path.join(folder_sample,sample_name + pre_ending_step_mesen)
    input_file_neural_tiff_15   = os.path.join(folder_sample,sample_name + pre_ending_step_neural)

    file_volume_Neural_3DSlicer     = os.path.join(folder_sample,sample_name + ending_neural_clean)

    output_file_neural_corrected_17 = os.path.join(folder_sample,sample_name + ending_step_neural)
    output_file_mesen_corrected_17  = os.path.join(folder_sample,sample_name + ending_step_mesen)
    output_file_tissues_corrected17 = os.path.join(folder_sample,sample_name + ending_step_tissues)

    #correct Neural, which is delete in Neural
    volume_Neural = functionReadTIFFMultipage(input_file_neural_tiff_15, 8)
    volume_Neural_3DSlicer = functionReadTIFFMultipage(file_volume_Neural_3DSlicer,8)
    
    if volume_Neural.shape == volume_Neural_3DSlicer.shape:

        labelValueNE = np.max(volume_Neural) # To keep the same values in the separated tiffs of each tissue
        volume_Neural_corrected = np.zeros_like(volume_Neural)
        volume_Neural_corrected = np.where(volume_Neural_3DSlicer > 0, labelValueNE, 0)
    
        functionSaveTIFFMultipage(volume_Neural_corrected, output_file_neural_corrected_17, 8)
    
        #correct neural, which is change label from Neural to Mesen
        diffVoxels = np.logical_xor(volume_Neural > 0, volume_Neural_corrected>0)
        del volume_Neural_corrected, volume_Neural, volume_Neural_3DSlicer
    
        volume_Mesen = functionReadTIFFMultipage(input_file_mesen_tiff_15,8)
        
        if volume_Mesen.shape == diffVoxels.shape:
    
            #extract label value
            labelValueMes = np.max(volume_Mesen) # To keep the same values in the separated tiffs of each tissue
            volume_Mesen_corrected = np.where(diffVoxels, labelValueMes, volume_Mesen) #where diffvoxels is 0,
            # get from volume_Mesen, where is true, assign the mesenchyme labels (that means, where in NE but moved to Mes)
            functionSaveTIFFMultipage(volume_Mesen_corrected, output_file_mesen_corrected_17, 8)
        
            #modify tissue volume. Mesen is the max value in the segmentation
            volume_Tissues_corrected = functionReadTIFFMultipage(input_file_tissues_tiff_15,8)
            volume_Tissues_corrected = np.where(diffVoxels, LABEL_MES, volume_Tissues_corrected)
            functionSaveTIFFMultipage(volume_Tissues_corrected, output_file_tissues_corrected17, 8)
            
            del volume_Tissues_corrected, volume_Mesen_corrected
            
            print ('-- ' + sample_name + ' CORRECTED', flush = True)
            
        else:
            print ('-- ' + sample_name + ' NOT CORRECTED, DIMENSIONS MISMATCH!', flush = True)
    
    else:
        
        print ('-- ' + sample_name + ' NOT CORRECTED, DIMENSIONS MISMATCH!', flush = True)
    