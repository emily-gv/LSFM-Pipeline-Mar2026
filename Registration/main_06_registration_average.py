import os
import glob
import sys
import SimpleITK as sitk
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_path)
from Registration.functionsRegistration import resliceVolumeITK, numpy_to_sitk, register_groupwise_samples, \
    register_groupwise_syn_samples, compute_IoU_samples, SuppressOutput
import Registration.functionsLandmarks as flms
from Registration.functionsAtlasCreation import create_atlases
from CellSegmentation.scriptStats import plot_IoU_tissue
from matplotlib.pylab import plt

from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
import numpy as np
import yaml
import time
import ants
import logging
os.environ["QT_QPA_PLATFORM"] = "offscreen"

#################### PARAMETERS ###############################

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

vector_sample_name = [sample["name"] for sample in config["samples"]]    
n_samples = len(vector_sample_name)
sample_groups = [sample["group"] for sample in config["samples"]]
n_groups = len(set(sample_groups))
sample_ages = [sample["age"] for sample in config["samples"]]
n_ages = len(set(sample_ages))

print('n samples: ' + str(n_samples), flush = True)
print('n groups: ' + str(n_groups), flush = True)
print('n ages: ' + str(n_ages), flush = True)

sample_dict = {s["name"]: s for s in config["samples"]}

# RegType_LMbased = 'Sim' # Use 'Sim' or 'Rigid'

#############################################################################

ending_moving_volumes = [config[key_moving_volume["moving_volume"]] for key_moving_volume in config["list_moving_volumes"]]
print(ending_moving_volumes)
n_ending_moving_volumes = len(ending_moving_volumes)

ending_LMbased_moved_volumes = [name_moved_volume["moved_volume"] for name_moved_volume in config["list_moved_lm_volumes"]]
print(ending_LMbased_moved_volumes)

ending_Sim_moved_volumes = [name_moved_volume["moved_volume"] for name_moved_volume in config["list_moved_sim_volumes"]]
print(ending_Sim_moved_volumes)

ending_Affine_moved_volumes = [name_moved_volume["moved_volume"] for name_moved_volume in config["list_moved_affine_volumes"]]
print(ending_Affine_moved_volumes)

ending_SyN_moved_volumes = [name_moved_volume["moved_volume"] for name_moved_volume in config["list_moved_syn_volumes"]]
print(ending_SyN_moved_volumes)

if (n_ending_moving_volumes != len(ending_LMbased_moved_volumes)) \
    or (n_ending_moving_volumes != len(ending_Sim_moved_volumes)) \
    or (n_ending_moving_volumes != len(ending_Affine_moved_volumes)) \
    or (n_ending_moving_volumes != len(ending_SyN_moved_volumes)):
    print('ERROR: No correspondance between Moving and Moved volumes. Check YML file.', flush=True)
    sys.exit(1)

n_samples = len(vector_sample_name)

constDivide = 1



################ INTENSITY-BASED SIMILARITY (ROT, TRANS, SCALING) REGISTRATION #####################

flag_make_aux_mask = False

ending_Sim_folder = config["ending_sim_folder_output"]
ending_Affine_folder = config["ending_affine_folder_output"]
ending_SyN_folder = config["ending_syn_folder_output"]

# scale_registration = 4 # Intensity Sim and Affine can be done in 2, but affine needs more downsampling
scale_registration = config["scale_registration"]

aff_iterations = (200, 100, 10, 10)
syn_iterations = (200, 100, 10, 10)

#############################################################################

        
def flip_volume_and_lms(folder_group, folder_sample, sample_name, config, ending_lms, ending_moving_volumes):
    
    sample_name_mirrored = sample_name + config["ending_mirror_volumes"]
    
    folder_mirrored = os.path.join(folder_group, sample_name_mirrored)
    
    if not os.path.exists(folder_mirrored): os.makedirs(folder_mirrored)
    
    fullpath_csv = os.path.join(folder_sample, sample_name + config["ending_lms"])
    flms.mirror_lms(fullpath_csv, folder_sample, sample_name, folder_mirrored, sample_name_mirrored, config["swap_pairs"], axis = 1)
    
    # Mirror all tiff in the folder
    tiff_files_endings = ending_moving_volumes
    for tiff_file_end in tiff_files_endings:
        tiff_file           = os.path.join(folder_sample, sample_name + tiff_file_end)
        tiff_file_mirrored  = os.path.join(folder_mirrored, sample_name + config["ending_mirror_volumes"] + tiff_file_end)
        # print(tiff_file)
        volume = functionReadTIFFMultipage(tiff_file, 8)
        volume_mirrored = np.flip(volume, axis=0)
        
        functionSaveTIFFMultipage(volume_mirrored, tiff_file_mirrored, 8)
        
    return sample_name_mirrored

def correct_labels_3d(array, valid_labels):
    """
    Corrects invalid values in a 3D NumPy array by replacing them with the nearest valid label.
    
    Parameters:
    - array: A 3D NumPy array containing integer labels.
    - valid_labels: A sorted list or array of known valid labels.
    
    Returns:
    - A new 3D NumPy array with corrected values.
    """
    valid_labels = np.array(valid_labels)  # Convert to NumPy array for efficiency
    corrected_array = array.copy()
    
    # Identify invalid values
    invalid_mask = ~np.isin(array, valid_labels)
    invalid_values = array[invalid_mask]
    
    if invalid_values.size > 0:
        # Find the closest valid label for each invalid value
        nearest_values = valid_labels[np.argmin(np.abs(valid_labels[:, None] - invalid_values), axis=0)]
        corrected_array[invalid_mask] = nearest_values  # Replace invalid values
    
    return corrected_array

def register_images_sim_ants(fixed, moving, aff_iterations, affine_global):
    """Perform affine registration and return the transformation."""
    # Convert images to float32
    
    # Convert images to float32
    fixed1_ants = ants.from_numpy(fixed)
    moving1_ants = ants.from_numpy(moving)

    reg1 = ants.registration(fixed = fixed1_ants, moving = moving1_ants, type_of_transform='Similarity', aff_metric='GC', \
                         aff_iterations=aff_iterations, aff_shrink_factors=(5, 3, 2, 1), aff_smoothing_sigmas=(30, 15, 10, 5), \
                                 verbose=False, initial_transform=["Identity"])

    affine1 = reg1['fwdtransforms'][0]  # Path to affine matrix
    tranform_back = reg1['invtransforms'][0]
    return affine1, tranform_back

def register_images_affine_ants(fixed, moving, aff_iterations, affine_global):
    """Perform affine registration and return the transformation."""

    # Convert images to float32
    fixed1_ants = ants.from_numpy(fixed)
    moving1_ants = ants.from_numpy(moving)
    
    with SuppressOutput():
        reg1 = ants.registration(fixed = fixed1_ants.clone(), moving = moving1_ants.clone(), type_of_transform='Affine', aff_metric='GC', \
                             aff_iterations=aff_iterations, aff_shrink_factors=(5, 3, 2, 1), aff_smoothing_sigmas=(30, 15, 10, 5), \
                                     verbose=False, initial_transform=["Identity"])

    affine1 = reg1['fwdtransforms'][0]  # Path to affine matrix
    tranform_back = reg1['invtransforms'][0]

    return affine1, tranform_back

def main():
    
    if config["flag_mirror_volumes"]:
        sampleNames_to_add = []
        for sample_name in vector_sample_name:
            print('Mirroring: ' + sample_name, flush = True)
            sample_name_mirrored = sample_name + config["ending_mirror_volumes"] 
            #flip_volume_and_lms(workFolder, sample_name, ending_lms = ending_original_38lms, flag_set_38 = True)
            folder_age = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"])
            folder_sample = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)
            
            if not(config["flag_skip_mirroring"]):
                flip_volume_and_lms(folder_age, folder_sample, sample_name, config, config["ending_lms"], ending_moving_volumes)
            sampleNames_to_add.append(sample_name_mirrored)
            # Addition to dictionary:
            sample_dict[sample_name_mirrored] = sample_dict[sample_name]
        vector_sample_name.extend(sampleNames_to_add)
    
    pathLMsGMObjective = os.path.join(config["folder_output"], config["reference_landmarks"])
    pathVolumeObjective = os.path.join(config["folder_output"], config["reference_volume_tiff"])
    regTypeFolder_LMbased = os.path.join(config["folder_output"], config["registration_type_LMbased"])
    n_volumes = n_ending_moving_volumes
    # nSamples = len(sampleNames)
    
    if not os.path.exists(regTypeFolder_LMbased):
        os.mkdir(regTypeFolder_LMbased)
    
    
    ############## SIM REG USING LMS ################
    
    if not(config["flag_skip_reg_lm"]):
        for sample_name in vector_sample_name:
            
            # print('---------------------------------------')
            print(regTypeFolder_LMbased + ' registration: ' + sample_name, flush = True)
        
            #-------------- Moving ------------------------------
            # folder = os.path.join(workFolder, sampleName)
            folder_sample = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)
            folderOutput_LMbased =  os.path.join(regTypeFolder_LMbased, sample_name + config["ending_lm_folder_output"])
        
            try:
                os.mkdir(folderOutput_LMbased)
            except OSError as error:
                print(error)
        
            pathLMsGMMoving =   os.path.join(folder_sample, sample_name + config["ending_lms"])
            pathLMsGMMoved = os.path.join(folderOutput_LMbased, sample_name + config["ending_lm_registered_lms"])
        
            pathTransformationTXT = os.path.join(folderOutput_LMbased, sample_name + config["ending_lm_transformation_txt"])
            pathTransformationTFM = os.path.join(folderOutput_LMbased, sample_name + config["ending_lm_transformation_tfm"])
        
            pathTransformationInvTXT = os.path.join(folderOutput_LMbased, sample_name + config["ending_lm_transformation_inverse_txt"])
            pathTransformationInvTFM = os.path.join(folderOutput_LMbased, sample_name + config["ending_lm_transformation_inverse_tfm"])
        
            #-------------- Moving ------------------------------
        
            pathVolumeFixedCorrected = os.path.join(folderOutput_LMbased,'VolumeFixedCorrected.tiff')
            
            fixed_im = sitk.ReadImage(pathVolumeObjective)
    
            sourcePoints = flms.convertLMsToITKPoints(pathLMsGMMoving, constDivide = constDivide)
            targetPoints = flms.convertLMsToITKPoints(pathLMsGMObjective , constDivide = constDivide)
        
        
            if 'Sim' in config["registration_type_LMbased"]:
                InitTx = sitk.Similarity3DTransform()
            elif 'Rigid' in config["registration_type_LMbased"]:
                InitTx = sitk.VersorRigid3DTransform()
            else:
                print('------- Transformation not correctly defined!!! -------', flush = True)
        
            landmarkTransformITKFilter = sitk.LandmarkBasedTransformInitializerFilter()
            landmarkTransformITKFilter.SetFixedLandmarks(targetPoints)
            landmarkTransformITKFilter.SetMovingLandmarks(sourcePoints)
            landmarkTransformITKFilter.SetReferenceImage(fixed_im)
            landmarkTransformITK = landmarkTransformITKFilter.Execute(InitTx)
        
            sitk.WriteTransform(landmarkTransformITK, pathTransformationTXT)
            sitk.WriteTransform(landmarkTransformITK, pathTransformationTFM)
        
            landmarkTransformITKInverse = landmarkTransformITK.GetInverse()
            sitk.WriteTransform(landmarkTransformITKInverse, pathTransformationInvTXT)
            sitk.WriteTransform(landmarkTransformITKInverse, pathTransformationInvTFM)
        
            if os.path.exists(pathLMsGMMoving):
                GMPoints = flms.getLMs(pathLMsGMMoving, constDivide = constDivide)
                LMsGMMoved = [landmarkTransformITKInverse.TransformPoint(p) for p in GMPoints]
                flms.saveLMs(pathLMsGMMoved, LMsGMMoved, constDivide = constDivide)
            else:
                print('-----No LMs!!!------', flush = True)
                
            for i in range(n_volumes):
                print(ending_moving_volumes[i])
                print(ending_LMbased_moved_volumes[i])
                pathMovingVolume = os.path.join(folder_sample, sample_name + ending_moving_volumes[i])
                pathResultVolume = os.path.join(folderOutput_LMbased , sample_name + ending_LMbased_moved_volumes[i])
                resliceVolumeITK(pathMovingVolume, pathResultVolume, pathVolumeObjective, pathVolumeFixedCorrected,
                                  landmarkTransformITK, constDivide = constDivide)
                
                #Analysis
                plot_IoU_tissue(pathResultVolume, folderOutput_LMbased, sample_name, str_description = ending_LMbased_moved_volumes[i], threshold = 1)
                plt.close('all')
              
        if config["flag_compute_similarity_all_volumes"]:
            compute_IoU_samples(vector_sample_name, ending_LMbased_moved_volumes, regTypeFolder_LMbased, config["ending_lm_folder_output"])
        else:
            compute_IoU_samples(vector_sample_name, [ending_LMbased_moved_volumes[0]], regTypeFolder_LMbased, config["ending_lm_folder_output"])
    
    ############## SIM REG USING Intensity ################
    print('----- SIM REG USING Intensity -----', flush = True)
    
    output_folder_intensity_sim = register_groupwise_samples(register_images_sim_ants, aff_iterations, sampleNames = vector_sample_name, input_folder = regTypeFolder_LMbased, \
                                                              ending_input_folder = config["ending_lm_folder_output"], \
                      ending_input_volumes = ending_LMbased_moved_volumes, \
                          ending_output_folder = config["ending_sim_folder_output"], ending_output_volumes = ending_Sim_moved_volumes, \
                              ending_input_landmarks = config["ending_lm_registered_lms"], ending_output_landmarks = config["ending_sim_registered_lms"], \
                                  scale_registration = scale_registration, path_volume_mask_registration = '', type_reg_str = 'Sim', logger=logger, flag_skip_processing = config["flag_skip_reg_sim_intensity"],\
                                      flag_compute_similarity_all_volumes = config["flag_compute_similarity_all_volumes"])
    
        
    ############## Affine REG USING Intensity and ANTS ################
    print('----- AFFINE REG USING Intensity -----', flush = True)
    
    output_folder_intensity_affine = register_groupwise_samples(register_images_affine_ants, aff_iterations, sampleNames = vector_sample_name, input_folder = output_folder_intensity_sim, \
                      ending_input_folder = config["ending_sim_folder_output"], ending_input_volumes = ending_Sim_moved_volumes, \
                          ending_output_folder = config["ending_affine_folder_output"], ending_output_volumes = ending_Affine_moved_volumes, \
                              ending_input_landmarks = config["ending_sim_registered_lms"], ending_output_landmarks = config["ending_affine_registered_lms"], \
                                  scale_registration = scale_registration, path_volume_mask_registration = '', type_reg_str = 'Affine', logger=logger, flag_skip_processing = config["flag_skip_reg_affine_intensity"],\
                                      flag_compute_similarity_all_volumes = config["flag_compute_similarity_all_volumes"])
    
    print('----- SyN REG USING Intensity -----', flush = True)
    output_folder_intensity_syn = \
        register_groupwise_syn_samples(sampleNames = vector_sample_name, syn_iterations = syn_iterations, input_folder = output_folder_intensity_affine, \
                      ending_input_folder = config["ending_affine_folder_output"], ending_input_volumes = ending_Affine_moved_volumes, \
                          ending_output_folder = config["ending_syn_folder_output"], ending_output_volumes = ending_SyN_moved_volumes, \
                              ending_input_landmarks = config["ending_affine_registered_lms"], ending_output_landmarks = config["ending_syn_registered_lms"], \
                                  scale_registration = scale_registration, type_reg_str = 'SyN', logger=logger, flag_skip_processing = config["flag_skip_reg_syn_intensity"],\
                                      flag_compute_similarity_all_volumes = config["flag_compute_similarity_all_volumes"])
            

    create_atlases(folder_atlas = output_folder_intensity_syn, group_name = sample_groups[0] + '_' + sample_ages[0], sampleNames = vector_sample_name, \
                   ending_folder = config["ending_syn_folder_output"], ending_SyN_moved_volumes = ending_SyN_moved_volumes)

# Using the special variable 
# __name__
if __name__=="__main__":
    logger = logging.getLogger(__name__)
    main()