import os
# import glob
import sys
import SimpleITK as sitk
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_path)
from Registration.functionsRegistration import resliceVolumeITK, numpy_to_sitk, register_samples_toReference, \
    register_syn_samples_to_reference, compute_IoU_samples, SuppressOutput
import Registration.functionsLandmarks as flms
from Registration.functionsAtlasCreation import create_atlases
from CellSegmentation.scriptStats import plot_IoU_tissue
from matplotlib.pylab import plt

from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
import numpy as np
import yaml
import ants
import logging
from datetime import datetime
os.environ["QT_QPA_PLATFORM"] = "offscreen"

#################### PARAMETERS ###############################

with open("config.yml", "r") as file:
    print('Configuration file found')
    config = yaml.safe_load(file)

vector_sample_name = [sample["name"] for sample in config["samples"]]    
n_samples = len(vector_sample_name)
sample_groups = [sample["group"] for sample in config["samples"]]
n_groups = len(set(sample_groups))
sample_ages = [sample["age"] for sample in config["samples"]]
n_ages = len(set(sample_ages))

# Reference sample name
sample_name_reference = config["name_reference"]

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

# flag_mirror_volumes = True

################ INTENSITY-BASED SIMILARITY (ROT, TRANS, SCALING) REGISTRATION #####################

flag_make_aux_mask = False

ending_Sim_folder = config["ending_sim_folder_output"]
ending_Affine_folder = config["ending_affine_folder_output"]
ending_SyN_folder = config["ending_syn_folder_output"]

# scale_registration = 4 # Intensity Sim and Affine can be done in 2, but affine needs more downsampling
scale_registration = 6

aff_iterations = (200, 100, 10, 10)
syn_iterations = (200, 100, 10, 10)

# aff_iterations = (2, 1, 1, 1)
# syn_iterations = (2, 1, 1, 1)

# In Debug, the IoU of Tissue in SyN was around 0.75

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

def register_images_sim_ants(fixed, moving, aff_iterations,affine_global, aff_shrink_factors=(5, 3, 2, 1), aff_smoothing_sigmas=(30, 15, 10, 5)):
    """Perform affine registration and return the transformation."""
    # Convert images to float32
    
    # Convert images to float32
    fixed1_ants = ants.from_numpy(fixed)
    moving1_ants = ants.from_numpy(moving)

    reg1 = ants.registration(fixed = fixed1_ants, moving = moving1_ants, type_of_transform='Similarity', aff_metric='GC', \
                         aff_iterations=aff_iterations, aff_shrink_factors=aff_shrink_factors, aff_smoothing_sigmas=aff_smoothing_sigmas, \
                                 verbose=False, initial_transform=["Identity"])

    affine1 = reg1['fwdtransforms'][0]  # Path to affine matrix
    tranform_back = reg1['invtransforms'][0]
    return affine1, tranform_back

def register_images_affine_ants(fixed, moving, aff_iterations,affine_global, aff_shrink_factors=(5, 3, 2, 1), aff_smoothing_sigmas=(30, 15, 10, 5)):
    """Perform affine registration and return the transformation."""

    # Convert images to float32
    fixed1_ants = ants.from_numpy(fixed)
    moving1_ants = ants.from_numpy(moving)
    
    with SuppressOutput():
        reg1 = ants.registration(fixed = fixed1_ants.clone(), moving = moving1_ants.clone(), type_of_transform='Affine', aff_metric='GC', \
                             aff_iterations=aff_iterations, aff_shrink_factors=aff_shrink_factors, aff_smoothing_sigmas=aff_smoothing_sigmas, \
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
            print(folder_age, flush = True)
            folder_sample = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)
            print(folder_sample, flush = True)
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
    # print(regTypeFolder_LMbased, flush = True)
    if not os.path.exists(regTypeFolder_LMbased):
        os.mkdir(regTypeFolder_LMbased)
    
    # Logger file:
    logger = logging.getLogger("log_to_reference")
    logger.setLevel(logging.INFO)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(regTypeFolder_LMbased, 'log_to_reference_' + timestamp + '.log')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    ############## SIM REG USING LMS ################
    
    if not(config["flag_skip_reg_lm"]):
        for sample_name in vector_sample_name:
            
            # print('---------------------------------------')
            print(regTypeFolder_LMbased + ' registration: ' + sample_name, flush = True)
            logger.info(regTypeFolder_LMbased + ' registration: ' + sample_name)
        
            #-------------- Moving ------------------------------
            # folder = os.path.join(workFolder, sampleName)
            folder_sample = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)
            folderOutput_LMbased =  os.path.join(regTypeFolder_LMbased, sample_name + config["ending_lm_folder_output"])
        
            try:
                os.mkdir(folderOutput_LMbased)
            except OSError as error:
                print(error)
                logger.info(error)
        
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
                logger.info('------- Transformation not correctly defined!!! -------')
        
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
                logger.info('-----No LMs!!!------')
                
            for i in range(n_volumes):
                # print(ending_moving_volumes[i])
                # print(ending_LMbased_moved_volumes[i])
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
    logger.info('----- SIM REG USING Intensity -----')
    output_folder_intensity_sim = register_samples_toReference(register_images_sim_ants, aff_iterations, sampleNames = vector_sample_name, sampleNameReference = sample_name_reference, input_folder = regTypeFolder_LMbased, \
                                                              ending_input_folder = config["ending_lm_folder_output"], \
                      ending_input_volumes = ending_LMbased_moved_volumes, \
                          ending_output_folder_toReference = config["ending_sim_folder_output"], ending_output_volumes_toReference = ending_Sim_moved_volumes, \
                              ending_input_landmarks = config["ending_lm_registered_lms"], ending_output_landmarks_toReference = config["ending_sim_registered_lms"], \
                                  scale_registration = scale_registration, path_volume_mask_registration = '', type_reg_str = 'Sim', logger = logger, flag_skip_processing = config["flag_skip_reg_sim_intensity"],\
                                      flag_compute_similarity_all_volumes = config["flag_compute_similarity_all_volumes"])
    
        
    ############## Affine REG USING Intensity and ANTS ################
    print('----- AFFINE REG USING Intensity -----', flush = True)
    logger.info('----- AFFINE REG USING Intensity -----')
    output_folder_intensity_affine = register_samples_toReference(register_images_affine_ants, aff_iterations, sampleNames = vector_sample_name, sampleNameReference = sample_name_reference, \
                                                                input_folder = output_folder_intensity_sim, \
                      ending_input_folder = config["ending_sim_folder_output"], ending_input_volumes = ending_Sim_moved_volumes, \
                          ending_output_folder_toReference = config["ending_affine_folder_output"], ending_output_volumes_toReference = ending_Affine_moved_volumes, \
                              ending_input_landmarks = config["ending_sim_registered_lms"], ending_output_landmarks_toReference = config["ending_affine_registered_lms"], \
                                  scale_registration = scale_registration, path_volume_mask_registration = '', type_reg_str = 'Affine', logger = logger, flag_skip_processing = config["flag_skip_reg_affine_intensity"],\
                                      flag_compute_similarity_all_volumes = config["flag_compute_similarity_all_volumes"])
    
    print('----- SyN REG USING Intensity -----', flush = True)
    logger.info('----- SyN REG USING Intensity -----')
    output_folder_intensity_syn = \
        register_syn_samples_to_reference(sampleNames = vector_sample_name, sampleNameReference = sample_name_reference, syn_iterations = syn_iterations, input_folder = output_folder_intensity_affine, \
                      ending_input_folder = config["ending_affine_folder_output"], ending_input_volumes = ending_Affine_moved_volumes, \
                          ending_output_folder_toReference = config["ending_syn_folder_output"], ending_output_volumes_toReference = ending_SyN_moved_volumes, \
                              ending_input_landmarks = config["ending_affine_registered_lms"], ending_output_landmarks_toReference = config["ending_syn_registered_lms"], \
                                  scale_registration = scale_registration, type_reg_str = 'SyN', logger = logger, flag_skip_processing = config["flag_skip_reg_syn_intensity"],\
                                      flag_compute_similarity_all_volumes = config["flag_compute_similarity_all_volumes"])
    
    logger.info('----- CREATING VOLUME FOR MASKING PURPOSES -----')
    create_atlases(folder_atlas = output_folder_intensity_syn, group_name = sample_groups[0] + '_' + sample_ages[0], sampleNames = vector_sample_name, \
                   ending_folder = config["ending_syn_folder_output"], ending_SyN_moved_volumes = ending_SyN_moved_volumes, logger = logger)


# Using the special variable 
# __name__
if __name__=="__main__":
    main()








    
