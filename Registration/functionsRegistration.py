import csv
import vtk
import SimpleITK as sitk
import numpy as np
from PIL import Image
import os
from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage
import scipy
import time
from skimage.transform import resize
import ants
import Registration.functionsLandmarks as flms
import pandas as pd
import nibabel as nib
from Registration.functionsRegistrationComplimentary import create_magnitude_displacement_volume, \
    save_affine_transform_info_to_txt, compute_IoU_samples, percentage_lms_in_surface, get_volume_edge
import gc


os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["ANTS_VERBOSE"] = "0"

import logging

logging.getLogger("ants").setLevel(logging.ERROR)

import sys
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

def applyTransformToPoints(pointsVector,final_transform):
    LMs = []
    nPoints = len(pointsVector)
    for i in range(nPoints):
        pAux = pointsVector[i]
        tInv = final_transform.GetInverse()
        pAuxT = tInv.TransformPoint(pAux)
        LMs.append([pAuxT[0],pAuxT[1],pAuxT[2]])
    return LMs

def resliceVolumeITK(pathTIFFMoving, pathTIFFMovingResult, pathVolumeFixed, pathVolumeFixedCorrected,
                     landmarkTransformITK, constDivide = 1):
    spacingEmbryoCorrection = [constDivide,constDivide,constDivide]
    moving_im = sitk.ReadImage(pathTIFFMoving)
    fixed_im = sitk.ReadImage(pathVolumeFixed)

    moving_im.SetSpacing(spacingEmbryoCorrection)
    fixed_im.SetSpacing(spacingEmbryoCorrection)

    moving_im.SetOrigin(fixed_im.GetSpacing())

    itk_resampled_im = sitk.Resample(moving_im, fixed_im, landmarkTransformITK, sitk.sitkNearestNeighbor, 0)
    itk_resampled_im.SetSpacing(spacingEmbryoCorrection)
    itk_resampled_im.SetOrigin(fixed_im.GetSpacing())

    sitk.WriteImage(sitk.Cast(itk_resampled_im, sitk.sitkUInt8), pathTIFFMovingResult)
    
    temp_file = pathTIFFMovingResult + "_temp_compressed.tiff"

    # Open the existing multipage TIFF
    with Image.open(pathTIFFMovingResult) as img:
        # Collect all frames in a list
        frames = []
        while True:
            frames.append(img.copy())
            try:
                img.seek(img.tell() + 1)  # Move to the next frame
            except EOFError:
                break
    
    # Save frames with LZW compression
    frames[0].save(
        temp_file,
        save_all=True,
        append_images=frames[1:],
        compression="tiff_lzw"
    )
    
    # Overwrite the original file with the compressed version
    os.replace(temp_file, pathTIFFMovingResult)


def resliceVolume(pathTIFFMoving, pathTIFFMovingResult, pathVolumeFixed, pathVolumeFixedCorrected, landmarkTransform,
                  constDivide = 0.35277777777777775 ):
    spacingEmbryoCorrection = [constDivide,constDivide,constDivide]

    readerVTKMoving = vtk.vtkTIFFReader()
    readerVTKMoving.SetFileName(pathTIFFMoving)
    print("Read VTK successful", pathTIFFMoving)
    #readerVTKMoving.SetOutputSpacing(spacingEmbryoCorrection[0], spacingEmbryoCorrection[1], spacingEmbryoCorrection[2])
    readerVTKMoving.Update()

    readerVTKFixed = vtk.vtkTIFFReader()
    readerVTKFixed.SetFileName(pathVolumeFixed)
    print("Read VTK successful", pathVolumeFixed)
    #readerVTKFixed.SetOutputSpacing(spacingEmbryoCorrection[0], spacingEmbryoCorrection[1], spacingEmbryoCorrection[2])
    readerVTKFixed.Update()
    
    originEmbryo2 = readerVTKFixed.GetOutput().GetOrigin()

    VTKMoving = vtk.vtkImageReslice()
    VTKMoving.SetInputData(readerVTKMoving.GetOutput())
    
    VTKMoving.SetOutputSpacing(spacingEmbryoCorrection[0], spacingEmbryoCorrection[1], spacingEmbryoCorrection[2])
    
    VTKMoving.Update()

    VTKFixed = vtk.vtkImageReslice()
    VTKFixed.SetInputData(readerVTKFixed.GetOutput())
    #VTKFixed.SetOutputOrigin(originEmbryo2[0], originEmbryo2[1], originEmbryo2[2])
    VTKFixed.SetOutputSpacing(spacingEmbryoCorrection[0], spacingEmbryoCorrection[1], spacingEmbryoCorrection[2])
    VTKFixed.SetOutputExtent(readerVTKFixed.GetOutput().GetExtent())
    VTKFixed.Update()

    resliceEmbryo0 = vtk.vtkImageReslice()
    resliceEmbryo0.SetInputData(VTKMoving.GetOutput())
    resliceEmbryo0.Update()
    print("extent output: ", resliceEmbryo0.GetOutputExtent(), " SetOutputOrigin: ", resliceEmbryo0.GetOutputOrigin(),
          " GetOutputSpacing: ", resliceEmbryo0.GetOutputSpacing())

    # Applying the transformation
    mat = landmarkTransform.GetMatrix()
    transform = vtk.vtkTransform()
    transform.SetMatrix(mat)

    #resliceEmbryo0.SetOutputExtent(extent2)
    #spacingEmbryo2 = [1, 1, 1]
    resliceEmbryo0.SetOutputOrigin(originEmbryo2[0], originEmbryo2[1], originEmbryo2[2])
    resliceEmbryo0.SetOutputSpacing(spacingEmbryoCorrection[0], spacingEmbryoCorrection[1], spacingEmbryoCorrection[2])
    #resliceEmbryo0.SetOutputExtent(VTKFixed.GetOutput().GetExtent())
    resliceEmbryo0.SetOutputExtent(VTKMoving.GetOutput().GetExtent())
    resliceEmbryo0.SetInterpolationModeToNearestNeighbor()
    resliceEmbryo0.Update()


    print("extent output: ", resliceEmbryo0.GetOutputExtent(), " SetOutputSpacing: ", resliceEmbryo0.GetOutputOrigin(),
          " GetOutputSpacing: ", resliceEmbryo0.GetOutputSpacing())
    #showVolume(None, resliceEmbryo0.GetOutput(), "Before transform")
    #resliceEmbryo0.SetResliceTransform(landmarkTransform) #not working

    resliceEmbryo1 = vtk.vtkImageReslice()
    resliceEmbryo1.SetInputData(resliceEmbryo0.GetOutput())
    #resliceEmbryo1.SetInputData(VTKMoving.GetOutput())
    #resliceEmbryo1.SetResliceTransform(transform.GetInverse())
    resliceEmbryo1.SetResliceTransform(transform.GetInverse())
    resliceEmbryo1.SetOutputOrigin(originEmbryo2[0], originEmbryo2[1], originEmbryo2[2])
    resliceEmbryo1.SetOutputSpacing(spacingEmbryoCorrection[0], spacingEmbryoCorrection[1], spacingEmbryoCorrection[2])
    resliceEmbryo1.SetOutputExtent(VTKFixed.GetOutput().GetExtent())
    resliceEmbryo1.SetInterpolationModeToNearestNeighbor()
    resliceEmbryo1.Update()

    resliceEmbryo0 = []

    castFilter = vtk.vtkImageCast()
    castFilter.SetInputData(resliceEmbryo1.GetOutput())
    castFilter.SetOutputScalarTypeToUnsignedChar()
    castFilter.Update()
    print("writing Image")
    writer = vtk.vtkTIFFWriter()
    #writer.SetCompressionToLZW() #LZW compression is patented outside US so it is disabled
    writer.SetFileName(pathTIFFMovingResult)
    writer.SetInputConnection(castFilter.GetOutputPort())
    writer.Write()

    castFilter = []
    resliceEmbryo1 = []

    #Fixed image corrected
    castFilter = vtk.vtkImageCast()
    castFilter.SetInputData(VTKFixed.GetOutput())
    castFilter.SetOutputScalarTypeToUnsignedChar()
    castFilter.Update()
    print("writing Image")
    writer = vtk.vtkTIFFWriter()
    #writer.SetCompressionToLZW() #LZW compression is patented outside US so it is disabled
    writer.SetFileName(pathVolumeFixedCorrected)
    writer.SetInputConnection(castFilter.GetOutputPort())
    writer.Write()


def resample_img(itk_image, out_spacing = [0.5,0.5,0.5], is_label=True):
    #from 0.35 to 0.5
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    print('original_spacing:' + str(original_spacing))
    print('original_size:' + str(original_size))

    out_size = [
        int(np.round(float(original_size[0]) * (original_spacing[0] / out_spacing[0]))),
        int(np.round(float(original_size[1]) * (original_spacing[1] / out_spacing[1]))),
        int(np.round(float(original_size[2]) * (original_spacing[2] / out_spacing[2])))]

    print('out_size:' + str(out_size))
    print('out_spacing:' + str(out_spacing))
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def numpy_to_sitk(image_np, spacing=(1.0, 1.0, 1.0)):
    """Convert a NumPy array to a SimpleITK image."""
    image_sitk = sitk.GetImageFromArray(image_np)
    image_sitk.SetSpacing(spacing)
    return image_sitk

def register_images_affine(fixed, moving):
    """Perform affine registration and return the transformation."""
    # Convert images to float32
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    # Initialize registration framework
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsGradientDescent(learningRate=10.0, numberOfIterations=100)#(learningRate=1.0, numberOfIterations=30)
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Use affine transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)
    registration.SetOptimizerScalesFromPhysicalShift()

    # Perform registration
    final_transform = registration.Execute(fixed, moving)
    
    # Ensure we extract the affine transform
    if isinstance(final_transform, sitk.CompositeTransform):
        final_transform = final_transform.GetBackTransform()
    
    return final_transform

def average_affine_transforms(transform_list):
    """Compute the average affine transformation from a list of SimpleITK AffineTransform(3) objects."""
    if not transform_list:
        raise ValueError("The transform list is empty.")

    # Extract all parameters into a NumPy array (shape: num_transforms x 12)
    params_array = np.array([list(t.GetParameters()) for t in transform_list])
    
    # Compute the mean along axis 0 (average each parameter)
    mean_params = np.mean(params_array, axis=0)

    # Create a new AffineTransform and set the averaged parameters
    mean_transform = sitk.AffineTransform(3)
    mean_transform.SetParameters(mean_params.tolist())

    return mean_transform

# def rescale_affine_transform(transform, scale_factors):
#     """Rescales an ants.antsTransforms.ANTsTransform to the original space."""

#     matrix = transform.parameters[0:9].reshape(3, 3).copy()  # Assuming 3D
#     offset = transform.parameters[9:12].copy()

#     scale_x, scale_y, scale_z = scale_factors

#     # Rescale the translation vector
#     offset[0] *= scale_x
#     offset[1] *= scale_y
#     offset[2] *= scale_z

#     # Update the transform parameters
#     transform.parameters[0:9] = matrix.flatten()
#     transform.parameters[9:12] = offset

#     return transform



def average_syn_transform(list_files_displacement_fields):
    n_files = len(list_files_displacement_fields)
    
    transformation_avg_nii = nib.load(list_files_displacement_fields[0])
    transformation_avg = transformation_avg_nii.get_fdata()/ n_files
    
    #To not overload memory, read one by one and divide it
    if n_files > 1:
        for disp_file in list_files_displacement_fields[1:]:
            print('-- Reading and averaging', flush = True)
            transformation_temp = nib.load(disp_file)
            transformation_temp = transformation_temp.get_fdata()
            transformation_temp = transformation_temp / n_files
            transformation_avg = transformation_avg + transformation_temp
            del transformation_temp
    
    print('Building averaged_avg_nii')
    averaged_avg_nii = nib.Nifti1Image(transformation_avg, transformation_avg_nii.affine, transformation_avg_nii.header)
    
    
    return averaged_avg_nii#, averaged_avg_ants#, averaged_avg_ants

def correct_resampled_labels(original_labels, resampled):
    
    # Find the unique values in the resampled volume
    resampled_labels = np.unique(resampled)

    # Identify new values in the resampled volume that are not in the original labels
    new_labels = np.setdiff1d(resampled_labels, original_labels)

    if len(new_labels) == 0:
        return resampled  # No correction needed

    # Create a KDTree for efficient nearest neighbor search
    tree = scipy.spatial.KDTree(original_labels.reshape(-1, 1))

    # Iterate through the resampled volume and correct new values
    corrected_resampled = resampled.copy()
    for new_label in new_labels:
        # Find the indices where the resampled volume has the new label
        indices = np.where(resampled == new_label)

        # Find the nearest original label for the new label
        dist, index = tree.query(new_label)
        nearest_original_label = original_labels[index]

        # Replace the new label with the nearest original label
        corrected_resampled[indices] = nearest_original_label

    return corrected_resampled

def rescaling_transformation_saving(affine_avg, scale_registration, affine_avg_path):
    #Re-scaling matrix-based transformation transformation
    affine_avg_new_parameters = affine_avg.parameters.copy()
    offset = affine_avg.parameters[9:12].copy()
    
    # Rescale the translation vector
    offset[0] = offset[0] * scale_registration
    offset[1] = offset[1] * scale_registration
    offset[2] = offset[2] * scale_registration

    affine_avg_new_parameters[9:12] = offset
    affine_avg.set_parameters(affine_avg_new_parameters)
    
    ants.write_transform(affine_avg, affine_avg_path)
    affine_avg = ants.read_transform(affine_avg_path)
    
    return affine_avg

affine_global = np.array([
[0.0, 1.0, 0.0, 0.0],
[1.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 1.0]
])


# affine_global = np.array([
# [0.0, -1.0, 0.0, 0.0],
# [-1.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 1.0, 0.0],
# [0.0, 0.0, 0.0, 1.0]
# ])

def transform_landmarks_to_ants_coords(landmarks, affine):
    """Transforms landmark coordinates to ANTs coordinate system."""
    transformed_landmarks = []
    transformed_landmarks_fwd = []
    for landmark in landmarks:
        # Add a homogeneous coordinate (1)
        homogeneous_landmark = np.array([landmark[0], landmark[1], landmark[2], 1])
        # Apply the inverse affine transformation
        transformed_landmark = np.linalg.inv(affine) @ homogeneous_landmark
        transformed_landmarks.append(transformed_landmark[:3].tolist()) #remove homogenous coordinate
        
        transformed_landmark_fwd = affine @ homogeneous_landmark
        transformed_landmarks_fwd.append(transformed_landmark_fwd[:3].tolist()) #remove homogenous coordinate
        
    return transformed_landmarks, transformed_landmarks_fwd

def transform_ants_coords_to_original(ants_points, affine):
    """Transforms ANTs point coordinates back to original coordinate system."""
    original_points = []
    original_points_inv = []
    for point in ants_points:
        # Add a homogeneous coordinate (1)
        homogeneous_point = np.array([point[0], point[1], point[2], 1])
        # Apply the affine transformation
        original_point = affine @ homogeneous_point
        original_points.append(original_point[:3].tolist()) #remove homogenous coordinate
        
        original_point_inv = np.linalg.inv(affine) @ homogeneous_point # affine @ homogeneous_point
        original_points_inv.append(original_point_inv[:3].tolist()) #remove homogenous coordinate
        
    return original_points, original_points_inv

def align_tiff_to_nifti(tiff_volume, nifti_affine, output_filename):
    """
    Aligns a TIFF-like volume to a NIfTI orientation by flipping the first two axes.

    Args:
        tiff_volume (numpy.ndarray): The 3D NumPy array from the TIFF stack.
        nifti_affine (numpy.ndarray): The affine matrix of the NIfTI volume.
        output_filename (str): The filename to save the aligned NIfTI volume.
    """

    # Flip the first two axes of the TIFF volume
    aligned_volume = np.flip(tiff_volume, axis=(0, 1))

    # Create a Nifti1Image object with the aligned data and the NIfTI affine
    aligned_nifti = nib.Nifti1Image(aligned_volume, nifti_affine)

    # Save the aligned NIfTI volume
    nib.save(aligned_nifti, output_filename)

def register_groupwise_samples(func_register_images, tuple_iterations, sampleNames, input_folder, ending_input_folder, ending_input_volumes, \
                     ending_output_folder, ending_output_volumes, ending_input_landmarks, ending_output_landmarks, \
                     scale_registration, path_volume_mask_registration, type_reg_str, logger, flag_skip_processing = False, flag_compute_similarity_all_volumes = True):
    print('SIM REG USING Intensity', flush=True)
    # transform_dict = {}
    nSamples = len(sampleNames)    
    nVolumes = len(ending_input_volumes)
    output_folder_intensity = os.path.join(input_folder, type_reg_str)
    logger.info('Skipping this process: ' + str(flag_skip_processing))
    if not flag_skip_processing:
        if not os.path.exists(output_folder_intensity):
            print('Outputs: ' + output_folder_intensity, flush=True)
            os.mkdir(output_folder_intensity)
            
        if (path_volume_mask_registration != ''):
            mask_registration = functionReadTIFFMultipage(path_volume_mask_registration,8) > 0
    
        for i in range(nSamples):
            # This is the moving volume
            sampleName1 = sampleNames[i]
            current_time_tuple = time.localtime()
            current_time_string = time.strftime("%H:%M:%S", current_time_tuple)
            print(sampleName1 + ' started at ' + current_time_string, flush = True)

            pathVolume1 = os.path.join(input_folder, sampleName1 + ending_input_folder, sampleName1 + ending_input_volumes[0])
            volume_moving_original_size = functionReadTIFFMultipage(pathVolume1,8)
            # volume1 = np.array(scipy.ndimage.zoom(volume_moving_original_size, zoom=1/scale_registration, order=0))
            volume1 = volume_moving_original_size[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
            
            folderOutput =  os.path.join(output_folder_intensity, sampleName1 + ending_output_folder)
            if not os.path.exists(folderOutput):
                os.mkdir(folderOutput)
            # Convert NumPy arrays to SimpleITK images
            moving_numpy = np.asarray(volume1)
            moving_numpy = np.float16(moving_numpy)
            del volume1
            if (path_volume_mask_registration != ''):
                mask_registration = resize(mask_registration, moving_numpy.shape, order=0, preserve_range=True, anti_aliasing=False)
                moving_masked_volume = np.float16(np.where(mask_registration, moving_numpy, np.zeros_like(moving_numpy)))
            else:
                moving_masked_volume = moving_numpy
            
            list_matrices = []
            list_backwards = []
            
            start_time = time.time()
            for j in range(nSamples):
                
                sampleName2 = sampleNames[j]
                key_transf = sampleName1 + '_to_' + sampleName2
                print('-- Computing ' + key_transf, flush = True)
                logger.info('-- Computing ' + key_transf)
                pathVolume2 = os.path.join(input_folder, sampleName2 + ending_input_folder, sampleName2 + ending_input_volumes[0])
                volume2 = functionReadTIFFMultipage(pathVolume2,8)
                # fixed = np.array(scipy.ndimage.zoom(volume2, zoom=1/scale_registration, order=0))
                fixed = volume2[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
                fixed = np.float16(fixed)
                
                if (path_volume_mask_registration != ''):
                    mask_registration = resize(mask_registration, fixed.shape, order=0, preserve_range=True, anti_aliasing=False)
                    fixed_masked_volume = np.float16(np.where(mask_registration, fixed, np.zeros_like(fixed)))
                else:
                    fixed_masked_volume = fixed
    
                transform1, tranform_back = func_register_images(fixed_masked_volume, moving_masked_volume, tuple_iterations,affine_global)
                list_matrices.append(transform1)
                list_backwards.append(tranform_back)
                del fixed, volume2, fixed_masked_volume
            
            gc.collect()
                
            end_time = time.time()
            elapsed_time_01 = end_time - start_time
            print(f"-- Elapsed time: {elapsed_time_01:.2f} seconds.", flush = True)
            del moving_masked_volume
            # Apply transform to the volume
            affine_avg = ants.average_affine_transform(list_matrices)
            affine_avg_back = ants.average_affine_transform(list_backwards)
            
            print('-- Transformations averaged', flush = True)
            
            affine_avg_path = os.path.join(folderOutput, sampleName1 + '_average_affine_fwd.mat')
            affine_avg = rescaling_transformation_saving(affine_avg, scale_registration, affine_avg_path)
            
            print('-- Starting resampling', flush = True)
            
            folderOutput_LMbased =  os.path.join(input_folder, sampleName1 + ending_input_folder)
            for k in range(nVolumes):
                print('transforming volumes of the sample')
                pathMovingVolume = os.path.join(folderOutput_LMbased , sampleName1 + ending_input_volumes[k])
                pathLMsGMMovedRigid = os.path.join(folderOutput_LMbased, sampleName1+ ending_input_landmarks)
                lms =  flms.getLMs(pathLMsGMMovedRigid)
                
                pathResultVolume = os.path.join(folderOutput , sampleName1 + ending_output_volumes[k])
                volume1 = functionReadTIFFMultipage(pathMovingVolume,8)
                # As nearestNeighbor from ants does not work correctly, I need to correct the volume after the transformation
                original_labels = np.unique(volume1)
                
                if k==0:
                    volume1_edge = get_volume_edge(volume1) # volume1>5 # 
                    xyz_0, yxz_0, zyx_0 = percentage_lms_in_surface(volume1_edge, lms)
                    # lms_new_coordinates, lms_new_coordinates_fwd_debug = transform_landmarks_to_ants_coords(lms, affine_global)
                    lms_new_coordinates, _ = transform_landmarks_to_ants_coords(lms, affine_global)
                    xyz_0_newCoord, yxz_0_newCoord, zyx_0_newCoord = percentage_lms_in_surface(volume1_edge, lms_new_coordinates)
                
                    del volume1_edge
                
                volume_ants = ants.from_numpy(volume1)
                                
                if k==0: #It is the tissues volume!!
                    output_txt_fwd = os.path.join(folderOutput, sampleName1 + '_average_affine_fwd.txt')
                    save_affine_transform_info_to_txt(affine_avg, output_txt_fwd, volume_ants)
                    output_txt_inv = os.path.join(folderOutput, sampleName1 + '_average_affine_inv.txt')
                    save_affine_transform_info_to_txt(affine_avg_back, output_txt_inv, volume_ants)
                    
                del volume1
                with SuppressOutput():

                    resampled = ants.apply_ants_transform_to_image(transform = affine_avg, \
                                                                image=volume_ants, \
                                                                    interpolation = 'nearestNeighbor', \
                                                                        reference = volume_ants)
                resampled_np = resampled.numpy()
                del volume_ants
                
                if k==0:
                    # Transform landmarks
                    
                    pathLMsGMMovedSim = os.path.join(folderOutput, sampleName1+ ending_output_landmarks)
                    
                    resampled_np_edge = get_volume_edge(resampled_np) # resampled_np > 5 #
                    affine_avg_inv = affine_avg.invert()
                    transformed_lms_sim_debug = [ants.apply_ants_transform_to_point(affine_avg_inv, np.asarray(p)) for p in lms]
                    flms.saveLMs(pathLMsGMMovedSim + '_Direct application.csv',transformed_lms_sim_debug)
                    xyz_1_debug, yxz_1_debug, zyx_1_debug = percentage_lms_in_surface(resampled_np_edge, transformed_lms_sim_debug)
                    
                    transformed_lms_sim_new_coordinates = [ants.apply_ants_transform_to_point(affine_avg_inv, np.asarray(p)) for p in lms_new_coordinates]
                    flms.saveLMs(pathLMsGMMovedSim + '_Orientation correction and application.csv',transformed_lms_sim_new_coordinates)
                    xyz_2, yxz_2, zyx_2 = percentage_lms_in_surface(resampled_np_edge, transformed_lms_sim_new_coordinates)
                    
                    transformed_lms_sim_new_coordinates_back_to_original, transformed_lms_sim_new_coordinates_back_to_original_inv \
                        = transform_ants_coords_to_original(transformed_lms_sim_new_coordinates, affine_global)
                    flms.saveLMs(pathLMsGMMovedSim + '_Orientation correction, application, and back to original coords.csv',transformed_lms_sim_new_coordinates_back_to_original)
                    flms.saveLMs(pathLMsGMMovedSim,transformed_lms_sim_new_coordinates_back_to_original)
                    xyz_3, yxz_3, zyx_3 = percentage_lms_in_surface(resampled_np_edge, transformed_lms_sim_new_coordinates_back_to_original)
                    
                    with open(pathLMsGMMovedSim + '_ratio_accuracy_lms', "w") as f:
                            f.write('xyz yxz zyx \n')
                            f.write('Original LMs in Not Moved Volume: ' + str((xyz_0, yxz_0, zyx_0)) + "\n")
                            f.write('Original LMs correction in Not Moved Volume: ' + str((xyz_0_newCoord, yxz_0_newCoord, zyx_0_newCoord)) + "\n")
                            f.write('Direct application: ' + str((xyz_1_debug, yxz_1_debug, zyx_1_debug)) + "\n")
                            f.write('Orientation correction and application: ' + str((xyz_2, yxz_2, zyx_2)) + "\n")
                            f.write('Orientation correction, application, and back to original coords CORRECT?: ' + str((xyz_3, yxz_3, zyx_3)) + "\n")
                            
                    del resampled_np_edge
                

                del resampled
                resampled_np = correct_resampled_labels(original_labels, resampled_np)
                functionSaveTIFFMultipage(resampled_np,pathResultVolume,8)
                
                # resampled_np = np.array(scipy.ndimage.zoom(resampled_np, zoom=1/scale_registration, order=0))
                resampled_np = resampled_np[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
                
                pathResultVolume_nii = pathResultVolume + '_ants_downsampled.nii.gz'
                
                image_nifti = nib.Nifti1Image(resampled_np.astype('uint8'), affine=affine_global)
                
                nib.save(image_nifti, pathResultVolume_nii)
                
                del resampled_np, image_nifti
            
            
            for temp_file_transform in list_matrices:
                os.remove(temp_file_transform)
                
            del moving_numpy, affine_avg # adjusted_transform
            
            gc.collect()
        
        print('Computing IoU', flush = True)
        if flag_compute_similarity_all_volumes:
            compute_IoU_samples(sampleNames, ending_output_volumes, output_folder_intensity, ending_output_folder)
        else:
            compute_IoU_samples(sampleNames, [ending_output_volumes[0]], output_folder_intensity, ending_output_folder)
    
    return output_folder_intensity

def register_samples_toReference(func_register_images, tuple_iterations, sampleNames, sampleNameReference, input_folder, ending_input_folder, ending_input_volumes, \
                     ending_output_folder_toReference, ending_output_volumes_toReference, ending_input_landmarks, ending_output_landmarks_toReference, \
                     scale_registration, path_volume_mask_registration, type_reg_str, logger, flag_skip_processing = False, flag_compute_similarity_all_volumes = True):
    print('SIM REG USING Intensity', flush=True)
    logger.info('SIM REG USING Intensity')
    # transform_dict = {}
    nSamples = len(sampleNames)    
    nVolumes = len(ending_input_volumes)
    output_folder_intensity = os.path.join(input_folder, type_reg_str)
    logger.info('Skipping this process: ' + str(flag_skip_processing))
    if not flag_skip_processing:
        if not os.path.exists(output_folder_intensity):
            print('Outputs: ' + output_folder_intensity, flush=True)
            logger.info('Outputs: ' + output_folder_intensity)
            os.mkdir(output_folder_intensity)
            
        if (path_volume_mask_registration != ''):
            mask_registration = functionReadTIFFMultipage(path_volume_mask_registration,8) > 0
    
        for i in range(nSamples):
            # This is the moving volume
            sampleName1 = sampleNames[i]
            current_time_tuple = time.localtime()
            current_time_string = time.strftime("%H:%M:%S", current_time_tuple)
            print(sampleName1 + ' started at ' + current_time_string, flush = True)
            logger.info(sampleName1 + ' started at ' + current_time_string)

            pathVolume1 = os.path.join(input_folder, sampleName1 + ending_input_folder, sampleName1 + ending_input_volumes[0])
            volume_moving_original_size = functionReadTIFFMultipage(pathVolume1,8)
            # volume1 = np.array(scipy.ndimage.zoom(volume_moving_original_size, zoom=1/scale_registration, order=0))
            volume1 = volume_moving_original_size[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
            
            folderOutput =  os.path.join(output_folder_intensity, sampleName1 + ending_output_folder_toReference)
            if not os.path.exists(folderOutput):
                os.mkdir(folderOutput)
            # Convert NumPy arrays to SimpleITK images
            moving_numpy = np.asarray(volume1)
            moving_numpy = np.float16(moving_numpy)
            del volume1
            if (path_volume_mask_registration != ''):
                mask_registration = resize(mask_registration, moving_numpy.shape, order=0, preserve_range=True, anti_aliasing=False)
                moving_masked_volume = np.float16(np.where(mask_registration, moving_numpy, np.zeros_like(moving_numpy)))
            else:
                moving_masked_volume = moving_numpy
            
            list_matrices = []
            list_backwards = []
            
            start_time = time.time()
            
            key_transf = sampleName1 + '_to_' + sampleNameReference
            print('-- Computing ' + key_transf, flush = True)
            logger.info('-- Computing ' + key_transf)
            
            pathVolume2 = os.path.join(input_folder, sampleNameReference + ending_input_folder, sampleNameReference + ending_input_volumes[0])
            volume2 = functionReadTIFFMultipage(pathVolume2,8)
            # fixed = np.array(scipy.ndimage.zoom(volume2, zoom=1/scale_registration, order=0))
            fixed = volume2[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
            fixed = np.float16(fixed)
            
            if (path_volume_mask_registration != ''):
                mask_registration = resize(mask_registration, fixed.shape, order=0, preserve_range=True, anti_aliasing=False)
                fixed_masked_volume = np.float16(np.where(mask_registration, fixed, np.zeros_like(fixed)))
            else:
                fixed_masked_volume = fixed

            transform1, tranform_back = func_register_images(fixed_masked_volume, moving_masked_volume, tuple_iterations,affine_global)
            list_matrices.append(transform1)
            list_backwards.append(tranform_back)
            del fixed, volume2, fixed_masked_volume
            
            gc.collect()
                
            end_time = time.time()
            elapsed_time_01 = end_time - start_time
            print(f"-- Elapsed time: {elapsed_time_01:.2f} seconds.", flush = True)
            logger.info(f"-- Elapsed time: {elapsed_time_01:.2f} seconds.")
            del moving_masked_volume
            # Apply transform to the volume
            affine_avg = ants.average_affine_transform(list_matrices)
            affine_avg_back = ants.average_affine_transform(list_backwards)
            
            print('-- Transformations averaged', flush = True)
            logger.info('-- Transformations averaged')
            affine_avg_path = os.path.join(folderOutput, sampleName1 + '_average_affine_fwd.mat')
            affine_avg = rescaling_transformation_saving(affine_avg, scale_registration, affine_avg_path)
            
            print('-- Starting resampling', flush = True)
            logger.info('-- Starting resampling')
            folderOutput_LMbased =  os.path.join(input_folder, sampleName1 + ending_input_folder)
            for k in range(nVolumes):
                print('transforming volumes of the sample', flush = True)
                logger.info('-- Transforming volumes ' + ending_input_volumes[k])
                try:
                    pathMovingVolume = os.path.join(folderOutput_LMbased , sampleName1 + ending_input_volumes[k])
                    pathLMsGMMovedRigid = os.path.join(folderOutput_LMbased, sampleName1+ ending_input_landmarks)
                    lms =  flms.getLMs(pathLMsGMMovedRigid)
                    
                    pathResultVolume = os.path.join(folderOutput , sampleName1 + ending_output_volumes_toReference[k])
                    volume1 = functionReadTIFFMultipage(pathMovingVolume,8)
                    # As nearestNeighbor from ants does not work correctly, I need to correct the volume after the transformation
                    original_labels = np.unique(volume1)
                    
                    if k==0:
                        volume1_edge = get_volume_edge(volume1) # volume1>5 # 
                        xyz_0, yxz_0, zyx_0 = percentage_lms_in_surface(volume1_edge, lms)
                        # lms_new_coordinates, lms_new_coordinates_fwd_debug = transform_landmarks_to_ants_coords(lms, affine_global)
                        lms_new_coordinates, _ = transform_landmarks_to_ants_coords(lms, affine_global)
                        xyz_0_newCoord, yxz_0_newCoord, zyx_0_newCoord = percentage_lms_in_surface(volume1_edge, lms_new_coordinates)
                    
                        del volume1_edge
                    
                    volume_ants = ants.from_numpy(volume1)
                                    
                    if k==0: #It is the tissues volume!!
                        output_txt_fwd = os.path.join(folderOutput, sampleName1 + '_average_affine_fwd.txt')
                        save_affine_transform_info_to_txt(affine_avg, output_txt_fwd, volume_ants)
                        output_txt_inv = os.path.join(folderOutput, sampleName1 + '_average_affine_inv.txt')
                        save_affine_transform_info_to_txt(affine_avg_back, output_txt_inv, volume_ants)
                        
                    del volume1
                    with SuppressOutput():
    
                        resampled = ants.apply_ants_transform_to_image(transform = affine_avg, \
                                                                    image=volume_ants, \
                                                                        interpolation = 'nearestNeighbor', \
                                                                            reference = volume_ants)
                    resampled_np = resampled.numpy()
                    del volume_ants
                    
                    if k==0:
                        # Transform landmarks
                        
                        pathLMsGMMovedSim = os.path.join(folderOutput, sampleName1+ ending_output_landmarks_toReference)
                        
                        resampled_np_edge = get_volume_edge(resampled_np) # resampled_np > 5 #
                        affine_avg_inv = affine_avg.invert()
                        transformed_lms_sim_debug = [ants.apply_ants_transform_to_point(affine_avg_inv, np.asarray(p)) for p in lms]
                        flms.saveLMs(pathLMsGMMovedSim + '_Direct application.csv',transformed_lms_sim_debug)
                        xyz_1_debug, yxz_1_debug, zyx_1_debug = percentage_lms_in_surface(resampled_np_edge, transformed_lms_sim_debug)
                        
                        transformed_lms_sim_new_coordinates = [ants.apply_ants_transform_to_point(affine_avg_inv, np.asarray(p)) for p in lms_new_coordinates]
                        flms.saveLMs(pathLMsGMMovedSim + '_Orientation correction and application.csv',transformed_lms_sim_new_coordinates)
                        xyz_2, yxz_2, zyx_2 = percentage_lms_in_surface(resampled_np_edge, transformed_lms_sim_new_coordinates)
                        
                        transformed_lms_sim_new_coordinates_back_to_original, transformed_lms_sim_new_coordinates_back_to_original_inv \
                            = transform_ants_coords_to_original(transformed_lms_sim_new_coordinates, affine_global)
                        flms.saveLMs(pathLMsGMMovedSim + '_Orientation correction, application, and back to original coords.csv',transformed_lms_sim_new_coordinates_back_to_original)
                        flms.saveLMs(pathLMsGMMovedSim,transformed_lms_sim_new_coordinates_back_to_original)
                        xyz_3, yxz_3, zyx_3 = percentage_lms_in_surface(resampled_np_edge, transformed_lms_sim_new_coordinates_back_to_original)
                        
                        with open(pathLMsGMMovedSim + '_ratio_accuracy_lms', "w") as f:
                                f.write('xyz yxz zyx \n')
                                f.write('Original LMs in Not Moved Volume: ' + str((xyz_0, yxz_0, zyx_0)) + "\n")
                                f.write('Original LMs correction in Not Moved Volume: ' + str((xyz_0_newCoord, yxz_0_newCoord, zyx_0_newCoord)) + "\n")
                                f.write('Direct application: ' + str((xyz_1_debug, yxz_1_debug, zyx_1_debug)) + "\n")
                                f.write('Orientation correction and application: ' + str((xyz_2, yxz_2, zyx_2)) + "\n")
                                f.write('Orientation correction, application, and back to original coords CORRECT?: ' + str((xyz_3, yxz_3, zyx_3)) + "\n")
                                
                        del resampled_np_edge
                
                    resampled_np = correct_resampled_labels(original_labels, resampled_np)
                    functionSaveTIFFMultipage(resampled_np,pathResultVolume,8)
                    
                    # resampled_np = np.array(scipy.ndimage.zoom(resampled_np, zoom=1/scale_registration, order=0))
                    resampled_np = resampled_np[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
                    
                    pathResultVolume_nii = pathResultVolume + '_ants_downsampled.nii.gz'
                    
                    image_nifti = nib.Nifti1Image(resampled_np.astype('uint8'), affine=affine_global)
                    
                    nib.save(image_nifti, pathResultVolume_nii)
                    
                    del resampled_np, image_nifti
                
                except OSError as error:
                    print(error)
                    logger.info(error)
                    logger.info('POSSIBLE SKIPPED VOLUME:' + ending_input_volumes[k])

                del resampled
            
            
            for temp_file_transform in list_matrices:
                os.remove(temp_file_transform)
                
            del moving_numpy, affine_avg # adjusted_transform
            
            gc.collect()
        
        print('Computing IoU', flush = True)
        logger.info('Computing IoU')
        if flag_compute_similarity_all_volumes:
            compute_IoU_samples(sampleNames, ending_output_volumes_toReference, output_folder_intensity, ending_output_folder_toReference)
        else:
            compute_IoU_samples(sampleNames, [ending_output_volumes_toReference[0]], output_folder_intensity, ending_output_folder_toReference)
    
    return output_folder_intensity

def register_images_syn(fixed, moving, key_transf,affine_global, reg_iterations=(2, 1, 1)):
    """Perform affine registration and return the transformation."""
    # Convert images to float32
    fixed1_ants = ants.from_numpy(np.float32(fixed)) # To avoid float64
    moving1_ants = ants.from_numpy(np.float32(moving)) # To avoid float64

    reg1 = ants.registration(fixed = fixed1_ants.clone(), moving = moving1_ants.clone(), type_of_transform='SyNOnly', aff_metric='GC', \
                         reg_iterations=reg_iterations, verbose=False, initial_transform=["Identity"])

    displacement = reg1['fwdtransforms'][0]  # Path registration
    displacement_inv = reg1['invtransforms'][0]  # Path registration
    
    return displacement, displacement_inv

def register_groupwise_syn_samples(sampleNames, syn_iterations, input_folder, ending_input_folder, ending_input_volumes, \
                     ending_output_folder, ending_output_volumes, ending_input_landmarks, ending_output_landmarks, \
                         scale_registration, type_reg_str, logger, flag_skip_processing = False, flag_compute_similarity_all_volumes = True):
    
    nSamples = len(sampleNames)    
    nVolumes = len(ending_input_volumes)
    output_folder_intensity = os.path.join(input_folder, type_reg_str)
    if not os.path.exists(output_folder_intensity):
        os.mkdir(output_folder_intensity)
    logger.info('Skipping this process: ' + str(flag_skip_processing))
    if not flag_skip_processing:
        for i in range(nSamples):
            # This is the moving volume
            sampleName1 = sampleNames[i]
            print(sampleName1)
            pathVolume1 = os.path.join(input_folder, sampleName1 + ending_input_folder, sampleName1 + ending_input_volumes[0])
            print(pathVolume1)
            volume_moving_original_size = functionReadTIFFMultipage(pathVolume1,8)
            # volume1 = np.array(scipy.ndimage.zoom(volume_moving_original_size, zoom=1/scale_registration, order=0))
            volume1 = volume_moving_original_size[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
            
            folderOutput =  os.path.join(output_folder_intensity, sampleName1 + ending_output_folder)
            if not os.path.exists(folderOutput):
                os.mkdir(folderOutput)
            # Convert NumPy arrays to SimpleITK images
            moving_numpy = np.asarray(volume1)
            moving_numpy = np.float16(moving_numpy)
            del volume1
            
            list_files_displacement_fields = []
            # list_files_displacement_fields_inv = []
            
            start_time = time.time()
            for j in range(nSamples):
                
                sampleName2 = sampleNames[j]
                key_transf = sampleName1 + '_to_' + sampleName2
                print('-- Computing ' + key_transf, flush = True)
                logger.info('-- Computing ' + key_transf)
                pathVolume2 = os.path.join(input_folder, sampleName2 + ending_input_folder, sampleName2 + ending_input_volumes[0])
                volume2 = functionReadTIFFMultipage(pathVolume2,8)
                # fixed_numpy = np.array(scipy.ndimage.zoom(volume2, zoom=1/scale_registration, order=0))
                fixed_numpy = volume2[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
                fixed_numpy = np.float16(fixed_numpy)
                
                reg_file, reg_file_inv = register_images_syn(fixed_numpy, moving_numpy, key_transf,affine_global, reg_iterations=syn_iterations)
                
                list_files_displacement_fields.append(reg_file)
                
                del fixed_numpy
    
            end_time = time.time()
            elapsed_time_01 = end_time - start_time
            print(f"-- Elapsed time: {elapsed_time_01:.2f} seconds.", flush = True)
            del moving_numpy
            # Apply transform to the volume
            print('-- Starting: average_syn_transform', flush = True)
            syn_avg = average_syn_transform(list_files_displacement_fields)

            # syn_inv_avg = average_syn_transform(list_files_displacement_fields_inv)
            
            # transformation_avg_nii = nib.load(syn_avg)
            transformation_avg = syn_avg.get_fdata()
            syn_avg_path = os.path.join(folderOutput, sampleName1 + '_average_SyN.nii.gz')
            syn_avg_path_inv  = os.path.join(folderOutput, sampleName1 + '_average_SyN_inv.nii')
            nib.save(syn_avg,syn_avg_path)
            
            transformation_avg_inv = transformation_avg * (-1)

            transformation_avg_inv_nii = nib.Nifti1Image(transformation_avg_inv, syn_avg.affine, syn_avg.header)
            nib.save(transformation_avg_inv_nii, syn_avg_path_inv)
            del transformation_avg_inv_nii
        
            create_magnitude_displacement_volume(syn_avg_path)
            
            # Transform the volume from the LM-based step to this step
            folderOutput_LMbased =  os.path.join(input_folder, sampleName1 + ending_input_folder)
            
            # Transform landmarks
            pathLMsGMMovedRigid = os.path.join(folderOutput_LMbased, sampleName1+ ending_input_landmarks)
            pathLMsGMMovedSim = os.path.join(folderOutput, sampleName1+ ending_output_landmarks)
            
            lms =  flms.getLMs(pathLMsGMMovedRigid, constDivide=1.0/scale_registration)
            
            for k in range(nVolumes):
                print('-- Transforming volumes of the sample', flush = True)
                #print('transforming volumes of the sample')
                try:
                    pathMovingVolume = os.path.join(folderOutput_LMbased , sampleName1 + ending_input_volumes[k])
                    pathResultVolume = os.path.join(folderOutput , sampleName1 + ending_output_volumes[k])
                    volume1 = functionReadTIFFMultipage(pathMovingVolume,8)
                    # volume1 = np.array(scipy.ndimage.zoom(volume1, zoom=1/scale_registration, order=0))
                    volume1 = volume1[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
                    functionSaveTIFFMultipage(volume1,pathResultVolume + '_downsizing_to_check_inter.tiff',8)
                    
                    if k==0:
                        volume1_edge = get_volume_edge(volume1, kernel_size = 2)
                        xyz_0, yxz_0, zyx_0 = percentage_lms_in_surface(volume1_edge, lms)
                        lms_new_coordinates, lms_new_coordinates_fwd_debug = transform_landmarks_to_ants_coords(lms, affine_global)
                        xyz_0_newCoord, yxz_0_newCoord, zyx_0_newCoord = percentage_lms_in_surface(volume1_edge, lms_new_coordinates)
                    
                        del volume1_edge
                    
                    with SuppressOutput():
                        # syn_avg_path
                        resampled = ants.apply_transforms(transformlist = [syn_avg_path], \
                                                                        fixed=ants.from_numpy(volume1), \
                                                                            interpolation = 'genericLabel', moving = ants.from_numpy(volume1))
                    resampled_np = resampled.numpy()
                    functionSaveTIFFMultipage(resampled_np,pathResultVolume,8)
                    
                    if k==0:
                        
                        df_lms = pd.DataFrame(data=lms_new_coordinates)
                        df_lms.columns = ['x', 'y', 'z']
                        df_lms_t =ants.apply_transforms_to_points(dim = 3, points = df_lms, transformlist = [syn_avg_path_inv]) #2. syn_avg_path_ants 1. syn_avg_path
                        transformed_lms_new_coordinates = df_lms_t.values.tolist()
                                            
                        resampled_np_edge = get_volume_edge(resampled_np, kernel_size = 2)
                        
                        transformed_lms_sim_new_coordinates_back_to_original, transformed_lms_sim_new_coordinates_back_to_original_inv \
                            = transform_ants_coords_to_original(transformed_lms_new_coordinates, affine_global)
                        flms.saveLMs(pathLMsGMMovedSim + '_newCoordinatesTransformed_backToOriginalCoords.csv',transformed_lms_sim_new_coordinates_back_to_original)
                        flms.saveLMs(pathLMsGMMovedSim,transformed_lms_sim_new_coordinates_back_to_original)
                        xyz_3, yxz_3, zyx_3 = percentage_lms_in_surface(resampled_np_edge, transformed_lms_sim_new_coordinates_back_to_original)
                        
                        with open(pathLMsGMMovedSim + '_ratio_accuracy_lms.txt', "w") as f:
                                f.write('xyz yxz zyx \n')
                                f.write('Original LMs in Not Moved Volume: ' + str((xyz_0, yxz_0, zyx_0)) + "\n")
                                f.write('Original LMs correction in Not Moved Volume: ' + str((xyz_0_newCoord, yxz_0_newCoord, zyx_0_newCoord)) + "\n")
                                f.write('Orientation correction, application, and back to original coords CORRECT?: ' + str((xyz_3, yxz_3, zyx_3)) + "\n")
                        
                        del resampled_np_edge
                    
                    del volume1, resampled
                
                except OSError as error:
                    print(error)
                    logger.info(error)
                    logger.info('POSSIBLE SKIPPED VOLUME:' + ending_input_volumes[k])
            
            for temp_file_transform in list_files_displacement_fields:
                os.remove(temp_file_transform)
                
            del syn_avg
            
        if flag_compute_similarity_all_volumes:
            compute_IoU_samples(sampleNames, ending_output_volumes, output_folder_intensity, ending_output_folder)
        else:
            compute_IoU_samples(sampleNames, [ending_output_volumes[0]], output_folder_intensity, ending_output_folder)
    
    return output_folder_intensity

def register_syn_samples_to_reference(sampleNames, sampleNameReference, syn_iterations, input_folder, ending_input_folder, ending_input_volumes, \
                     ending_output_folder_toReference, ending_output_volumes_toReference, ending_input_landmarks, ending_output_landmarks_toReference, \
                         scale_registration, type_reg_str, logger, flag_skip_processing = False, flag_compute_similarity_all_volumes = True):
    
    nSamples = len(sampleNames)    
    nVolumes = len(ending_input_volumes)
    output_folder_intensity = os.path.join(input_folder, type_reg_str)
    if not os.path.exists(output_folder_intensity):
        os.mkdir(output_folder_intensity)
    
    logger.info('Skipping this process: ' + str(flag_skip_processing))
    if not flag_skip_processing:
        for i in range(nSamples):
            # This is the moving volume
            sampleName1 = sampleNames[i]
            # print(sampleName1)
            pathVolume1 = os.path.join(input_folder, sampleName1 + ending_input_folder, sampleName1 + ending_input_volumes[0])
            # print(pathVolume1)
            volume_moving_original_size = functionReadTIFFMultipage(pathVolume1,8)
            # volume1 = np.array(scipy.ndimage.zoom(volume_moving_original_size, zoom=1/scale_registration, order=0))
            volume1 = volume_moving_original_size[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
            
            folderOutput =  os.path.join(output_folder_intensity, sampleName1 + ending_output_folder_toReference)
            if not os.path.exists(folderOutput):
                os.mkdir(folderOutput)
            # Convert NumPy arrays to SimpleITK images
            moving_numpy = np.asarray(volume1)
            moving_numpy = np.float16(moving_numpy)
            del volume1
            
            start_time = time.time()
            
            key_transf = sampleName1 + '_to_' + sampleNameReference
            print('-- ' + key_transf, flush = True)
            logger.info('-- ' + key_transf)
            
            pathVolume2 = os.path.join(input_folder, sampleNameReference + ending_input_folder, sampleNameReference + ending_input_volumes[0])
            volume2 = functionReadTIFFMultipage(pathVolume2,8)
            # fixed_numpy = np.array(scipy.ndimage.zoom(volume2, zoom=1/scale_registration, order=0))
            fixed_numpy = volume2[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
            fixed_numpy = np.float16(fixed_numpy)
            
            reg_file, reg_file_inv = register_images_syn(fixed_numpy, moving_numpy, key_transf,affine_global, reg_iterations=syn_iterations)
            
            del fixed_numpy
    
            end_time = time.time()
            elapsed_time_01 = end_time - start_time
            print(f"-- Elapsed time: {elapsed_time_01:.2f} seconds.", flush = True)
            logger.info(f"-- Elapsed time: {elapsed_time_01:.2f} seconds.")
            del moving_numpy
            # Apply transform to the volume
            print('-- Starting: average_syn_transform', flush = True)
            logger.info('-- Starting: average_syn_transform')
            syn_avg = average_syn_transform([reg_file])

            transformation_avg = syn_avg.get_fdata()
            syn_avg_path = os.path.join(folderOutput, sampleName1 + '_to_'+sampleNameReference+'_SyN.nii.gz')
            syn_avg_path_inv  = os.path.join(folderOutput, sampleName1 + '_to_'+sampleNameReference+'_SyN_inv.nii')
            nib.save(syn_avg,syn_avg_path)
            
            transformation_avg_inv = transformation_avg * (-1)

            transformation_avg_inv_nii = nib.Nifti1Image(transformation_avg_inv, syn_avg.affine, syn_avg.header)
            nib.save(transformation_avg_inv_nii, syn_avg_path_inv)
            del transformation_avg_inv_nii
        
            create_magnitude_displacement_volume(syn_avg_path)
            
            # Transform the volume from the LM-based step to this step
            folderOutput_LMbased =  os.path.join(input_folder, sampleName1 + ending_input_folder)
            
            # Transform landmarks
            pathLMsGMMovedRigid = os.path.join(folderOutput_LMbased, sampleName1+ ending_input_landmarks)
            pathLMsGMMovedSim = os.path.join(folderOutput, sampleName1+ ending_output_landmarks_toReference)
            
            lms =  flms.getLMs(pathLMsGMMovedRigid, constDivide=1.0/scale_registration)
            
            for k in range(nVolumes):
                print('-- Transforming volumes of the sample', flush = True)
                logger.info('-- Transforming volumes ' + ending_input_volumes[k])
                try:
                    #print('transforming volumes of the sample')
                    pathMovingVolume = os.path.join(folderOutput_LMbased , sampleName1 + ending_input_volumes[k])
                    pathResultVolume = os.path.join(folderOutput , sampleName1 + ending_output_volumes_toReference[k])
                    volume1 = functionReadTIFFMultipage(pathMovingVolume,8)
                    # volume1 = np.array(scipy.ndimage.zoom(volume1, zoom=1/scale_registration, order=0))
                    volume1 = volume1[::np.int16(scale_registration), ::np.int16(scale_registration), ::np.int16(scale_registration)]
                    functionSaveTIFFMultipage(volume1,pathResultVolume + '_downsizing_to_check_inter.tiff',8)
                    
                    if k==0:
                        volume1_edge = get_volume_edge(volume1, kernel_size = 2)
                        xyz_0, yxz_0, zyx_0 = percentage_lms_in_surface(volume1_edge, lms)
                        lms_new_coordinates, lms_new_coordinates_fwd_debug = transform_landmarks_to_ants_coords(lms, affine_global)
                        xyz_0_newCoord, yxz_0_newCoord, zyx_0_newCoord = percentage_lms_in_surface(volume1_edge, lms_new_coordinates)
                    
                        del volume1_edge
                    
                    with SuppressOutput():
                        # syn_avg_path
                        resampled = ants.apply_transforms(transformlist = [syn_avg_path], \
                                                                        fixed=ants.from_numpy(volume1), \
                                                                            interpolation = 'genericLabel', moving = ants.from_numpy(volume1))
                    resampled_np = resampled.numpy()
                    functionSaveTIFFMultipage(resampled_np,pathResultVolume,8)
                    
                    if k==0:
                        
                        df_lms = pd.DataFrame(data=lms_new_coordinates)
                        df_lms.columns = ['x', 'y', 'z']
                        df_lms_t =ants.apply_transforms_to_points(dim = 3, points = df_lms, transformlist = [syn_avg_path_inv]) #2. syn_avg_path_ants 1. syn_avg_path
                        transformed_lms_new_coordinates = df_lms_t.values.tolist()
                                            
                        resampled_np_edge = get_volume_edge(resampled_np, kernel_size = 2)
                        
                        transformed_lms_sim_new_coordinates_back_to_original, transformed_lms_sim_new_coordinates_back_to_original_inv \
                            = transform_ants_coords_to_original(transformed_lms_new_coordinates, affine_global)
                        flms.saveLMs(pathLMsGMMovedSim + '_newCoordinatesTransformed_backToOriginalCoords.csv',transformed_lms_sim_new_coordinates_back_to_original)
                        flms.saveLMs(pathLMsGMMovedSim,transformed_lms_sim_new_coordinates_back_to_original)
                        xyz_3, yxz_3, zyx_3 = percentage_lms_in_surface(resampled_np_edge, transformed_lms_sim_new_coordinates_back_to_original)
                        
                        with open(pathLMsGMMovedSim + '_ratio_accuracy_lms.txt', "w") as f:
                                f.write('xyz yxz zyx \n')
                                f.write('Original LMs in Not Moved Volume: ' + str((xyz_0, yxz_0, zyx_0)) + "\n")
                                f.write('Original LMs correction in Not Moved Volume: ' + str((xyz_0_newCoord, yxz_0_newCoord, zyx_0_newCoord)) + "\n")
                                f.write('Orientation correction, application, and back to original coords CORRECT?: ' + str((xyz_3, yxz_3, zyx_3)) + "\n")
                        
                        del resampled_np_edge
                    
                except OSError as error:
                    print(error)
                    logger.info(error)
                    logger.info('POSSIBLE SKIPPED VOLUME:' + ending_input_volumes[k])
                
                del volume1, resampled
            
            os.remove(reg_file)
                
            del syn_avg
        
        logger.info('Computing IoU')            
        if flag_compute_similarity_all_volumes:
            compute_IoU_samples(sampleNames, ending_output_volumes_toReference, output_folder_intensity, ending_output_folder_toReference)
        else:
            compute_IoU_samples(sampleNames, [ending_output_volumes_toReference[0]], output_folder_intensity, ending_output_folder_toReference)
    
    return output_folder_intensity


