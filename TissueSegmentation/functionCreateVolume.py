import os
import cv2
from PIL import Image
from PIL.TiffTags import TAGS_V2
import numpy as np
from PIL import ImageSequence


def functionCreateVolume(folder_input, dest_file_tiff, resX = 911, resY = 911, resZ = 2917, bith_depth = 8, label = -1 ):
    # in nanometers
    # search the voxel dimensions in Superhero Embryos, Sheet 2

    # processing
    scale = resX / resZ

    tiffs = []  # list of files (tiles) corresponding to channel
    for file in os.listdir(folder_input):
        # os.chdir(currentDir + filesep + source_folder)
        fileExtension = os.path.splitext(file)
        if fileExtension[1] == ".png":
            tiffs.append(''.join(fileExtension))
    tiffs = sorted(tiffs)

    numImgs = len(tiffs)  # number of tiles

    multifile_tiff = []
    for i in range(0, numImgs):

        full_path_source = os.path.join(folder_input, tiffs[i])
        #print(full_path_source)
        slice_original = cv2.imread(full_path_source, cv2.IMREAD_GRAYSCALE)
        if slice_original is None:
            print('Error reading: ' + os.path.basename(full_path_source))
        
        if label != -1:
            slice_original = slice_original == label
            slice_original = np.uint8(slice_original) * 255

        height, width = np.shape(slice_original)
        resized = (int(scale * width), int(scale * height))
        img_downsample = cv2.resize(slice_original, resized, cv2.INTER_NEAREST)
        if bith_depth == 16:
            img_downsample = np.uint16(img_downsample)
        else:
            img_downsample = np.uint8(img_downsample)
        img_downsample = Image.fromarray(img_downsample)
        multifile_tiff.append(img_downsample)

    firstSlice = multifile_tiff[0]
    firstSlice.save(dest_file_tiff, save_all=True, append_images=multifile_tiff[1:], compression="tiff_lzw")
    #ImageSequence.tiffsave(firstSlice,dest_file_tiff,save_all=True,append_images=multifile_tiff[1:],compression="tiff_lzw")
    #print("Finished!")
    
def functionIsotropicVolume(list_anisotropic, dest_file_tiff, resX = 913.89, resY = 913.89, resZ = 4940, bith_depth = 8, label = -1):
    # processing
    scale = resX / resZ
    n_slices = len(list_anisotropic)
    
    multifile_tiff = []
    for i in range(0, n_slices):

        slice_original = list_anisotropic[i]
        
        if label != -1:
            slice_original = slice_original == label
            slice_original = np.uint8(slice_original) * 255

        height, width = np.shape(slice_original)
        resized = (int(scale * width), int(scale * height))
        img_downsample = cv2.resize(slice_original, resized, cv2.INTER_NEAREST)
        if bith_depth == 16:
            img_downsample = np.uint16(img_downsample)
        else:
            img_downsample = np.uint8(img_downsample)
        img_downsample = Image.fromarray(img_downsample)
        multifile_tiff.append(img_downsample)

    firstSlice = multifile_tiff[0]
    firstSlice.save(dest_file_tiff, save_all=True, append_images=multifile_tiff[1:], compression="tiff_lzw")
