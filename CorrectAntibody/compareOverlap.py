from TissueSegmentation.data_loader import get_images_from_path
import cv2
import os 
import numpy as np
from PIL import Image
from scipy import ndimage

def compute_image_overlay(image1, image2, output_folder, str_description):
    """
    Create a merged image of overlap between CC3 and pHH3. Will only compare between images of the same shape.

    Image 1 = Red
    Image 2 = Green
    Overlap = Yellow

    Args:
        image1 (str) = Full filepath to image slice of cell marker #1
        image2 (str) = Full filepath to image slice of cell marker #2
        output_folder (str) = Full filepath where you want the output
        str_description (str) = Will be added to output image (eg. 'cc3_phh3_comparison')
    """

    mask1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE) == 255  
    mask2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE) == 255
    # print("Shape of image1: ", mask1.shape)
    # print("Shape of image2: ", mask2.shape)
    if mask1.shape != mask2.shape:
        print("Images are not same shape: " + image1 + ', ' + image2)
    else:
        h,w = mask1.shape
        output = np.zeros((h,w,3), dtype=np.uint8) # create empty rgb image
        output[mask1 & mask2] = [0,255,255] # yellow
        output[mask1 & ~mask2] = [0,0,255] # red
        output[~mask1 & mask2] = [0,255,0] # green
        base = os.path.splitext(os.path.basename(image1))[0]
        output_path = os.path.join(output_folder, base + '_' + str_description + '_merged.png')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(output_path, output)

def compute_image_overlay_folder(folder_slices_1, folder_slices_2, output_folder, str_description):
        """
    Iterate through two folders and create merged images of overlap between CC3 and pHH3.
    Assumed image names are the same between folders because get_images_from_path() returns whatever with sorted()

    folder_slices_1 = Red
    folder_slices_2 = Green
    Overlap = Yellow

    Args:
        folder_slices_1 (str) = Full filepath to folder of cell marker #1
        folder_slices_2 (str) = Full filepath to folder of cell marker #2
        output_folder (str) = Full filepath where you want the output
        str_description (str) = Will be added to output images (eg. 'cc3_phh3_comparison')
    """
    list_slices_1 = get_images_from_path(folder_slices_1)
    list_slices_2 = get_images_from_path(folder_slices_2)
    n_list_slices_1 = len(list_slices_1)
    n_list_slices_2 = len(list_slices_2)
    if n_list_slices_1 != n_list_slices_2:
        print('Folders with different number of slices')
        print('folder 1:' + str(n_list_slices_1))
        print('folder 2:' + str(n_list_slices_2))
        return
    for i in range(n_list_slices_1):
        img_slice_path_1 = list_slices_1[i]
        img_slice_path_2 = list_slices_2[i]
        compute_image_overlay(img_slice_path_1, img_slice_path_2, output_folder, str_description)

# https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
# Shift up = array[:-x, :]
# Shift down = array[:x, ]
# Shift left = array[:,:-x]
# Shift right = array[;,:x]
def correct_shift(folder_slices_toShift, output_folder, shift_value):
    """
    Iterates through folder to correct shift in a hardcoded direction and saves to a new folder.
    Replace "img_shifted[:-shift_value,:] = img_original[shift_value:, :]" with desired behaviour as outlined in the comments directly above.
    ** I think I made this to use black pixels and not wrap around but verify that before using **

    Args: 
        folder_slices_toShift (str): Full filepath
        output_folder (str): Full filepath
        shift_value (int): Manually determined pixels to shift by


    Outputs a folder of BINARY images.
    """
    # output = np.zeros_like(image)
    # output[:-shift_value, :] = image[shift_value:, :]
    list_slices = get_images_from_path(folder_slices_toShift)
    n_list_slices = len(list_slices)
    for i in range(n_list_slices):
        img_slice_path = list_slices[i]
        img_original = cv2.imread(img_slice_path, cv2.IMREAD_GRAYSCALE) == 255
        
        img_shifted = np.zeros_like(img_original)
        # THE BELOW SHIFTS IMAGE UP
        img_shifted[:-shift_value,:] = img_original[shift_value:, :]
        img_shifted_bw = (img_shifted.astype(np.uint8)) * 255

        base = os.path.splitext(os.path.basename(img_slice_path))[0]
        output_path = os.path.join(output_folder, base + '_shifted.png')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        cv2.imwrite(output_path, img_shifted_bw)


def correct_shift_label(folder_slices_toShift, output_folder, shift_value):
    """
    Same as correct_shift() but for label images. NO POINT IN USING. 
    Cellpose labels are kinda bad, correct_cells() script ends up calculating its own labels and using those. 
    """
    list_slices = get_images_from_path(folder_slices_toShift)
    n_list_slices = len(list_slices)
    for i in range(n_list_slices):
        img_slice_path = list_slices[i]
        img_original = cv2.imread(img_slice_path, cv2.IMREAD_UNCHANGED)

        img_shifted = np.zeros_like(img_original)
        # THE BELOW SHIFTS IMAGE UP
        img_shifted[:-shift_value,...] = img_original[shift_value:, ...]

        base = os.path.splitext(os.path.basename(img_slice_path))[0]
        output_path = os.path.join(output_folder, base + '_shifted.png')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        cv2.imwrite(output_path, img_shifted)

def main():
    # image1 = '/home/emilygv/Desktop/CC3_pHH3_comparison/Aug28_2025_27/Step25b_pHH3_slices_binary/Aug28_2025_27_slice_0350.png'
    # image2 = '/home/emilygv/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_binary/Aug28_2025_27_slice_0350.png' 
    # compute_image_overlay(image1, image2, output_folder, str_description)

    # folder_slices_toShift = '/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_binary'
    # output_folder = '/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_binary_shifted'
    # correct_shift(folder_slices_toShift, output_folder, 12)
    
    # folder_slices_1 = '/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/Step25b_pHH3_slices_binary'
    # folder_slices_2 = '/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_binary_shifted' 
    # output_folder = '/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/CC3_pHH3_comparison_shifted'
    # str_description = 'phh3_cc3'
    # compute_image_overlay_folder(folder_slices_1, folder_slices_2, output_folder, str_description)

    # folder_slices_toShift = '/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_label'
    # output_folder = '/home/emily/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_label_shifted'
    # correct_shift_label(folder_slices_toShift, output_folder, 12)
    
if __name__=="__main__":
    main()
