from TissueSegmentation.data_loader import get_images_from_path
import cv2
import os 
import numpy as np

def compute_image_overlay(image1, image2, output_folder, str_description):
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
        output_path = os.path.join(output_folder, image1 + str_description + '_merged.png')
        cv2.imwrite(output_path, output)


def compute_image_overlay_folder(folder_slices_1, folder_slices_2, output_folder, str_description):
    list_slices_1 = get_images_from_path(folder_slices_1)
    list_slices_2 = get_images_from_path(folder_slices_2)
    n_list_slices_1 = len(list_slices_1)
    n_list_slices_2 = len(list_slices_2)
    if n_list_slices_1 != n_list_slices_2:
        print('Cannot compute IoU between folder with different number of slices')
        print('folder 1:' + str(n_list_slices_1))
        print('folder 2:' + str(n_list_slices_2))
        return
    for i in range(n_list_slices_1):
        img_slice_path_1 = list_slices_1[i]
        img_slice_path_2 = list_slices_2[i]
        compute_image_overlay(img_slice_path_1, img_slice_path_2, output_folder, str_description)
    
def main():
    # image1 = '/home/emilygv/Desktop/CC3_pHH3_comparison/Aug28_2025_27/Step25b_pHH3_slices_binary/Aug28_2025_27_slice_0350.png'
    # image2 = '/home/emilygv/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_binary/Aug28_2025_27_slice_0350.png' 
    folder_slices_1 = '/home/emilygv/Desktop/CC3_pHH3_comparison/Aug28_2025_27/Step25b_pHH3_slices_binary'
    folder_slices_2 = '/home/emilygv/Desktop/CC3_pHH3_comparison/Aug28_2025_27/cc3_slices_binary' 
    output_folder = '/home/emilygv/Desktop/CC3_pHH3_comparison'
    str_description = '_phh3_cc3'
    # compute_image_overlay(image1, image2, output_folder, str_description)
    compute_image_overlay_folder(folder_slices_1, folder_slices_2, output_folder, str_description)

if __name__=="__main__":
    main()
