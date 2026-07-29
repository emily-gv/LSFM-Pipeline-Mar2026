import os
import numpy as np
from PIL import Image

import os
import numpy as np
from PIL import Image


def load_and_resize(fullpath, target_size=3000):
    """Load image as RGB and upscale (if needed) on the
    long side, using nearest-neighbour so no new blended colours appear."""
    img = Image.open(fullpath).convert("RGB")

    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), resample=Image.NEAREST)

    return np.asarray(img)


def function_encode_annotation_semantic(fullpath_orig, filename, folder_dest,
                                         red_thresh=15, green_thresh=15,
                                         target_size=3000):
    """
    Encodes an RGB annotation image into a semantic label mask:
        0 = background
        1 = red
        2 = green
    """

    # img = Image.open(fullpath_orig).convert('RGB')
    # numpydata = np.asarray(img).astype(int)

    numpydata = load_and_resize(fullpath_orig, target_size=target_size).astype(int)

    r = numpydata[:, :, 0]
    g = numpydata[:, :, 1]
    b = numpydata[:, :, 2]

    is_red = (r - g > red_thresh) & (r - b > red_thresh)
    is_green = (g - r > green_thresh) & (g - b > green_thresh)

    labels = np.zeros(r.shape, dtype=np.uint8)
    labels[is_red] = 1
    labels[is_green] = 2

    fullpath_dest = os.path.join(folder_dest, filename + '_mask.png')
    mask_encoded = Image.fromarray(labels)
    mask_encoded.save(fullpath_dest)


def function_encode_annotations(folder_rgb_annotations, folder_masks, target_size=3000):
    if not os.path.exists(folder_masks):
        os.makedirs(folder_masks)

    path_images = []
    file_ending = ''
    for file in os.listdir(folder_rgb_annotations):
        if file.endswith(".png") or file.endswith(".bmp") or file.endswith(".tif"):
            filename_tmp = file[0:-4]
            file_ending = file[-4:]
            path_images.append(filename_tmp)
    nFiles = len(path_images)

    for imgNumber in range(nFiles):
        fullpath_rgb = os.path.join(folder_rgb_annotations, path_images[imgNumber] + file_ending)
        function_encode_annotation_semantic(fullpath_orig=fullpath_rgb, filename=path_images[imgNumber], folder_dest=folder_masks, target_size=target_size)


if __name__ == "__main__":
    # function_encode_annotations(
    #     folder_rgb_annotations="/home/emilygv/Desktop/INTER_OBS_ANALYSIS/TestSet_Obs2",
    #     folder_masks="/home/emilygv/Desktop/INTER_OBS_ANALYSIS/TestSet_Obs2_3000",
    # )
    # output_folder = "/home/emily/Desktop/TestSet_Atiksha_3000"
    output_folder = "/home/emily/Desktop/June2026Annotations_Emily_complete_encoded3000"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input_folder = "/home/emily/Desktop/June2026Annotations_Emily_complete"

    function_encode_annotations(input_folder, output_folder)

    # path_images = []
    # file_ending = ''
    # for file in os.listdir(input_folder):
    #     if file.endswith(".png") or file.endswith(".bmp") or file.endswith(".tif"):
    #         filename_tmp = file[0:-4]
    #         file_ending = file[-4:]
    #         path_images.append(filename_tmp)
    # nFiles = len(path_images)

    # for imgNumber in range(nFiles):
    #     fullpath_input = os.path.join(input_folder, path_images[imgNumber] + file_ending)
    #     # print(fullpath_input)
    #     numpyarray = load_and_resize(fullpath_input)
    #     fullpath_dest = os.path.join(output_folder, path_images[imgNumber] + '.png')
    #     image = Image.fromarray(numpyarray)
    #     image.save(fullpath_dest)
