import os
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

# import glymur
# from PIL import Image
import cv2

# Configuration - edit these or specify directly when calling compress_TIFFs_parallel in main

# source_folder = "/mnt/BHNasLightsheet/Emily_Thesis_Winter2026/Nosip_Jan152026_12/PHH3_C" 
# dest_folder = "/mnt/BHNasLightsheet/Emily_Thesis_Winter2026/Nosip_Jan152026_12/PHH3_C_Compressed" 
# stats_file = "/home/emily/Documents/LSFM-Segmentation-Refactoring/stats_cv2.txt"

def compress_TIFFs_cv2(source_folder, dest_folder):
    """
    Compress tiffs to jp2 images using cv2, calculate metrics, and save compressed images to specified folder
        - CV2 doesn't support .j2k


    Compression ratio can be specified in the line:
        cv2.imwrite(dest_file, img_original, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])

    Args:
        source_folder (str): Filepath to folder containing uncompressed files
        dest_folder (str): Filepath to output folder for compressed files
        
    Returns:
        N/A
    """

    if not os.path.exists(source_folder):
        print("Input file does not exist")
        return
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    tiffs = [f for f in os.listdir(source_folder) if f.endswith(".tif")]
    numImgs = len(tiffs)

    # Intermediate variables for statistics
    sizeTIFFs = np.zeros(numImgs) # MB
    sizeJPEG2000s = np.zeros(numImgs) # MB
    lossJPEG2000s = np.zeros(numImgs) # Mean non-zero valued pixel difference

    printcounter = 0
    start_time = time.time()

    for imgNumber in range(numImgs):
        printcounter += 1

        full_path_source = os.path.join(source_folder, tiffs[imgNumber])

        # Read tiff using cv2 - IMREAD_UNCHANGED preserves bit depth
        img_original = cv2.imread(full_path_source, cv2.IMREAD_UNCHANGED)
        
        # Handle color images (convert to grayscale if needed). Shape 3 would be colour channel?
        if len(img_original.shape) == 3:
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        height, width = img_original.shape[:2] # ignore channels

        # Split filename without extension & change to .jp2
        name, ext = os.path.splitext(tiffs[imgNumber])
        dest_file = os.path.join(dest_folder, name + '.jp2')
        
        # Save as JPEG2000 with 10x compression ratio
        # 1000 / compression ratio (=10) = 100
        cv2.imwrite(dest_file, img_original, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])

        sizeTIFFs[imgNumber] = os.path.getsize(full_path_source) / (1024 * 1024)
        sizeJPEG2000s[imgNumber] = os.path.getsize(dest_file) / (1024 * 1024)

        # Read compressed image as numpy array
        img_j2k = cv2.imread(dest_file, cv2.IMREAD_UNCHANGED) 
        
        # Array of all pixels >1=
        ANoZero = img_original[img_original > 1] 

        # Compute average intensity of all non-zero pixels. If divide by zero, set 1.0
        valMeansNoZero = float(np.mean(ANoZero)) if np.any(ANoZero) else 1.0 

        # Compute loss
        differenceJPEG2000 = img_original[img_original > 1].astype(float) - img_j2k[img_original > 1].astype(float)
        differenceJPEG2000 = np.abs(differenceJPEG2000)
        differenceJPEG2000 = differenceJPEG2000.sum()
        lossJPEG2000s[imgNumber] = differenceJPEG2000 / (width*height*valMeansNoZero)

        if printcounter % 10 == 0:
            print(f"[{imgNumber+1}/{numImgs}] compressed")

    meanSizeTIFF = sizeTIFFs.mean()
    meanCompJPEG2000 = 1 - sizeJPEG2000s.mean() / meanSizeTIFF
    time_elapsed = time.time() - start_time
    
    print(f"Time elapsed: {int(time_elapsed)} seconds")
    print("Statistics: ")
    print(f"Compression JPEG2000: {meanCompJPEG2000}")
    print(f"Loss JPEG2000: {lossJPEG2000s.mean()} ({lossJPEG2000s.std()})")


def compress_chunk_cv2(source_folder, dest_folder, tiffs, index):
    """
    FOR USE IN compress_TIFFs_parallel()
    Compress tiffs to jp2 images & calculates loss of current batch
        - CV2 doesn't support .jp2 images
    Compression ratio can be specified in the line:
        cv2.imwrite(dest_file, img_original, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])

    Args:
        source_folder (str): Filepath to folder containing uncompressed files
        dest_folder (str): Filepath to output folder for compressed files
        tiffs (list):  List of tiffs in source_folder
        index (int):  Subsection of tiffs to compress

    Returns:
        local_sizeTIFFs (list): Sizes of original TIFFs in chunk
        local_sizeJPEG2000s (list): Sizes of jp2 images
        local_lossJPEG2000s (list): Loss for each image
    """

    local_sizeTIFFs = []
    local_sizeJPEG2000s = []
    local_lossJPEG2000s = []

    for imgNumber in index:
        full_path_source = os.path.join(source_folder, tiffs[imgNumber])

        # Read tiff using cv2 - IMREAD_UNCHANGED preserves bit depth
        img_original = cv2.imread(full_path_source, cv2.IMREAD_UNCHANGED)
        
        # Handle color images (convert to grayscale if needed). Shape 3 would be colour channel?
        if len(img_original.shape) == 3:
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        height, width = img_original.shape[:2] # ignore channels
        
        name, ext = os.path.splitext(tiffs[imgNumber])
        dest_file = os.path.join(dest_folder, name + '.jp2')

        # Save as JPEG2000 with 10x compression ratio
        # 1000 / compression ratio (=10) = 100
        cv2.imwrite(dest_file, img_original, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])

        local_sizeTIFFs.append(os.path.getsize(full_path_source) / (1024 * 1024))
        local_sizeJPEG2000s.append(os.path.getsize(dest_file) / (1024 * 1024))

        # Read compressed image as numpy array
        img_j2k = cv2.imread(dest_file, cv2.IMREAD_UNCHANGED)

        # Array of all pixels >1
        ANoZero = img_original[img_original > 1] 
        # Compute average intensity of all non-zero pixels. If divide by zero, set 1.0
        valMeansNoZero = float(np.mean(ANoZero)) if np.any(ANoZero) else 1.0 
        differenceJPEG2000 = np.abs(img_original[img_original > 1].astype(float) - img_j2k[img_original > 1].astype(float)).sum()
        local_lossJPEG2000s.append(differenceJPEG2000 / (width*height*valMeansNoZero))

    return local_sizeTIFFs, local_sizeJPEG2000s, local_lossJPEG2000s


def compress_TIFFs(source_folder, dest_folder):
    """
    ** Same thing using glymur/pillow instead of cv2 **

    Compress tiffs to j2k images & computes metrics. Save results to specified folder

    Compression ratio can be specified in the line:
        glymur.Jp2k(dest_file, data=img_original, cratios=[10])

    Args:
        source_folder (str): Filepath to folder containing uncompressed files
        dest_folder (str): Filepath to output folder for compressed files
        
    Returns:
        N/A
    """
    if not os.path.exists(source_folder):
        print("Input file does not exist")
        return
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    tiffs = [f for f in os.listdir(source_folder) if f.endswith(".tif")]
    numImgs = len(tiffs)

    # Intermediate variables for statistics
    sizeTIFFs = np.zeros(numImgs) # MB
    sizeJPEG2000s = np.zeros(numImgs) # MB
    lossJPEG2000s = np.zeros(numImgs) # Mean non-zero valued pixel difference

    printcounter = 0
    start_time = time.time()

    for imgNumber in range(numImgs):
        printcounter += 1

        full_path_source = os.path.join(source_folder, tiffs[imgNumber])

        # Read tiff as numpy array for glymur
        img_original = np.array(Image.open(full_path_source))
        height, width = img_original.shape[:2] # ignore channels

        name, ext = os.path.splitext(tiffs[imgNumber])
        dest_file = os.path.join(dest_folder, name + '.j2k')
        
        # Save as JPEG2000 with 10 compression ratio
        glymur.Jp2k(dest_file, data=img_original, cratios=[10])

        sizeTIFFs[imgNumber] = os.path.getsize(full_path_source) / (1024 * 1024)
        sizeJPEG2000s[imgNumber] = os.path.getsize(dest_file) / (1024 * 1024)
        ANoZero = img_original[img_original > 1] # Array of all pixels >1
        valMeansNoZero = float(np.mean(ANoZero)) if np.any(ANoZero) else 1.0 # Compute average intensity of all non-zero pixels. If divide by zero, set 1.0
        img_j2k = glymur.Jp2k(dest_file)[:] # Read compressed image as numpy array
        differenceJPEG2000 = img_original[img_original > 1].astype(float) - img_j2k[img_original > 1].astype(float)
        differenceJPEG2000 = np.abs(differenceJPEG2000)
        differenceJPEG2000 = differenceJPEG2000.sum()
        lossJPEG2000s[imgNumber] = differenceJPEG2000 / (width*height*valMeansNoZero)

        if printcounter % 10 == 0:
            print(f"[{imgNumber+1}/{numImgs}] compressed")

    meanSizeTIFF = sizeTIFFs.mean()
    meanCompJPEG2000 = 1 - sizeJPEG2000s.mean() / meanSizeTIFF
    time_elapsed = time.time() - start_time
    
    print(f"Time elapsed: {int(time_elapsed)} seconds")
    print("Statistics: ")
    print(f"Compression JPEG2000: {meanCompJPEG2000}")
    print(f"Loss JPEG2000: {lossJPEG2000s.mean()} ({lossJPEG2000s.std()})")


def compress_chunk(source_folder, dest_folder, tiffs, index):
    """
    ** Same thing using glymur/pillow instead of cv2 **

    FOR USE IN compress_TIFFs_parallel()

    Compress tiffs to j2k images & computes metrics
    Compression ratio can be specified in the line:
        glymur.Jp2k(dest_file, data=img_original, cratios=[10])

    Args:
        source_folder (str): Filepath to folder containing uncompressed files
        dest_folder (str): Filepath to output folder for compressed files
        tiffs (list):  List of tiffs in source_folder
        index (int):  Subsection of tiffs to compress
        
    Returns:
        local_sizeTIFFs (list): Sizes of original TIFFs in chunk
        local_sizeJPEG2000s (list): Sizes of jp2 images
        local_lossJPEG2000s (list): Loss for each image
    """

    local_sizeTIFFs = []
    local_sizeJPEG2000s = []
    local_lossJPEG2000s = []

    for imgNumber in index:
        full_path_source = os.path.join(source_folder, tiffs[imgNumber])

        # Read tiff using pillow
        img_pil = Image.open(full_path_source)

        # Convert to grayscale if not already
        if img_pil.mode != 'L':
            img_pil = img_pil.convert('L')

         # Read tiff as numpy array for glymur
        img_original = np.array(img_pil)
        height, width = img_original.shape[:2] # ignore channels
        
        name, ext = os.path.splitext(tiffs[imgNumber])
        dest_file = os.path.join(dest_folder, name + '.j2k')

        # Save as JPEG2000 with 10 compression ratio
        glymur.Jp2k(dest_file, data=img_original, cratios=[10])

        local_sizeTIFFs.append(os.path.getsize(full_path_source) / (1024 * 1024))
        local_sizeJPEG2000s.append(os.path.getsize(dest_file) / (1024 * 1024))
        
        ANoZero = img_original[img_original > 1] # Array of all pixels >1
        valMeansNoZero = float(np.mean(ANoZero)) if np.any(ANoZero) else 1.0 # Compute average intensity of all non-zero pixels. If divide by zero, set 1.0
        img_j2k = glymur.Jp2k(dest_file)[:] # Read compressed image as numpy array
        differenceJPEG2000 = np.abs(img_original[img_original > 1].astype(float) - img_j2k[img_original > 1].astype(float)).sum()
        local_lossJPEG2000s.append(differenceJPEG2000 / (width*height*valMeansNoZero))

    return local_sizeTIFFs, local_sizeJPEG2000s, local_lossJPEG2000s


def compress_TIFFs_parallel(source_folder, dest_folder, n_workers=5):
    """
    Parallize TIFF compression & compute overall metrics
    Save results to specified folder

    Args:
        source_folder (str): Filepath to folder containing uncompressed files
        dest_folder (str): Filepath to output folder for compressed files
        n_workers = number of CPU cores to use

    Returns:
        N/A
    """
    
    if not os.path.exists(source_folder):
        print("Input file does not exist")
        return
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    tiffs = [f for f in os.listdir(source_folder) if f.endswith(".tif")]
    numImgs = len(tiffs)

    indices = np.array_split(np.arange(numImgs), n_workers)

    total_sizeTIFFs = []
    total_sizeJPEG2000s = []
    total_lossJPEG2000s = []
    
    start_time = time.time()
    print("Started compressing files")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for idx in indices:
            # CHANGE THIS BASED ON CV2 VS GLYMUR IMPLEMENTATION
            future = executor.submit(compress_chunk_cv2, source_folder, dest_folder, tiffs, idx)
            futures.append(future)

        for f in futures:
            local_sizeTIFFs, local_sizeJPEG2000s, local_lossJPEG2000s = f.result()
            total_sizeTIFFs.extend(local_sizeTIFFs)
            total_sizeJPEG2000s.extend(local_sizeJPEG2000s)
            total_lossJPEG2000s.extend(local_lossJPEG2000s)

    all_sizeTIFFs = np.array(total_sizeTIFFs)
    all_sizeJPEG2000s = np.array(total_sizeJPEG2000s)
    all_lossJPEG2000s = np.array(total_lossJPEG2000s)

    meanSizeTIFF = all_sizeTIFFs.mean()
    meanCompJPEG2000 = 1 - all_sizeJPEG2000s.mean() / meanSizeTIFF
    time_elapsed = time.time() - start_time

    print(f"Time elapsed: {int(time_elapsed)} seconds")
    print("Statistics: ")
    print(f"Compression JPEG2000: {meanCompJPEG2000}")
    print(f"Loss JPEG2000: {all_lossJPEG2000s.mean()} ({all_lossJPEG2000s.std()})")

    # with open(stats_file, "a") as f:
    #     f.write(f"Source: {source_folder}\n")
    #     f.write(f"Time elapsed: {int(time_elapsed)} seconds\n")
    #     f.write("Statistics: \n")
    #     f.write(f"Compression JPEG2000: {meanCompJPEG2000}\n")
    #     f.write(f"Loss JPEG2000: {all_lossJPEG2000s.mean()} ({all_lossJPEG2000s.std()})\n")


# if __name__ == "__main__":
    # compress_TIFFs_parallel(source_folder, dest_folder,  n_workers=5)
    # compress_TIFFs_parallel("/home/emily/Desktop/Test-ClearHoechst", "/home/emily/Desktop/Test-ClearHoechst/Nuclear_Compressed" ,  n_workers=5)
    # compress_TIFFs_parallel('/mnt/BHNasLightsheet/Emily_Thesis_Winter2026/Nosip_Jan152026_12/CC3_C', '/mnt/BHNasLightsheet/Emily_Thesis_Winter2026/Nosip_Jan152026_12_cv2/CC3_Compressed', n_workers=5)
