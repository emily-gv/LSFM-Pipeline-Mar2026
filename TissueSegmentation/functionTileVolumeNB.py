from skimage import exposure
import os
os.environ['OPENCV_IO_ENABLE_JASPER']='True'
import cv2
import numpy as np
from TissueSegmentation.functionPercNorm import functionPercNorm
from PIL import Image
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def functionCopyImageAsPNG(folder_input, folder_output, sample_name):
    listing = os.listdir(folder_input)
    listing = sorted(listing)

    fileFormat = '.png'
    nFiles = len(listing)
    for imgNumber in range(nFiles):
        sliceNumber = imgNumber + 1
        #print('Processing:' + str(imgNumber + 1))
        fullpathOrig = os.path.join(folder_input, listing[imgNumber])

        if fullpathOrig.endswith('.j2k'): #CV2 does not read j2k, but PILLOW does
            img = Image.open(fullpathOrig)
            if img is None:
                print('Error reading: ' + os.path.basename(fullpathOrig))
            img = np.array(img)
        else:
            img = cv2.imread(fullpathOrig, cv2.IMREAD_ANYDEPTH)
            if img is None:
                print('Error reading: ' + os.path.basename(fullpathOrig))
        
        imgNorm = functionPercNorm(img)  # Normalization fixed
        imgNorm = exposure.rescale_intensity(imgNorm, out_range=(0, 255))
        
        outputName = os.path.join(folder_output, sample_name)
        name = outputName + '_slice_' + str(sliceNumber).zfill(4) + fileFormat
        # print(name)
        cv2.imwrite(name, imgNorm.astype(np.uint8))  # saves as 8 bit image
        
def process_png_range(args):
    folder_input, folder_output, sample_name, start_idx, end_idx, listing = args
    
    fileFormat = '.png'
    processed_count = 0
    for imgNumber in range(start_idx, end_idx):
        sliceNumber = imgNumber + 1
        #print('Processing:' + str(imgNumber + 1))
        fullpathOrig = os.path.join(folder_input, listing[imgNumber])

        if fullpathOrig.endswith('.j2k'): #CV2 does not read j2k, but PILLOW does
            img = Image.open(fullpathOrig)
            if img is None:
                print('Error reading: ' + os.path.basename(fullpathOrig))
            img = np.array(img)
        else:
            img = cv2.imread(fullpathOrig, cv2.IMREAD_ANYDEPTH)
            if img is None:
                print('Error reading: ' + os.path.basename(fullpathOrig))
        
        imgNorm = functionPercNorm(img)  # Normalization fixed
        imgNorm = exposure.rescale_intensity(imgNorm, out_range=(0, 255))
        
        outputName = os.path.join(folder_output, sample_name)
        name = outputName + '_slice_' + str(sliceNumber).zfill(4) + fileFormat
        # print(name)
        cv2.imwrite(name, imgNorm.astype(np.uint8))  # saves as 8 bit image
        
        processed_count += 1
    return processed_count

def functionCopyImageAsPNG_parallel(folder_input, folder_output, sample_name, n_workers=None):
    listing = os.listdir(folder_input)
    listing = sorted(listing)
    nFiles = len(listing)

    if n_workers is None:
        n_workers = os.cpu_count()

    chunk_size = nFiles // n_workers

    args_list = []
    start_idx = 0

    for i in range(n_workers):
        if i==n_workers - 1:
            end_idx = nFiles
        else:
            end_idx = start_idx + chunk_size
        
        if start_idx < nFiles:
            args_list.append((
                folder_input,
                folder_output,
                sample_name,
                start_idx,
                end_idx,
                listing
            ))
        
        start_idx = end_idx
        print(f'Converting {nFiles} images to PNG across {n_workers} workers...')

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_png_range, args) for args in args_list]
        
        with tqdm(total=nFiles, desc="Converting to PNG", unit="image") as pbar:
            for future in as_completed(futures):
                images_processed = future.result()
                pbar.update(images_processed)



def functionTileVolumeNB(folder_input, folder_output, sample_name, patchSize = 3000):
    # output parameters
    # nbits = 8, can be 16
    nChannels = 1  # can be 3
    fileFormat = '.png'

      # patchSize = 512, patchSize = 1024
    spx = np.uint16(patchSize * 0.1)

    listing = os.listdir(folder_input)
    listing = sorted(listing)
    nFiles = len(listing)
    usefulSize = patchSize - (2 * spx)

    outputName = os.path.join(folder_output, sample_name)
    #print('Total files: ' + str(nFiles))
    #nFiles_debug = 100
    for imgNumber in range(nFiles):
        sliceNumber = imgNumber + 1
        #print('Processing:' + str(imgNumber + 1))
        fullpathOrig = os.path.join(folder_input, listing[imgNumber])

        if fullpathOrig.endswith('.j2k'): #CV2 does not read j2k, but PILLOW does
            img = Image.open(fullpathOrig)
            if img is None:
                print('Error reading: ' + os.path.basename(fullpathOrig))
            img = np.array(img)
        else:
            img = cv2.imread(fullpathOrig, cv2.IMREAD_ANYDEPTH)
            if img is None:
                print('Error reading: ' + os.path.basename(fullpathOrig))
        hOrig, wOrig = np.shape(img)
        #
        imgNorm = functionPercNorm(img)  # Normalization fixed
        imgNorm = exposure.rescale_intensity(imgNorm, out_range=(0, 255))
        #
        hImgAugmented = hOrig + patchSize
        wImgAugmented = wOrig + patchSize
        imgAugmented = np.zeros([hImgAugmented, wImgAugmented])
        #
        imgAugmented[(spx + 1):(spx + hOrig + 1), (spx + 1):(spx + wOrig + 1)] = imgNorm

        # Saving the tiles
        numBlocksR = 0
        for r in range(0, hOrig, usefulSize):  # iteration inside the true image
            numBlocksC = 0
            numBlocksR += 1
            for c in range(0, wOrig, usefulSize):
                numBlocksC += 1
                # add neighbourhood to tile

                # top left corner
                r1 = r
                c1 = c

                # upper neighbourhood in the true image
                r0 = r1 - spx
                c0 = c1 - spx

                # bottom right neighbourhood in the true image
                r2 = r0 + patchSize - 1
                c2 = c0 + patchSize - 1

                # translation to augmented image
                r0_a = r0 + spx
                c0_a = c0 + spx
                r2_a = r2 + spx
                c2_a = c2 + spx

                block = imgAugmented[r0_a:(r2_a + 1), c0_a:(c2_a + 1)]

                name = (outputName + '_slice_' + str(sliceNumber).zfill(4) + '_block_' +
                        str(numBlocksR).zfill(2) + '_' + str(numBlocksC).zfill(2) + fileFormat)
                if nChannels == 3:
                    rgb = np.concatenate(3, block, block, block, block)
                else:
                    rgb = block

                cv2.imwrite(name, rgb.astype(np.uint8))  # saves as 8 bit image
                # io.imsave(name, img_as_ubyte(rgb), check_contrast = False)

        # save slice data
        infoName = outputName + '_slice_' + str(sliceNumber).zfill(4) + '_info.txt'

        fid = open(infoName, 'w')
        fid.write('sampleName:' + sample_name +
                  '\nimgNumber:' + str(sliceNumber) +
                  '\nhOrig:' + str(hOrig) +
                  ' \nwOrig:' + str(wOrig) +
                  '\nhImgAugmented:' + str(hImgAugmented) +
                  '\nwImgAugmented:' + str(wImgAugmented) +
                  '\npatchSize:' + str(patchSize) +
                  '\nnumBlocksR:' + str(numBlocksR) +
                  '\nnumBlocksC:' + str(numBlocksC) +
                  '\nspx:' + str(spx) +
                  '\nusefulSize:' + str(usefulSize))

        fid.close()


def functionTileVolume_range(args):
    folder_input, folder_output, sample_name, patchSize, start_idx, end_idx, listing = args
    
    fileFormat = '.png'
    spx = int(patchSize * 0.1)
    usefulSize = patchSize - (2 * spx)
    outputName = os.path.join(folder_output, sample_name)


    for imgNumber in range(start_idx, end_idx):
        sliceNumber = imgNumber + 1
        #print('Processing:' + str(imgNumber + 1))
        fullpathOrig = os.path.join(folder_input, listing[imgNumber])

        if fullpathOrig.endswith('.j2k'): #CV2 does not read j2k, but PILLOW does
            img = Image.open(fullpathOrig)
            if img is None:
                print('Error reading: ' + os.path.basename(fullpathOrig))
            img = np.array(img)
        else:
            img = cv2.imread(fullpathOrig, cv2.IMREAD_ANYDEPTH)
            if img is None:
                print('Error reading: ' + os.path.basename(fullpathOrig))
        
        hOrig, wOrig = np.shape(img)
        
        imgNorm = functionPercNorm(img)  # Normalization fixed
        imgNorm = exposure.rescale_intensity(imgNorm, out_range=(0, 255))
        
        hImgAugmented = hOrig + patchSize
        wImgAugmented = wOrig + patchSize
        imgAugmented = np.zeros([hImgAugmented, wImgAugmented])
        imgAugmented[(spx + 1):(spx + hOrig + 1), (spx + 1):(spx + wOrig + 1)] = imgNorm

        # Saving the tiles
        numBlocksR = 0
        for r in range(0, hOrig, usefulSize):  # iteration inside the true image
            numBlocksC = 0
            numBlocksR += 1
            for c in range(0, wOrig, usefulSize):
                numBlocksC += 1

                r1 = r
                c1 = c
                r0 = r1 - spx
                c0 = c1 - spx
                r2 = r0 + patchSize - 1
                c2 = c0 + patchSize - 1
                r0_a = r0 + spx
                c0_a = c0 + spx
                r2_a = r2 + spx
                c2_a = c2 + spx

                block = imgAugmented[r0_a:(r2_a + 1), c0_a:(c2_a + 1)]

                name = (outputName + '_slice_' + str(sliceNumber).zfill(4) + '_block_' +
                        str(numBlocksR).zfill(2) + '_' + str(numBlocksC).zfill(2) + fileFormat)

                rgb = block

                cv2.imwrite(name, rgb.astype(np.uint8))
        
        infoName = outputName + '_slice_' + str(sliceNumber).zfill(4) + '_info.txt'
        fid = open(infoName, 'w')
        fid.write('sampleName:' + sample_name +
                  '\nimgNumber:' + str(sliceNumber) +
                  '\nhOrig:' + str(hOrig) +
                  ' \nwOrig:' + str(wOrig) +
                  '\nhImgAugmented:' + str(hImgAugmented) +
                  '\nwImgAugmented:' + str(wImgAugmented) +
                  '\npatchSize:' + str(patchSize) +
                  '\nnumBlocksR:' + str(numBlocksR) +
                  '\nnumBlocksC:' + str(numBlocksC) +
                  '\nspx:' + str(spx) +
                  '\nusefulSize:' + str(usefulSize))
        fid.close()
    
    return end_idx - start_idx

def functionTileVolumeNB_parallel(folder_input, folder_output, sample_name, patchSize=3000, n_workers=None):
    listing = os.listdir(folder_input)
    listing = sorted(listing)
    nFiles = len(listing)

    if n_workers is None:
        n_workers = os.cpu_count()

    # Split files evenly among workers, with remaineder going to last worker
    chunk_size = nFiles // n_workers

    args_list = []
    start_idx = 0

    for i in range(n_workers):
        if i==n_workers - 1:
            end_idx = nFiles
        else:
            end_idx = start_idx + chunk_size
        
        if start_idx < nFiles:
            args_list.append((
                folder_input,
                folder_output,
                sample_name,
                patchSize,
                start_idx,
                end_idx,
                listing
            ))
        
        start_idx = end_idx
    
    print(f'Tiling {nFiles} slices across {n_workers} workers...')

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(functionTileVolume_range, args) for args in args_list]
        
        # Progress bar tracking slices
        # REMOVE TQDM STUFF
        with tqdm(total=nFiles, desc="Processing slices", unit="slice") as pbar:
            for future in as_completed(futures):
                slices_processed = future.result()
                pbar.update(slices_processed)
