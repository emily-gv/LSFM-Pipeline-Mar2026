import numpy as np
import os
import cv2

from concurrent.futures import ThreadPoolExecutor

def parse_tile_metadata(filepath):
    with open(filepath) as file:
        values = [line.strip().split(":", 1)[1] for line in file.readlines()]
    return {
        'sampleName': values[0],
        'sliceNumber': int(values[1]),
        'hOrig': int(values[2]),
        'wOrig': int(values[3]),
        'hImgAugmented': int(values[4]),
        'wImgAugmented': int(values[5]),
        'patchSize': int(values[6]),
        'numBlocksR': int(values[7]),
        'numBlocksC': int(values[8]),
        'spx': int(values[9]),
        'usefulSize': int(values[10])
    }

def load_tile_image(file_path, patchSize, spx, dtype):
    img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not read {file_path}")

    if img.shape[0] != patchSize:
        img = cv2.resize(img, (patchSize, patchSize), interpolation=cv2.INTER_NEAREST)

    img = img[spx:(patchSize - spx), spx:(patchSize - spx)]
    return img.astype(dtype)

def functionMergeProcessedTilesNB(folder_input, folder_output, folder_tiles_original, fileformat='.png'):
    nbits = 8  # or 16
    dtype = np.uint8 if nbits == 8 else np.uint16

    txt_files = sorted(f for f in os.listdir(folder_tiles_original) if f.endswith('.txt'))

    for txt_file in txt_files:
        metadata = parse_tile_metadata(os.path.join(folder_tiles_original, txt_file))
        sampleName = metadata['sampleName']
        sliceNumber = metadata['sliceNumber']
        hOrig = metadata['hOrig']
        wOrig = metadata['wOrig']
        patchSize = metadata['patchSize']
        numBlocksR = metadata['numBlocksR']
        numBlocksC = metadata['numBlocksC']
        spx = metadata['spx']
        usefulSize = metadata['usefulSize']

        mergedImg = np.zeros((hOrig, wOrig), dtype=dtype)
        base_name = os.path.join(folder_input, f"{sampleName}_slice_{sliceNumber:04d}")

        def get_tile_info(r, c):
            file_path = f"{base_name}_block_{r:02d}_{c:02d}{fileformat}"
            rStart = (r - 1) * usefulSize
            cStart = (c - 1) * usefulSize
            return r, c, file_path, rStart, cStart

        tile_tasks = [get_tile_info(r, c) for r in range(1, numBlocksR + 1) for c in range(1, numBlocksC + 1)]

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(load_tile_image, file_path, patchSize, spx, dtype): (r, c, rStart, cStart)
                for (r, c, file_path, rStart, cStart) in tile_tasks
            }

            for future in futures:
                r, c, rStart, cStart = futures[future]
                img = future.result()
                endR = min(rStart + usefulSize, hOrig)
                endC = min(cStart + usefulSize, wOrig)
                patchEndR = endR - rStart
                patchEndC = endC - cStart
                mergedImg[rStart:endR, cStart:endC] = img[:patchEndR, :patchEndC]

        output_path = os.path.join(folder_output, f"{sampleName}_slice_{sliceNumber:04d}{fileformat}")
        cv2.imwrite(output_path, mergedImg)


# def functionMergeProcessedTilesNB(folder_input, folder_output, folder_tiles_original, fileformat = '.png'):
#     #output parameters
#     nbits = 8 # can be 16
#     #------------------------------------------------

#     # look for all the .txt where information is stored
#     filesToSearch = []
#     for file in os.listdir(folder_tiles_original):
#         fileExtension = os.path.splitext(file)
#         if fileExtension[1] == '.txt':
#             filesToSearch.append(''.join(fileExtension))
#     nFiles = len(filesToSearch)
#     filesToSearch = sorted(filesToSearch)

#     # for each file in the directory
#     for i in range(nFiles):
#         filename = os.path.join(folder_tiles_original, filesToSearch[i])
#         #print(filename)
#         with open(filename) as file:
#             lines = [line.rstrip() for line in file]
#         file.close()
#         lines = [line.split(':', 1) for line in lines]
#         lines = np.array(lines).reshape(11, 2)

#         #read the ouput name and imgNumber
#         sampleName = lines[0][1]
#         #print(sampleName)
#         sliceNumber = int(lines[1][1])
#         #print(sliceNumber)
#         hOrig = int(lines[2][1])
#         #print(hOrig)
#         #print("hOrig:" + str(hOrig))
#         wOrig = int(lines[3][1])
#         #print("wOrig:" + str(wOrig))
#         hImgAugmented = int(lines[4][1])
#         wImgAugmented = int(lines[5][1])
#         patchSize = int(lines[6][1])
#         #print("patchSize:"+str(patchSize))
#         numBlocksR = int(lines[7][1])
#         #print("numBlocksR:"+str(numBlocksR))
#         numBlocksC = int(lines[8][1])
#         #print("numBlocksC:" + str(numBlocksC))
#         spx = int(lines[9][1])
#         #print("spx:" + str(spx))
#         usefulSize = int(lines[10][1])
#         #print("usefulSize:" + str(usefulSize))

#         # build the first part of the string
#         firstPatPath = os.path.join(folder_input, sampleName + '_slice_' + str(sliceNumber).zfill(4))
#         #print(firstPatPath)
#         #print("Processing: " + str(sliceNumber).zfill(4))

#         # read the original hOrig and wOrig, create an empty matirx of zeros
#         if nbits == 8:
#             mergedImg = np.uint8(np.zeros([hOrig, wOrig]))
#         else:
#             mergedImg = np.uint16(np.zeros([hOrig, wOrig]))

#         # read the numBlocksR and numBlocksC
#         for r in range(1, numBlocksR+1):
#             #print("row:"+str(r))
#            for c in range(1, numBlocksC+1):
#                 #print("col:" + str(c))
#                 # for r in Rows and c in Cols
#                 # reconstruct te png filename
#                 processedFile = firstPatPath + '_block_' + str(r).zfill(2) + '_' + str(c).zfill(2) + fileformat
#                 #print(processedFile)
#                 # read the file
#                 img = cv2.imread(processedFile, cv2.IMREAD_ANYDEPTH)
#                 if img is None:
#                     print('Error reading: ' + processedFile)
#                 #print(np.shape(img))
#                 hProcessed, wProcessed = np.shape(img)
#                 if hProcessed != patchSize:
#                     img = cv2.resize(img, [patchSize, patchSize], interpolation = cv2.INTER_NEAREST)

#                 #img = img[(spx + 1):(len(img) - spx + 1), (spx + 1):(len(img[0]) - spx + 1)]

#                 img = img[spx:(patchSize - spx), spx:(patchSize - spx)]
#                 #print("after cropping overlapping:" + str(np.shape(img)))
#                 if r ==1 and c==1: # first iteration
#                     if img.dtype == np.uint16:
#                         mergedImg = mergedImg.astype(np.uint16)
#                     else:
#                         mergedImg = mergedImg.astype(np.uint8) # only one 8bits channel

#                 # multiply the r and c per patchSize
#                 rOriginal = (r-1) * usefulSize
#                 cOriginal = (c-1) * usefulSize

#                 #print("inic rOriginal:" + str(rOriginal) + " cOriginal:" + str(cOriginal) )

#                 endRowOriginal = rOriginal + usefulSize
#                 endColOriginal = cOriginal + usefulSize
#                 endRowPatch = usefulSize
#                 endColPatch = usefulSize

#                 #print("endRowOriginal:" + str(endRowOriginal) + " endColOriginal:" + str(endColOriginal))
#                 #print("endRowPatch:" + str(endRowPatch) + " endColPatch:" + str(endColPatch))

#                 # fill the matrix of zeros in the correct position with the tile
#                 if endRowOriginal > hOrig:
#                     endRowPatch = endRowPatch - (endRowOriginal - hOrig)
#                     endRowOriginal = hOrig
#                     #print("endRowOriginal:" + str(endRowOriginal) + " endColOriginal:" + str(endColOriginal))

#                 if endColOriginal > wOrig:
#                     endColPatch = endColPatch - (endColOriginal - wOrig)
#                     endColOriginal = wOrig

#                 cropPatch = img[0:(endRowPatch), 0:(endColPatch)]
#                 #print("cropPatch shape: " + str(np.shape(cropPatch)))
#                 mergedImg[rOriginal : (endRowOriginal), cOriginal:(endColOriginal)] = cropPatch

#         #save the completed slice with the output name and imgNumber .png
#         name = os.path.join(folder_output,sampleName + '_slice_' + str(sliceNumber).zfill(4) + fileformat)
#         cv2.imwrite(name, mergedImg)
    
    