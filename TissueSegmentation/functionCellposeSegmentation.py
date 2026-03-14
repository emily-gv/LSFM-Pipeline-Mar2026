import cv2
from cellpose import models
from cellpose.io import imread
from TissueSegmentation.data_loader import get_image_array, get_pairs_from_paths, get_images_from_path
import os
import logging
import numpy as np
import torch
import time



def functionCellposeSegmentation(folder_input, folder_output, model_trained):

    imgs = get_images_from_path(folder_input)
    a = models.CellposeModel(pretrained_model=model_trained, gpu=True)
    for img in imgs:

        img_name = os.path.basename(img)  # this gets the image name out of the path
        #print(img)
        #print(img_name)
        img = imread(img)
        if img is None:
            print('Error reading: ' + img_name)

        masks, flows, styles = a.eval(img, diameter=None, channels= [[0,0]])
        binary = masks > 0
        binary = binary * 255.0
        cv2.imwrite(os.path.join(folder_output, img_name), binary)
    return

def functionCellposeSegmentation_label(folder_input, folder_output_labels, folder_output_binary, model_trained, flag_gpu=True):
    imgs = get_images_from_path(folder_input)
    a = models.CellposeModel(pretrained_model=model_trained, gpu=True)
    for img in imgs:

        img_name = os.path.basename(img)  # this gets the image name out of the path
        #print(img)
        #print(img_name)
        img = imread(img)
        if img is None:
            print('Error reading: ' + img_name)

        masks, flows, styles = a.eval(img, diameter=None, channels= [[0,0]])
        cv2.imwrite(os.path.join(folder_output_labels, img_name), masks)
        binary = masks > 0
        binary = binary * 255.0
        cv2.imwrite(os.path.join(folder_output_binary, img_name), binary)
    return

def functionTileSliceOptimizedNB(img, patchSize=131):
    spx = int(patchSize * 0.2)
    usefulSize = patchSize - (2 * spx)
    hOrig, wOrig = img.shape

    # Create an augmented image with padding
    hAugmented = hOrig + 2 * spx
    wAugmented = wOrig + 2 * spx
    imgAugmented = np.zeros((hAugmented, wAugmented), dtype=img.dtype)

    # Copy the original image into the center of the augmented image
    imgAugmented[spx:spx + hOrig, spx:spx + wOrig] = img

    # Calculate the number of blocks
    numBlocksR = (hOrig + usefulSize - 1) // usefulSize  # Ceiling division
    numBlocksC = (wOrig + usefulSize - 1) // usefulSize  # Ceiling division

    # Initialize tiles as a NumPy array for faster appends
    tiles = []

    # Extract overlapping patches
    for r in range(numBlocksR):
        for c in range(numBlocksC):
            r0 = r * usefulSize
            c0 = c * usefulSize

            # Extract patch with overlap
            block = imgAugmented[
                r0:r0 + patchSize,
                c0:c0 + patchSize
            ]

            tiles.append(block)

    return tiles, hOrig, wOrig, numBlocksR, numBlocksC, patchSize, spx, usefulSize

def functionTileSliceNB(img, patchSize=131):
    spx = int(patchSize * 0.2)
    usefulSize = patchSize - 2 * spx
    hOrig, wOrig = img.shape

    pad_width = ((spx, spx), (spx, spx))
    img_aug = np.pad(img, pad_width, mode='constant', constant_values=0)

    tiles = []
    numBlocksR = 0
    numBlocksC = 0
    for r in range(0, hOrig, usefulSize):
        numBlocksR += 1
        numBlocksC = 0
        for c in range(0, wOrig, usefulSize):
            numBlocksC += 1
            r0, r1 = r, r + patchSize
            c0, c1 = c, c + patchSize
            block = img_aug[r0:r1, c0:c1]
            if block.shape != (patchSize, patchSize):
                pad_r = patchSize - block.shape[0]
                pad_c = patchSize - block.shape[1]
                block = np.pad(block, ((0, pad_r), (0, pad_c)), mode='constant')
            tiles.append(block)

    return tiles, hOrig, wOrig, numBlocksR, numBlocksC, patchSize, spx, usefulSize


def functionMergeProcessedTilesNB(tiles, hOrig, wOrig, numBlocksR, numBlocksC, patchSize, spx, usefulSize):
    dtype = tiles[0].dtype
    mergedImg = np.zeros((hOrig, wOrig), dtype=dtype)

    tile_idx = 0
    for r in range(numBlocksR):
        for c in range(numBlocksC):
            img = tiles[tile_idx]
            tile_idx += 1

            crop = img[spx:spx+usefulSize, spx:spx+usefulSize]

            rStart = r * usefulSize
            cStart = c * usefulSize

            rEnd = min(rStart + usefulSize, hOrig)
            cEnd = min(cStart + usefulSize, wOrig)

            patchR = rEnd - rStart
            patchC = cEnd - cStart

            mergedImg[rStart:rEnd, cStart:cEnd] = crop[:patchR, :patchC]

    return mergedImg

def functionCellposeSegmentation_tiling(folder_input, folder_output_labels, folder_output_binary, model_trained, flag_gpu=True, tile_window=131):
    imgs = get_images_from_path(folder_input)
    model = models.CellposeModel(pretrained_model=model_trained, gpu=flag_gpu)

    for img_path in imgs:
        img_name = os.path.basename(img_path)
        img = imread(img_path)
        if img is None:
            print(f'Error reading: {img_name}')
            continue

        tiles, hOrig, wOrig, numBlocksR, numBlocksC, patchSize, spx, usefulSize = functionTileSliceNB(img, patchSize=tile_window)
        masks, _, _ = model.eval(tiles, diameter=None, channels=[[0, 0]])
        merged = functionMergeProcessedTilesNB(masks, hOrig, wOrig, numBlocksR, numBlocksC, patchSize, spx, usefulSize)
        cv2.imwrite(os.path.join(folder_output_labels, img_name), merged)

        binary = (merged > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(folder_output_binary, img_name), binary)

    del model
    torch.cuda.empty_cache()
    








