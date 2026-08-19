import numpy as np
import cv2

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_path)

from Cellpose.functionF1Score import functionF1Score
from skimage import measure

gtFolder = "/home/emilygv/Documents/EmilyGV_Thesis/Jan9/TissueMasks"
segFolder = "/home/emilygv/Documents/EmilyGV_Thesis/Jan9/TissueSegmentations"
filesep = '/' 

listingGT = os.listdir(gtFolder)
listingGT.sort()
listingSeg = os.listdir(segFolder)
listingSeg.sort()
nFiles = len(listingGT)

vArea = np.zeros(nFiles)
vCellCount = np.uint16(np.zeros(nFiles))
vSegF1 = np.zeros(nFiles)
vSegDetection = np.zeros(nFiles)
vSegIoU = np.zeros(nFiles)
vSegArea = np.zeros(nFiles)
vSegAcc = np.zeros(nFiles)
vSegReca = np.zeros(nFiles)
vSegPrec = np.zeros(nFiles)
vSegMCC = np.zeros(nFiles)
vAreaUnion = np.uint32(np.zeros(nFiles))

with open("metrics.txt", "w") as file:
    file.write("--- METRICS ---\n\n")

#for i in range(2, nFiles):
for i in range(nFiles):
    fullPathGT = gtFolder + filesep + listingGT[i]
    fullPathSeg = segFolder + filesep + listingSeg[i]

    imgSeg = cv2.imread(fullPathSeg, cv2.IMREAD_UNCHANGED) # GREYSCALE > UNCHANGED to preserve bit depth
    print(np.shape(imgSeg))
    imgSeg = np.where(imgSeg>0, 1, 0) # anywhere where image is NOT black (pixel != 0)
    shape = np.shape(imgSeg)
    print(shape)

    imgGT = np.int16(cv2.imread(fullPathGT, cv2.IMREAD_UNCHANGED)) # why int16??
    imgGT = np.where(imgGT>0, 1, 0)
    #imgGT = cv2.resize(imgGT, [shape[0], shape[1]], cv2.INTER_NEAREST)

    # print(np.shape(imgGT))
    # print(type(imgGT[i]))
    # print(type(imgSeg[i]))
    vArea[i] = np.sum(np.where(imgGT>0, 1, 0))
    # print(vArea[i])
    vSegArea[i] = np.sum(np.where(imgSeg>0, 1, 0))
    vAreaUnion[i] = np.sum(np.uint32(np.where(np.logical_or(imgSeg>0, imgGT>0), 1, 0)))

    cells = np.where(imgGT > 0, 1, 0)
    # print(cells.shape)
    connectedcomps, vCellCount[i] = measure.label(cells, background=0, return_num = True)

    GTCells = np.where(imgGT > 0, 1, 0)
    SegCells = np.where(imgSeg > 0, 1, 0)

    # Debugging for divide by zero error
    if np.count_nonzero(GTCells) == 0:
        print(f"Skipping {listingGT[i]} - empty ground truth (mask file)")
        continue
    elif np.count_nonzero(SegCells) == 0:
        print(f"Skipping {listingGT[i]} - empty segmentation")
        continue

    f1, precison, recall, accuracy, f1beta, f1beta2, dice, IoU = functionF1Score(imgSeg, imgGT)
    # STILL NEED TO CHECK f1beta, f2beta, dice, IoU, mcc

    vSegF1[i] = f1
    vSegIoU[i] = IoU
    vSegAcc[i] = accuracy
    # vSegMCC[i] = mcc

    with open("metrics.txt", "a") as file:
        file.write(listingGT[i] + "-" + listingSeg[i] + "\n")
        file.write("GT Area (non-zero pixels): " + str(vArea[i]) + "\n")
        file.write("GT Cells: " + str(vCellCount[i]) + "\n")
        file.write("Seg Area (non-zero pixels): " + str(vSegArea[i]) + "\n")
        file.write("F1: " + str(f1) + "\n")
        file.write("Precision: " + str(precison) + "\n")
        file.write("Recall: " + str(recall) + "\n")
        file.write("Accuracy: " + str(accuracy) + "\n\n\n")