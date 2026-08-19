#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:18:31 2023

@author: lucas
"""

import cv2
from cellpose import models
from cellpose.io import imread
import matplotlib.pyplot as plt
import time
import numpy as np

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_path)

from AuxFunctions.functionPercNorm import functionPercNorm

###################################   PARAMETERS   #########################

folder_input = r"C:\Users\Emily Garcia-Volk\Documents\EmilyGV_Thesis\CC3_Validation\CC3_Greyscale"
folder_output = r"C:\Users\Emily Garcia-Volk\Documents\EmilyGV_Thesis\CC3_Validation\CC3_Segmentations"
ending_segmentation = '_cellpose.png'
path_model_trained = r"C:\Users\Emily Garcia-Volk\Documents\EmilyGV_Thesis\Models\CC3_20251103.252221"
channels = [[0,0]] #Same channels as training
diameter = 20 # Use model diameter
flag_show = False
flag_gpu = False
flag_normalize = True

##############################################################################


def functionCellposeSegmentation(folder_input, folder_output, path_model_trained,\
                                 channels = [[0,0]], diameter = None, flag_show = False, flag_gpu = False):
    
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    model_trained = models.CellposeModel(pretrained_model=path_model_trained, gpu=flag_gpu)
    
    file_images = []
    for file in os.listdir(folder_input):
        if file.endswith(".png") or file.endswith(".bmp")  or file.endswith(".tif")  or file.endswith(".jpg"):
            file_images.append(file)
    
    for file_image in file_images:
        
        start_time = time.time()

        path_image = os.path.join(folder_input, file_image)
        img = imread(path_image)
        
        img = get_one_channel(img)
        #Load image (first channel)
        if flag_normalize:
            img = functionPercNorm( np.single(img))
        
        masks, flows, styles = model_trained.eval(img, diameter=diameter, channels= channels)
        cv2.imwrite(os.path.join(folder_output, file_image + ending_segmentation), masks)
        
        print("--- %s seconds ---" % (time.time() - start_time))
    
    if flag_show:
        plt.subplot(1, 2, 1)
        plt.imshow(img,cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(masks)
        plt.show()
    
    return

def main():
    functionCellposeSegmentation(folder_input, folder_output, path_model_trained,\
                                 channels = channels, diameter = diameter, flag_show = flag_show, flag_gpu = flag_gpu)
    
    
if __name__ == "__main__":
    main()