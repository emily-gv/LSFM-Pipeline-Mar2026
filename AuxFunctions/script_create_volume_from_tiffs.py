#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:49:02 2026

@author: lucas
"""

#------------------------------ PARAMETERS ---------------------------------

folder_input = '/media/lucas/Seagate Backup Plus Drive/LIGHTSHEET/2023_Development_article/TIFFs_WT/E10.0/Dec2_E10_11/Nuclear_C' # Path to folder with tiff files. They should be grayscale images. In Windows, place an r before the ''.
dest_file_tiff = 'Dec2_E10_11_nuclear.tiff' # Fullpath to the multipage tiff you want to produce. In Windows, place an r before the ''.


bith_depth = 8 # Change for the bit depth of your tiff files, usually 16bit. 
resX = 911 # What is the pixel size? in nanometers
resY = 911 # What is the pixel size? in nanometers
resZ = 4940 # What is the spacing between slices? in nanometers

#---------------------------------------------------------------------------

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_path)
from TissueSegmentation.functionCreateVolume import functionCreateVolume

functionCreateVolume(folder_input, dest_file_tiff, resX = resX, resY = resY, resZ = resZ)