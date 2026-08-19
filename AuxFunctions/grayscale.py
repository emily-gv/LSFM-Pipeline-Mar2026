from skimage import io, exposure, img_as_ubyte
import os
import cv2
import numpy as np
from functionPercNorm import functionPercNorm  

folderVolume = "/mnt/BHNasLightsheet/Emily_Summer2026/Retraining_Slices/ModelTrainingImages_May2026/TrainingSet_Cropped"
folderDest = "/home/emily/Desktop/Model_retraining_workspace/PNG_Good"

os.makedirs(folderDest, exist_ok=True)

# output parameters
# nbits = 8, can be 16
nChannels = 1 # can be 3
fileFormat = '.png'

#------------------------------------------------------------

listing = sorted(os.listdir(folderVolume))

for img_name in listing:
    full_path = os.path.join(folderVolume, img_name)
    print(f"Processing: {img_name}")

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_ANYDEPTH)

    # Normalization and rescaling 
    imgNorm = functionPercNorm(img)
    img_ubyte = img_as_ubyte(imgNorm) #idk why Script1 uses im_as_ubyte(rgb)

    # Save image
    name = img_name 
    save_path = os.path.join(folderDest, name) # instead of chdir
    io.imsave(save_path, img_ubyte, check_contrast=False)

