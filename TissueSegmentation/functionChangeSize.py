import os
import cv2
import numpy as np

def functionChangeSize(folder_input, folder_output, dest_size = 192, ending = '.png'):

    listing = os.listdir(folder_input)
    listing = sorted(listing)
    nFiles = len(listing)

    #print("Number of tiles: " + str(nFiles))
    for i in range(nFiles):
        if listing[i].endswith(ending):
            #print(listing[i])
            fullPathOrig = os.path.join(folder_input,listing[i])#os.path.abspath(listing[i])
            #print(fullPathOrig)
            imgRGB = cv2.imread(fullPathOrig, cv2.IMREAD_ANYDEPTH)
            if imgRGB is None:
                print('Error reading: ' + os.path.basename(fullPathOrig))
            sizeRGB = np.shape(imgRGB)
            #print(sizeRGB)
            if (len(sizeRGB) >= 3):
                img = imgRGB[:,:,1]
            else:
                img = imgRGB

            img = cv2.resize(img, (dest_size, dest_size), interpolation=cv2.INTER_NEAREST)
            fullPathDest = os.path.join(folder_output,listing[i])
            cv2.imwrite(fullPathDest, img)