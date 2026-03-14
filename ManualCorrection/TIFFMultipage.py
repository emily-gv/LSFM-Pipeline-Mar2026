import cv2
import numpy as np
from PIL import Image

def functionReadTIFFMultipage(dirImage, bitdepth):
    img = Image.open(dirImage)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))

    del img

    height, width = np.shape(images[0])
    numImgs = len(images)
    #print(height, width, numImgs)

    if bitdepth == 8:
        volume = np.uint8(np.zeros((height, width, numImgs)))
    else:
        volume = np.uint16(np.zeros((height, width, numImgs)))

    for i in range(numImgs):
        sliceSingle = images[i]
        volume[:, :, i] = sliceSingle
    
    return volume

def functionSaveTIFFMultipage(volume, fileNameOutput, bitdepth):
    h, w, d = np.shape(volume)

    if bitdepth == 8:
        volume = np.uint8(volume)
    else:
        volume = np.uint16(volume)

    imlist = []
    #for m in volume:
    for i in range(d):
        #imlist.append(Image.fromarray(m))
        m = volume[:,:,i]
        imlist.append(Image.fromarray(m))

    imlist[0].save(fileNameOutput, compression="tiff_lzw", save_all=True,
                   append_images=imlist[1:])
    
    
def functionSaveTIFFMultipage_w(volume, fileNameOutput, bitdepth):
    h, w, d = np.shape(volume)

    if bitdepth == 8:
        volume = np.uint8(volume)
    else:
        volume = np.uint16(volume)

    imlist = []
    #for m in volume:
    for i in range(w):
        #imlist.append(Image.fromarray(m))
        m = volume[:,i,:]
        imlist.append(Image.fromarray(m))

    imlist[0].save(fileNameOutput, compression="tiff_lzw", save_all=True,
                   append_images=imlist[1:])
    
def functionSaveTIFFMultipage_h(volume, fileNameOutput, bitdepth):
    h, w, d = np.shape(volume)

    if bitdepth == 8:
        volume = np.uint8(volume)
    else:
        volume = np.uint16(volume)

    imlist = []
    #for m in volume:
    for i in range(h):
        #imlist.append(Image.fromarray(m))
        m = volume[i,:,:]
        imlist.append(Image.fromarray(m))

    imlist[0].save(fileNameOutput, compression="tiff_lzw", save_all=True,
                   append_images=imlist[1:])