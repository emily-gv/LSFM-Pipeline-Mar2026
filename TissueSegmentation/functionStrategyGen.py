#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:14:24 2024

@author: lucas
"""

import cv2
import numpy as np
import os
import tensorflow as tf
# from tensorflow.keras.models import load_model # Commented to make a SavedModel work in a newer Tf Keras
from TissueSegmentation.GAN_FUNCTIONS.functionsDiscriminators import get_features_discriminator#, get_image
from TissueSegmentation.data_loader import get_images_from_path
import gc
from pickle import load
from scipy import ndimage
import shutil
from TissueSegmentation.functionCreateVolume import functionIsotropicVolume

LABEL_GOOD = 0
LABEL_SHADOW = 1
LABEL_BLUR = 2
LABEL_SL = 3

LABEL_MES = 50
LABEL_NE = 100

def read_image_for_model(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        print('Error reading: ' + os.path.basename(image_path))
    img = img.astype(np.float32)
    img = np.array(img)
    #img = img.reshape((1,) + img.shape)

    img = np.interp(img, (0, 255), (-1, +1))
    
    # Add batch and channel dimensions → shape: (1, H, W, 1)
    img = np.expand_dims(img, axis=0)  # batch
    img = np.expand_dims(img, axis=-1) # channel

    # Optional: convert to tf.Tensor (if you want to ensure dtype)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    
    return img

def from_model_to_img(image_output):
    preds_test = image_output
    preds_test = np.interp(preds_test, (-1, +1), (0, 255))
    return preds_test

def compute_perc_diff_images(folder_1, folder_2, output_folder_npy ,output_folder_png):
    
    list_1 = get_images_from_path(folder_1)
    list_2 = get_images_from_path(folder_2)
    
    n_list_1 = len(list_1)
    n_list_2 = len(list_2)
    
    if n_list_1 != n_list_2:
        print("DIFFERENT NUMBER OF IMAGES")
    else:
        for i in range(n_list_1):
            #print(list_1[i])
            #print('diff:')
            img1_path = list_1[i]
            # img2_path = list_2[i]
            img1_filename = os.path.basename(img1_path)
            
            img1 = cv2.imread(img1_path, 0)
            if img1 is None:
                print('Error reading: ' + os.path.basename(img1_path))
            img1 = img1.astype(np.float32)
            img1 = np.array(img1)
            #print(img1_filename)
            
            img2_path = os.path.join(folder_2, img1_filename)
            img2 = cv2.imread(img2_path, 0)
            if img2 is None:
                print('Error reading: ' + os.path.basename(img2_path))
            img2 = img2.astype(np.float32)
            img2 = np.array(img2)
            #print(os.path.basename(img2_path))
            
            diff = np.float32(img2 - img1)
            #Need to control zeros
            # print('np.min(diff):' + str(np.min(diff)))
            # print('np.max(diff):' + str(np.max(diff)))
            
            diff[img1<5] = 0 #If the value is 0, 0 diff to avoid problems with division
            
            # Need to cast to avoid large files
            diff_perc = np.divide(np.float16(diff), np.float16(img1) + np.float16(0.000001))
            
            outputName = os.path.join(output_folder_npy, img1_filename + '.npz')
            np.savez_compressed(outputName, diff_perc = np.uint8(diff_perc)) #Need to sacrifice precision to save space in disk
            
            #This is just to visualize
            # print('np.max(diff_perc):' + str(np.max(diff_perc)))
            diff_perc = np.absolute(diff_perc)
            diff_perc = (np.divide(diff_perc, np.max(diff_perc))) * 256.0 #Normalized and multiplied
            diff_perc = np.where(diff_perc > 255.0, 255, diff_perc)
            diff_perc = np.uint8(diff_perc)
            # outputName = os.path.join(output_folder_png, 'diff_' + img1_filename)
            # cv2.imwrite(outputName, diff_perc)
            
            outputName = os.path.join(output_folder_png, img1_filename)
            
            diff_perc_uniform = ndimage.uniform_filter(diff_perc, size=30)
            cv2.imwrite(outputName, diff_perc_uniform)
            
            del diff, diff_perc
            

def compute_abs_diff_images(folder_1, folder_2, output_folder):
    
    list_1 = get_images_from_path(folder_1)
    list_2 = get_images_from_path(folder_2)
    
    n_list_1 = len(list_1)
    n_list_2 = len(list_2)
    
    if n_list_1 != n_list_2:
        print("DIFFERENT NUMBER OF IMAGES")
    else:
        for i in range(n_list_1):
            #print(list_1[i])
            #print('diff:')
            img1_path = list_1[i]
            #img2_path = list_2[i]
            img1_filename = os.path.basename(img1_path)
            
            img1 = cv2.imread(img1_path, 0)
            if img1 is None:
                print('Error reading: ' + os.path.basename(img1_path))
            img1 = img1.astype(np.float32)
            img1 = np.array(img1)
            #print(img1_filename)
            
            img2_path = os.path.join(folder_2, img1_filename)
            img2 = cv2.imread(img2_path, 0)
            if img2 is None:
                print('Error reading: ' + os.path.basename(img2_path))
            img2 = img2.astype(np.float32)
            img2 = np.array(img2)
            #print(os.path.basename(img2_path))
            
            diff = np.absolute(img2 - img1)
            outputName = os.path.join(output_folder, img1_filename)
            cv2.imwrite(outputName, diff)

def filter_images(folder_orig, folder_dest):
    list_1 = get_images_from_path(folder_orig)
    n_list_1 = len(list_1)
    for i in range(n_list_1):
        img1_path = list_1[i]
        img1_filename = os.path.basename(img1_path)
        img1 = cv2.imread(img1_path, 0)
        if img1 is None:
            print('Error reading: ' + os.path.basename(img1_path))
        img1 = img1.astype(np.float32)
        img1 = np.array(img1)
        filtered_image = ndimage.uniform_filter(img1, size=30)
        outputName = os.path.join(folder_dest, img1_filename)
        cv2.imwrite(outputName, filtered_image)
        

def StrategyGenImage(img_path, folder_dest, model_generator_signal_loss, model_generator_shadow, model_generator_blur, \
                model_discriminator_signal_loss, model_discriminator_shadow, model_discriminator_blur, random_forest_classifier):
    img_name = os.path.basename(img_path)
    img_for_model_gen = read_image_for_model(img_path)
    img_for_model_disc = read_image_for_model(img_path) #get_image(img_path)
    outputName = os.path.join(folder_dest, img_name)
    
    img_gen_sl = model_generator_signal_loss(img_for_model_gen)
    img_gen_shadow = model_generator_shadow(img_for_model_gen)
    img_gen_blur = model_generator_blur(img_for_model_gen)
    
    # print('passed gens')
    
    #img_gen_sl = tf.squeeze(img_gen_sl)
    #img_gen_shadow = tf.squeeze(img_gen_shadow)
    #img_gen_blur = tf.squeeze(img_gen_blur)
    
    #img_for_model_disc = tf.squeeze(img_for_model_disc)
    
    v_disc_features_sl = get_features_discriminator(model_discriminator_signal_loss, img_for_model_disc, img_gen_sl)
    v_disc_features_shadow = get_features_discriminator(model_discriminator_shadow, img_for_model_disc, img_gen_shadow)
    v_disc_features_blur = get_features_discriminator(model_discriminator_blur, img_for_model_disc, img_gen_blur)
    
    v_disc_features = np.array([v_disc_features_blur[0], v_disc_features_sl[0], v_disc_features_shadow[0], v_disc_features_blur[1], v_disc_features_sl[1], v_disc_features_shadow[1],\
                       v_disc_features_blur[2], v_disc_features_sl[2], v_disc_features_shadow[2]])
    
    v_disc_features = v_disc_features.reshape(1, -1)
    
    prediction_img_artifact = random_forest_classifier.predict(v_disc_features)    
    
    prediction = LABEL_GOOD
    if prediction_img_artifact == LABEL_SL: #Is Signal loss to transform and save?
        img_to_save = from_model_to_img(img_gen_sl)
        img_to_save= np.squeeze(img_to_save)
        cv2.imwrite(outputName, img_to_save)
        prediction = LABEL_SL
        # print('Is SL')
    else:
        
        if prediction_img_artifact == LABEL_SHADOW: #Is Shadow?
            img_to_save = from_model_to_img(img_gen_shadow)
            img_to_save= np.squeeze(img_to_save)
            cv2.imwrite(outputName, img_to_save)
            prediction = LABEL_SHADOW
            # print('Is Shadow')
        else: #No change
            img_to_save = from_model_to_img(img_for_model_gen)
            img_to_save= np.squeeze(img_to_save)
            cv2.imwrite(outputName, img_to_save)
            prediction = LABEL_GOOD
            # print('No action')
    
    return prediction
            
def functionGenerateNuclear(folder_input, folder_dest, model_generator_signal_loss_path, model_generator_shadow_path, model_generator_blur_path,\
                     model_discriminator_signal_loss_path, model_discriminator_shadow_path, model_discriminator_blur_path, random_forest_path):
    
    model_generator_signal_loss = tf.saved_model.load(model_generator_signal_loss_path)
    model_generator_shadow = tf.saved_model.load(model_generator_shadow_path)
    model_generator_blur = tf.saved_model.load(model_generator_blur_path)

    model_discriminator_signal_loss = tf.saved_model.load(model_discriminator_signal_loss_path)
    model_discriminator_shadow = tf.saved_model.load(model_discriminator_shadow_path)
    model_discriminator_blur = tf.saved_model.load(model_discriminator_blur_path)
    
    with open(random_forest_path, 'rb') as pickle_file_rf:
        random_forest_classifier = load(pickle_file_rf)
    
    dictionary_prediction_artifact = {} #img_basename, LABEL...
    imgs = get_images_from_path(folder_input)
    for img_path in imgs:
        basename = os.path.basename(img_path)
        prediction = StrategyGenImage(img_path, folder_dest, model_generator_signal_loss, model_generator_shadow, model_generator_blur, \
                        model_discriminator_signal_loss, model_discriminator_shadow, model_discriminator_blur, random_forest_classifier)
        dictionary_prediction_artifact[basename] = prediction
    
    del model_generator_signal_loss, model_generator_shadow, model_discriminator_signal_loss, model_discriminator_shadow
    del model_generator_blur, model_discriminator_blur
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Copy the data of the tiles
    
    # Iterate over all files in the source directory
    for filename in os.listdir(folder_input):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            # Construct full file paths
            source_file = os.path.join(folder_input, filename)
            destination_file = os.path.join(folder_dest, filename)
            
            # Copy the file to the destination directory
            shutil.copy2(source_file, destination_file)
    
    return dictionary_prediction_artifact
        
def function_overlay_artifacts(folder_base_images, folder_diff_strategy, folder_diff_blur, folder_dest, th_diff = 10):
    list_1 = get_images_from_path(folder_base_images)
    n_list_1 = len(list_1)
    for i in range(n_list_1):
        img1_path = list_1[i]
        img1_filename = os.path.basename(img1_path)
        
        img_base = cv2.imread(img1_path, 0)
        if img_base is None:
            print('Error reading: ' + os.path.basename(img1_path))
        img_base = img_base.astype(np.float32)
        img_base = np.array(img_base)
        #print(img1_filename)
        
        img_diff_strategy_path = os.path.join(folder_diff_strategy, img1_filename)
        img_strategy = cv2.imread(img_diff_strategy_path, 0)
        if img_strategy is None:
            print('Error reading: ' + os.path.basename(img_diff_strategy_path))
        img_strategy = img_strategy.astype(np.float32)
        img_strategy = np.array(img_strategy)
        
        img_diff_blur_path = os.path.join(folder_diff_blur, img1_filename)
        img_blur = cv2.imread(img_diff_blur_path, 0)
        if img_blur is None:
            print('Error reading: ' + os.path.basename(img_diff_blur_path))
        img_blur = img_blur.astype(np.float32)
        img_blur = np.array(img_blur)
        
        #Only paint important differences, biggert than the th
        img_strategy[img_strategy<th_diff] = 0
        img_blur[img_blur<th_diff] = 0
        
        # Normalize images to range [0, 1]
        img_strategy /= 255.0
        img_blur /= 255.0
        img_base /= 255.0
        
        # Create an empty RGB image
        img_rgb = np.zeros((img_base.shape[0], img_base.shape[1], 3), dtype=np.float32)
        
        # Paint img_strategy (orange: red + green channels)
        img_rgb[:, :, 0] += img_strategy  # Red channel
        img_rgb[:, :, 1] += img_strategy  # Green channel
        
        # Paint img_blur (purple: red + blue channels)
        img_rgb[:, :, 0] += img_blur      # Red channel
        img_rgb[:, :, 2] += img_blur      # Blue channel
        
        # Add the base image to all channels
        img_rgb[:, :, 0] += img_base  # Red channel
        img_rgb[:, :, 1] += img_base  # Green channel
        img_rgb[:, :, 2] += img_base  # Blue channel
        
        # Normalize the result to keep values between 0 and 1
        img_rgb = np.clip(img_rgb, 0, 1)
        
        # Convert back to 8-bit format
        img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Save the result
        output_path = os.path.join(folder_dest, img1_filename + '.jpg')
        cv2.imwrite(output_path, img_rgb)
        
def create_volume_artifact_prediction(folder_tiles_original, dest_file_tiff, dictionary_prediction_artifact, fileformat = '.png', \
                                      resX = 913.89, resY = 913.89, resZ = 4940, bith_depth = 8, label = -1):
    
    list_anisotropic = []
    
    #output parameters
    nbits = 8 # can be 16
    #------------------------------------------------

    # look for all the .txt where information is stored
    filesToSearch = []
    for file in os.listdir(folder_tiles_original):
        fileExtension = os.path.splitext(file)
        if fileExtension[1] == '.txt':
            filesToSearch.append(''.join(fileExtension))
    nFiles = len(filesToSearch)
    filesToSearch = sorted(filesToSearch)
    
    # for each file in the directory
    for i in range(nFiles):
        filename = os.path.join(folder_tiles_original, filesToSearch[i])
        #print(filename)
        with open(filename) as file:
            lines = [line.rstrip() for line in file]
        file.close()
        lines = [line.split(':', 1) for line in lines]
        lines = np.array(lines).reshape(11, 2)

        #read the ouput name and imgNumber
        sampleName = lines[0][1]
        #print(sampleName)
        sliceNumber = int(lines[1][1])
        #print(sliceNumber)
        hOrig = int(lines[2][1])
        #print(hOrig)
        #print("hOrig:" + str(hOrig))
        wOrig = int(lines[3][1])
        #print("wOrig:" + str(wOrig))
        hImgAugmented = int(lines[4][1])
        wImgAugmented = int(lines[5][1])
        patchSize = int(lines[6][1])
        #print("patchSize:"+str(patchSize))
        numBlocksR = int(lines[7][1])
        #print("numBlocksR:"+str(numBlocksR))
        numBlocksC = int(lines[8][1])
        #print("numBlocksC:" + str(numBlocksC))
        spx = int(lines[9][1])
        #print("spx:" + str(spx))
        usefulSize = int(lines[10][1])
        #print("usefulSize:" + str(usefulSize))
        
        # slice_anisotropic = np.zeros((hOrig, wOrig))

        # build the first part of the string
        #firstPatPath = os.path.join(folder_input, sampleName + '_slice_' + str(sliceNumber).zfill(4))
        #print(firstPatPath)
        #print("Processing: " + str(sliceNumber).zfill(4))

        # read the original hOrig and wOrig, create an empty matirx of zeros
        if nbits == 8:
            mergedImg = np.uint8(np.zeros([hOrig, wOrig]))
        else:
            mergedImg = np.uint16(np.zeros([hOrig, wOrig]))

        # read the numBlocksR and numBlocksC
        for r in range(1, numBlocksR+1):
            #print("row:"+str(r))
           for c in range(1, numBlocksC+1):
                #print("col:" + str(c))
                # for r in Rows and c in Cols
                # reconstruct te png filename
                processedFile = sampleName + '_slice_' + str(sliceNumber).zfill(4) + '_block_' + str(r).zfill(2) + '_' + str(c).zfill(2) + fileformat
                prediction = dictionary_prediction_artifact[processedFile]
                #print(processedFile)
                # read the file
                #img = cv2.imread(processedFile, cv2.IMREAD_ANYDEPTH)
                img = np.ones((hImgAugmented, wImgAugmented)) * prediction
                #print(np.shape(img))
                hProcessed, wProcessed = np.shape(img)
                if hProcessed != patchSize:
                    img = cv2.resize(img, [patchSize, patchSize], interpolation = cv2.INTER_NEAREST)

                #img = img[(spx + 1):(len(img) - spx + 1), (spx + 1):(len(img[0]) - spx + 1)]

                img = img[spx:(patchSize - spx), spx:(patchSize - spx)]
                #print("after cropping overlapping:" + str(np.shape(img)))
                if r ==1 and c==1: # first iteration
                    if img.dtype == np.uint16:
                        mergedImg = mergedImg.astype(np.uint16)
                    else:
                        mergedImg = mergedImg.astype(np.uint8) # only one 8bits channel

                # multiply the r and c per patchSize
                rOriginal = (r-1) * usefulSize
                cOriginal = (c-1) * usefulSize

                #print("inic rOriginal:" + str(rOriginal) + " cOriginal:" + str(cOriginal) )

                endRowOriginal = rOriginal + usefulSize
                endColOriginal = cOriginal + usefulSize
                endRowPatch = usefulSize
                endColPatch = usefulSize

                #print("endRowOriginal:" + str(endRowOriginal) + " endColOriginal:" + str(endColOriginal))
                #print("endRowPatch:" + str(endRowPatch) + " endColPatch:" + str(endColPatch))

                # fill the matrix of zeros in the correct position with the tile
                if endRowOriginal > hOrig:
                    endRowPatch = endRowPatch - (endRowOriginal - hOrig)
                    endRowOriginal = hOrig
                    #print("endRowOriginal:" + str(endRowOriginal) + " endColOriginal:" + str(endColOriginal))

                if endColOriginal > wOrig:
                    endColPatch = endColPatch - (endColOriginal - wOrig)
                    endColOriginal = wOrig

                cropPatch = img[0:(endRowPatch), 0:(endColPatch)]
                #print("cropPatch shape: " + str(np.shape(cropPatch)))
                mergedImg[rOriginal : (endRowOriginal), cOriginal:(endColOriginal)] = cropPatch

        #save the completed slice with the output name and imgNumber .png
        #name = os.path.join(folder_output,sampleName + '_slice_' + str(sliceNumber).zfill(4) + fileformat)
        #cv2.imwrite(name, mergedImg)

        list_anisotropic.append(mergedImg)
    
    functionIsotropicVolume(list_anisotropic, dest_file_tiff, resX = resX, resY = resY, resZ = resZ, bith_depth = bith_depth, label = label)
        
def generateMesenchymeEdge(folder_tissues_slices, folder_edge_mesen):
    
    if not os.path.exists(folder_edge_mesen):     os.makedirs(folder_edge_mesen)
    
    # Iterate through all PNG files in the input folder
    for filename in os.listdir(folder_tissues_slices):
        if filename.endswith(".png"):
            # Read the image
            filepath = os.path.join(folder_tissues_slices, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Failed to read {filename}. Skipping.")
                continue

            # Create the mask for the mesenchyme
            mask_mesen = (image == LABEL_MES).astype(np.uint8) * 255

            # Define a 200x200 kernel for erosion
            kernel = np.ones((200, 200), np.uint8)

            # Erode the mask, to get the interior of the mesenchyme
            eroded_mask = cv2.erode(mask_mesen, kernel, iterations=1)

            # Edge of mesenchyme
            xor_mask = cv2.bitwise_xor(mask_mesen, eroded_mask)
            
            #Mask of the NE, as it is in the interior, it is prone to segmentation errors too:
            mask_ne = (image == LABEL_NE).astype(np.uint8) * 255
            dilated_ne = cv2.dilate(mask_ne, kernel, iterations=1)
            
            xor_mask[dilated_ne>0] = False #Remove segmentation close to the NE
            
            # Save the resulting mask to the output directory
            output_filepath = os.path.join(folder_edge_mesen, filename)
            cv2.imwrite(output_filepath, xor_mask)

            # print(f"Processed and saved: {filename}")

    print("Processing complete.")

# Example usage
# process_images("path_to_input_folder", "path_to_output_folder")
        
        
        
