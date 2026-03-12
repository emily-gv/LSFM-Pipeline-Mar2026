#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:21:08 2024

@author: lucas
"""
import os
import cv2
import numpy as np
import gc
import tensorflow as tf
from TissueSegmentation.functionStrategyGen import read_image_for_model, from_model_to_img
#from TissueSegmentation.GAN_FUNCTIONS.functionsDiscriminators import get_image
# from tensorflow.keras.models import load_model # commented to make SavedModel work in newer TF Keras
from TissueSegmentation.data_loader import get_images_from_path

def functionGenBlurImage(img_path, folder_dest, model_generator_blur):
    img_name = os.path.basename(img_path)
    img_for_model_gen = read_image_for_model(img_path)
    outputName = os.path.join(folder_dest, img_name)
    
    img_gen = model_generator_blur(img_for_model_gen)
    
    img_to_save = from_model_to_img(img_gen)
    img_to_save= np.squeeze(img_to_save)
    cv2.imwrite(outputName, img_to_save)

def functionGenBlur(folder_CSGreen_tiles, folder_CSGreen_tiles_d, path_model_generator_blur):
    model_generator_blur = tf.saved_model.load(path_model_generator_blur)
    
    imgs = get_images_from_path(folder_CSGreen_tiles)
    for img_path in imgs:
        functionGenBlurImage(img_path, folder_CSGreen_tiles_d, model_generator_blur)
    
    del model_generator_blur
    gc.collect()
    tf.keras.backend.clear_session()