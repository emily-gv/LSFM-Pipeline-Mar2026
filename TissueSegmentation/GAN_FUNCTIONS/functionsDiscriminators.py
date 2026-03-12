#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:24:10 2024

@author: lucas
"""

# import cv2
import numpy as np
import tensorflow as tf

# def get_image(img_path):
#     img = cv2.imread(img_path, 0)
#     img = img.astype(np.float32)
#     img = np.array(img)
#     img= np.interp(img, (0, 255), (-1, +1))
#     return img

def discriminator_result(model, img_input, img_target):
    
    # img2 = [img_input[tf.newaxis, ...], img_target[tf.newaxis, ...]]
    img2 = [img_input, img_target]
    preds_test = model(img2)
    preds_test = preds_test[0]

    return preds_test


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(
        tf.ones_like(disc_real_output), disc_real_output)
    
    generated_loss = loss_object(tf.zeros_like(
        disc_generated_output), disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

def get_features_discriminator(model_discriminator, img_original, img_transformed):
    
    disc_real_output = discriminator_result(model_discriminator, img_original, img_original)
    disc_generated_output = discriminator_result(model_discriminator, img_original, img_transformed)
    
    total_disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    return np.array([np.mean(disc_generated_output), np.mean(disc_real_output), total_disc_loss.numpy()])