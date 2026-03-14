#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 15:11:12 2025

@author: lucas
"""
import os
import numpy as np
from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage, functionSaveTIFFMultipage

LABEL_BACK = 0
LABEL_MES = 50
LABEL_NE = 100

str_volumes_for_majority = ['_Tissues','_NE','_Mes','_artifact']
str_volumes_for_mean = ['_pHH3','_DiffGen']

# def find_files_ending_substring(root_dir, substring):
#     matches = []
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename.endswith(substring):
#                 matches.append(os.path.join(dirpath, filename))
#     return matches

def function_create_mean_volume(folder_atlas, files_for_atlas, atlas_name):
    mean_volume = None
    n_volumes = len(files_for_atlas)
    for pathVolume1 in files_for_atlas:
        
        volume_temp = functionReadTIFFMultipage(pathVolume1,8)
        
        if mean_volume is None: # First iteration
            (h, w, z) = volume_temp.shape
            mean_volume = np.double(np.zeros_like(volume_temp))
        
        mean_volume = mean_volume + np.double(volume_temp)

        del volume_temp

    mean_volume = mean_volume / n_volumes
    
    path_majority = os.path.join(folder_atlas, atlas_name)
    functionSaveTIFFMultipage(np.uint8(mean_volume), path_majority, bitdepth = 8)

def function_create_majority_volume(folder_atlas, files_for_atlas, atlas_name, labels, margin_label = 0):
    voting_volumes = {}
    #labels = [0, 1, 2, 3, LABEL_MES, LABEL_NE]

    for pathVolume1 in files_for_atlas:
        # print(len(voting_volumes))
        # print(pathVolume1)
        tissue_volume_temp = functionReadTIFFMultipage(pathVolume1,8)
        
        if len(voting_volumes) == 0: # First iteration
            (h, w, z) = tissue_volume_temp.shape
            for label in labels:
                voting_volume = np.zeros_like(tissue_volume_temp)
                voting_volumes[label] = voting_volume
        
        for label in labels:
            voting_volumes[label] = voting_volumes[label] + np.logical_and(tissue_volume_temp >= label-margin_label, tissue_volume_temp <= label+margin_label)

        del tissue_volume_temp

    # Create volumeSum with three channels
    list_volumes_votes = [voting_volumes[label] for label in labels]
    volume_sum = np.stack(list_volumes_votes, axis=-1)
    list_volumes_votes.clear()
    del list_volumes_votes
    # print(volume_sum.shape)

    argmax = np.argmax(volume_sum, axis=-1) # Convert to 1-based index

    volume_output = np.zeros((h, w, z), dtype=np.uint8)

    for i_l in range(len(labels)):
        volume_output[argmax == i_l] = labels[i_l]

    path_majority = os.path.join(folder_atlas, atlas_name)
    functionSaveTIFFMultipage(volume_output, path_majority, bitdepth = 8)
    
def create_atlases(folder_atlas, group_name, sampleNames, ending_folder, ending_SyN_moved_volumes, logger):
    
    for ending_tiff in ending_SyN_moved_volumes:
        try:
            files_for_atlas = [] # find_files_ending_substring(folder_atlas, ending_tiff)
            for sample_name in sampleNames:
                files_for_atlas.append( os.path.join(folder_atlas, sample_name + ending_folder, sample_name + ending_tiff) )
            
            match = next((s for s in str_volumes_for_majority if s in ending_tiff), None)
            if match: # The atlas has to be computed by majority
                # logger.info('---match: ' + match + ' --- ' + ending_tiff)
                
                logger.info('files for atlases: ' + str(len(files_for_atlas)))
                if match=='_Tissues':
                    function_create_majority_volume(folder_atlas, files_for_atlas, group_name + '_' + ending_tiff[1:], labels = [LABEL_BACK, LABEL_MES, LABEL_NE], margin_label = 10)
                elif match=='_NE' or match=='_Mes':
                    function_create_majority_volume(folder_atlas, files_for_atlas, group_name + '_' + ending_tiff[1:], labels = [0, 255], margin_label = 10)
                else:
                    function_create_majority_volume(folder_atlas, files_for_atlas, group_name + '_' + ending_tiff[1:], labels = [0, 1, 2, 3], margin_label = 0)
            
            match = next((s for s in str_volumes_for_mean if s in ending_tiff), None)
            if match: #It has to be computed by mean
                # logger.info('---match: ' + match + ' --- ' + ending_tiff)
                
                logger.info('files for atlases: ' + str(len(files_for_atlas)))
                function_create_mean_volume(folder_atlas, files_for_atlas, group_name + '_' + ending_tiff[1:])
        except OSError as error:
            print(error)
            logger.info(error)
            logger.info('POSSIBLE FAILED ATLAS: ' + ending_tiff)
    
    
    
    