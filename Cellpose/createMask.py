#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_path)

from AuxFunctions.CC_Cells import function_encode_annotations

input_folder = r"C:\Users\Emily Garcia-Volk\Documents\EmilyGV_Thesis\CC3_Validation\CC3_Annotations"
output_folder = r"C:\Users\Emily Garcia-Volk\Documents\EmilyGV_Thesis\CC3_Validation\CC3_Masks"

function_encode_annotations(input_folder, output_folder, flag_binary_mask=False)