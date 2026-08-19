#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from aux_functions.CC_Cells import function_encode_annotations

input_folder = r"C:\Users\Emily Garcia-Volk\Documents\EmilyGV_Thesis\CC3_Validation\CC3_Annotations"
output_folder = r"C:\Users\Emily Garcia-Volk\Documents\EmilyGV_Thesis\CC3_Validation\CC3_Masks"

function_encode_annotations(input_folder, output_folder, flag_binary_mask=False)