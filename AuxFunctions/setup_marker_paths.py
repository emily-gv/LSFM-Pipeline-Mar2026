import os
import sys

def setup_marker_paths_main_02(marker: str, config, sample_dict, sample_name):
    """
    Setup paths necessary for main02 for a specified marker and sample
    
    Args:
        marker (str): Marker name. Case sensitive, must be consistent across all instances in config file
        config (dict): Parsed YAML config
        sample_dict (dict): Metadata per sample name
        sample_name (str): Sample name

    Returns:
        paths (dict): All required paths for a specified marker
    """
    
    paths = {}

    folder_architectures = config["folder_CNN_architectures"]
    paths["path_cnn"] = os.path.join(folder_architectures,config[f"filename_CNN_{marker}"])

    # Load folder with marker TIFFs
    folder_input_slices = os.path.join(sample_dict[sample_name]["folder_slices"],sample_dict[sample_name][f"subfolder_{marker}"])
    if not os.path.exists(folder_input_slices):
        print(f"Directory does not exist: {folder_input_slices}")
        sys.exit(1)
    paths["folder_input_slices"] = folder_input_slices

    # Base output folder
    # I load this in main so this could just be passed in as a param
    folder_sample_output = os.path.join(config["folder_output"],sample_dict[sample_name]["group"],sample_dict[sample_name]["age"],sample_name)

    # Tiling output folder
    folder_tiles= os.path.join(folder_sample_output, config[f"folder_{marker}_tiles"])
    if not os.path.exists(folder_tiles): os.makedirs(folder_tiles)
    paths["folder_tiles"] = folder_tiles

    # Segmentation output folder (tiles)
    paths["folder_tiles_segmented_label"] = os.path.join(folder_sample_output, config[f"folder_{marker}_segmented_tiles_label"]) # folder_phh3_segmented_label_debug
    paths["folder_tiles_segmented_binary"] = os.path.join(folder_sample_output, config[f"folder_{marker}_segmented_tiles_binary"])

    # Reconstructed slices output folder
    paths["folder_slices_label"] = os.path.join(folder_sample_output, config[f"folder_{marker}_slices_label"])
    paths["folder_slices_binary"] = os.path.join(folder_sample_output, config[f"folder_{marker}_slices_binary"])

    # Masks output folder
    paths["folder_masked_label"] = os.path.join(folder_sample_output, config[f"folder_{marker}_masked_label"])
    paths["folder_masked_binary"] = os.path.join(folder_sample_output, config[f"folder_{marker}_masked_binary"])

    # TIFF outputs 
    # Script currently only saves the TIFF of the first slice per sample
    paths["dest_file_label_tiff"] = os.path.join(folder_sample_output, sample_name + config[f"dest_file_{marker}_label_tiff"])
    paths["dest_file_binary_tiff"] = os.path.join(folder_sample_output, sample_name + config[f"dest_file_{marker}_binary_tiff"])

    # Mesenchyme edge masks output folder
    paths["folder_edge_mesen"] = os.path.join(folder_sample_output,config["folder_edge_mesen"])
    paths["folder_mesen_masked_label"] = os.path.join(paths["folder_edge_mesen"], f"{marker}_masked_edge_label")
    paths["folder_mesen_masked_binary"] = os.path.join(paths["folder_edge_mesen"], f"{marker}_masked_edge_binary")

    return paths


def setup_marker_paths_main_03(marker: str, config, folder_sample, sample_name):
    """
    Setup paths necessary for main03 for a specified marker and sample
    
    Args:
        marker (str): Marker name. Case sensitive, must be consistent across all instances in config file
        config (dict): Parsed YAML config
        sample_dict (dict): Metadata per sample name
        sample_name (str): Sample name

    Returns:
        paths (dict): All required paths for a specified marker
    """
    
    paths = {}

    paths["origin_file_label_tiff"] = os.path.join(folder_sample,sample_name + config[f"dest_file_{marker}_label_tiff"])
    paths["origin_file_binary_tiff"]= os.path.join(folder_sample,sample_name + config[f"dest_file_{marker}_binary_tiff"])
    paths["dest_file_label_tiff"] = os.path.join(folder_sample,sample_name + config[f"dest_file_{marker}_label_tiff_maskout_sedimentation"])
    paths["dest_file_binary_tiff"] = os.path.join(folder_sample,sample_name + config[f"dest_file_{marker}_binary_tiff_maskout_sedimentation"])
    return paths

def setup_marker_paths_main_05(marker: str, config, folder_sample, sample_name):

    paths = {}

    paths["pre_ending_binary"] = os.path.join(folder_sample,sample_name + config[f"dest_file_{marker}_binary_tiff_maskout_sedimentation"])
    paths["ending_step_tissues"] = os.path.join(folder_sample,sample_name + config["dest_file_tiff_tissue_segmentation_ne_correction"])
    paths["ending_density"] = os.path.join(folder_sample,sample_name + config[f"dest_file_{marker}_density"])
    paths["ending_density_histnorm"] = os.path.join(folder_sample,sample_name + config[f"dest_file_{marker}_density_histnorm"])
    paths["ending_density_isotropic"] = os.path.join(folder_sample,sample_name + config[f"dest_file_{marker}_density_isotropic"])

    return paths