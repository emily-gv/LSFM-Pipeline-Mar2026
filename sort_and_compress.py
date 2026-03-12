from AuxFunctions.sort_files import sort_files
from AuxFunctions.compress_TIFFs import compress_TIFFs_parallel
from AuxFunctions.config_loader import load_config
import os

config_filename = "config.yml" 
config, sample_dict = load_config(config_filename)
n_samples = len(sample_dict)
markers_dict = config["markers"]

for i in range(n_samples):
    sample_name = list(sample_dict.keys())[i] 
    sort_input = sample_dict[sample_name]["folder_slices"]
    sort_files(sort_input)

    for m in markers_dict:
        if m.get("flag_compress"):
            compress_TIFFs_parallel(os.path.join(sample_dict[sample_name]["folder_slices"],f"{m}"), os.path.join(sample_dict[sample_name]["folder_slices"],f"{m}_compressed"),  n_workers=5) 
