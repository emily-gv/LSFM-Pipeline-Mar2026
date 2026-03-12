# LSFM-Pipeline-Mar2026

## DEPENDENCIES
Linux. Try running this on Windows at your own peril.

Models are available at: https://uofc-my.sharepoint.com/:f:/g/personal/emily_garciavolk_ucalgary_ca/IgDeEwgZK4uhR6BKKOrLGUiYAZehPUhQGZ5A3WzhrBwwbqM?e=ATwkec

## INSTRUCTIONS

### SET-UP

1. Clone the repo

2. Download your files as a z-stack

    - On Zen Blue, select 'Method: Image Export' with the following parameters
        - File Type: 'Tagged Image File Format (TIFF)'
        - NO CLICK: 'Burn-in Annotations'
        - CLICK: 'Short Format'
            - NO CLICK: 'Use Channel Names'

3.  Setup your conda environment

    - In terminal, navigate to the folder 'LSFM-Pipeline-Mar2026'
    - Run:
        - ```conda env create -f lsfm_pipeline.yml``` 
    - Activate the env using: 
        - ```conda activate lsfm-pipeline```

3. Edit the config file

    - Edit the following:
        - samples: (see notes in the file)
        - folder_output: (where you want your output files stored locally)
        - folder_CNN_architectures: (where the models are stored locally)
    - Edit the other things above the dotted line if you are adding a new marker

4. Sort and compress your files

    - For all the marker images you want to compress, edit ```config.yml``` and under ```markers:``` set ```flag_compress: True```

    - In terminal, run:
        - ```python sort_and_compress.py```

    - If for whatever reason you want to run the pipeline on uncompressed images, comment out everything under ```sort_files(sort_input)``` in ```sort_and_compress.py```. Also edit each ```subfolder:``` in ```config.yml``` to remove '_compressed'

### TISSUE SEGMENTATION

Try running as is, and if it crashes severely try editting the following in main_01:
- os.environ['XLA_FLAGS'] 
- os.environ['PATH']
- os.environ['LD_LIBRARY_PATH']

1. In terminal, run:
    - ```python main_01_tissue_segmentation```


    
