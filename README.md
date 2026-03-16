# LSFM-Pipeline-Mar2026

## DEPENDENCIES

Linux. Try running this on Windows at your own peril.

Models are available at: https://uofc-my.sharepoint.com/:f:/g/personal/emily_garciavolk_ucalgary_ca/IgDeEwgZK4uhR6BKKOrLGUiYAZehPUhQGZ5A3WzhrBwwbqM?e=ATwkec

Subset of volumes to try the registration process: https://uofc-my.sharepoint.com/:f:/r/personal/lucasdaniel_lovercio_ucalgary_ca/Documents/Emily/Volumes_to_make_atlas?csf=1&web=1&e=PGjvZM

Link to subset of images: https://uofc-my.sharepoint.com/:f:/g/personal/emily_garciavolk_ucalgary_ca/IgAH3rSB2olwSoDI3khwia1sATwWJqeZyWEmWHZJvpwOCSo?e=GjDBsL

## INSTRUCTIONS

### SET-UP

1. Clone the repo

2. Download your files as a z-stack
   - On Zen Blue, select 'Method: Image Export' with the following parameters
     - File Type: 'Tagged Image File Format (TIFF)'
     - NO CLICK: 'Burn-in Annotations'
     - CLICK: 'Short Format'
       - NO CLICK: 'Use Channel Names'

3. Setup your conda environment (this is a Linux-format environment with Linux-specific packages)
   - In terminal, navigate to the folder 'LSFM-Pipeline-Mar2026'
   - Run:
     - `conda env create -f lsfm_pipeline.yml`
   - Activate the env using:
     - `conda activate lsfm-pipeline`

4. Edit the config file
   - Edit the following:
     - samples: (see notes in the file)
     - folder_output: (where you want your output files stored locally)
     - folder_CNN_architectures: (where the models are stored locally)
   - Edit the other things above the dotted line if you are adding a new marker

5. Sort and compress your files
   - For all the marker images you want to compress, edit `config.yml` and under `markers:` set `flag_compress: True`

   - In terminal, run:
     - `python sort_and_compress.py`

   - If for whatever reason you want to run the pipeline on uncompressed images, comment out everything under `sort_files(sort_input)` in `sort_and_compress.py`. Also edit each `subfolder:` in `config.yml` to remove '\_compressed'

### TISSUE SEGMENTATION

Try running as is, and if it crashes severely try editing the following in main_01:

- os.environ['XLA_FLAGS']
- os.environ['PATH']
- os.environ['LD_LIBRARY_PATH']

1. In terminal, run:
   - `python main_01_tissue_segmentation.py`

### CELL SEGMENTATION

1. For each marker you want to segment, toggle `flag_segment:true` in `config.yml`
   - If you want to track marker cells in the edge of the mesenchyme, toggle `flag_mesenchyme`

2. In terminal, run:
   - `python main_02_tissue_segmentation.py`

### MANUAL CORRECTIONS

1. Using 3DSlicer, create a mask of just the sample (ie. removing sedimentation)
   - Save it as `<original-tissue-TIFF-name>_Mask.tiff`
   - eg. Aug28_2025_27_Step06a_Tissues_Mask.tiff

2. For each marker you want to remove sedimentation for, toggle `flag_remove_sedimentation:true` in `config.yml` (must have previously segmented cell marker)

3. In terminal, run:
   - `python main_03_remove_sedimentation.py`

4. Additionally, using 3DSlicer make a mask of the neural ectoderm (remove misc mesenchyme in the volume)
   - Save it as `<sample>_NE_corrected`
   - eg. Aug28_2025_27_NE_corrected.tiff

5. In terminal, run:
   - `python main_04_move_ne_to_mesenchyme.py`

### SOFTWARE REQUIRED

- Python. To create a Virtual Environment using lsfm_pipeline.yml, you can follow the example: https://github.com/lucaslovercio/ACHRI_Workshop_Cellpose/tree/main/environments (avoid Step 2)
- 3DSlicer
- Recommended: Paraview
