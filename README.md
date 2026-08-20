# LSFM-Pipeline-Mar2026

Models are available at: https://uofc-my.sharepoint.com/:f:/g/personal/emily_garciavolk_ucalgary_ca/IgDeEwgZK4uhR6BKKOrLGUiYAZ7RzuBoWA8PQdZ1mWnbg3Y?e=Ilkdh6

Subset of volumes to try the registration process: [https://uofc-my.sharepoint.com/:f:/r/personal/lucasdaniel_lovercio_ucalgary_ca/Documents/Emily/Volumes_to_make_atlas?csf=1&web=1&e=PGjvZM](https://uofc-my.sharepoint.com/:f:/g/personal/lucasdaniel_lovercio_ucalgary_ca/IgCo5T0yGBPvR697UtUkcYQ7AcRoi0B_61iSLJPOumIs9Q4?e=pDqv07)

Link to subset of images: https://uofc-my.sharepoint.com/:f:/g/personal/emily_garciavolk_ucalgary_ca/IgAH3rSB2olwSoDI3khwia1sATwWJqeZyWEmWHZJvpwOCSo?e=GjDBsL


### SET-UP

1. Download your files as a z-stack
   - On Zen Blue, select 'Method: Image Export' with the following parameters
     - File Type: 'Tagged Image File Format (TIFF)'
     - NO CLICK: 'Burn-in Annotations'
     - CLICK: 'Short Format'
       - NO CLICK: 'Use Channel Names'

2. Setup your conda environment (this is a Linux-format environment with Linux-specific packages)
   - Follow the install instructions found at https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
   - In terminal, navigate to the folder 'LSFM-Pipeline-Mar2026'
   - Run:
     - `conda env create -f lsfm_pipeline.yml`
   - Activate the env using:
     - `conda activate lsfm-pipeline`
   - With the env activated, run:
      - `conda install -c nvidia cuda-toolkit=12.2`
      - `conda install -c conda-forge libxcb`

3. Edit the config file
   - Edit the following:
     - samples: (see notes in the file)
     - folder_output: (where you want your output files stored locally)
     - folder_CNN_architectures: (where the models are stored locally)
   - Edit the other things above the dotted line if you are adding a new marker

4. Sort and compress your files
   - If you're adding a new marker:
      - In `sort_files.py`, below `"c3.tif": os.path.join(folder_input, "phh3")` add another
         - Format should be `c4.tif` (or whatever number you're on) and replace `"phh3"` with your marker name the SAME as in the config

   - For all the marker images you want to compress, edit `config.yml` and under `markers:` set `flag_compress: True`

   - In terminal, run:
     - `python sort_and_compress.py`


### TISSUE + CELL SEGMENTATION

Once everything is set up, you should be able to run main_01 & main_02. For a full overview of the pipeline, check out the [PDF guide](./LSFM-Pipeline-Guide.pdf)


### SOFTWARE REQUIREMENTS

Linux. Try running this on anything else at your own peril.

- Python. To create a Virtual Environment using lsfm_pipeline.yml, you can follow the example: https://github.com/lucaslovercio/ACHRI_Workshop_Cellpose/tree/main/environments (avoid Step 2)
- 3DSlicer
- Recommended: Paraview
   - If you're using newer versions of Ubuntu > 24.04, you can get Paraview to work by opening it through terminal running the following commands:
      `export HWLOC_COMPONENTS=-gl`
      `paraview`
