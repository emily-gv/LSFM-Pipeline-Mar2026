import os
import time 
import shutil

def sort_files(folder_input):
    """
    Sort files into folders based on channel.
    For new markers, add a key/value pair under output_folders based format denoted
    
    Args:
        folder_input (str): File path to folder containing unsorted files

    Returns:
        N/A
    """

    if not os.path.exists(folder_input):
        print("Input file does not exist")
        return

    # Create list of all files within input folder
    listing = [f for f in os.listdir(folder_input) if os.path.isfile(os.path.join(folder_input, f))]
    nFiles = len(listing)

    output_folders = {
        # <COMMON_END>.<FILETYPE> : os.path.join(folder_input, <MARKER>_C)
        "c1.tif": os.path.join(folder_input, "nuclei"),
        "c2.tif": os.path.join(folder_input, "cc3"),
        "c3.tif": os.path.join(folder_input, "phh3")
    }

    # Create output folders if they don't exist
    for f in output_folders.values():
        if not os.path.exists(f):
            os.makedirs(f)

    printcounter = 0

    for imgNumber in range(nFiles):
        dest = None
   
        # Get destination folder
        # key, value = items
        for marker, folder in output_folders.items():
            if listing[imgNumber].endswith(marker):
                dest = os.path.join(folder, listing[imgNumber])
                break

        # Move file to destination folder
        if dest is not None:
            shutil.move(os.path.join(folder_input, listing[imgNumber]), dest)

        printcounter += 1

        if printcounter % 10 == 0:
            print(f"[{imgNumber+1}/{nFiles}] moved")

# sort_files("/mnt/BHNasLightsheet/Emily_Thesis_Winter2026/Nosip_Jan152026_1_redo")

        
        