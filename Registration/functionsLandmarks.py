import os
import numpy as np
import vtk
import SimpleITK as sitk
import csv
from ManualCorrection.TIFFMultipage import functionReadTIFFMultipage
from glob import glob
import pandas as pd

def getLMsOutputpointstxt(fileOutputpointstxt, constDivide = 0.35277777777777775):
    text_file = open(fileOutputpointstxt, "r")
    # read whole file to a string
    data = text_file.read()
    # close file
    text_file.close()
    vector1 = data.split("OutputPoint = [ ")
    LMs = []
    firstline = True
    for line in vector1:
        #print(line)
        if firstline:
            firstline = False
        else:
            row = line.split()#space as delimiter
            vPoint = [float(row[0]),float(row[1]),float(row[2])]
            #print(str(vPoint))
            LMs.append(vPoint) #No const divide here, already with that factor
            #oddLine = True #the next line is odd number
    return LMs

#def getLMsPTS(filePathPTS, constDivide = 0.35277777777777775):
def getLMsPTS(filePathPTS, constDivide=1):
    LMs = []
    with open(filePathPTS, 'r') as file:
        reader = csv.reader(file)
        thirdLine = 2 #the first 2 lines are point and the dimension, which is the number of points
        for row in reader:
            if thirdLine==0:
                #format is 1,136.832489,68.25724792,270.9500122
                #print("row:",row)
                row = row[0].split() #space is the delimiter, everything is in the first element of the list
                LMs.append([np.float64(row[0]) * constDivide,np.float64(row[1]) * constDivide,np.float64(row[2]) * constDivide])
                #print("LMs:", LMs)
            else:
                thirdLine = thirdLine - 1
    file.close()
    return LMs

def saveLMsPTS(filePathPTS,LMs, constDivide=1.0):
    nPoints = len(LMs)
    with open(filePathPTS, 'w') as file:
        file.write("point\n"+str(nPoints)+"\n")
        for row in LMs:
            file.write(str(row[0] / constDivide) + ' ' + str(row[1] / constDivide) + ' ' + str(row[2]/ constDivide)+'\n')
    file.close()
    return LMs

def getLMs(filePathCSV, constDivide=1):
    LMs = []
    with open(filePathCSV, 'r') as file:
        reader = csv.reader(file)
        firstLine = True
        
        for row in reader:
            
            #Check how many columns has the CSV and which column has X
            if firstLine:
                first_column_with_x = next(
                    (i for i, value in enumerate(row) if 'x' in value.lower()), 
                    None
                )
                if (first_column_with_x is not None) and first_column_with_x >= 0:
                    firstLine = False
            else:
                LMs.append([np.float64(row[first_column_with_x]) * constDivide,np.float64(row[first_column_with_x+1]) * constDivide,np.float64(row[first_column_with_x+2]) * constDivide])
                
    file.close()
    return LMs

def saveLMs(filePathCSV,LMs, constDivide=1):
    with open(filePathCSV, 'w') as file:
        #writer = csv.writer(file)
        #writer.writerow(",X,Y,Z")
        file.write(",X,Y,Z\n")
        iLM = 1
        for row in LMs:
            #print(str(row))
            #format is 1,136.832489,68.25724792,270.9500122
            #LMs.append([float(row[1]) * constDivide,float(row[2]) * constDivide,float(row[3]) * constDivide])
            #writer.writerow(str(iLM)+','+str(row[0])+','+str(row[1])+','+str(row[2]))
            file.write(str(iLM) + ',' + str(row[0] / constDivide) + ',' + str(row[1] / constDivide) + ',' + str(row[2]/ constDivide)+'\n')
            iLM = iLM + 1
    file.close()
    return LMs

def convertVTKPointsToLMs(pointsVTK, constDivide=1):
    LMs = []
    nPoints = pointsVTK.GetNumberOfPoints()
    print("Number of points: " + str(nPoints))
    for i in range(nPoints):
        pAux = pointsVTK.GetPoint(i)
        LMs.append([pAux[0],pAux[1],pAux[2]])
        #print(LMs)
    return LMs

def convertLMsToVTKPoints(filePathCSV, constDivide=1):
    LMs = getLMs(filePathCSV,constDivide)
    pointsVTK = vtk.vtkPoints()
    for point in LMs:
        pointsVTK.InsertNextPoint(point)
    return pointsVTK

def convertLMsToITKPoints(filePathCSV, constDivide=1):
    LMs = getLMs(filePathCSV,constDivide)
    flatLMs = []
    for point in LMs:
        flatLMs.append(point[0])
        flatLMs.append(point[1])
        flatLMs.append(point[2])
    return flatLMs

def modifyOrigin(pointsVector,XInic,YInic,ZInic, constDivide=1, scale = 1.0):
    LMs = []
    nPoints = len(pointsVector)
    for i in range(nPoints):
        pAux = pointsVector[i]
        pModif = [pAux[0] - (XInic*constDivide), pAux[1] - (YInic*constDivide), pAux[2] - (ZInic*constDivide)]
        #pModif = float(pModif) * scale #Added due to the scaling of the E11.5s
        pModif2 = [x * scale for x in pModif] #Added due to the scaling of the E11.5s
        LMs.append(pModif2)
    return LMs



def transformAndSaveLMsElastix(moving_image_cropped, XInic, YInic, ZInic, folderOutput,
                               p, TransformationDirTXT,
                               pathLMsMoving, pathLMsModifiedForTransform, pathLMsGMCorrected, pathLMsMoved,
                               scaleLMs = 1.0, constDivide = 1):
    # To extract the origin of the transformation
    parameterMap = sitk.ReadParameterFile(TransformationDirTXT)
    origin_out = np.asarray(parameterMap["Origin"]).astype(np.float)
    print("---origin of the image")
    print(str(origin_out.tolist()))
    moving_image_cropped.SetOrigin(
        origin_out.tolist())  # For some reason, the origin changes, and that info is in the parameter map

    print('-----Modifiying origin------')
    #Correct landmarks
    points = getLMsPTS(pathLMsMoving)
    #Because of the cropping
    pointsVector = modifyOrigin(points, float(XInic), float(YInic), float(ZInic))
    saveLMsPTS(pathLMsGMCorrected, pointsVector)

    #Because of the origin of the image, already in the correct spacing
    #keep the constant divide to 1. If it is not working for downsampled volumes, change it
    pointsVector = modifyOrigin(pointsVector,-origin_out.tolist()[0],-origin_out.tolist()[1],-origin_out.tolist()[2], 1,
                                scale = scaleLMs)

    saveLMsPTS(pathLMsModifiedForTransform, pointsVector, 1)

    #Invert the Transformation

    print('-----Starting elastixImageFilter------')
    elastixImageFilter = sitk.ElastixImageFilter()
    print(TransformationDirTXT)
    #elastixImageFilter.SetParameterMap(p)
    elastixImageFilter.SetInitialTransformParameterFileName(TransformationDirTXT)
    elastixImageFilter.SetFixedImage(moving_image_cropped)
    elastixImageFilter.SetMovingImage(moving_image_cropped)
    elastixImageFilter.LogToConsoleOff()
    #affine_pm = sitk.GetDefaultParameterMap("affine") #it should be affine. I tried rigid, just in case, it did not work
    affine_pm = affine_transf_default()
    elastixImageFilter.SetParameterMap(affine_pm)
    elastixImageFilter.SetParameter('HowToCombineTransforms', 'Compose')
    elastixImageFilter.SetParameter('Metric', 'DisplacementMagnitudePenalty')
    elastixImageFilter.SetParameter('MaximumNumberOfIterations', '5000')


    print('-----Executing DisplacementMagnitudePenalty to invert------')
    #elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.Execute()
    Tx = elastixImageFilter.GetTransformParameterMap()
    Tx[0]['InitialTransformParametersFileName'] = ('NoInitialTransform',)

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(Tx)
    transformixImageFilter.SetMovingImage(moving_image_cropped)
    transformixImageFilter.SetFixedPointSetFileName(pathLMsModifiedForTransform)
    #transformixImageFilter.SetMovingPointSetFileName(pathLMsMovingCorrected)
    transformixImageFilter.SetOutputDirectory(folderOutput)
    print('-----Inverting points------')
    transformixImageFilter.Execute()

    #transformixImageFilter.SetFixedPointSetFileName(pathLMsCorrected)
    #transformixImageFilter.ExecuteInverse() #will save in folderOutput
    outputTempFile = folderOutput+"outputpoints.temp"
    os.rename(folderOutput+"outputpoints.txt", outputTempFile)
    LMs = getLMsOutputpointstxt(outputTempFile)
    print("----transform back from the origin----")
    LMs = modifyOrigin(LMs, origin_out.tolist()[0], origin_out.tolist()[1],
                       origin_out.tolist()[2], constDivide = constDivide,
                                scale = 1.0/scaleLMs)
    saveLMsPTS(pathLMsMoved,LMs)

def affine_transf_default():
    p = sitk.GetDefaultParameterMap("affine")
    p["Transform"] = ["AffineTransform"]
    p["Metric"] = ["AdvancedMattesMutualInformation"]
    p["NumberOfResolutions"] = ["5"]
    p["Registration"] = ["MultiResolutionRegistration"]
    p["AutomaticTransformInitialization"] = ["true"]
    p["AutomaticTransformInitializationMethod"] = ["Origins"]
    p["AutomaticParameterEstimation"] = ["true"]
    p["CheckNumberOfSamples"] = ["true"]
    p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["DefaultPixelValue"] = ["0"]
    p["Interpolator"] = ["LinearInterpolator"]
    p["FinalBSplineInterpolationOrder"] = ["1"]
    p["Resampler"] = ["DefaultResampler"]
    p["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    p["MaximumNumberOfIterations"] = ["20000"]
    p["ImageSampler"] = ["RandomCoordinate"]
    p["NumberOfSamplesForExactGradient"] = ["4096"]
    p["NumberOfSpatialSamples"] = ["4096"]
    p["MaximumNumberOfSamplingAttempts"] = ["8"]
    p["NewSamplesEveryIteration"] = ["true"]
    return p


def functionSwapLMs(mat_xyz, right_lm, left_lm):
    """
    Swap two rows in the mat_xyz array based on the given indices.
    
    Parameters:
    - mat_xyz (numpy.ndarray): The array containing landmarks.
    - right_lm (int): Index of the right landmark (1-based indexing as in MATLAB).
    - left_lm (int): Index of the left landmark (1-based indexing as in MATLAB).
    
    Returns:
    - numpy.ndarray: The updated array with the rows swapped.
    """
    # Convert 1-based MATLAB indexing to 0-based Python indexing
    right_idx = right_lm
    left_idx = left_lm

    # Swap the rows
    mat_xyz[[right_idx, left_idx]] = mat_xyz[[left_idx, right_idx]]
    
    return mat_xyz


def mirror_lms(fullpath_csv, folder_sample, sample_name, folder_sample_flipped, sample_name_flipped, swap_pairs, axis = 1):
    
    pat_tiff = os.path.join(folder_sample, '*.tiff')
    list_tiff = glob(pat_tiff)
    any_tiff = list_tiff[0]  # Assuming at least one TIFF file exists
    
    # Read the TIFF volume (dummy function, replace with actual implementation)
    volume = functionReadTIFFMultipage(any_tiff, 8)
    h, w, z = volume.shape
    
    
    del volume
    
    name_file = os.path.basename(fullpath_csv)
        
    new_str = name_file.replace(sample_name, sample_name_flipped)
    fullpath_csv_flipped = os.path.join(folder_sample_flipped, new_str)
    # print(fullpath_csv_flipped)
    
    # Read the CSV into a DataFrame
    df = pd.read_csv(fullpath_csv)
    
    # Drop the header row and transform to a NumPy array
    # Assuming the columns are ['Index', 'X', 'Y', 'Z']
    mat_xyz = df[['X', 'Y', 'Z']].to_numpy()
    
    
    # Adjust Y to create the flipped version
    mat_xyz[:, 1] = h - mat_xyz[:, 1]  # Assuming Y is the second column
    
    # Write Mirror
    # print(mat_xyz)
    h, w = mat_xyz.shape
    
    # MATLAB-style indexing (1-based)
    swap_pairs = [(a - 1, b - 1) for a, b in swap_pairs] # to numpy indexing
    for a, b in swap_pairs:
        mat_xyz = functionSwapLMs(mat_xyz, a, b)
    
    # Save flipped CSV
    saveLMs(fullpath_csv_flipped,mat_xyz, constDivide=1)
    # with open(fullpath_csv_flipped, 'w') as fid_dest:
    #     fid_dest.write(",X,Y,Z\n")
    #     for j in range(h):
    #         row = mat_xyz[j]
    #         line = f"{j + 1},{row[0]},{row[1]},{row[2]}\n"
    #         fid_dest.write(line)
