'''
Module: stemdiff.dbase
----------------------
Functions to calculate, save and re-read the database of 4D-STEM datafiles.

* The database is saved as a pandas.DataFrame object,
  which contains the following data for each datafile:
  filename, XY-center coordinates, MaximumIntensity, NoOfPeaks, ShannonEntropy.
* The database enables fast filtering of datafiles
  and fast access to datafile features,
  which do not have to be re-calculated repeatedly.
'''

import numpy as np
import pandas as pd
import stemdiff.io
from skimage import measure
    
def calc_database(SDATA, DIFFIMAGES):
    """
    Read 4D-STEM datafiles and calculate database of all files, which contains
    [filename, XY-center coordinates, MaxIntensity, NoOfPeaks, ShannonEntropy]
    for each datafile.
        
    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describes source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        The object describing the diffraction images/patterns.
        
    Returns
    -------
    df : pandas DataFrame object
        Database contains [filename, xc, yc, MaxInt, NumPeaks, S]
        for each datafile in the dataset.
    """
    # Prepare variables before we run for-cycle to go through datafiles
    # (initialize coordinates of the center + rescale/upscale coefficient
    xc,yc = (None,None)
    R = SDATA.detector.upscale
    # Initialize list of datafiles
    # (appending to lists is more efficient than appending to np/pd-structures
    list_of_datafiles = []
    # Pre-calculate additional variables
    # (it would be wasteful to calculate them in each for-cycle loop
    neighborhood_matrix = np.ones((DIFFIMAGES.peak_dist,DIFFIMAGES.peak_dist))
    # Go through datafiles and calculate their entropy
    for datafile in SDATA.filenames:
        # a) Read datafile
        arr = stemdiff.io.Datafiles.read(SDATA,datafile)
        # b) Get datafile name
        datafile_name = datafile.relative_to(SDATA.data_dir)
        # c) Calculate intensity center
        # (to get high precision
        # (the center must be determined for rescaled/upscaled array
        if DIFFIMAGES.ctype == 2:
            xc,yc = stemdiff.io.Arrays.find_center(
                stemdiff.io.Arrays.rescale(arr, R, order=3),  # rescaled array
                DIFFIMAGES.csquare*R, DIFFIMAGES.cintensity)  # central region
        elif (DIFFIMAGES.ctype == 1) and (xc == None):
            xc,yc = stemdiff.io.Array.find_center(
                stemdiff.io.Arrays.rescale(arr, R, order=3),  # rescaled array
                DIFFIMAGES.csquare*R, DIFFIMAGES.cintensity)  # central region
        elif (DIFFIMAGES.ctype == 0) and (xc == None):
            geometric_center = round(SDATA.detector.detector_size*R/2)
            xc,yc = (geometric_center,geometric_center)
        # d) Determine maximum intensity
        # NOTE: max_intensity must be saved as float
        # Reason: when we later convert to pandas.Dataframe
        # ...the datatypes are inferred from the first value in each column
        # ...which may not be correct if the first integer is a small number
        # ...as the estimated datatype may be int16
        # ...and this may not work for larger values in the next datafiles
        max_intensity = float(np.max(arr))
        # e) Estimate number of peaks (local maxima)
        no_of_maxima = stemdiff.io.Arrays.number_of_peaks(arr,
            peak_height = DIFFIMAGES.peak_height, 
            neighborhood_matrix=neighborhood_matrix)
        # f) Calculate Shannon entropy of the datafile
        entropy = measure.shannon_entropy(arr)
        entropy = round(entropy,2)
        # d) Append all calculated values to list
        list_of_datafiles.append(
            [datafile_name, xc, yc, max_intensity, no_of_maxima, entropy])
    # Convert list to pandas DataFrame
    df = pd.DataFrame(
        list_of_datafiles,
        columns=['DatafileName','Xcenter','Ycenter', 'MaxInt', 'Peaks','S'])
    
    # Return the dataframe containing names of datafiles + their entropies
    return(df)

def save_database(df, output_file):
    """
    Save the database of 4D-STEM datafiles;
    the dbase is a pandas.DataFrame object saved as pickled object/zip-file.
        
    Parameters
    ----------
    df : pandas.DataFrame object
        This object is a database of all 4D-STEM datafiles,
        which contains basic characteristics (filename, XY-center, entropy ...)
        of each datafile in 4D-STEM dataset.
    output_file : str
        Filename of the output file (without extension).
        The database is saved as a pickled object/zip-file.

    Returns
    -------
    None
        The output is the database *df* saved as *output_file* on a disk.
    """
    df.to_pickle(output_file)
    
def read_database(input_database):
    """
    Read the calculated database of 4D-STEM datafiles.
    
    Parameters
    ----------
    input_database : str or pandas.DataFrame
        * str = filename of the input file that contains the database.
        * pandas.Dataframe = dataframe that contains the database.
        * Why two possile types of input?
          If the database is in memory in the form of pandas.DataFrame
          (which is quite common when using STEMDIFF package),
          it is useles to re-read it from file.
          
    Returns
    -------
    df : pandas.DataFrame object
        Database that has been read from disk or pandas.Dataframe.
    """
    # Read database from [input_file]
    # NOTE: input_database is either saved/pickled file or pd.DataFrame
    # Reason: sometimes it is more efficient to use pd.DataFrame directly 
    if type(input_database) == pd.DataFrame:
        df = input_database
    else:
        df = pd.read_pickle(input_database)
    return(df)
