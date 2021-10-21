'''
stemdiff.dbase
--------------
Read 4D-STEM datafiles and create database of all files.

The database contains [filename, S-entropy and XY-center] of each datafile.
    
* The database enables fast filtering of datafiles
  and fast access to datafile features.
* S-entropy = Shannon entropy = a fast-to-calculate image feature;
  datafiles with high S contain strong diffractions and *vice versa*.
* XY-center = two values = X- and Y-coordinate of the central spot
  (primary beam) for given datafile. 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
import stemdiff.io
from stemdiff.const import DET_SIZE, RESCALE
    
def calc_database(DATAFILES,CENTERING):
    """
    Read 4D-STEM datafiles and create database of all files,
    which contains [filename, S-entropy and XY-center] for each datafile.
    
    Parameters
    ----------
    DATAFILES : glob object
        Names of all datafiles from 4D-STEM dataset.
        The [glob object] is usually defined in the master script as follows:
        
            >>> from pathlib import Path
            >>> DATA_DIR  = Path('D:/DATA/AU')
            >>> DATAFILES = DATA_DIR.glob('*.dat')
            
    CENTERING : centering object
        Object containing parameters for finding center of diffractograms.
        The [centering object] is usually defined in the master script:
        
            >>> import stemdiff.const
            >>> CENTERING = stemdiff.const.centering(
            >>>     ctype=1, csquare=30, cintensity=0.8)

    Returns
    -------
    df : pandas DataFrame object
        Database containing [filename, ShannonEntropy, XY-center]
        of each datafile of 4D-STEM dataset.
    """
    # Prepare coordinates of the center
    xc,yc = (None,None)
    # Prepare empty list
    # (appending to lists is more efficient than appending to np/pd-structures
    list_of_datafiles = []
    # Go through datafiles and calculated their entropy
    for datafile in DATAFILES:
        # a) Read datafile
        arr = stemdiff.io.read_datafile(datafile)
        # b) Calculated and print Shannon entropy of the datafile
        entropy_value = measure.shannon_entropy(arr)
        # c) Calculate center of the datafile
        # Note: the center is calculated for enlarged/rescaled array
        # (this gives better accuracy for rescaled arrays
        # (for original arrays, divide center coordinates by RESCALE
        if CENTERING.ctype == 2:
            arr = stemdiff.io.rescale_array(arr, RESCALE)
            xc,yc = stemdiff.io.find_array_center(
                arr, CENTERING.csquare, CENTERING.cintensity)
        elif (CENTERING.ctype == 1) and (xc == None):
            arr = stemdiff.io.rescale_array(arr, RESCALE)
            xc,yc = stemdiff.io.find_array_center(
                arr, CENTERING.csquare, CENTERING.cintensity)
        elif (CENTERING.ctype == 0) and (xc == None):
            xc,yc = (round(DET_SIZE*RESCALE/2),round(DET_SIZE*RESCALE/2))
        # d) Append all calculated values to list
        list_of_datafiles.append([datafile, entropy_value, xc, yc])
    # Convert list to pandas DataFrame
    df = pd.DataFrame(list_of_datafiles,
                      columns=['DatafileName','Entropy','Xcenter','Ycenter'])
    # Return the dataframe containing names of datafiles + their entropies
    return(df)

def save_database(df, output_file):
    """
    Save database, which contains [filenames, entropies and XY-centers]
    of all 4D-STEM datafiles; the dbase is saved az pickled object/zip-file.

    Parameters
    ----------
    df : pandas DataFrame object
        This object is a database of all datafiles,
        which contains [filenames, entropies and centers] of diffractograms
        (the database is created by function stemdiff.dbase.calc_database).
        
    output_file : str
        Filename of the output file (without extension).

    Returns
    -------
    None.
    
    * The output is the saved file with name [output_file].zip
    * The output file = pickled data in zip format;
      this file is read back as an input to other functions,
      such as calculation of PSF function and summation of datafiles.
    """
    df.to_pickle(output_file)
    
def read_database(input_file):
    """
    Read database, which contains [filenames, entropies and centers]
    of all 4D-STEM datafiles.
    
    * the database is saved as pickled object/zip-file
    * it is created by functions *calc_database* and *save_database*

    Parameters
    ----------
    input_file : str
        Filename of the input file that contains the database.

    Returns
    -------
    df : pandas DataFrame object
        Database that has been read from disk.
    """
    # Read database from [input_file]
    # NOTE: input_file is either saved/pickled file or pd.DataFrame
    # Reason: sometimes it is more efficient to use pd.DataFrame directly 
    if type(input_file) == pd.DataFrame:
        df = input_file
    else:
        df = pd.read_pickle(input_file)
    return(df)

def get_all_datafiles(dbase):
    """
    Get filenames and parameters of all datafiles
    from a database, which contains [filenames, entropies and centers]
    of all 4D-STEM datafiles; the dbase is saved as pickled object/zip-file. 

    Parameters
    ----------
    dbase : str
        Filename of the database file
        = zip-file containg pickled database object.

    Returns
    -------
    DataFrame iterator
        The complete database/DataFrame is huge => return DataFrame iterator;     
        the iterator gradually returns all items/datafiles from the database.
    """
    # Read database file into pandas.DataFrame
    df = read_database(dbase)
    # Return complete database as DataFrame iterator
    # (whole DataFrame is huge and not iterable directly
    return(df.iterrows())

def get_high_S_files(dbase, S=None, P=None, N=None):
    """
    Get filenames and parameters of high-entropy datafiles
    from a database, which contains [filenames, entropies and centers]
    of all 4D-STEM datafiles; the dbase is saved as pickled object/zip-file. 

    Parameters
    ----------
    dbase : str
        Filename of the database file
        = zip-file containg pickled database object.    
    S : float
        Shannon entropy value;
        if S is given, we get only the files with entropy > S.
    P : float
        Percent of files with the lowest entropy;
        if P is given, we get only P% of files with the highest entropy.
    N : integer
        Number of files with the lowest entropy;
        if N is given, we get only N files with the highest entropy.

    Returns
    -------
    DataFrame iterator
        The complete database/DataFrame is huge => return DataFrame iterator;     
        the iterator gradually returns all items/datafiles with high entropy.
        
    Note:
    -----
    Priority of parameters : S > P > N
        i.e. if S is given, P and N are ignored etc.
    """
    # Read database file into pandas.DataFrame
    df = read_database(dbase)
    # Calculate entropy limit (then we can get files with higher entropy)
    entropy_limit = get_entropy_limit(df,S,P,N,high_entropy_files=True)
    # Calculate and return reduced database containing only high-entropy files
    df2 = df[df.Entropy >= entropy_limit]
    # Return adjusted/filtered database as DataFrame iterator
    # (whole DataFrame is huge and not iterable directly
    return(df2.iterrows())

def get_low_S_files(dbase, S=None, P=None, N=None):
    """
    Get filenames and parameters of low-entropy datafiles
    from a database, which contains [filenames, entropies and centers]
    of all 4D-STEM datafiles; the dbase is saved as pickled object/zip-file. 

    Parameters
    ----------
    dbase : str
        Filename of the database file
        = zip-file containg pickled database object.    
    S : float
        Shannon entropy value;
        if S is given, we get only the files with entropy < S.
    P : float
        Percent of files with the lowest entropy;
        if P is given, we get only P% of files with the lowest entropy.
    N : integer
        Number of files with the lowest entropy;
        if N is given, we get only N files with the lowest entropy.

    Returns
    -------
    DataFrame iterator
        The complete database/DataFrame is huge => return DataFrame iterator;     
        the iterator gradually returns all items/datafiles with a low entropy.
        
    Note:
    -----
    Priority of parameters : S > P > N
        i.e. if S is given, P and N are ignored etc.
    """
    # Read database file into pandas.DataFrame
    df = read_database(dbase)
    # Calculate entropy limit (then we can get files with lower entropy)
    entropy_limit = get_entropy_limit(df,S,P,N,high_entropy_files=False)
    # Calculate and return reduced database containing only low-entropy files
    df2 = df[df.Entropy <= entropy_limit]
    # Return adjusted/filtered database as DataFrame iterator
    # (whole DataFrame is huge and not iterable directly
    return(df2.iterrows())

def get_entropy_limit(df,S,P,N,high_entropy_files=True):
    """
    Determine Shannon entropy limit;
    which separates high- and low-entropy files;
    (the entropy limit can be determined from S or P or N parameter).

    Parameters
    ----------
    df : pandas DataFrame object
        This object is a database of all 4D-STEM datafiles,
        which contains [filenames, entropies and centers] of diffractograms.
    S : Shannon entropy
        that separaters high- and low-S files;
        (trivial case: the function just returns the value of S).
    P : Percent of files
        that determines how many percent of high- or low-S files we want;
        (here we calculate the S-value separating the high/low-S files and
        the calculated value depends on parameter high_entropy_files below).
    N : Number of files
        that determines how many high- or low-S files we want;
        (here we calculate the S-value separating the high/low-S files and
        the calculated value depends on parameter high_entropy_files below).
    high_entropy_files : boolean, optional, default=True.
        * If the parameter is True, the S-value separates high-entropy files.
        * If the parameter is False, the S-value separates low-entropy files.
        * Example: if we have P=20, we can separate either
          20% of the high-S files (if high_entropy_files = True)
          or 20% of the low-S files (if the high_entropy_files = False).
          
    Returns
    -------
    entropy limit : float
        The value of Shannon entropy,
        that separates high- and low-entropy files.
        
    Note:
    -----
    Priority of parameters : S > P > N
        i.e. if S is given, P and N are ignored etc.
    """
    # Calculate histogram, pdf and cdf
    # (we need high precision: bins=1000
    # (in order to get correct/non-approximate number of files for switch N
    counts,bins,pdf,cdf = calculate_entropy_histogram(df,bins=1000)
    # A) Entropy limit is given by S
    if S:
        entropy_limit = S
    # B) Entropy limit is given by P or N 
    else:
        # 1) determine percent of files - this is given by P or N
        # P = percent of high- or low-entropy files
        # N = number of  high- or low-entropy files
        if P: percent = P
        else: percent = N / np.sum(counts) * 100
        # 2) if we want high-entropy files, percent = 100-percent
        if high_entropy_files: percent = 100-percent 
        # 3) Now that we have percent of files, we can calculate entropy limit
        cdf_limit = percent/100
        entropy_limit = bins[np.argmax(cdf>=cdf_limit)]
    return(entropy_limit)

def calculate_entropy_histogram(df, bins):
    """
    Calculate parameters,
    which can be used for plotting the entropy histogram.

    Parameters
    ----------
    df : pandas DataFrame object
        This object is a database of (pre-selected) 4D-STEM datafiles,
        which contains [filenames, entropies and centers] of diffractograms.
    bins : int
        Number of bins ~ intervals of the histogram.

    Returns
    -------
    counts, bins, pdf, cdf
        * counts = array; number of files in bins
        * bins = array containing bins ~ intervals ~ limits between values
        * pdf = probability distribution function = counts normalized to 1
        * cdf = cummulative distribution function - calculated by np.cumsum
    """
    counts, bins = np.histogram(df.Entropy, bins=bins)
    pdf = counts / sum(counts)
    cdf = np.cumsum(pdf)
    return(counts,bins,pdf,cdf)

def plot_entropy_histogram(df, bins):
    """
    Create plot/histogram of entropy values of all datafiles
    and show it on the screen.

    Parameters
    ----------
    df : pandas DataFrame object
        Database containing all datafiles;
        df object is obtained as the output from the function calc_database.
        
    bins : integer
        Number of bins = intervals in the histogram; typical value is 100.

    Returns
    -------
    None.
        The result is the entropy histogram on the screen.
    """
    counts,bins,pdf,cdf = calculate_entropy_histogram(df, bins)
    plt.plot(bins[1:],pdf, 'b-', label='PDF')
    plt.plot(bins[1:],cdf, 'r-', label='CDF')
    plt.title('Shannon entropy distribution')
    plt.legend()
    plt.grid()
    plt.show()
