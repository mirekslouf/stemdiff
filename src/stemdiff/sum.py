'''
stemdiff.sum
------------
Sum 4D-STEM datafiles to create one 2D powder diffraction file.
'''

import numpy as np
import stemdiff.io
import stemdiff.dbase
import stemdiff.radial
from stemdiff.const import DET_SIZE, RESCALE
from skimage import transform, restoration

def sum_all(DBASE,SUMMATION,R=RESCALE):
    """
    Sum all datafiles from 4D-STEM dataset.

    Parameters
    ----------
    DBASE : str or pathlib object
        Name of the database file that contains
        [filename, entropy and XY-center] of each datafile of 4D-STEM dataset.
    
    SUMMATION : summation object
        The summation object contains parameters for the summation.
        The object is usually defined in advance as follows:
            
            >>> import stemdiff.const
            >>> SUMMATION = stemdiff.const.summation(
            >>>     psfsize=130,imgsize=125,iterate=30)
            
        More information about the summation parameters:
                
            >>> import stemdiff.const
            >>> help(stemdiff.const.summation)        
            
    R : int, optional, default=stemdiff.const.RESCALE
        * The final diffractogram size = size-of-original-array * RESCALE.
        * The RESCALE parameter is defined/imported from stemdiff.const
        * The optimal value of R is 4 ~ we use 4-times upscaling.
        * It is not recommened to change this default.
        
    Returns
    -------
    arr : 2D-numpy array
        The array represents the calculated 2D-PNBD diffraction pattern.
    """
    all_datafiles = stemdiff.dbase.get_all_datafiles(DBASE)
    arr = stemdiff.sum.sum_4Dstem_datafiles(
        all_datafiles, R=R, imgsize=SUMMATION.imgsize)
    return(arr)

def sum_highS(DBASE,SUMMATION,R=RESCALE, S=None, P=None, N=None):
    """
    Sum high-entropy datafiles from 4D-STEM dataset;
    the number of high-entropy files is determined by parameter S, P or N.

    Parameters
    ----------
    DBASE : str or pathlib object
        Name of the database file that contains
        [filename, entropy and XY-center] of each datafile of 4D-STEM dataset.
    
    SUMMATION : summation object
        The summation object contains parameters for the summation.
        The object is usually defined in advance as follows:
            
            >>> import stemdiff.const
            >>> SUMMATION = stemdiff.const.summation(
            >>>     psfsize=130,imgsize=125,iterate=30)
        
        More information about the summation parameters:
        
            >>> import stemdiff.const
            >>> help(stemdiff.const.summation)        
         
    R : int, optional, default=stemdiff.const.RESCALE
        * The final diffractogram size = size-of-original-array * RESCALE.
        * The RESCALE parameter is defined/imported from stemdiff.const
        * The optimal value is R = 4, i.e. we use 4-times upscaling.
        * It is not recommened to change this default.
    
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
    
    Returns
    -------
    arr : 2D-numpy array
        The array represents the calculated 2D-PNBD diffraction pattern.
    """
    highS_datafiles = stemdiff.dbase.get_high_S_files(DBASE, P=P, N=N, S=S)
    arr = stemdiff.sum.sum_4Dstem_datafiles(
        highS_datafiles, R=R, imgsize=SUMMATION.imgsize)
    return(arr)

def sum_highS_deconv(DBASE,SUMMATION,PSF,R=RESCALE, S=None, P=None, N=None):
    """
    Sum high-entropy datafiles from 4D-STEM dataset with deconvolution.
    
    * the number of high-S files is determined by parameter S, P or N
    * the number of deconvolution iterations is in SUMMATION parameter

    Parameters
    ----------
    DBASE : str or pathlib object
        Name of the database file that contains
        [filename, entropy and XY-center] of each datafile of 4D-STEM dataset.
        The database is usually created by functions in stemdiff.dbase module.
    
    SUMMATION : summation object
        The summation object contains parameters for the summation.
        The object is usually defined in advance as follows:
        
            >>> import stemdiff.const
            >>> SUMMATION = stemdiff.const.summation(
            >>>     psfsize=130,imgsize=125,iterate=30)
        
        More information about the summation parameters:
        
            >>> import stemdiff.const
            >>> help(stemdiff.const.summation)        
    
    PSF : str or pathlib object
        Name of the file that contains 2D-PSF function.
        The PSF is saved in NPY-file (numpy array in numpy format).
        The PSF is usually created by: stemdiff.psf.psf_from_lowS_files
        
    R : int, optional, default=stemdiff.const.RESCALE
        * The final diffractogram size = size-of-original-array * RESCALE.
        * The RESCALE parameter is defined/imported from stemdiff.const
        * The optimal value is R = 4, i.e. we use 4-fold upscaling.
        * It is not recommened to change this default.
    
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
    
    Returns
    -------
    arr : 2D-numpy array
        The array represents the calculated 2D-PNBD diffraction pattern.
    """ 
    highS_datafiles = stemdiff.dbase.get_high_S_files(
        DBASE, P=P, S=S, N=N)
    arr = stemdiff.sum.sum_4Dstem_datafiles(
        highS_datafiles, R, PSF,
        itr=SUMMATION.iterate, imgsize=SUMMATION.imgsize)
    return(arr)

def sum_4Dstem_datafiles(df,R=RESCALE, PSF=None, itr=None, imgsize=None):
    '''
    Sum input datafiles from a 4D-STEM dataset.
    This function can be called directly, but typically it is called
    from functions *sum_all*, *sum_highS*, and *sum_highS_deconv*.
    
    Parameters
    ----------
    df: pandas DataFrame row iterator
        DataFrame columns: DatafileName,Entropy,Xcenter,Ycenter.
    R : integer
        Rescale coefficient;
        the size of the final array is rescaled/multiplied by factor R.
        If PSF is given, the rescaling is performed before deconvolution.
        Note: PSF should the same rescale coefficient as given here!
    PSF : 2D numpy array
        PSF = point spread function for deconvolution
    itr : integer 
        Number of iterations during R-L deconvolution
    imgsize: integer
        Size of array read from the detector is reduced to imgsize.
        If imgsize is given, we sum only the central square with edge=imgsize.
        Smaller central area gives higher speed during deconvolution,
        while the outer area usually contains just the weakest diffractions.
        
    Returns
    -------
    final_arr : 2D numpy array
        The array is a sum of datafiles;
        if the datafiles are pre-filtered, we get sum of filtered datafiles,
        if PSF is given, we get sum of datafiles with PSF deconvolution.
    '''
    # Prepare variables ......................................................
    n = 0
    if imgsize:
        arr_size = imgsize
        xc,yc = (None,None)
    else:
        arr_size = DET_SIZE
    # Sum without deconvolution ..............................................
    if type(PSF)==type(None):
        # Prepare files
        # (arr size = detector size => rescaling at the end - see below
        sum_arr   = np.zeros((arr_size,arr_size), dtype=np.float)
        final_arr = np.zeros((arr_size,arr_size), dtype=np.uint16)
        # Sum datafiles
        # (rescaling can be done AFTER summation if there is no deconvolution
        for index,datafile in df:
            arr = stemdiff.io.read_datafile(datafile.DatafileName)
            # If rsize was given, reduce array size
            if imgsize:
                xc,yc = (round(datafile.Xcenter/R),round(datafile.Ycenter/R))
                arr = stemdiff.io.reduce_array_size(arr,imgsize,xc,yc)
            sum_arr += arr
            n += 1
        # Rescale final datafile
        norm_const = np.max(sum_arr)
        sum_arr = transform.rescale(sum_arr, R, order=3)
        sum_arr = sum_arr/np.max(sum_arr) * norm_const
    # Sum with deconvolution .................................................
    else:
        # Read PSF and normalize it
        # (normalization of PSF is necessary for deconvolution algorithm
        psf = stemdiff.psf.read_psf(PSF)
        psf = psf/np.sum(psf)
        # Prepare files
        # (array size = detector size * R => rescaling during summation
        # (if imsize was given, detector size is reduced to imsize
        sum_arr   = np.zeros((arr_size*R,arr_size*R), dtype=np.float)
        final_arr = np.zeros((arr_size*R,arr_size*R), dtype=np.uint16)
        # Sum datafiles with rescaling and deconvolution
        # (rescaling must be done during summagion BEFORE deconvolution
        for index,datafile in df:
            print('.',end='')
            arr = stemdiff.io.read_datafile(datafile.DatafileName)
            # Rescale array
            # (rescaling is done before reducing detector/array size
            # (this should result in more precise center determination
            arr = stemdiff.io.rescale_array(arr, R)
            # If rsize parameter was given, reduce detector area
            if imgsize:
                xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
                arr = stemdiff.io.reduce_array_size(arr,imgsize*R,xc,yc)
            # Normalize array
            # (normalization must be done before deconvolution
            # (BUT we save norm.const to restore orig.intensity at the end            
            norm_const = np.max(arr)
            arr = arr/np.max(arr)
            # Deconvolution
            arr = restoration.richardson_lucy(arr, psf, iterations=itr)
            # Multiply by the saved normalization constant
            arr = arr * norm_const
            # Add rescaled and deconvoluted file to summation
            sum_arr += arr
            n += 1
        print()
    # Calculate final array ..................................................
    # (divide sum by number of summed files in order to get reasonable values
    final_arr = np.round(sum_arr/n).astype(np.uint16)
    # Return final array
    return(final_arr)

def save_results(arr, output, icut=300, itype='8bit', rdist=True):
    '''
    Save results of summation (final 2D-image + optional 1D-radial profile).

    Parameters
    ----------
    arr : 2D numpy array
        Array representing 2D-PNBD diffractogram.
    output : string
        Filename of output 2D-diffratogram and its 1D-radial distribution;
        name of 2D-diffractogram will be [output.png],
        name of 1D-radial distribution will be [output.txt].
    icut : integer, optional, default=300
        Cut of intensity;
        if icut = 300, all image intensities > 300 will be set equal to 300.
    itype : string ('8bit'  or '16bit'), optional, default='8bit'
        Type of the image with 2D-PNBD diffractogram: 8 or 16 bit grayscale.   
    rdist : boolean, optional, default =True
        If rdist=True, calculate and save also 1D-radial distribution.
        The saved file = [output].txt, where output is the argument above.
        
    Returns
    -------
    None.
        The outputs are saved 2D-diffractogram and its 1D-profile.
    '''
    # Prepare filenames
    output_2d_image   = output + '.png'
    output_1d_profile = output + '.txt'
    # a) Save summation = 2D-array as image
    stemdiff.io.save_array(arr, output_2d_image, icut, itype)
    # b) Calculate and save 1D-radial profile of the image
    if rdist:
        stemdiff.radial.save_radial_distribution(arr, output_1d_profile)
