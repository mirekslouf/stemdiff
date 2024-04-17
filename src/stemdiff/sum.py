'''
Module: stemdiff.sum
--------------------
The summation of 4D-STEM datafiles to create one 2D powder diffraction file.

* stemdiff.sum = this module, which runs on a single core (serial processing)
* stemdiff.summ = sister module running on multiple cores (parallel processing)

To perform the summation, we just call function sum_datafiles:

* serial   : stemdiff.sum.sum_datafiles(SDATA, DIFFIMAGES, df, deconv, ...)
* parallel : stemdiff.summ.sum_datafiles(SDATA, DIFFIMAGES, df, deconv, ...)

The initial arguments are:

* SDATA = stemdiff.gvars.SourceData object = description of source data
* DIFFIMAGES = stemdiff.gvars.DiffImages object = description of diffractograms
* df = pre-calculated database of datafiles/diffratograms to sum

Key argument is deconv, which determines the processing type:
    
* deconv=0 = sum *without* deconvolution
* deconv=1 = R-L deconvolution with global PSF from low-diffraction datafiles
* deconv=2 = subtract background + R-L deconvolution with PSF from the center
'''


import numpy as np
import stemdiff.io
import stemdiff.dbase
import idiff
from skimage import restoration
import tqdm
import sys


def sum_datafiles(SDATA, DIFFIMAGES, df, deconv=0, psf=None, iterate=10):
    """
     Sum datafiles from a 4D-STEM dataset to get 2D powder diffractogram.
 
     Parameters
     ----------
     SDATA : stemdiff.gvars.SourceData object
         The object describes source data (detector, data_dir, filenames).
     DIFFIMAGES : stemdiff.gvars.DiffImages object
         Object describing the diffraction images/patterns.
     df : pandas.DataFrame object
         Pre-calculated atabase with datafiles to be summed.
         Each row of the database contains
         [filename, xc, yc, MaxInt, NumPeaks, S].
     deconv : int, optional, default is 0
         Deconvolution type:
         0 = no deconvolution,
         1 = deconvolution based on external PSF,
         2 = deconvolution based on PSF from central region,
     psf : 2D-numpy array or None, optional, default is None
         Array representing 2D-PSF function.
         Relevant only for deconv = 1.
     iterate : integer, optional, default is 10
         Number of iterations during the deconvolution.
 
     Returns
     -------
     final_arr : 2D numpy array
         The array is a sum of datafiles;
         if the datafiles are pre-filtered,
         we get the sum of filtered datafiles.
         Additional arguments determin the (optional) type of deconvolution.
 
     Technical notes
     ---------------
     * This function works as a signpost.
     * It reads the summation parameters and calls a more specific summation 
       functions (which aren NOT called directly by the end-user).
     * It employs progress bar, handles possible exceptions,
       and returns the final array (= post-processed and normalized array).
    """

    # (1) Prepare variables for summation 
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafiles = [datafile[1] for datafile in df.iterrows()] 
    sum_arr = np.zeros((img_size * R, img_size * R), dtype=np.float32)

    # (2) Prepare variables for tqdm
    # (to create a single progress bar for the entire process
    total_tasks = len(datafiles)
    sys.stderr = sys.stdout

    # (3) Run summations
    # (summations will run with tqdm
    # (we will use several types of summations
    # (each summations uses datafiles prepared in a different way
    with tqdm.tqdm(total=total_tasks, desc="Processing ") as pbar:
        try:
            # Process each image in the database
            for index, datafile in df.iterrows():
                # Deconv0 => sum datafiles without deconvolution
                if deconv == 0:
                    sum_arr += dfile_without_deconvolution(
                        SDATA, DIFFIMAGES, datafile)
                # Deconv1 => sum datafiles with DeconvType1
                elif deconv == 1:
                    sum_arr += dfile_with_deconvolution_type1(
                        SDATA, DIFFIMAGES, datafile, psf, iterate)
                # Deconv2 => sum datafiles with DeconvType2
                elif deconv == 2:
                    sum_arr += dfile_with_deconvolution_type2(
                        SDATA, DIFFIMAGES, datafile, iterate)
                # Update the progress bar for each processed image
                pbar.update(1)
        except Exception as e:
            print(f"Error processing a task: {str(e)}")

    # (4) Move to the next line after the progress bar is complete
    print('')

    # (5) Post-process the summation and return the result
    return sum_postprocess(sum_arr, len(df))


def sum_postprocess(sum_of_arrays, n):
    """
    Normalize and convert the summed array to 16-bit unsigned integers.
    
    Parameters
    ----------
    sum_of_arrays : np.array
        Sum of the arrays -
        usually from stemdiff.sum.sum_datafiles function.
    n : int
        Number of summed arrays -
        usually from stemdiff.sum.sum_datafiles function.
    
    Returns
    -------
    arr : np.array
        Array representing final summation.
        The array is normalized and converted to unsigned 16bit integers.
    """
    arr = np.round(sum_of_arrays/n).astype(np.uint16)
    return(arr)

    
def dfile_without_deconvolution(SDATA, DIFFIMAGES, datafile):
    """
    Prepare datafile for summation without deconvolution (deconv=0).

    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        The bject describing the diffraction images/patterns.
    datafile : one row from the prepared database of datafiles
        The database of datafiles is created
        in stemdiff.dbase.calc_database function.
        Each row of the database contains
        [filename, xc, yc, MaxInt, NumPeaks, S].
    
    Returns
    -------
    arr : 2D numpy array
        The datafile in the form of the array,
        which is ready for summation (with DeconvType0 => see Notes below). 
    
    Notes
    -----
    * The parameters are transferred from the `sum_datafiles` function.
    * DeconvType0 = no deconvolution,
      just summation of the prepared datafiles (upscaled, centered...).
    """
        
    # (0) Prepare variables
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    
    # (2) Rescale/upscale datafile and THEN remove border region
    # (a) upscale datafile
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    # (b) get the accurate center of the upscaled datafile
    # (the center coordinates for each datafile are saved in the database
    # (note: our datafile is one row from the database => we know the coords!
    xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
    # (c) finally, the borders can be removed with respect to the center
    arr = stemdiff.io.Arrays.remove_edges(arr,img_size*R,xc,yc)
    # (Important technical notes:
    # (* This 3-step procedure is necessary to center the images precisely.
    # (  The accurate centers from upscaled images are saved in database.
    # (  The centers from original/non-upscaled datafiles => wrong results.
    # (* Some border region should ALWAYS be cut, for two reasons:
    # (  (i) weak/zero diffractions at edges and (ii) detector edge artifacts
    
    # (3) Return the datafile as an array that is ready for summation
    return(arr)


def dfile_with_deconvolution_type1(SDATA, DIFFIMAGES, datafile, psf, iterate):
    """
    Prepare datafile for summation with deconvolution type1 (deconv=1).

    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        The bject describing the diffraction images/patterns.
    datafile : one row from the prepared database of datafiles
        The database of datafiles is created
        in stemdiff.dbase.calc_database function.
        Each row of the database contains
        [filename, xc, yc, MaxInt, NumPeaks, S].
    psf : 2D-numpy array
        Array representing the 2D-PSF function.
    iterate : int
        Number of iterations during the deconvolution.

    Returns
    -------
    arr : 2D numpy array
        The datafile in the form of the array,
        which is ready for summation (with DeconvType1 => see Notes below). 
    
    Notes
    -----
    * The parameters are transferred from the `sum_datafiles` function.
    * DeconvType1 = Richardson-Lucy deconvolution using PSFtype1.
    * PSFtype1 = 2D-PSF estimated from files with negligible diffractions.
    """
       
    # (0) Prepare variables
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    
    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

    # (2) Rescale/upscale datafile and THEN remove border region
    # (a) upscale datafile
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    # (b) get the accurate center of the upscaled datafile
    # (the center coordinates for each datafile are saved in the database
    # (note: our datafile is one row from the database => we know the coords!
    xc,yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    # (c) finally, the borders can be removed with respect to the center
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size*R,xc,yc)        
    # (Important technical notes:
    # (* This 3-step procedure is necessary to center the images precisely.
    # (  The accurate centers from upscaled images are saved in database.
    # (  The centers from original/non-upscaled datafiles => wrong results.
    # (* Some border region should ALWAYS be cut, for two reasons:
    # (  (i) weak/zero diffractions at edges and (ii) detector edge artifacts
        
    # (3) Deconvolution: Richardson-Lucy using a global PSF
    # (a) save np.max, normalize
    # (reason: deconvolution algorithm requires normalized arrays...
    # (...and we save original max.intensity to re-normalize the result
    norm_const = np.max(arr)
    arr_norm = arr/np.max(arr)
    psf_norm = psf/np.max(psf)
    # (b) perform the deconvolution
    arr_deconv = restoration.richardson_lucy(
        arr_norm, psf_norm, num_iter=iterate)
    # (c) restore original range of intensities = re-normalize
    arr = arr_deconv * norm_const
    
    # (4) Return the deconvolved datafile
    # as an array that is ready for summation
    return arr


def dfile_with_deconvolution_type2(SDATA, DIFFIMAGES, datafile, iterate):
    """
    Prepare datafile for summation with deconvolution type2 (deconv=2).

    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        The bject describing the diffraction images/patterns.
    datafile : one row from the prepared database of datafiles
        The database of datafiles is created
        in stemdiff.dbase.calc_database function.
        Each row of the database contains
        [filename, xc, yc, MaxInt, NumPeaks, S].
    iterate : int
        Number of iterations during the deconvolution.

    Returns
    -------
    arr : 2D numpy array
        The datafile in the form of the array,
        which is ready for summation (with DeconvType1 => see Notes below). 
    
    Notes
    -----
    * The parameters are transferred from the `sum_datafiles` function.
    * DeconvType2 = Richardson-Lucy deconvolution
      using PSFtype2 + simple background subtraction. 
    * PSFtype2 = 2D-PSF estimated from central region of the datafile
      AFTER background subtraction.
    """
    
    # (0) Prepare variables
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    psf_size = DIFFIMAGES.psfsize
    
    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name) 
    
    # (2) Rescale/upscale datafile and THEN remove border region
    # (a) upscale datafile
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    # (b) get the accurate center of the upscaled datafile
    # (the center coordinates for each datafile are saved in the database
    # (note: our datafile is one row from the database => we know the coords!
    xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
    # (c) finally, the borders can be removed with respect to the center
    arr = stemdiff.io.Arrays.remove_edges(arr,img_size*R,xc,yc)        
    # (Important technical notes:
    # (* This 3-step procedure is necessary to center the images precisely.
    # (  The accurate centers from upscaled images are saved in database.
    # (  The centers from original/non-upscaled datafiles => wrong results.
    # (* Some border region should ALWAYS be cut, for two reasons:
    # (  (i) weak/zero diffractions at edges and (ii) detector edge artifacts
    
    # (3) Remove background
    arr = idiff.bcorr.rolling_ball(arr, radius=20)
    
    # (4) Prepare PSF from the center of given array
    # (recommended parameters:
    # (psf_size => to be specified in the calling script ~ 30
    # (circular => always True - square PSF causes certain artifacts
    psf = idiff.psf.PSFtype2.get_psf(arr, psf_size, circular=True)
    
    # (5) Deconvolution
    # (a) save np.max, normalize
    # (reason: deconvolution algorithm requires normalized arrays...
    # (...and we save original max.intensity to re-normalize the result
    norm_const = np.max(arr)
    arr_norm = arr/np.max(arr)
    psf_norm = psf/np.max(psf)
    # (b) perform the deconvolution
    arr_deconv = restoration.richardson_lucy(
        arr_norm, psf_norm, num_iter=iterate)
    # (c) restore original range of intensities = re-normalize
    arr = arr_deconv * norm_const

    # (6) Return the deconvolved datafile
    # as an array that is ready for summation
    return(arr)
