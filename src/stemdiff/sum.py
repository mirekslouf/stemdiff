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
import idiff
import ediff.center
from skimage import restoration
from scipy.ndimage import shift
import tqdm
import sys


def sum_datafiles(SDATA, DIFFIMAGES, df_sum, df_psf=None, bkg=0, deconv=False,
                  peaks=False, iterate=10, nn_path=None):
    """
     Sum datafiles from a 4D-STEM dataset to get 2D powder diffractogram.
 
     Parameters
     ----------
     SDATA : stemdiff.gvars.SourceData object
         The object describes source data (detector, data_dir, filenames).
     DIFFIMAGES : stemdiff.gvars.DiffImages object
         Object describing the diffraction images/patterns.
     df_sum : pandas.DataFrame object
         Pre-calculated database with datafiles to be summed.
         Each row of the database contains
         [filename, xc, yc, MaxInt, NumPeaks, S].
     df_psf : pandas.DataFrame object, optional
         Database with datafiles to calculate PSF.
         If None, PSF is calculated from central region of each datafile.
     bkg : int, optional, default is 0
         Background subtraction type:
         0 = no background subtraction,
         1 = rolling ball,
         2 = neural network.

         Neural network also needs the nn_path argument.
     deconv : int, optional, default is False
         Use deconvolution base on PSF determined by parameter df_psf.
     peaks : bool, optional, default is False
         If true, run peak detection algorithm.
     iterate : integer, optional, default is 10
         Number of iterations during the deconvolution.
     nn_path : str, optional
         Path to neural network for background subtraction.
 
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
    datafiles = [datafile[1] for datafile in df_sum.iterrows()] 
    sum_arr = np.zeros((img_size * R, img_size * R), dtype=np.float32)

    if nn_path != None:
        nn = idiff.bkg2d.NeuralNetwork(nn_path)
    else:
        if bkg == 2:
            raise ValueError("Argument nn_path must be specified, if bkg=2.")
        nn = None

    if df_psf != None:
        psf = idiff.psf.PSFtype1.get_psf(SDATA, DIFFIMAGES, df_psf)
    else:
        psf = None

    # (2) Prepare variables for tqdm
    # (to create a single progress bar for the entire process
    total_tasks = len(datafiles)
    sys.stderr = sys.stdout

    # (3) Run summations
    # (summations will run with tqdm
    # (we will use several types of summations
    # (each summations uses datafiles prepared in a different way
    with tqdm.tqdm(total=total_tasks, desc="Processing ") as pbar:
        # try:
            # Process each image in the database
            for index, datafile in df_sum.iterrows():
                sum_arr += prepare_dfile(SDATA, DIFFIMAGES, datafile, psf, bkg,
                                         deconv, peaks, iterate, nn)
                
                # Update the progress bar for each processed image
                pbar.update(1)
        # except Exception as e:
        #     print(f"Error processing a task: {str(e)}")

    # (4) Move to the next line after the progress bar is complete
    print('')

    # (5) Post-process the summation and return the result
    return sum_postprocess(sum_arr, len(df_sum))


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

    
def prepare_dfile(SDATA, DIFFIMAGES, datafile, psf, bkg, deconv, peaks,
                  iterate, nn):
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
    psf_size = DIFFIMAGES.psfsize

    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

    # (2) Remove background
    if bkg == 1:
        arr = idiff.bkg2d.rolling_ball(arr, radius=3)
        arr[arr < 50] = 0
    elif bkg == 2:
        arr = nn.predict(arr)
    elif bkg == 3:
        arr = idiff.bkg2d.tophat(arr)
    
    # (3) Rescale/upscale datafile and THEN remove border region
    # (a) upscale datafile
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    # (b) get the accurate center of the upscaled datafile
    # (the center coordinates for each datafile are saved in the database
    # (note: our datafile is one row from the database => we know the coords!
    # (c) finally, recenter the image and zero the edges
    # if bkg >= 2:
    #     arr = recenter_on_max(arr)
    # else:
    #     center = ediff.center.CenterLocator(
    #             arr, "intensity", final_print=False)
    #     xc, yc = round(center.x), round(center.y)
    #     arr = recenter(arr, xc, yc)
    center = ediff.center.CenterLocator(
            arr, "intensity", final_print=False)
    xc, yc = round(center.x), round(center.y)
    arr = recenter(arr, xc, yc)
    arr = zero_spatial_edges(arr)
    # (Important technical notes:
    # (* This 3-step procedure is necessary to center the images precisely.
    # (  The accurate centers from upscaled images are saved in database.
    # (  The centers from original/non-upscaled datafiles => wrong results.
    # (* Some border region should ALWAYS be cut, for two reasons:
    # (  (i) weak/zero diffractions at edges and (ii) detector edge artifacts

    # (4) Prepare PSF from the center of given array, if not given as parameter
    # (recommended parameters:
    # (psf_size => to be specified in the calling script ~ 30
    # (circular => always True - square PSF causes certain artifacts
    if psf == None and deconv:
        # remove more background for psf
        psf = idiff.psf.PSFtype2.get_psf(arr, psf_size, circular=True)

    # (5) Deconvolution
    # (a) save np.max, normalize
    # (reason: deconvolution algorithm requires normalized arrays...
    # (...and we save original max.intensity to re-normalize the result
    if deconv:
        norm_const = np.max(arr)
        arr_norm = arr/np.max(arr)
        psf_norm = psf/np.max(psf)
        # (b) perform the deconvolution
        arr_deconv = restoration.richardson_lucy(
            arr_norm, psf_norm, num_iter=iterate)
        # (c) restore original range of intensities = re-normalize
        arr = arr_deconv * norm_const

    # (6) Detect peaks
    if peaks:
        arr = idiff.peaks.run_regions(arr)

    # (7) Return the datafile as an array that is ready for summation
    return arr

def recenter_on_max(img):
    """
    Finds the maximum value in a 2D array and centers the image on it.
    """
    # 1. Find the 2D coordinates of the maximum value
    # np.argmax gives the flat index; unravel_index converts it to (row, col)
    max_y, max_x = np.unravel_index(np.argmax(img), img.shape)
    
    h, w = img.shape
    
    # 2. Calculate the shift required to move (max_y, max_x) to (h//2, w//2)
    shift_y = (h // 2) - max_y
    shift_x = (w // 2) - max_x
    
    # 3. Apply the shift with zero-padding
    # order=0 preserves the original pixel values (nearest neighbor)
    recentered_img = shift(img, shift=[shift_y, shift_x], mode='constant', cval=0, order=0)
    
    return recentered_img

def recenter(img, center_x, center_y):
    """
    Recenters the image by shifting (center_x, center_y) to the array center.
    Empty edges are filled with zeros.
    """
    h, w = img.shape
    
    # Calculate the required displacement
    # shift_y = target_y - current_y
    shift_y = (h // 2) - center_y
    shift_x = (w // 2) - center_x
    
    # mode='constant' fills the boundary with cval (default is 0.0)
    # order=0 uses nearest-neighbor (keeps pixel values exact)
    # order=1 uses bilinear interpolation (smoother, better for sub-pixel)
    shifted_img = shift(img, shift=[shift_y, shift_x], mode='constant', cval=0,
                        order=0)
    
    return shifted_img

def zero_spatial_edges(data, border_width=10):
    """
    Zeros the edges of an array with shape (..., H, W).
    Works for (C, H, W) and (B, C, H, W).
    """
    res = data
    w = border_width
    
    # Zero Top and Bottom
    res[..., :w, :] = 0      # All batches/channels, first 'w' rows
    res[..., -w:, :] = 0     # All batches/channels, last 'w' rows
    
    # Zero Left and Right
    res[..., :, :w] = 0      # All batches/channels, first 'w' columns
    res[..., :, -w:] = 0     # All batches/channels, last 'w' columns
    
    return res