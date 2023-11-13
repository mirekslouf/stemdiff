'''
stemdiff.sum
------------
The summation of 4D-STEM datafiles to create one 2D powder diffraction file.

In stemdiff, we can sum datafiles in with or without 2D-PSF deconvolution.
We just call function sum_datafiles with various arguments as explained below.
The key argument determining type of deconvolution is deconv:
    
* deconv=0 = sum *without* deconvolution
* deconv=1 = sum deconvolution, fixed PSF from selected datafiles
* deconv=2 = sum with deconvolution, individual PSF from central region
* deconv=3 = sum with deconvolution, individual PSF from whole datafile
'''

import numpy as np
import stemdiff.io
import stemdiff.dbase
from skimage import restoration
import tqdm


def sum_postprocess(dat, n):
    """
    Normalize and convert the sum array to 16-bit unsigned integers.
    """
    return np.round(dat / n).astype(np.uint16)

    
def sum_datafiles(SDATA, DIFFIMAGES, 
                  df, deconv=0, iterate=10, psf=None, cake=None, subtract=None):
    """
     Sum datafiles from a 4D-STEM dataset.
 
     Parameters
     ----------
     SDATA : stemdiff.gvars.SourceData object
         The object describes source data (detector, data_dir, filenames).
     DIFFIMAGES : stemdiff.gvars.DiffImages object
         Object describing the diffraction images/patterns.
     df : pandas.DataFrame object
         Database with datafile names and characteristics.
     deconv : int, optional, default is 0
         Deconvolution type:
         0 = no deconvolution,
         1 = deconvolution based on external PSF,
         2 = deconvolution based on PSF from central region,
     iterate : integer, optional, default is 10
         Number of iterations during the deconvolution.
     psf : 2D-numpy array or None, optional, default is None
         Array representing 2D-PSF function.
         Relevant only for deconv = 1.
 
     Returns
     -------
     final_arr : 2D numpy array
         The array is a sum of datafiles;
         if the datafiles are pre-filtered, we get the sum of filtered datafiles,
         if PSF is given, we get the sum of datafiles with PSF deconvolution.
 
     Technical notes
     ---------------
     This function works as a signpost.
     It reads the summation parameters and calls a more specific summation 
     function.
     Handles exceptions during processing, closes the progress bar, 
     and returns the post-processed result.
     """

    # Prepare variables ....................................................... 
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    sum_arr = np.zeros((img_size * R, img_size * R), dtype=np.float32)

    progress_bar = tqdm.tqdm(total=len(df), 
                             desc="Processing Database", 
                             unit="image")

    try:
        # Process each image in the database
        for index, datafile in df.iterrows():
            if deconv == 0:
                sum_arr += no_deconvolution(datafile, 
                                            SDATA, 
                                            DIFFIMAGES)
            elif deconv == 1:
                sum_arr += deconvolution_type1(datafile, 
                                               SDATA, 
                                               DIFFIMAGES, 
                                               psf, 
                                               iterate)
            elif deconv == 2:
                sum_arr += deconvolution_type2(datafile, 
                                               SDATA, 
                                               DIFFIMAGES, 
                                               psf, 
                                               iterate)

            # Update progress bar for each image
            progress_bar.update(1)

    except Exception as e:
        print("Error during processing:", e)

    finally:
        # Close the progress bar after processing the entire database
        progress_bar.close()

    # Move to the next line after the progress bar is complete
    print('')

    # Post-process and return the result
    return sum_postprocess(sum_arr, len(df))


    
def no_deconvolution(datafile, SDATA, DIFFIMAGES):
    """
    Sum datafiles without deconvolution.

    Parameters
    ----------
    datafile : Datafile
        Datafile information.
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        Object describing the diffraction images/patterns.
    
    Returns
    -------
    arr : 2D numpy array
        The sum of datafiles without deconvolution.
    
    Technical notes
    ---------------
    The parameters are transferred from the `sum_files` function.
    """
    
    # # Check threading
    # thread_name = threading.current_thread().name
    # print(f"{thread_name} started deconvolution on {datafile.DatafileName}")
    
    # Prepare variables .......................................................
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    # SUM DATAFILES ...........................................................
    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    
    # Rescale/upscale datafile and THEN remove border region
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)

    # (i) the accurate image center must be taken from the upscaled image
    xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))

    # (ii) then the borders can be removed with respect to the center
    arr = stemdiff.io.Arrays.remove_edges(arr,img_size*R,xc,yc)

    # (This procedure is necessary to center the images precisely.
    # (The accurate centers from upscaled images are saved in database.
    # (Some border region should ALWAYS be cut, for two reasons:
    # (i) weak/zero diffractions at edges and (ii) detector edge artifacts

    return(arr)


def deconvolution_type1(datafile,SDATA,DIFFIMAGES,psf,iterate):
    """
    Sum datafiles with Richardson-Lucy deconvolution using a 2D-PSF of type 1.

    Parameters
    ----------
    datafile : Datafile
        Datafile information.
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        Object describing the diffraction images/patterns.
    psf : 2D-numpy array
        Array representing the 2D-PSF function.
    iterate : int
        Number of iterations during the deconvolution.

    Returns
    -------
    arr : 2D numpy array
        The sum of datafiles with Richardson-Lucy deconvolution.
    
    Technical notes
    ---------------
    The parameters are transferred from the `sum_datafiles` function.
    Deconvolution type 1 refers to Richardson-Lucy deconvolution
    using a 2D-PSF estimated from files with negligible diffractions.
    """
    
    # # Check threading
    # thread_name = threading.current_thread().name
    # print(f"{thread_name} started deconvolution on {datafile.DatafileName}")
    
    # Prepare variables .......................................................
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    
    # DECONVOLUTION ...........................................................
    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

    # (2) Rescale/upscale datafile and THEN remove border region
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    # (i) the accurate image center must be taken from the upscaled image
    xc,yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    # (ii) then the borders can be removed with respect to the center
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size*R,xc,yc)        
    # (This procedure is necessary to center the images precisely.
    # (The accurate centers from upscaled images are saved in database.
    # (Some border region should ALWAYS be cut, for two reasons:
    # (i) weak/zero diffractions at edges and (ii) detector edge artifacts
    
    # (3) Deconvolute using the external PSF
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
    
    # Return deconvolved image ................................................
    return arr


def deconvolution_type2(datafile, SDATA, DIFFIMAGES, psf, iterate):
    """
    Sum datafiles with Richardson-Lucy deconvolution using a 2D-PSF of type 2.

    Parameters
    ----------
    datafile : Datafile
        Datafile information.
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        Object describing the diffraction images/patterns.
    psf : 2D-numpy array
        Array representing the 2D-PSF function.
    iterate : int
        Number of iterations during the deconvolution.

    Returns
    -------
    arr : 2D numpy array
        The sum of datafiles with Richardson-Lucy deconvolution.
    
    Technical notes
    ---------------
    The parameters are transferred from the `sum_datafiles` function.
    Deconvolution type 2 refers to Richardson-Lucy deconvolution
    using a 2D-PSF estimated from the central region of each datafile.
    """
    
    # Prepare variables .......................................................
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    psf_size = DIFFIMAGES.psfsize
    
    # SUM DATAFILES ...........................................................
    # (we sum datafiles cut, rescaled, and deconvoluted
    # (rescaling DURING the summation => smoother deconvolution function
    
    # Sum with deconvolution, PSF from center of image ........................
    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name) 
    
    # (2) Rescale/upscale datafile and THEN remove border region
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)

    # (i) the accurate image center must be taken from the upscaled image
    xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
    
    # (ii) then the borders can be removed with respect to the center
    arr = stemdiff.io.Arrays.remove_edges(arr,img_size*R,xc,yc)        

    # (This procedure is necessary to center the images precisely.
    # (The accurate centers from upscaled images are saved in database.
    # (Some border region should ALWAYS be cut, for two reasons:
    # (i) weak/zero diffractions at edges and (ii) detector edge artifacts
    
    # (3) Prepare PSF from the center of given array
    psf = stemdiff.psf.PSFtype2.get_psf(arr, psf_size, circular=True)
    
    # (4) DECONVOLUTION
    # (a) save np.max, normalize
    norm_const = np.max(arr)
    arr_norm = arr/np.max(arr)
    psf_norm = psf/np.max(psf)
    # (reason: deconvolution algorithm requires normalized arrays...
    # (...and we save original max.intensity to re-normalize the result

    # (b) perform the deconvolution
    arr_deconv = restoration.richardson_lucy(
        arr_norm, psf_norm, num_iter=iterate)
    
    # (c) restore original range of intensities = re-normalize
    arr = arr_deconv * norm_const

    return(arr)


