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


def sum_datafiles(
        SDATA, DIFFIMAGES,
        df, deconv=0, iterate=10, psf=None, cake=None, subtract=None):
    '''
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
        3 = deconvolution based on PSF from whole datafile.    
    iterate : integer, optional, default is 10  
        Number of iterations during the deconvolution.
    psf : 2D-numpy array or None, optional, default is None
        Array representing 2D-PSF function.
        Relevant only for deconv = 1.
        
    Returns
    -------
    final_arr : 2D numpy array
        The array is a sum of datafiles;
        if the datafiles are pre-filtered, we get sum of filtered datafiles,
        if PSF is given, we get sum of datafiles with PSF deconvolution.
    
    Technical notes
    ---------------
    This function works as signpost.
    It reads the summation parameters and
    calls more specific summation function.
    '''
    if deconv == 0:
        arr = sum_without_deconvolution(
            SDATA, DIFFIMAGES, df)
    elif deconv == 1:
        arr = sum_with_deconvolution_type1( 
            SDATA, DIFFIMAGES, df, psf, iterate)
    elif deconv == 2:
        arr = sum_with_deconvolution_type2(
            SDATA, DIFFIMAGES, df, iterate)
    elif deconv == 3:
        arr = sum_with_deconvolution_type3(
            SDATA, DIFFIMAGES, df, iterate, cake, subtract)
    else:
        print(f'Unknown deconvolution type: deconv={deconv}')
        print('Nothing to do.')
        return(None)
    return(arr)

def sum_without_deconvolution(SDATA, DIFFIMAGES, df):
    '''
    Sum datafiles wihtout deconvolution.

    * Parameters of the function:
        - This function is usually called from stemdiff.sum.sum_files.
        - The parameters are transferred from the sum_files function
'''
    # Prepare variables
    n = 0  # number of summed datafiles
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    # Prepare array for summation
    # (for the precise center determination, we must sum upscaled arrays
    sum_arr   = np.zeros((img_size*R,img_size*R), dtype=np.float32)
    # Sum datafiles
    for index,datafile in df.iterrows():
        # Read datafile to array
        datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
        arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
        # Rescale/upscale datafile and THEN remove border region
        # (i) the accurate image center must be taken from the upscaled image
        # (ii) then the borders can be removed with respect to the center
        # (This procedure is necessary to center the images precisely.
        # (The accurate centers from upscaled images are saved in database.
        # (Some border region should ALWAYS be cut, for two reasons:
        # (i) weak/zero diffractions at edges and (ii) detector edge artifacts
        arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
        xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
        arr = stemdiff.io.Arrays.remove_edges(arr,img_size*R,xc,yc)
        # Add current datafile/array to summation
        sum_arr += arr
        # Update n = number of summed arrays
        n += 1
    # Prepare final array = (1) normalize summation and (2) convert to final
    sum_arr = sum_arr/n
    final_arr = np.round(sum_arr).astype(np.uint16)
    # Return the final array
    return(final_arr)

def sum_with_deconvolution_type1(SDATA, DIFFIMAGES, df, psf, iterate):
    '''
    Sum datafiles with 2D-PSF deconvolution of type1.
    
    This function is usually called from stemdiff.sum.sum_datafiles.
    For argument description see the abovementioned function.
    
    * What is deconvolution type1:
        - Richardson-Lucy deconvolution.
        - 2D-PSF function estimated from files with negligible diffractions.
        - Therefore, the 2D-PSF function is the same for all summed datafiles.
    * Parameters of the function:
        - This function is usually called from stemdiff.sum.sum_files.
        - The parameters are transferred from the sum_files function
    '''
    # Prepare variables .......................................................
    n = 0
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    # Prepare array for summation
    # (for better precision, we use deconvolution on rescaled/upscaled array
    sum_arr = np.zeros((img_size*R,img_size*R), dtype=np.float32)
    # Sum datafiles
    # (we sum datafiles cut, rescaled, and deconvoluted
    # (rescaling DURING the summation => smoother deconvolution function
    # Sum with deconvolution, external PSF from low-diffracting files .........
    for index,datafile in df.iterrows():
        # (0) Simple progress indicator
        print('.', end='')
        # (1) Read datafile
        datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
        arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
        # (2) Rescale/upscale datafile and THEN remove border region
        # (i) the accurate image center must be taken from the upscaled image
        # (ii) then the borders can be removed with respect to the center
        # (This procedure is necessary to center the images precisely.
        # (The accurate centers from upscaled images are saved in database.
        # (Some border region should ALWAYS be cut, for two reasons:
        # (i) weak/zero diffractions at edges and (ii) detector edge artifacts
        arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
        xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
        arr = stemdiff.io.Arrays.remove_edges(arr,img_size*R,xc,yc)        
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
        # (4) Add rescaled and deconvoluted array to summation
        sum_arr += arr
        # (5) Updated n = number of summed arrays
        n += 1
    # Finalize and return result ..............................................
    # (a) terminate simple progress indicator
    print('End')
    # (b) normalize the final array
    sum_arr = sum_arr/n
    # (c) convert to final array with integer values
    # (why integer values? => arr with int's can be plotted as image and saved
    final_arr = np.round(sum_arr).astype(np.uint16)
    # (d) return the finalized array
    return(final_arr)


def sum_with_deconvolution_type2(SDATA, DIFFIMAGES, df, iterate):
    '''
    Sum datafiles with 2D-PSF deconvolution of type2.
    
    This function is usually called from stemdiff.sum.sum_datafiles.
    For argument description see the abovementioned function.

    * What is deconvolution type2:
        - Richardson-Lucy deconvolution.
        - The 2D-PSF function estimated from central region of each datafile.
    * Parameters of the function:
        - This function is usually called from stemdiff.sum.sum_files.
        - The parameters are transferred from the sum_files function
    '''
    # Prepare variables .......................................................
    n = 0
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    psf_size = DIFFIMAGES.psfsize
    # Prepare array for summation
    # (for better precision, we use deconvolution on rescaled/upscaled array
    sum_arr = np.zeros((img_size*R,img_size*R), dtype=np.float32)
    # Sum datafiles
    # (we sum datafiles cut, rescaled, and deconvoluted
    # (rescaling DURING the summation => smoother deconvolution function
    # Sum with deconvolution, PSF from center of image ........................
    for index,datafile in df.iterrows():
        # (0) Simple progress indicator
        print('.', end='')
        # (1) Read datafile
        datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
        arr = stemdiff.io.Datafiles.read(SDATA, datafile_name) 
        # (2) Rescale/upscale datafile and THEN remove border region
        # (i) the accurate image center must be taken from the upscaled image
        # (ii) then the borders can be removed with respect to the center
        # (This procedure is necessary to center the images precisely.
        # (The accurate centers from upscaled images are saved in database.
        # (Some border region should ALWAYS be cut, for two reasons:
        # (i) weak/zero diffractions at edges and (ii) detector edge artifacts
        arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
        xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
        arr = stemdiff.io.Arrays.remove_edges(arr,img_size*R,xc,yc)        
        # (3) Prepare PSF from the center of given array
        psf = stemdiff.psf.PSFtype2.get_psf(arr, psf_size, circular=True)
        # (4) Deconvolute
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
        # (5) Add rescaled and deconvoluted array to summation
        sum_arr += arr
        # (6) Updated n = number of summed arrays
        n += 1
    # Finalize and return result ..............................................
    # (a) terminate simple progress indicator
    print('End')
    # (b) normalize the final array
    sum_arr = sum_arr/n
    # (c) convert to final array with integer values
    # (why integer values? => arr with int's can be plotted as image and saved
    final_arr = np.round(sum_arr).astype(np.uint16)
    # (d) return the finalized array
    return(final_arr)

def sum_with_deconvolution_type3(
        SDATA, DIFFIMAGES, df, iterate, cake, subtract):
    '''
    Sum datafiles with 2D-PSF deconvolution of type3.
    
    * What deconvolution type3:
        - Richardson-Lucy deconvolution.
        - The 2D-PSF function is estimated from each (whole) datafile.
        - The diffractions in 2D-PSF are removed by means of "cake method".
    * Parameters of the function:
        - This function is usually called from stemdiff.sum.sum_files.
        - The parameters are transferred from the sum_files function
    '''
    
    # Prepare variables .......................................................
    n = 0
    R = SDATA.detector.upscale
    # ! img_size and psf_size must by multiplied by R wherever relevant
    img_size = DIFFIMAGES.imgsize 
    psf_size = DIFFIMAGES.psfsize
    # Prepare array for summation
    # (for better precision, we use deconvolution on rescaled/upscaled array
    sum_arr = np.zeros((img_size*R,img_size*R), dtype=np.float32)
    # Sum datafiles
    # (we sum datafiles cut, rescaled, and deconvoluted
    # (rescaling DURING the summation => smoother deconvolution function
    # Sum with deconvolution, PSF from center of image ........................
    for index,datafile in df.iterrows():
        # (0) Simple progress indicator
        print('.', end='')
        # (1) Read datafile
        datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
        arr = stemdiff.io.Datafiles.read(SDATA, datafile_name) 
        # (2) Rescale/upscale datafile and THEN remove border region
        # (i) the accurate image center must be taken from the upscaled image
        # (ii) then the borders can be removed with respect to the center
        # (This procedure is necessary to center the images precisely.
        # (The accurate centers from upscaled images are saved in database.
        # (Some border region should ALWAYS be cut, for two reasons:
        # (i) weak/zero diffractions at edges and (ii) detector edge artifacts
        arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
        xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
        arr = stemdiff.io.Arrays.remove_edges(arr,img_size*R,xc,yc)        
        # (3) Prepare PSF from the center of given array
        # ! psf_size must by multiplied by R
        psf = stemdiff.psf.PSFtype3.get_psf(arr, psf_size*R, cake, subtract)   
        if subtract:    
            # Individual background (PSF) subtraction
            arr = arr - psf
            # All negative values shoud go to zero!
            # (the negative values have result in many side effects and errors!
            arr = np.where(arr < 0, 0, arr)
        # (4) Deconvolute
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
        # (5) Add rescaled and deconvoluted array to summation
        sum_arr += arr
        # (6) Updated n = number of summed arrays
        n += 1
    # Finalize and return result ..............................................
    # (a) terminate simple progress indicator
    print('End')
    # (b) normalize the final array
    sum_arr = sum_arr/n
    # (c) convert to final array with integer values
    # (why integer values? => arr with int's can be plotted as image and saved
    final_arr = np.round(sum_arr).astype(np.uint16)
    # (d) return the finalized array
    return(final_arr)

def sum_with_deconvolution_type4(
        DETECTOR, DATAFILES, DIFFIMAGES, df, iterate):
    '''
    Sum datafiles with 2D-PSF deconvolution of type4.
    
    * What is deconvolution type4: 
        - Richardson-Lucy deconvolution.
        - The 2D-PSF function estimated from whole datafile (~type3).
        - The 2D-PSF subtracted from the datafile (background removal).
        - Final deconvolution with 2D-PSF from central region (~type2).
    * Parameters of the function:
        - This function is usually called from stemdiff.sum.sum_files.
        - The parameters are transferred from the sum_files function
    '''
    pass
    

