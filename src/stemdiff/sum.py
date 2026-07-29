'''
Module: stemdiff.sum
--------------------
The summation of 4D-STEM datafiles to create one 2D powder diffraction file.

* stemdiff.sum = this module, which runs on a single core (serial processing)
* stemdiff.summ = sister module running on multiple cores (parallel processing)

To perform the summation, we just call function sum_datafiles:

* serial   : stemdiff.sum.sum_datafiles(SDATA, DIFFIMAGES, df, ...)
* parallel : stemdiff.summ.sum_datafiles(SDATA, DIFFIMAGES, df, ...)

The initial arguments are:

* SDATA = stemdiff.gvars.SourceData object = description of source data
* DIFFIMAGES = stemdiff.gvars.DiffImages object = description of diffractograms
* df_sum = pre-calculated database of datafiles/diffratograms to sum

Key arguments are `deconv` and `bkg`, which determine the processing type.
'''


import numpy as np
import stemdiff.io
import idiff
import ediff.center
from skimage import restoration
import tqdm
import sys


def sum_datafiles(SDATA, DIFFIMAGES, df_sum, df_psf=None, bkg=0, bkgp={}, 
                  deconv=0, deconvp={"num_iter": 10}, peaks=0, peaksp={},
                  center=None, centerp={}):
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
        If None, PSF is used from deconvp or calculated from central region
        of each datafile.
    bkg : int, optional, default is 0
        Use background subtraction from `idiff.bkg2d`.
        Background subtraction type:
        * 0 = no background subtraction,
        * 1 = rolling ball,
        * 2 = tophat,
        * 3 = gaussian,
        * 4 = neural network.

        Neural network needs `path` argument.
    bkgp : dictionary, optional, default is {}
        Parameters for the background subtraction method.
    deconv : int, optional, default is 0
        Use deconvolution.
        Deconvolution type:
        * 0 = no deconvolution,
        * 1 = Richardson-Lucy deconvolution from skimage,
        * 2 = Richardson-Lucy deconvolution from idiff,
        * 3 = Richardson-Lucy deconvolution from idiff with 
        Tikhonov regularization,
        * 4 = Richardson-Lucy deconvolution from idiff with 
        L1 regularization.

        Which PSF is used for the deconvolution is determined by this order: 
        1. `"psf"` argument in deconvp - this array is directly used as PSF
        after normalization
        2. `df_psf` parameter is used to calculate the PSF
        3. central region (after bkg subtraction) of each array is used as PSF 
        (every array has its own individual PSF)

    deconvp : dictionary, optional, default is {"num_iter": 10}
        Parameters for the deconvolution, default uses 10 iterations.
    peaks : int, optional, default is 0
        Run peaks detection algorithm on the processed NBD pattern. 
        Every peak is replaced with a single pixel with intensity equal to
        the sum of the peaks intensities.
        Possible values:
        * 0 = no peaks detection
        * 1 = `idiff.peaks.run_regions`
        * 2 = `idiff.peaks.run_log`
        * 3 = `idiff.peaks.run_doh`
        * 4 = `idiff.peaks.run_pcbr`

        `run_regions` or `run_log` are recommended.

    peaksp : dictionary, optional, default is {}
        Parameters for the peaks detection method.
    center : string or None, optional, default is None
        Detect center for each image. If None, use the centers from the 
        database. For possible values refer to `ediff.center.CenterLocator`.
    centerp : dictionary, optional, default is {}
        Parameters for the center detection method

    Returns
    -------
    final_arr : 2D numpy array
        The array is a sum of datafiles;
        if the datafiles are pre-filtered,
        we get the sum of filtered datafiles.
        Additional arguments determine the (optional) processing.

    Technical notes
    ---------------
    * This function works as a signpost.
    * It reads the summation parameters and calls a more specific summation 
    functions (which are NOT called directly by the end-user).
    * It employs progress bar, handles possible exceptions,
    and returns the final array (= post-processed and normalized array).
    """

    # (1) Prepare variables for summation 
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafiles = [datafile[1] for datafile in df_sum.iterrows()] 
    sum_arr = np.zeros((img_size * R, img_size * R), dtype=np.float32)

    if bkg == 4:
        nn = idiff.bkg2d.NeuralNetwork(**bkgp)
    else:
        nn = None

    if df_psf is not None and "psf" not in deconvp:
        psf = idiff.psf.PSFtype1.get_psf(SDATA, DIFFIMAGES, df_psf)
        deconvp["psf"] = psf

    # (2) Prepare variables for tqdm
    # (to create a single progress bar for the entire process
    total_tasks = len(datafiles)
    stderr_original = sys.stderr
    sys.stderr = sys.stdout

    # (3) Run summations
    # (summations will run with tqdm
    # (we will use several types of summations
    # (each summations uses datafiles prepared in a different way
    with tqdm.tqdm(total=total_tasks, desc="Processing ") as pbar:
        try:
            # Process each image in the database
            for index, datafile in df_sum.iterrows():
                sum_arr += prepare_dfile(SDATA, DIFFIMAGES, datafile, bkg,
                                         bkgp, deconv, deconvp, peaks, peaksp,
                                         nn, center, centerp)
                
                # Update the progress bar for each processed image
                pbar.update(1)
        except Exception as e:
            print(f"Error processing a task: {str(e)}")
            
    sys.stderr = stderr_original

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

    
def prepare_dfile(SDATA, DIFFIMAGES, datafile, bkg, bkgp, deconv, deconvp,
                  peaks, peaksp, nn, center_detection, centerp):
    """
    Prepare datafile for summation.
    This function is not supposed to be used directly.
    Use `sum_datafiles`.

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
        The datafile in the form of the array.
    
    Notes
    -----
    * The parameters are transferred from the `sum_datafiles` function.
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
        arr = idiff.bkg2d.rolling_ball(arr, **bkgp)
    elif bkg == 2:
        arr = idiff.bkg2d.tophat(arr, **bkgp)
    elif bkg == 3:
        arr = idiff.bkg2d.gaussian(arr, **bkgp)
    elif bkg == 4:
        arr = nn.predict(arr)
    
    # (3) Rescale/upscale datafile and THEN remove border region
    # (a) upscale datafile
    arr = stemdiff.io.Arrays.rescale_fast(arr, R, inter=2)
    arr = stemdiff.io.Arrays.zero_spatial_edges(arr, border_width=10)
    # (b) get the accurate center of the upscaled datafile
    # (the center coordinates for each datafile are saved in the database
    # (note: our datafile is one row from the database => we know the coords!
    # optionally, if user requests, run the provided center detection
    if center_detection == None:
        xc, yc = (round(datafile.Xcenter),round(datafile.Ycenter))
    elif center_detection == "intensity":
        center_locator = ediff.center.IntensityCenter()
        xc, yc = center_locator.center_of_intensity(arr, **centerp)
        xc, yc = round(xc), round(yc)
    else:
        center = ediff.center.CenterLocator(
            arr, center_detection, final_print=False, **centerp)
        xc, yc = round(center.x), round(center.y)
    # (c) finally, recenter the image and zero the edges
    arr = stemdiff.io.Arrays.recenter(arr, xc, yc)
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
    if deconv > 0 and "psf" not in deconvp:
        psf = idiff.psf.PSFtype2.get_psf(arr, psf_size, circular=True)
    elif deconv > 0:
        psf = deconvp["psf"] # get psf for the next step
    

    # (5) Deconvolution
    if deconv > 0:
        # (a) save np.max, normalize
        # (reason: deconvolution algorithm requires normalized arrays...
        # (...and we save original max.intensity to re-normalize the result
        norm_const = np.max(arr)
        arr_norm = arr/norm_const

        # PSF sould sum to 1
        psf_norm = psf/np.sum(psf)

        deconvp = deconvp.copy() # avoid alteration for next iterations
        deconvp.pop("psf", None) # remove psf from deconvp

    # (b) perform the deconvolution
    if deconv == 1:
        deconvp["psf"] = psf_norm # add normalized psf to the arguments
        arr_deconv = restoration.richardson_lucy(arr_norm, **deconvp)
    elif deconv == 2:
        rl = idiff.deconv.RichardsonLucy(**deconvp)
        arr_deconv = rl.deconvRL(arr_norm, psf_norm)
    elif deconv == 3:
        rl = idiff.deconv.RichardsonLucy(**deconvp)
        arr_deconv = rl.deconvRLTM(arr_norm, psf_norm)
    elif deconv == 4:
        rl = idiff.deconv.RichardsonLucy(**deconvp)
        arr_deconv = rl.deconvRLTV(arr_norm, psf_norm)

    # (c) restore original range of intensities = re-normalize
    if deconv > 0:
        arr = arr_deconv * norm_const

    # (6) Detect peaks
    if peaks == 1:
        arr = idiff.peaks.run_regions(arr, **peaksp)
    elif peaks == 2:
        arr = idiff.peaks._run_log(arr, **peaksp)
    elif peaks == 3:
        arr = idiff.peaks._run_doh(arr, **peaksp)
    elif peaks == 4:
        arr = idiff.peaks._run_pcbr(arr, **peaksp)

    # (7) Return the datafile as an array that is ready for summation
    return arr