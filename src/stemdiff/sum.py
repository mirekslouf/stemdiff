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

    Serial (single-threaded) version -- processes datafiles one at a time
    in a plain loop. Use this when a stable, easy-to-debug/step-through
    reference implementation is preferred over speed. For the same
    functionality running in parallel across multiple threads, see the
    corresponding function in `stemdiff.summ`.

    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describes source data (detector, data_dir, filenames).

    DIFFIMAGES : stemdiff.gvars.DiffImages object
        Object describing the diffraction images/patterns.

    df_sum : pandas.DataFrame object
        Pre-calculated database with datafiles to be summed.
        Each row of the database contains
        ['DatafileName', 'Xcenter', 'Ycenter', 'MaxInt', 'Peaks', 'S'].

    df_psf : pandas.DataFrame object, optional
        Database with datafiles to calculate a single, shared PSF (same
        column structure as df_sum). If None and deconv > 0, a PSF is
        taken from deconvp['psf'] if given there, or otherwise computed
        independently for each datafile from its own central region
        (see the priority order under `deconv` below).

    bkg : int, optional, default is 0
        Background subtraction method, from `idiff.bkg2d`:
        * 0 = no background subtraction
        * 1 = rolling ball -- bkgp: {radius=20}
        * 2 = tophat -- bkgp: {thr=40, area_size=5, radius=2}
        * 3 = gaussian -- bkgp: {thr=20, area_size=5, sigma=2, normalize=False}
        * 4 = neural network -- bkgp: {path} (required: path to an .onnx
          model file; any other kwargs are passed through to
          `onnxruntime.InferenceSession`)

    bkgp : dict, optional, default is {}
        Parameters for the chosen background-subtraction method -- see
        the per-method breakdown under `bkg` above. Passed as **kwargs
        directly to the corresponding `idiff.bkg2d` function/class.

    deconv : int, optional, default is 0
        Deconvolution method:
        * 0 = no deconvolution
        * 1 = Richardson-Lucy from skimage
        * 2 = Richardson-Lucy from idiff -- deconvp: {num_iter=30, cuda=False}
        * 3 = Richardson-Lucy from idiff with Tikhonov regularization
        * 4 = Richardson-Lucy from idiff with L1 regularization

        Which PSF is used is determined by this priority order:
        1. `deconvp['psf']` -- used directly (after normalization) if given
        2. `df_psf` -- used to calculate one shared PSF for every datafile,
           if given
        3. otherwise -- the central region of each individual datafile
           (after background subtraction, before deconvolution) is used
           as that datafile's own PSF

    deconvp : dict, optional, default is {"num_iter": 10}
        Parameters for the chosen deconvolution method -- see `deconv`
        above. `num_iter` controls iteration count for all Richardson-Lucy
        variants; `psf`, if supplied here, takes priority as described above.

    peaks : int, optional, default is 0
        Run peak-detection on the processed pattern; every detected peak
        is replaced with a single pixel carrying the summed intensity of
        that peak.
        * 0 = no peak detection
        * 1 = `idiff.peaks.run_regions` -- peaksp: {threshold=1}
        * 2 = `idiff.peaks.run_log` -- peaksp: {downsample_factor=4,
          threshold_factor=3.5, peak_min_sigma=6, ...}
        * 3 = `idiff.peaks.run_doh` -- peaksp: {downsample_factor=4,
          threshold_factor=3.5, peak_min_sigma=6, ...}
        * 4 = `idiff.peaks.run_pcbr` -- peaksp: {sigma=3.0,
          lambda_thresh=0.5, response_thresh_rel=0.1, ...}

        `run_regions` or `run_log` are recommended.

    peaksp : dict, optional, default is {}
        Parameters for the chosen peak-detection method -- see `peaks` above.

    center : str or None, optional, default is None
        Center-detection method, re-run independently for each datafile.
        If None, use the (Xcenter, Ycenter) already stored for that file
        in df_sum instead of re-detecting it. The special value
        `"intensity"` routes to `ediff.center.IntensityCenter` (a
        lightweight intensity-centroid method); any other string is
        passed as the `detection` argument to the more general
        `ediff.center.CenterLocator`.

    centerp : dict, optional, default is {}
        Parameters for the chosen center-detection method -- passed as
        **kwargs to `IntensityCenter.center_of_intensity` (if
        center="intensity") or to `CenterLocator` (otherwise).

    Returns
    -------
    arr : 2D numpy array, dtype uint16
        Sum of the (optionally background-subtracted, deconvolved,
        peak-filtered) datafiles, normalized by the number of datafiles
        summed and rounded to the nearest integer (see
        `sum_postprocess`). Shape is (imgsize*R, imgsize*R), where R is
        SDATA.detector.upscale -- i.e. the *upscaled* resolution, not
        DIFFIMAGES.imgsize itself.
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
    sum_of_arrays : np.array, dtype float32
        Sum of the arrays -- usually from `stemdiff.sum.sum_datafiles`
        (its running `sum_arr` accumulator, before normalization).

    n : int
        Number of arrays the sum should be normalized by -- usually
        `len(df_sum)`, the number of datafiles requested. Note this is
        the *requested* count, not necessarily the number that actually
        succeeded and were added to `sum_of_arrays` (see the technical
        note in `sum_datafiles`).

    Returns
    -------
    arr : np.array, dtype uint16
        `sum_of_arrays / n`, rounded to the nearest integer and cast to
        uint16. Same shape as `sum_of_arrays`.
    """
    arr = np.round(sum_of_arrays/n).astype(np.uint16)
    return(arr)

    
def prepare_dfile(SDATA, DIFFIMAGES, datafile, bkg, bkgp, deconv, deconvp,
                  peaks, peaksp, nn, center_detection, centerp):
    """
    Read, process, and center a single datafile, ready for summation.

    Not intended to be called directly -- use `sum_datafiles`, which
    calls this once per datafile and passes all of these parameters
    through from its own same-named arguments.

    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).

    DIFFIMAGES : stemdiff.gvars.DiffImages object
        The object describing the diffraction images/patterns.

    datafile : one row from the prepared database of datafiles
        The database of datafiles is created in
        `stemdiff.dbase.calc_database`. Each row contains
        ['DatafileName', 'Xcenter', 'Ycenter', 'MaxInt', 'Peaks', 'S'].

    bkg, bkgp : int, dict
        Background-subtraction method and its parameters -- see
        `sum_datafiles` for the full per-method breakdown.

    deconv, deconvp : int, dict
        Deconvolution method and its parameters, and the PSF-selection
        priority order -- see `sum_datafiles` for details. Note:
        `deconvp` is copied internally before use, so the caller's own
        dict (and any shared `psf` entry in it) is not mutated between
        calls for different datafiles.

    peaks, peaksp : int, dict
        Peak-detection method and its parameters -- see `sum_datafiles`
        for the full per-method breakdown.

    nn : idiff.bkg2d.NeuralNetwork or None
        Pre-initialized neural network background-subtraction session,
        already loaded once by the caller (`sum_datafiles`) and reused
        across every datafile -- avoids reloading the ONNX model on
        every single call. Required (non-None) if bkg == 4; ignored
        otherwise.

    center_detection : str or None
        Center-detection method for this datafile. If None, use the
        already-stored (Xcenter, Ycenter) from `datafile` directly
        instead of re-detecting. `"intensity"` routes to
        `ediff.center.IntensityCenter`; any other string is passed as
        the `detection` argument to `ediff.center.CenterLocator`.

    centerp : dict
        Parameters for the chosen center-detection method -- passed as
        **kwargs to whichever of the two center-detection paths above
        is used.

    Returns
    -------
    arr : 2D numpy array
        The single datafile, upscaled, background-subtracted/
        deconvolved/peak-filtered as requested, and recentered -- ready
        to be added directly into the running sum.

    Notes
    -----
    * Processing order: read -> background subtraction -> upscale ->
      zero border edges -> determine center -> recenter -> (build PSF
      if needed) -> deconvolution -> peak detection.
    * Background subtraction happens *before* upscaling/recentering,
      but deconvolution's own PSF (when derived from the datafile's own
      central region, i.e. PSF priority option 3) is taken *after*
      recentering -- so that per-datafile PSF is always centered
      correctly regardless of where the diffraction pattern originally
      sat in the raw frame.
    * Centers are always taken from the *upscaled* image coordinates
      (never the original/non-upscaled datafile) -- using centers
      computed on non-upscaled data here would recenter incorrectly.
    * Some border region is always zeroed after upscaling, for two
      reasons: (i) weak/zero diffraction signal at the frame edges, and
      (ii) detector edge artifacts.
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
        arr = idiff.peaks.run_log(arr, **peaksp)
    elif peaks == 3:
        arr = idiff.peaks.run_doh(arr, **peaksp)
    elif peaks == 4:
        arr = idiff.peaks.run_pcbr(arr, **peaksp)

    # (7) Return the datafile as an array that is ready for summation
    return arr