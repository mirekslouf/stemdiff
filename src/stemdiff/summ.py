'''
Module: stemdiff.summ
---------------------
The summation of 4D-STEM datafiles to create one 2D powder diffraction file.

* The summation runs on all available cores (parallel processing).
* This module takes functions from semdiff.sum, but runs them in parallel. 

The key function of the module (for a user) = stemdiff.summ.sum_datafiles:
                  
* The function takes the same arguments as stemdiff.sum.sum_datafiles.
* The only difference consists in that  the summation runs on multiple cores.

How it works:

* This module contains just one function:
    - `summ.sum_datafiles` - runs the summation on multiple cores
* The rest is done with the functions of sister module *stemdiff.sum*.
    - i.e. the summ.multicore_sum calls functions from stemdiff.sum
    - but the functions run within this module, using multiple cores
* Summary:
    - `sum.sum_datafiles`  - runs on a single core (docs in stemdiff.sum)
    - `summ.sum_datafiles` - runs on multiple cores, arguments are identical 
'''

import os
import sys
import tqdm
import stemdiff.sum
import concurrent.futures as future
import idiff
import numpy as np

def sum_datafiles(SDATA, DIFFIMAGES, df_sum, df_psf=None, bkg=0, bkgp={}, 
                  deconv=0, deconvp={"num_iter": 10}, peaks=0, peaksp={},
                  center=None, centerp={}):
    '''
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
        ['DatafileName', 'Xcenter', 'Ycenter', 'MaxInt', 'Peaks', 'S'].

    df_psf : pandas.DataFrame object, optional
        Database with datafiles to calculate PSF (same column structure as
        df_sum). If None, PSF is used from deconvp['psf'] if given, or
        otherwise calculated independently from the central region of each
        individual datafile (see the priority order under `deconv` below).

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
        Parameters for the chosen background-subtraction method -- see the
        per-method breakdown under `bkg` above. Passed as **kwargs directly
        to the corresponding `idiff.bkg2d` function/class.

    deconv : int, optional, default is 0
        Deconvolution method:
        * 0 = no deconvolution
        * 1 = Richardson-Lucy from skimage
        * 2 = Richardson-Lucy from idiff -- deconvp: {num_iter=30, cuda=False}
        * 3 = Richardson-Lucy from idiff with Tikhonov regularization
        * 4 = Richardson-Lucy from idiff with L1 regularization

        Which PSF is used is determined by this priority order:
        1. `deconvp['psf']` -- used directly (after normalization) if given
        2. `df_psf` -- used to calculate a single shared PSF, if given
        3. otherwise -- the central region of each array (after background
           subtraction) is used as that array's own individual PSF

    deconvp : dict, optional, default is {"num_iter": 10}
        Parameters for the chosen deconvolution method -- see `deconv`
        above. `num_iter` controls iteration count for all Richardson-Lucy
        variants; `psf`, if supplied here, takes priority as described above.

    peaks : int, optional, default is 0
        Peak-detection method to run on the processed pattern; every
        detected peak is replaced with a single pixel carrying the summed
        intensity of that peak. `run_regions` or `run_log` are recommended
        (more robust to the typically low peak counts/high noise of
        individual 4D-STEM frames than `run_doh`/`run_pcbr`).
        * 0 = no peak detection
        * 1 = `idiff.peaks.run_regions` -- peaksp: {threshold=1}
        * 2 = `idiff.peaks.run_log` -- peaksp: {downsample_factor=4,
          threshold_factor=3.5, peak_min_sigma=6, ...}
        * 3 = `idiff.peaks.run_doh` -- peaksp: {downsample_factor=4,
          threshold_factor=3.5, peak_min_sigma=6, ...}
        * 4 = `idiff.peaks.run_pcbr` -- peaksp: {sigma=3.0,
          lambda_thresh=0.5, response_thresh_rel=0.1, ...}

    peaksp : dict, optional, default is {}
        Parameters for the chosen peak-detection method -- see `peaks` above.

    center : str or None, optional, default is None
        Detection method for `ediff.center.CenterLocator`'s `detection`
        argument, run independently for each image. If None, use the
        (xc, yc) center already stored in df_sum instead of re-detecting it.

    centerp : dict, optional, default is {}
        Parameters for the chosen center-detection method -- see `ediff.
        center.CenterLocator` for the full set of options per method.

    Returns
    -------
    final_arr : 2D numpy array, dtype float32
        Sum of the (optionally background-subtracted, deconvolved,
        peak-filtered) datafiles. Shape is (imgsize*R, imgsize*R), where R
        is SDATA.detector.upscale -- i.e. the *upscaled* resolution, not
        DIFFIMAGES.imgsize itself.

    '''
    
    # (0) Initialize
    num_workers = os.cpu_count()  # Number of concurrent workers
    datafiles = [datafile[1] for datafile in df_sum.iterrows()] 
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    sum_arr = np.zeros((img_size * R, img_size * R), dtype=np.float32)
    n_arr_summed = 0

    if bkg == 4:
        nn = idiff.bkg2d.NeuralNetwork(**bkgp)
    else:
        nn = None

    if df_psf is not None and "psf" not in deconvp:
        psf = idiff.psf.PSFtype1.get_psf(SDATA, DIFFIMAGES, df_psf)
        deconvp["psf"] = psf
    
    # (1) Use ThreadPool to perform multicore summation
    with future.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # (a) Prepare variables
        futures = set()
        total_tasks = len(datafiles)
        # (b) Submit tasks to the executor            
        for i, file in enumerate(datafiles): 
            try:
                func = stemdiff.sum.prepare_dfile
                future_obj = executor.submit(func, SDATA, DIFFIMAGES, file, bkg,
                                             bkgp, deconv, deconvp, peaks,
                                             peaksp, nn, center, centerp)
                futures.add(future_obj)
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
        # (c) Use tqdm to create a progress bar
        stderr_original = sys.stderr
        sys.stderr = sys.stdout
        with tqdm.tqdm(total=total_tasks, 
                       desc="Processing ") as pbar:
            # ...wait for tasks to complete
            for future_obj in future.as_completed(futures):
                # (d) Obtain the processed array and add it to the result
                try:
                    sum_arr += future_obj.result()
                    n_arr_summed += 1
                except Exception as e:
                    print(f"Error processing a task: {str(e)}")
                finally:
                    # Remove future from the set to free memory
                    futures.remove(future_obj)
                    # Clear the local loop variable
                    del future_obj 
                pbar.update(1)
            sys.stderr = stderr_original
    
    # Print a new line to complete the progress bar
    print()
    
    # (2) Perform post-processing
    final_arr = stemdiff.sum.sum_postprocess(sum_arr, n_arr_summed)
    
    # (3) Return final array = sum of datafiles with (optional) deconvolution
    return final_arr