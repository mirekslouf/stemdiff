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
        [filename, xc, yc, MaxInt, NumPeaks, S].
    df_psf : pandas.DataFrame object, optional
        Database with datafiles to calculate PSF.
        If None, PSF is used from deconvp or calculated from central region
        of each datafile.
    bkg : int, optional, default is 0
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
        PSF priority: 
        
        1. `"psf"` argument in deconvp - this array is directly use as PSF
        after normalization
        2. `df_psf` parameter is used to calculate the PSF
        3. central region (after bkg subtraction) of each array is used as PSF 
        (every array has its own individual PSF)

        Deconvolution type:
        * 0 = no deconvolution,
        * 1 = Richardson-Lucy deconvolution.
    deconvp : dictionary, optional, default is {"num_iter": 10}
        Parameters for the deconvolution, default uses 10 iterations.
    peaks : int, optional, default is 0
        Possible values:
        * 0 = no peaks detection
        * 1 = idiff.peaks.run_regions
        * 2 = idiff.peaks._run_log
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
    * This function is a wrapper.
    * It calls stemdiff.sum.prepare_dfile in parallel.
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