'''
Module: stemdiff.summ
---------------------
The summation of 4D-STEM datafiles to create one 2D powder diffraction file.

* The summation runs on all available cores (parallel processing).
* This module takes functions from semdiff.sum, but runs them parallelly. 

The key function of the module (for a user) = stemdiff.summ.sum_datafiles:
                  
* The function takes the same arguments as stemdiff.sum.sum_datafiles.
* The only difference consists in that  the summation runs on multiple cores.

How it works:

* This module contains just two functions:
    - `summ.sum_datafiles` - wrapper for the next function
    - `summ.multicore_sum` - runs the summation on multiple cores
* The rest is done with the functions of sister module *stemdiff.sum*.
    - i.e. the summ.multicore_sum calls functions from stemdiff.sum
    - but the functions run within this module, using multiple cores
* Summary:
    - `sum.sum_datafiles`  - runs on a single core (docs in stemdiff.sum)
    - `summ.sum_datafiles` - runs on multiple cores, aguments are identical 
'''

import os
import sys
import tqdm
import stemdiff.sum
import concurrent.futures as future
import idiff


def sum_datafiles(SDATA, DIFFIMAGES, df_sum, df_psf=None, bkg=0, deconv=False, 
                  peaks=False, iterate=10, nn_path=None):
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
        if the datafiles are pre-filtered, we get the sum of filtered datafiles,
        if PSF is given, we get the sum of datafiles with PSF deconvolution.
    
    Technical notes
    ---------------
    * This function is a wrapper.
    * It calls stemdiff.summ.multicore_sum with correct arguments:
        - all relevant original arguments
        - one additional argument: the *function for summation*
        - the *function for summation* depends on the deconvolution type
    '''
    
    # (0) Initialize
    num_workers = os.cpu_count()  # Number of concurrent workers
    datafiles = [datafile[1] for datafile in df_sum.iterrows()] 



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
    
    # (1) Use ThreadPool to perform multicore summation  
    with future.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # (a) Prepare variables
        futures = []
        total_tasks = len(datafiles)
        # (b) Submit tasks to the executor            
        for i, file in enumerate(datafiles): 
            # try:
                func = stemdiff.sum.prepare_dfile
                future_obj = executor.submit(func, SDATA, DIFFIMAGES, file, 
                                             psf, bkg, deconv, peaks, iterate,
                                             nn)
                futures.append(future_obj)
            # except Exception as e:
            #     print(f"Error processing file {file}: {str(e)}")
        # (c) Use tqdm to create a progress bar
        stderr_original = sys.stderr
        sys.stderr = sys.stdout
        with tqdm.tqdm(total=total_tasks, 
                       desc="Processing ") as pbar:
            # ...wait for all tasks to complete
            for future_obj in future.as_completed(futures):
                try:
                    future_obj.result()
                except Exception as e:
                    print(f"Error processing a task: {str(e)}")
                pbar.update(1)
            sys.stderr = stderr_original
    
    # (2) Summation done, collect the results
    # (a) Print a new line to complete the progress bar
    print()
    # (b) Collect results
    deconvolved_data = [f.result() for f in futures]
    
    # (3) Results collected, perform post-processing
    # (a) Sum results = the processed/deconvolved files from previous steps
    sum_arr = sum(deconvolved_data)    
    # (b) Run post-processing routine = normalization, 
    final_arr = stemdiff.sum.sum_postprocess(sum_arr,len(deconvolved_data))
    
    # (4) Return final array = sum of datafiles with (optional) deconvolution
    return(final_arr)