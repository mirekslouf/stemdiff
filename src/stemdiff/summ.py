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


def sum_datafiles(
        SDATA, DIFFIMAGES,
        df, deconv=0, psf=None, iterate=10):
    '''
    Sum datafiles from a 4D-STEM dataset to get 2D powder diffractogram.
    
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
    psf : 2D-numpy array or None, optional, default is None
        Array representing 2D-PSF function.
        Relevant only for deconv = 1.
    iterate : integer, optional, default is 10  
        Number of iterations during the deconvolution.
        
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
    
    if deconv == 0:
        arr = multicore_sum(
            SDATA, DIFFIMAGES, df, psf, iterate,
            func = stemdiff.sum.dfile_without_deconvolution)
    elif deconv == 1:
        arr = multicore_sum(
            SDATA, DIFFIMAGES, df, psf, iterate,
            func = stemdiff.sum.dfile_with_deconvolution_type1)
    elif deconv == 2:
        arr = multicore_sum(
            SDATA, DIFFIMAGES, df, psf, iterate,
            func = stemdiff.sum.dfile_with_deconvolution_type2)
    else:
        print(f'Unknown deconvolution type: deconv={deconv}')
        print('Nothing to do.')
        return None
    return arr


def multicore_sum(SDATA, DIFFIMAGES, df, psf, iterate, func):
    '''
    Execute concurrent data processing using a thread pool.

    This function processes a list of datafiles using a thread pool 
    for parallel execution. The number of concurrent workers is determined 
    by subtracting 1 from the available CPU cores.

    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describes source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        Object describing the diffraction images/patterns.
    df : pandas.DataFrame object
        Database with datafile names and characteristics.
    psf : 2D-numpy array or None, optional, default is None
        Array representing 2D-PSF function.
        Relevant only for deconv = 1.
    iterate : integer, optional, default is 10  
        Number of iterations during the deconvolution.
    func : a function from stemdiff.sum module to be used for summation
        A function from sister module stemdiff.sum,
        which will be used for summation on multiple cores.
        This argument is (almost always) passed from the calling function
        stemdiff.summ.sum_datafiles so that it corresponded to
        the user-selected deconvolution type.
    
    Returns
    -------
    final_arr : 2D numpy array
        The array is a sum of datafiles;
        if the datafiles are pre-filtered, we get sum of filtered datafiles,
        if PSF is given, we get sum of datafiles with PSF deconvolution.
    
    Technical notes
    ---------------
    * This function is NOT to be called directly.
    * It is called by wrapper function stemdiff.summ.sum_datafiles.
    * The two functions work as follows:
        - calling function = stemdiff.summ.sum_datafiles
            - passes all relevant arguments including function for summation
        - this function = stemdiff.summ.multicore_sum
            - runs the summation on multiple cores and returns the result
    '''
    
    # (0) Initialize
    num_workers = os.cpu_count()  # Number of concurrent workers
    datafiles = [datafile[1] for datafile in df.iterrows()] 
    
    # (1) Use ThreadPool to perform multicore summation  
    with future.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # (a) Prepare variables
        futures = []
        total_tasks = len(datafiles)
        # (b) Submit tasks to the executor            
        for i, file in enumerate(datafiles): 
            try:
                if func == stemdiff.sum.dfile_without_deconvolution:
                    future_obj = executor.submit(
                        func, SDATA, DIFFIMAGES, file)
                elif func == stemdiff.sum.dfile_with_deconvolution_type1:
                    future_obj = executor.submit(
                        func, SDATA, DIFFIMAGES, file, psf, iterate)
                elif func == stemdiff.sum.dfile_with_deconvolution_type2:
                    future_obj = executor.submit(
                        func, SDATA, DIFFIMAGES, file, iterate)
                else:
                    raise Exception("Uknown deconvolution function!")
                futures.append(future_obj)
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
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
