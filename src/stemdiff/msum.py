
import numpy as np
import os
import concurrent.futures as future
import ssum 
import tqdm




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
       [ 3 = deconvolution based on PSF from whole datafile.]
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
        arr = run_sums(
            SDATA, DIFFIMAGES, df, psf, iterate, cake, subtract,
            func = ssum.no_deconvolution)
    elif deconv == 1:
        arr = run_sums(
            SDATA, DIFFIMAGES, df, psf, iterate, cake, subtract,
            func = ssum.deconvolution_type1)
    elif deconv == 2:
        arr = run_sums(
            SDATA, DIFFIMAGES, df, psf, iterate, cake, subtract,
            func = ssum.deconvolution_type2)
    elif deconv == 3:
        arr = run_sums(
            SDATA, DIFFIMAGES, df, psf, iterate, cake, subtract,
            func = ssum.deconvolution_type3)
    else:
        print(f'Unknown deconvolution type: deconv={deconv}')
        print('Nothing to do.')
        return(None)
    return(arr)


def run_sums(SDATA, DIFFIMAGES, df, psf, iterate, cake, subtract, func):
    '''
    Execute concurrent data processing using a thread pool.

    This function processes a list of datafiles using a thread pool 
    for parallel execution. The number of concurrent workers is determined 
    by subtracting 1 from the available CPU cores.

        
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
    
    num_workers = os.cpu_count()  # Number of concurrent workers
    datafiles = [datafile[1] for datafile in df.iterrows()] 
    
    with future.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor            
        futures = []
        total_tasks = len(datafiles)
        
        for i, file in enumerate(datafiles): 
            if func == ssum.no_deconvolution:
                future_obj = executor.submit(func, 
                                             file, 
                                             SDATA, 
                                             DIFFIMAGES)
            
            elif func == ssum.deconvolution_type3:
                future_obj = executor.submit(func, 
                                             file, 
                                             SDATA, 
                                             DIFFIMAGES, 
                                             psf, 
                                             iterate, 
                                             cake,
                                             subtract)
                
            else:
                future_obj = executor.submit(func, 
                                             file, 
                                             SDATA, 
                                             DIFFIMAGES, 
                                             psf, 
                                             iterate)
                
            futures.append(future_obj)
        
        # Use tqdm to create a progress bar
        with tqdm.tqdm(total=total_tasks, desc=f"Deconvolving using {func.__name__}") as pbar:
            # Wait for all tasks to complete
            for future_obj in future.as_completed(futures):
                future_obj.result()
                pbar.update(1)
    
    # Print a new line to complete the progress bar
    print()
        
    # Collect results
    deconvolved_data = [f.result() for f in futures]

    # POST-PROCESSING
    # (a) sum deconvoluted data
    sum_arr = sum(deconvolved_data)
    
    # (b) normalize the final array
    sum_arr = sum_arr/len(deconvolved_data)
    
    # (c) convert to final array with integer values
    # (why integer values? => arr with int's can be plotted as image and saved)
    final_arr = np.round(sum_arr).astype(np.uint16)
    
    return final_arr



"""
def run_sums(SDATA, DIFFIMAGES, df, psf, iterate, cake, subtract, func):
    '''
    Execute concurrent data processing using a thread pool.

    This function processes a list of datafiles using a thread pool 
    for parallel execution. The number of concurrent workers is determined 
    by subtracting 1 from the available CPU cores.

        
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
    
    num_workers = os.cpu_count()  # Number of concurrent workers
    datafiles = [datafile[1] for datafile in df.iterrows()] 
    
    
    with future.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor            

        futures = []
        for i, file in enumerate(datafiles): 
            if func == ssum.no_deconvolution:
                future_obj = executor.submit(func, 
                                              file, 
                                              SDATA, 
                                              DIFFIMAGES)
                # sumc.print_progress_bar(i + 1, total_tasks)
            
            elif func == ssum.deconvolution_type3:
                future_obj = executor.submit(func, 
                                              file, 
                                              SDATA, 
                                              DIFFIMAGES, 
                                              psf, 
                                              iterate, 
                                              cake,
                                              subtract)
                
                # sumc.print_progress_bar(i + 1, total_tasks)
            
            else:
                future_obj = executor.submit(func, 
                                              file, 
                                              SDATA, 
                                              DIFFIMAGES, 
                                              psf, 
                                              iterate)
                # sumc.print_progress_bar(i + 1, total_tasks)
                
            futures.append(future_obj)
            
    
        # Wait for all tasks to complete
        future.wait(futures, return_when=future.ALL_COMPLETED)
    
    # Print a new line to complete the progress bar
    print()
        
    # Collect results
    deconvolved_data = [f.result() for f in futures]

    
    # POST-PROCESSING
    # (a) sum deconvoluted data
    sum_arr = sum(deconvolved_data)
    
    # (b) normalize the final array
    sum_arr = sum_arr/len(deconvolved_data)
    
    # (c) convert to final array with integer values
    # (why integer values? => arr with int's can be plotted as image and saved)
    final_arr = np.round(sum_arr).astype(np.uint16)
    
    return final_arr
    
"""