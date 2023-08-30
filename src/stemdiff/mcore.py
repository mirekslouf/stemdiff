'''
stemdiff.mcore
--------------
Multicore processing via multi-threading:
      managing a group of worker threads (cores) that are ready to perform tasks. 
      Instead of creating and destroying threads for every task (inefficient),
      a thread pool keeps a set of threads alive and ready to work on tasks 
      as needed. 

TECHNICAL NOTES:
    - original sum.py restructuralized for the purpose of parallel processing
    - done for deconvolution types 0, 1, 2 
        (type 3 requires cakes - I suppose it is connected to the Radim's 
         cake method which might not be used eventually)
    - individual cores perform deconvolution on available data at the same time
  
Average speed up for each deconvolution type (using 12 cores)
    type 0: 10x faster
    type 1:  6x faster
    type 2:  4x faster

_______________________________________________________________________________
Parallelizing code across multiple cores doesn't necessarily mean a linear 
speedup, especially when it comes to tasks like deconvolution. 

There are several factors that can influence the speedup:
    - Overhead: Parallel execution comes with some overhead, such as task 
                scheduling, data splitting, and communication between threads. 
                This overhead can become significant for small tasks or tasks 
                that are not easily parallelizable.
    - Amdahl's Law: Amdahl's Law states that the speedup of a program from 
                    parallelization is limited by the portion of the program 
                    that cannot be parallelized. If a significant portion 
                    of the code is inherently sequential, the speedup won't be 
                    as dramatic as the number of cores used.
    - Shared Resources: If the tasks are competing for shared resources, 
                        like memory bandwidth or cache, the performance gain 
                        from parallelization might be limited.
    - Data Dependencies: If the tasks have dependencies on each other, 
                         they might need to wait for certain resources 
                         or results before they can proceed, limiting 
                         the degree of parallelism.
    - Task Granularity: If the tasks are too fine-grained, the overhead 
                        of managing them can outweigh the benefits of parallel 
                        execution.
    - Scaling Limitations: Not all algorithms and computations can scale well 
                           to a large number of cores. Some problems are 
                           inherently complex and cannot be divided into smaller 
                           tasks that execute efficiently in parallel.

'''

import numpy as np
import stemdiff.io
import stemdiff.dbase
from skimage import restoration
import time
import os
import concurrent.futures as future
import threading


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
    print("++++")
    
    if deconv == 0:
        arr = run_sums(
            SDATA, DIFFIMAGES, df, psf, iterate,
            func = sum_without_deconvolution)
    elif deconv == 1:
        arr = run_sums(
            SDATA, DIFFIMAGES, df, psf, iterate,
            func = sum_with_deconvolution_type1)
    elif deconv == 2:
        arr = run_sums(
            SDATA, DIFFIMAGES, df, psf, iterate,
            func = sum_with_deconvolution_type2)
    # elif deconv == 3:
    #     arr = run_sums(
    #         SDATA, DIFFIMAGES, df, psf, iterate,
    #         func = sum_with_deconvolution_type3)
    else:
        print(f'Unknown deconvolution type: deconv={deconv}')
        print('Nothing to do.')
        return(None)
    return(arr)

def run_sums(SDATA, DIFFIMAGES, df, psf, iterate, func):
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
    
    start_time = time.time()
    with future.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor
        if func != sum_without_deconvolution:
            futures = [executor.submit(func, file, 
                                       SDATA, 
                                       DIFFIMAGES, 
                                       psf, iterate) for file in datafiles]
        else: 
            futures = [executor.submit(func, file, 
                                       SDATA, 
                                       DIFFIMAGES) for file in datafiles]

        # Wait for all tasks to complete
        future.wait(futures, return_when=future.ALL_COMPLETED)
        
    # Collect results
    deconvolved_data = [f.result() for f in futures]

    end_time = time.time()  # Record end time
    duration = end_time - start_time
    print("Total time taken:", duration, "seconds")
    
    # POST-PROCESSING
    # (a) sum deconvoluted data
    sum_arr = sum(deconvolved_data)
    # (b) normalize the final array
    sum_arr = sum_arr/len(deconvolved_data)
    # (c) convert to final array with integer values
    # (why integer values? => arr with int's can be plotted as image and saved
    final_arr = np.round(sum_arr).astype(np.uint16)
    
    return final_arr
    
def sum_without_deconvolution(datafile, SDATA, DIFFIMAGES):
    '''
    Sum datafiles wihtout deconvolution.

    * Parameters of the function:
        - This function is usually called from stemdiff.sum.sum_files.
        - The parameters are transferred from the sum_files function
    '''
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

def sum_with_deconvolution_type1(datafile,SDATA,DIFFIMAGES,psf,iterate):
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


def sum_with_deconvolution_type2(datafile, SDATA, DIFFIMAGES, df, iterate):
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

