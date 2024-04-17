'''
Module: stemdiff.io
-------------------
Input/output functions for package stemdiff.

Three types of stemdiff.io objects

* Datafiles = files on disk, saved directly from a 2D-STEM detector.
* Arrays = the datafiles converted to numpy array objects.
* Images = the datafiles converted to PNG files.

Additional stemdiff.io utilities
    
* Plots = easy multiplots from selected datafiles/arrays/images.
* set_plot_parameters = a general function available without subclassing.

General strategy for working with stemdiff.io objects
    
* Datafiles and Images are usually
  not used directly, but just converted to np.array objects.
* All data manipulation (showing, scaling, saving ...)
  is done within np.array objects.
* Datafiles and Images have (intentionally) just a limited amount of methods,
  the most important of which is read - this method simply reads
  Datafile/Image to a np.array.

Examples how to use Datafiles, Arrays and Images

>>> # Show a datafile
>>> # (basic operation => there is Datafiles.function for it
>>> stemdiff.io.Datafiles.show(SDATA, filename)

>>> # Read a datafile to array
>>> # (basic operation => there is Datafiles.function for it
>>> arr = stemdiff.io.Datafiles.read(SDATA, filename)

>>> # Describe AND show the datafile
>>> # (more complex operation:
>>> # (1) read datafile to array - using Datafiles.read
>>> # (2) do what you need (here: describe, show) - using Arrays.functions
>>> arr = stemdiff.io.Datafiles.read(SDATA, datafile)
>>> stemdiff.io.Arrays.describe(arr, csquare=20)
>>> stemdiff.io.Arrays.show(arr, icut=1000, cmap='gray')
'''


import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform, measure, morphology


def set_plot_parameters(size=(12,9), dpi=75, fontsize=10, my_rcParams=None):
    '''
    Set global plot parameters (mostly for plotting in Jupyter).

    Parameters
    ----------
    size : tuple of two floats, optional, the default is (12,9)
        Size of the figure (width, height) in [cm].
    dpi : int, optional, the defalut is 75
        DPI of the figure.
    fontsize : int, optional, the default is 10
        Size of the font used in figure labels etc.
    my_rcParams : dict, optional, default is None
        Dictionary in plt.rcParams format
        containing any other allowed global plot parameters.

    Returns
    -------
    None
        The result is a modification of the global plt.rcParams variable.
    '''
    # This function just calls the final function in Plotting module.
    # (left in the main namespace of the package as it is frequently used
    Plots.set_plot_parameters(size, dpi, fontsize, my_rcParams)
    
    
class Datafiles:
    '''
    Datafiles class = a collection of functions
    that work with datafiles from 2D-STEM detector
    (assumption: the datafiles were saved as standard files on a disk).
    '''

    def read(SDATA, filename):
        '''
        Read a datafile from STEM detector to an array.
        
        Parameters
        ----------
        SDATA : stemdiff.gvars.SourceData object
            The object describes source data (detector, data_dir, filenames).
        filename : string or pathlib object
            Name of datafile to be read into numpy 2D array.
            Filename can be given as an absolute path or
            a path relative to SDATA.data_dir 
            (i.e. data directory path, which is saved in SDATA object).
            
        Returns
        -------
        arr : 2D numpy array
            The array converted from datafile with given *filename*.
        '''
        # The reading of the datafile depends on the filename argument
        # (it can be an absolute path or a path relative to SDATA.data_dir
        if os.path.isfile(filename):
            # Filename given as absolute path:
            # Everything Ok => just read and return
            arr = SDATA.detector.read_datafile(filename)
            return(arr)
        else:
            # Filename not given as relative path
            # Perhaps it is a relative path - test if it exits
            complete_filename = SDATA.data_dir.joinpath(filename)
            if os.path.isfile(complete_filename):
                # Filename was given as relative path and it exists
                # Everything Ok => read and return the completed filename
                arr = SDATA.detector.read_datafile(complete_filename)
                return(arr)
            else:
                # Filename not found
                # (it was not a correct absolute or relative path
                print(f'Datafile [{filename}] not found!')
                sys.exit()
                

    def show(SDATA, filename,
             icut=None, itype='8bit', R=None, cmap='gray',
             center=False, csquare=20, cintensity=0.8):
        '''
        Show datafile/diffractogram with basic characteristics.
        
        Parameters
        ----------
        SDATA : stemdiff.gvars.SourceData object
            The object describes source data (detector, data_dir, filenames).
        filename : str or Path
            Name of datafile to be shown.
        icut : integer, optional, default is None
            Cut of intensity;
            if icut = 300, all image intensities > 300 will be equal to 300.
        itype : string, optional, '8bit' or '16bit', default is None
            Type of the image - 8-bit or 16-bit.
            If itype equals None or '16-bit' the image is treated as 16-bit.
        R : integer, optional, default is None
            Rescale coefficient;
            the input array is rescaled (usually upscaled) R-times.
            For typical 2D-STEM detector with size 256x256 pixels,
            the array should be processed with R=4
            in order to get sufficiently large image for further processing.
        cmap : str - matplotlib.colormap name, optional, the default is 'gray'
            Matplotlib colormap for plotting of the array.
            Other interesting or high-contrast options:
            'viridis', 'plasma', 'magma' ...
            The full list of matplotlib colormaps:
            `matplotlib.pyplot.colormaps()`
        center : bool, optional, default is False
            If True, intensity center is drawn in the final image.
        csquare : integer, optional, default is 20
            Edge of a central square, from which the center will be determined.
            Ignored if center == False.
        cintensity : float in interval 0--1, optional, default is 0.8
            The intensity < maximum_intensity * cintensity is regarded as 0
            (a simple temporary background removal in the central square).
            Ignored if center == False.
            
        Returns
        -------
        Nothing
            The output is the datafile shown as an image on the screen.
            
        Technical note
        --------------
        * This function just combines Datafiles.read + Arrays.show functions.
        '''
        # Read datafile to array
        arr = Datafiles.read(SDATA, filename)
        # Describe datafile/array
        Arrays.show(arr,
            icut, itype, R, cmap,
            center, csquare, cintensity)


    def show_from_disk(SDATA,
                       interactive=True, max_files=None,
                       icut=1000, itype=None, R=None, cmap='gray',
                       center=True, csquare=20, cintensity=0.8,
                       peak_height=100, peak_distance=9):
        '''
        Show datafiles (stored in a disk) from 2D-STEM detector. 
        
        Parameters
        ----------
        SDATA : stemdiff.gvars.SourceData object
            The object describes source data (detector, data_dir, filenames).
        interactive: bool, optional, the defailt is True
            If True, images are shown interactively,
            i.e. any key = show next image, 'q' = quit.
        max_files: integer, optional, the default is None
            If not(interactive==True) and max_files > 0,
            show files non-interactively = in one run, until
            number of files is less than max_files limit.
        icut : integer, optional, default is None
            Cut of intensity;
            if icut = 300, all image intensities > 300 will be equal to 300.
        itype : string, optional, None or '8bit' or '16bit'
            Type of the image - 8-bit or 16-bit.
            If itype equals None or '16-bit' the image is treated as 16-bit.
        R : integer, optional, default is None
            Rescale coefficient;
            the input array is rescaled (usually upscaled) R-times.
            For typical 2D-STEM detector with size 256x256 pixels,
            the array should be processed with R=4
            in order to get sufficiently large image for further processing.
        cmap : str - matplotlib.colormap name, optional, the default is 'gray'
            Matplotlib colormap for plotting of the array.
            Other interesting or high-contrast options:
            'viridis', 'plasma', 'magma', ...
            The full list of matplotlib colormaps:
            `matplotlib.pyplot.colormaps()`
        center : bool, optional, default is False
            If True, intensity center is drawn in the final image.
        csquare : integer, optional, default is 20
            Edge of a central square, from which the center will be determined.
            Ignored if center == False.
        cintensity : float in interval 0--1, optional, default is 0.8
            The intensity < maximum_intensity * cintensity is regarded as 0
            (a simple temporary background removal in the central square).
            Ignored if center == False.
        peak_height : int, optional, default is 100
            Minimal height of the peak to be detected.
        peak_distance : int, optional, default is 5
            Minimal distance between two peaks so that they were separated.
            
        Returns
        -------
        Nothing
            The output are the files and their characteristics on the screen.
        
        Technical note
        --------------
        * This function uses Datafiles.read and than Arrays.functions.
        '''
        # Initialization
        file_counter = 0
        # Iterate through the files
        for datafile in SDATA.filenames:
            # Read datafile from disk to array
            arr = Datafiles.read(SDATA, datafile)
            # Print datafile name
            datafile_name = datafile.relative_to(SDATA.data_dir)
            print('Datafile:', datafile_name)
            # Describe the datafile/array
            Arrays.describe(arr,
                csquare, cintensity, peak_height, peak_distance)
            # Show the datafile/array
            Arrays.show(arr,
                icut, itype, R, cmap,
                center, csquare, cintensity)
            # Decide if we should stop the show
            if interactive:
                # Wait for keyboard input...
                choice = str(input('[Enter] to show next, [q] to quit...\n'))
                # Break if 'q' was pressed and continue otherwise...
                if choice == 'q': break
            elif max_files:
                file_counter += 1
                if file_counter >= max_files: break

    
    def show_from_database(SDATA, df,
                           interactive=True, max_files=None,
                           icut=1000, itype='8bit', cmap='gray'):
        '''
        Show datafiles (pre-selected in a database) from 2D-STEM detector.

        Parameters
        ----------
        SDATA : stemdiff.gvars.SourceData object
            The object describes source data (detector, data_dir, filenames).
        df : pandas.DataFrame object
            Pre-calculated atabase with datafiles to be shown.
            Each row of the database contains
            [filename, xc, yc, MaxInt, NumPeaks, S].
        interactive: bool, optional, the defailt is True
            If True, images are shown interactively,
            i.e. any key = show next image, 'q' = quit.
        max_files: integer, optional, the default is None
            If not(interactive==True) and max_files > 0,
            show files non-interactively = in one run, until
            number of files is less than max_files limit.
        icut : integer, optional, default is None
            Cut of intensity;
            if icut = 300, all image intensities > 300 will be equal to 300.
        itype : string, optional, '8bit' or '16bit', default is '8bit'
            Type of the image - 8 or 16 bit grayscale.   
        cmap : str - matplotlib.colormap name, optional, the default is 'gray'
            Matplotlib colormap for plotting of the array.
            Other interesting or high-contrast options:
            'viridis', 'plasma', 'magma', ...
            The full list of matplotlib colormaps:
            `matplotlib.pyplot.colormaps()`

        Returns
        -------
        Nothing
            The output are the files and their characteristics on the screen.
        
        Technical notes
        ---------------
        * This function uses Datafiles.read function
          to read data from database.
        * As the function uses database data,
          it cannot use standard Arrays functions. 
        '''
        # Initialize file counter
        file_counter = 0
        # Show the files and their characteristics saved in the database
        for index,datafile in df.iterrows():
            # Read datafile
            datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
            arr = Datafiles.read(SDATA, datafile_name) 
            # Print datafile characteristics from the database
            print(f'Datafile: {datafile.DatafileName}')
            print(f'Center (x,y): \
                  ({datafile.Xcenter:.1f},{datafile.Ycenter:.1f})')
            print(f'Maximum intensity: {datafile.MaxInt}')
            print(f'Number of peaks: {datafile.Peaks}')
            print(f'Shannon entropy: {datafile.S}')
            # Show datafile (and draw XY-center from the database data)
            arr = np.where(arr>icut, icut, arr)
            plt.imshow(arr, cmap=cmap)
            # Draw center
            # (we read data from database
            # (files are shown without any rescaling
            # (but database contains Xcenter,Ycenter from upscaled images
            # (=> we have to divide Xcenter,Ycenter by rescale coefficient!
            R = SDATA.detector.upscale
            plt.plot(
                datafile.Ycenter/R,
                datafile.Xcenter/R,
                'r+', markersize=20)
            plt.show()
            # Increase file counter & stop if max_files limit was reached
            file_counter += 1
            if file_counter >= max_files: break



class Arrays:
    '''
    Arrays class = a collection of functions
    that work with 2D-arrays, which represent datafiles from 2D-STEM detector.
    '''


    def show(arr,
             icut=None, itype=None, R=None, cmap=None,
             center=False, csquare=20, cintensity=0.8,
             plt_type='2D', plt_size=None, colorbar=False):
        '''
        Show 2D-array as an image.
        
        Parameters
        ----------
        arr : 2D numpy array
            Array to show.
        icut : integer, optional, default is None
            Cut of intensity;
            if icut = 300, all image intensities > 300 will be equal to 300.
        itype : string, optional, None or '8bit' or '16bit'
            Type of the image - 8-bit or 16-bit.
            If itype equals None or '16-bit' the image is treated as 16-bit.
        R : integer, optional, default is None
            Rescale coefficient;
            the input array is rescaled (usually upscaled) R-times.
            For typical 2D-STEM detector with size 256x256 pixels,
            the array should be processed with R=4
            in order to get sufficiently large image for further processing.
        cmap : str - matplotlib.colormap name, optional, the default is None
            Matplotlib colormap for plotting of the array.
            Interesting or high-contrast options:
            'gray', 'viridis', 'plasma', 'magma', ...
            The full list of matplotlib colormaps:
            `matplotlib.pyplot.colormaps()`
        center : bool, optional, default is False
            If True, intensity center is drawn in the final image.
        csquare : integer, optional, default is 20
            Edge of a central square, from which the center will be determined.
            Ignored if center == False.
        cintensity : float in interval 0--1, optional, default is 0.8
            The intensity < maximum_intensity * cintensity is regarded as 0
            (a simple temporary background removal in the central square).
            Ignored if center == False.
        plt_type : str, '2D' or '3D', optional, default is '2D'
            Type of the plot: 2D-dimensional or 3D-dimensional/surface plot.
        plt_size : int, optional, default is not
            If given, we plot only the central region with size = *plt_size*.
            For central region we use a geometric center - see Technical notes.
        colorbar : bool, optional, the default is False
            If True, a colorbar is added to the plot.
    
        Returns
        -------
        Nothing
            The output is the array shown as an image on the screen.
        
        Technical notes
        ---------------
        * In this function, we *do not* center the image/array.
          Center can be drawn to 2D-image, but array *is not centered*.
        * Edges can be removed (using plt_size argument),
          but only only with respect to the geometrical center,
          which means that the function shows a *non-cenered central region*.
        * If you need to show *centered central region* of an array,
          combine Arrays.find_center + Arrays.remove_edges + Arrays.show
        '''        
        # Prepare array for saving
        arr = Arrays.prepare_for_show_or_save(arr, icut, itype, R)
        # Remove edges of the plot, if requested
        # (just simple removal of edges based on geometrical center!
        # (reason: simplicity; for centering/edge removal we have other funcs
        if plt_size:
            Xsize,Ysize = arr.shape
            xc,yc = (int(Xsize/2),int(Ysize/2))
            if plt_size:
                arr = Arrays.remove_edges(arr,plt_size,xc,yc)    
        # Plot array as image
        # (a) Prepare 2D plot (default)
        if plt_type=='2D':  
            if cmap==None:  # if cmap not selected, set default for 2D maps
                cmap='viridis'
            plt.imshow(arr, cmap=cmap)
            if colorbar:  # Add colorbar
                plt.colorbar()
            if center==True:  # Mark intensity center in the plot
                xc,yc = Arrays.find_center(arr,csquare, cintensity)
                plt.plot(yc,xc, 'r+', markersize=20) # switch xc,yc for img! 
        # (b) Prepare 3D plot (option; if plt_type is not the default '2D')
        else:  
            if cmap==None:  # if cmap not selected, set default for 3D maps
                cmap='coolwarm'
            # Prepare meshgrid for 3D-plotting
            Xsize,Ysize = arr.shape
            X = np.arange(Xsize)
            Y = np.arange(Ysize)
            Xm,Ym = np.meshgrid(X,Y)
            # Create 3D-plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(
                Xm,Ym,arr, cmap=cmap, linewidth=0, antialiased=False)
            plt.tight_layout()
        # (c) Show the plot
        plt.show()


    def describe(arr,
                 csquare=20, cintensity=0.8,
                 peak_height=100, peak_distance=5):
        '''
        Describe 2D-array = print XY-center, MaxIntensity, Peaks, Sh-entropy.

        Parameters
        ----------
        arr : 2D numpy array
            Array to describe.
        csquare : integer, optional, default is None
            Edge of a central square, from which the center will be determined.
        cintensity : float in interval 0--1, optional, default is None
            The intensity < maximum_intensity * cintensity is regarded as 0
            (a simple temporary background removal in the central square).
        peak_height : int, optional, default is 100
            Minimal height of the peak to be detected.
        peak_distance : int, optional, default is 5
            Minimal distance between two peaks so that they were separated.
            
        Returns
        -------
        Nothing
            The array characteristics are just printed.
            
        Technical note
        --------------
        This function is just a wrapper
        around several np.functions and Arrays.functions.
        To get the values, use the individual functions instead.
        '''
        # Determine center (of intensity)
        x,y = Arrays.find_center(arr, csquare, cintensity)
        print(f'Center (x,y): ({x:.1f},{y:.1f})')
        # Determine maximum intensity
        max_intensity = np.max(arr)
        print(f'Maximum intensity = {max_intensity:d}')
        # Estimate number of peaks (local maxima)
        no_of_maxima = Arrays.number_of_peaks(arr, peak_height, peak_distance)
        print(f'Number of peaks = {no_of_maxima}')
        # Calculate Shannon entropy of the datafile
        entropy_value = measure.shannon_entropy(arr)
        print(f'Shannon entropy = {entropy_value:.2f}')
            

    def find_center(arr, csquare=None, cintensity=None):
        '''
        Determine center of mass for 2D numpy array.
        Array center = mass/intensity center ~ position of central spot.
        Warning: In most cases, geometric center is NOT mass/intensity center.
    
        Parameters
        ----------
        arr : numpy 2D array
            Numpy 2D array, whose center (of mass ~ intensity) we want to get.
        csquare : integer, optional, default is None
            Edge of a central square, from which the center will be determined.
        cintensity : float in interval 0--1, optional, default is None
            The intensity < maximum_intensity * cintensity is regarded as 0
            (a simple temporary background removal in the central square).
            
        Returns
        -------
        xc,yc : integers
            Coordinates of the intesity center = position of the primary beam.
        '''
        # Calculate center of array
        if csquare:
            # If csquare was given,
            # calculate center only for the square in the center,
            # in which we set background intensity = 0 to get correct results.
            # a) Calculate array corresponding to central square
            xsize,ysize = arr.shape
            xborder = (xsize - csquare) // 2
            yborder = (ysize - csquare) // 2
            arr2 = arr[xborder:-xborder,yborder:-yborder].copy()
            # b) Set intensity lower than maximum*coeff to 0 (background removal)
            coeff = cintensity or 0.8
            arr2 = np.where(arr2>np.max(arr2)*coeff, arr2, 0)
            # c) Calculate center of intensity (and add borders at the end)
            M = measure.moments(arr2,1)
            (xc,yc) = (M[1,0]/M[0,0], M[0,1]/M[0,0])
            (xc,yc) = (xc+xborder,yc+yborder)
            (xc,yc) = np.round([xc,yc],2)
        else:
            # If csquare was not given,
            # calculate center for the whole array.
            # => Wrong position of central spot for non-centrosymmetric images!
            M = measure.moments(arr,1)
            (xc,yc) = (M[1,0]/M[0,0], M[0,1]/M[0,0])
            (xc,yc) = np.round([xc,yc],2)
        # Return final values
        # IMPORTANT: the values works for arrays => switch xc,yc for images!            
        return(xc,yc)
    

    def number_of_peaks(arr, peak_height,
                        peak_distance=None, neighborhood_matrix=None):
        '''
        Estimate number of peaks in given array.

        Parameters
        ----------
        arr : 2D numpy array
            Array, for which we want to determine the number of peaks.
        peak_height : float
            Height of peak with respect to background; obligatory argument.
        peak_distance : int, optional, default is None
            Distance between two neighboring peaks.
            If given, distance matrix is calculated out of it.
        neighborhood_matrix : numpy 2D-array, optional, default is None
            The neighborhood expressed as a 2D-array of 1's and 0's.
            The neighborhood matrix can be eigher given directly
            (using argument neighborhood_matrix) or indirectly
            (calculated from argument peak_distance).
            
        Returns
        -------
        no_of_peaks : int
            Estimated number of peaks (local maxima) in given array.

        '''
        # If peak distance was given, calculate distance matrix from it.
        if peak_distance:
            neighborhood_matrix = np.ones(
                (peak_distance,peak_distance), dtype=np.uint8)
        # Determine number of peaks
        no_of_peaks = np.sum(morphology.h_maxima(
            arr, h=peak_height, footprint=neighborhood_matrix))
        # Return number of peaks
        return(no_of_peaks)

    def rescale(arr, R, order=None):
        '''
        Rescale 2D numpy array (which represents an image).
        
        Parameters
        ----------
        arr : 2D numpy array
            Numpy array representing DAT-file/image.
        R : integer
            Rescale parameter: new_size_of the array = original_size * R
    
        Returns
        -------
        arr : 2D numpy array
            The array has `new_size = original_size * R`.
        '''
        # Keep original value of array maximum.
        arr_max = np.max(arr)
        # Rescale the array.
        arr = transform.rescale(arr, R, order)
        # Restore the original value of array maximum.
        arr = arr/np.max(arr) * arr_max
        # Return the rescaled array.
        return(arr)


    def remove_edges(arr,rsize,xc,yc):
        '''
        Cut array to rsize by removing edges; center of new array = (xc,yc).
       
        Parameters
        ----------
        arr : numpy 2D array
            The original array, whose size should be reduced.
        rsize : integer
            The size of reduced array.
        xc,yc : integers
            The center of original array;
            the reduced array is cut to rsize, center of new array is in xc,yc.
    
        Returns
        -------
        arr : 2D numpy array
            The array with reduced size.
        '''
        halfsize = int(rsize/2)
        if (rsize % 2) == 0:
            arr = arr[xc-halfsize:xc+halfsize, yc-halfsize:yc+halfsize]
        else:
            arr = arr[xc-halfsize:xc+halfsize+1, yc-halfsize:yc+halfsize+1]
        return(arr)
    

    def save_as_image(arr, output_image, icut=None, itype='8bit', R=None):
        '''
        Save 2D numpy array as grayscale image.
        
        Parameters
        ----------
        arr : 2D numpy array
            Array or image object to save.
        output_image : string or pathlib object
            Name of the output/saved file.
        icut : integer
            Cut of intensity;
            if icut = 300, all image intensities > 300 will be equal to 300.
        itype: string ('8bit'  or '16bit')
            Type of the image: 8 or 16 bit grayscale.   
        R: integer
            Rescale coefficient;
            the input array is rescaled/enlarged R-times.
            For typical 2D-STEM detector with size 256x256 pixels,
            the array should be saved with R = 2 (or 4)
            in order to get sufficiently large image for further processing.
    
        Returns
        -------
        Nothing
            The output is *arr* saved as *output_image* on a disk.
        '''
        # Prepare array for saving
        arr = Arrays.prepare_for_show_or_save(arr, icut, itype, R)
        # Prepare image object (8bit or 16bit)
        if itype == '8bit':
            img = Image.fromarray(arr, 'L')
        else:
            img = Image.fromarray(arr)
        # Save image
        img.save(output_image)
        

    def save_as_datafile(SDATA, arr, filename):
        '''
        Save array as a datafile (in current detector format).

        Parameters
        ----------
        SDATA : stemdiff.gvars.SourceData object
            The object describes source data (detector, data_dir, filenames).
        arr : 2D numpy array
            Array or image object to save.
        filename : str or path
            Name of the file to save.

        Returns
        -------
        None
            The output is *arr* saved as a datafile named *filename*.
            The format of the saved datafile corresponds to current detector.
        
        Technical notes
        ---------------
        * This function io.Arrays.save_as_datafile
          is just a wrapper that calls *save_datafile* function
          for given detector.
        * The detector object is accessible in this function
          thanks to SDATA argument as *SDATA.detector*.
        '''
        SDATA.detector.save_datafile(arr, filename) 


    def prepare_for_show_or_save(arr, icut=None, itype=None, R=None):
        '''
        Prepare 2D numpy array (which contains a 2D-STEM datafile)
        for showing/saving as grayscale image.
        
        Parameters
        ----------
        arr : 2D numpy array
            Array or image object to save.
        icut : integer, optional, default is None
            Cut of intensity;
            if icut = 300, all image intensities > 300 will be equal to 300.
        itype: string, optional, '8bit' or '16bit', default is None.
            Type of the image - 8-bit or 16-bit grayscale.
            If none, then the image is saved as 16-bit.
        R: integer, optional, default is None
            Rescale coefficient;
            the input array is rescaled/enlarged R-times.
            For smaller 2D-STEM detectors with size 256x256 pixels,
            the array should be saved with R = 2 (or 4)
            in order to get sufficiently large image for further processing.
    
        Returns
        -------
        arr : 2D numpy array
            The modified array ready for showing or saving on a disk.
        '''
        # Cut intensity
        if icut:
            arr = np.where(arr>icut, icut, arr)
        # Rescale
        if R:
            arr_max = np.max(arr)
            arr = transform.rescale(arr, R)
            arr = arr/np.max(arr) * arr_max
        # Prepare for showing/saving as 8bit or 16 bit
        if itype == '8bit':
            arr = np.round(arr * (255/np.max(arr))).astype(dtype=np.uint8)
        else:
            arr = arr.astype('uint16')
        # Return the modified array
        return(arr)



class Images:
    '''
    Images class = a collection of functions
    that work with images representing datafiles from 2D-STEM detector.
    '''


    def read(image_name, itype='8bit'):
        '''
        Read grayscale image into 2D numpy array.
        
        Parameters
        ----------
        image_name : string or pathlib object
            Name of image that should read into numpy 2D array.
        itype: string ('8bit'  or '16bit')
            type of the image: 8 or 16 bit grayscale    
            
        Returns
        -------
        arr : 2D numpy array
            The array converted from *image_name*.
        '''
        img = Image.open(image_name)
        if itype=='8bit':
            arr = np.asarray(img, dtype=np.uint8)
        else:
            arr = np.asarray(img, dtype=np.uint16)
        return(arr)


    def show(image_name,
             icut=None, itype='8bit', R=None, cmap='gray',
             center=False, csquare=20, cintensity=0.8):
        '''
        Read and display image from disk.
        
        Parameters
        ----------
        image_name : str or path-like object
            Name of image to read.
        icut : integer, optional, default is None
            Cut of intensity;
            if icut = 300, all image intensities > 300 will be equal to 300.
        itype : string, optional, '8bit' or '16bit', default is '8bit'
            Type of the image - 8 or 16 bit grayscale.   
        R : integer, optional, default is None
            Rescale coefficient;
            the input array is rescaled (usually upscaled) R-times.
            For typical 2D-STEM detector with size 256x256 pixels,
            the array should be processed with R=4
            in order to get sufficiently large image for further processing.
        cmap : str - matplotlib.colormap name, optional, the default is 'gray'
            Matplotlib colormap for plotting of the array.
            Other interesting or high-contrast options:
            'viridis', 'plasma', 'magma', ...
            The full list of matplotlib colormaps:
            `matplotlib.pyplot.colormaps()`
        center : bool, optional, default is False
            If True, intensity center is drawn in the final image.
        csquare : integer, optional, default is 20
            Edge of a central square, from which the center will be determined.
            Ignored if center == False.
        cintensity : float in interval 0--1, optional, default is 0.8
            The intensity < maximum_intensity * cintensity is regarded as 0
            (a simple temporary background removal in the central square).
            Ignored if center == False.
        
        Returns
        -------
        Nothing
            The output is *image_name* shown on the screen.
        '''
        # Read Image to array.
        arr = Images.read(image_name, itype=itype)
        # Show the array using pre-defined stemdiff.io.Array.show function.
        Arrays.show(arr,
                    icut, itype, R, cmap,
                    center, csquare, cintensity)


class Plots:
    '''
    Plots class = a collection of functions
    for easy plotting of diffractograms and/or graphs.
    '''
    # The functions in this sub-package are not TOO general.
    # This is intentional - writing too general plotting funcs makes no sense.
    # The exception is the 1st func, which sets plot parameters for any plot.

    def set_plot_parameters(size=(12,9), dpi=75, 
                            fontsize=10, my_rcParams=None):
        '''
        Set global plot parameters (mostly for plotting in Jupyter).
    
        Parameters
        ----------
        size : tuple of two floats, optional, the default is (12,9)
            Size of the figure (width, height) in [cm].
        dpi : int, optional, the defalut is 75
            DPI of the figure.
        fontsize : int, optional, the default is 10
            Size of the font used in figure labels etc.
        my_rcParams : dict, optional, default is None
            Dictionary in plt.rcParams format
            containing any other allowed global plot parameters.
    
        Returns
        -------
        None; the result is a modification of the global plt.rcParams variable.
        '''
        # We test all arguments, if they exist.
        # (Theoretically, user could have redefined them as None
        # (In such a case we would change nothing and leave default values
        if size:  # Figure size
            # Convert size in [cm] to required size in [inch]
            size = (size[0]/2.54, size[1]/2.54)
            plt.rcParams.update({'figure.figsize' : size})
        if dpi:  # Figure dpi
            plt.rcParams.update({'figure.dpi' : dpi})
        if fontsize:  # Global font size
            plt.rcParams.update({'font.size' : fontsize})
        if my_rcParams:  # Other possible rcParams in the form of dictionary
            plt.rcParams.update(my_rcParams)
    
    
    def plot_2d_diffractograms(data_to_plot,
                               icut=None, cmap='viridis',
                               output_file=None, dpi=300):
        '''
        Plot a few selected 2D diffraction patterns in a row, one-by-one.
    
        Parameters
        ----------
        data_to_plot : list of lists
            This object is a list of lists.
            The number of rows = the number of plotted diffractograms.
            Each row contains two elements:
            (i) data for diffractogram to plot and (ii) title of the plot.
            The data (first element of each row) can be:
            (i) PNG-file or (ii) 2D-array containing the 2D diffractogram. 
        icut : integer, optional, default is None
            Cut of intensity;
            if icut = 300, all image intensities > 300 will be equal to 300
            (this is used just for plotting; the array values are not changed).
        cmap : matplotlib.colormap name, optional, the default is 'viridis'
            Matplotlib colormap for plotting of the diffractogram.
            Other interesting or high-contrast options:
            'gray', 'plasma', 'magma', ...
            The full list of matplotlib colormaps:
            `matplotlib.pyplot.colormaps()` 
        output_file : str, optional, default is None
            If this argument is given,
            the plot is also saved in *output_file* image
            with *dpi* resolution (dpi is specified by the following argument).
        dpi : int, optional, default is 300
            The (optional) argument gives resolution of
            (the optional) output image. 
    
        Returns
        -------
        None
            The output is the plot of the diffraction patterns on the screen.
            If argument *ouput_file* is given, the plot is saved as an image. 
        '''
        # Initialize
        n = len(data_to_plot)
        diffs = data_to_plot
        fig,ax = plt.subplots(nrows=1, ncols=n)
        # Plot 2D-diffractograms
        for i in range(n):
            # Read data to plot
            data = diffs[i][0]
            # Test input data...
            if type(data) == str: # ...String - we suppose an image
                if data.lower().endswith('.png'):  # PNG file, 2D-diffractogram
                    # we read image as '16bit'
                    # in this case, it works for 8bit images as well    
                    arr = Images.read(data, itype='16bit')
                else:  # Other than PNG files are not supported now
                    print(f'Unsuported format of file {data}!')
                    sys.exit()
            elif type(data) == np.ndarray:  # Numpy array
                if data.shape[0] == data.shape[1]:  # square array, 2D-diff.pat
                    arr = data
                else: # Non-square arrays are not supported now
                    print('Non-square arrays to plot - not supported.')
                    sys.exit()
            # Read plot parameters
            my_title = diffs[i][1]
            # Plot i-th datafile/array
            ax[i].imshow(arr, vmax=icut, cmap=cmap)
            ax[i].set_title(my_title)
        # Finalize plot
        for i in range(n): ax[i].axis('off')
        fig.tight_layout()
        if output_file: fig.savefig(output_file, dpi=dpi)

    
    def plot_datafiles_with_NS(SDATA, df, N=None, S=None,
                               icut=None, rsize=None, cmap='viridis',
                               output_file=None, dpi=300):
        '''
        Plot datafiles with selected (N,S)=(NoOfPeaks,Entropy) in a row.
    
        Parameters
        ----------
        SDATA : stemdiff.gvars.SourceData object
            The object describes source data (detector, data_dir, filenames).
        df : pandas.DataFrame object
            Pre-calculated atabase with datafiles to be shown.
            Each row of the database contains
            [filename, xc, yc, MaxInt, NumPeaks, S].
        N : list of integers, arbitrary lenght, but should it not be empty
            List of N values, where N = estimated NumberOfPeaks in a datafile.
            This function will plot datafiles for given combination of (N,S).
            The sister argument S is described below.
            The (N,S) lists must have the same length.
        S : list of integers, arbitrary lenght, but should it not be empty
            List of S values; S = calculated ShannonEntropy of a datafile.
            This function will plot datafiles for given combination of (N,S).
            The sister argument N is described above.
            The (N,S) lists must have the same length.
        icut : integer, optional, default is None
            Cut of intensity;
            if icut = 300, all image intensities > 300 will be equal to 300
            (this is used just for plotting; the array values are not changed).
        rsize : int, optional, default is None
            Reduced size of an image.
            If rsize = 100, size of all datafiles/images is reduced
            so that only the central square with size = rsize is plotted.
            The center coordinates for given datafile
            are taken from the (obligatory) df argument (see above).
        cmap : matplotlib.colormap name, optional, the default is 'viridis'
            Matplotlib colormap for plotting of the diffractogram.
            Other interesting or high-contrast options:
            'gray', 'plasma', 'magma', ...
            The full list of matplotlib colormaps:
            `matplotlib.pyplot.colormaps()` 
        output_file : str, optional, default is None
            If this argument is given,
            the plot is also saved in *output_file* image
            with *dpi* resolution (dpi is specified by the following argument).
        dpi : int, optional, default is 300
            The (optional) argument gives resolution of
            (the optional) output image. 

        Returns
        -------
        None
            The output are the datafiles plotted on the screen.
            If *ouput_file* is given, the plot is also saved as an image.
        
        Technical note
        --------------
        The function has no return statement => it returns None.
        In Jupyter and Spyder, the plot/figure is drawn on the screen.
        In CLI Python, the figure should be saved => argument *output_file*.
        '''
        
        # STEP 1: Prepare multiplot.
        fig,ax = plt.subplots(nrows=1, ncols=len(N))
        
        # STEP 2: Find datafiles with selected N,S combinations,
        # and insert them in the prepared multiplot one-by-one.
        #
        # The finding of the datafiles:
        #   In given database (argument df),
        #   we localize datafiles with selected combination of (N,S)
        #   a) N = Peaks = NumberOfPeaks
        #   b) S = index of file sorted according to Shannon entropy
        #      => S = 1        : the file with given N and the highest S
        #      => S = 2,3...   : like previous, but the 2nd, 3rd highest S ...
        #      => S = -1       : the file with given N and the lowest S
        #      => S = -2,-3... : like previous, but the 2nd,3rd lowest S ...

        # Go through (N,S) pairs,
        # find datafiles and insert them in the muptiplot.
        # * We combine enumerate and zip
        #   (https://stackoverflow.com/q/16326853
        #   reason for enumerate => index i = plot/axes number = index in ax[i]
        #   reason for zip => we need (N,S) pairs = two values together
        for i,(n,s) in enumerate(zip(N,S)):
            # (1) Create sub-database, which ...
            #  a) contains just entries for given n = N = Peaks = NoOfPeaks
            #  b) is sorted according to S = ShannonEtropy descendingly
            dfN = df[df.Peaks == n].sort_values(
                by='S', ascending=False, ignore_index=True)
            # (2) Modify value of s according to logic ...
            if s >= 1:
                # If s >= 1, decrease its value
                # Reason: Python indexing start from 0, we start from 1
                s = s - 1
            elif s > len(dfN):
                # If s > lenght_of_database_with_given_N, set it to maximum
                # Reason: if s = 10 and len(dfN) is just 8 => s = 8
                s = len(dfN)
            elif s < - len(dfN):
                # If s < -lenght_of_database_with_given_N, set it to maximum
                # Reason: if s = -10 and len(dfN) is just 8 => s = -8
                s = -len(dfN)
            # (3) Get datafile name
            # (the datafile name is extracted from df by means of iloc
            # (the 1st index = 1 => col-index: col 1 in df/dfN = datafile names
            # (the 2nd index = s => row-index: dfN was sorted according to 'S'
            datafile = dfN.iloc[s,0]
            # (4) Read the datafile to an array
            arr = Datafiles.read(SDATA, datafile)
            # (5) Reduce size if requested
            # (remove edges and keep just central square with 
            if rsize:
                # Find datafile center coordinates (xc,yc)
                # (xc,yc are in the database in columns 2 and 3, respectively
                R = SDATA.detector.upscale
                xc = int(np.round(dfN.iloc[s,1]/R,0))
                yc = int(np.round(dfN.iloc[s,2]/R,0))
                # Keep just central square with size = rsize
                arr = Arrays.remove_edges(arr, rsize, xc, yc)
            # (6) Draw datafile in the plot
            # (a) Get parameters for plot title
            # (the plot title should be something like N=10,S=1,M=1200
            # (where N,S,M are NoOfPeaks, ShannonEntropy and MaxIntensity
            max_intensity = dfN.iloc[s,3]
            peaks         = dfN.iloc[s,4]
            entropy       = dfN.iloc[s,5]
            # (b) Prepare title
            plot_title = \
                f'N={peaks:d}  S={entropy:.2f}  M={int(max_intensity):d}'
            # (c) Create the plot
            ax[i].imshow(arr, vmax=icut, cmap=cmap)
            ax[i].set_title(plot_title)
            
        # STEP 3: Finalize the figure (and save it, if requested)
        # Finalize figure
        for i in range(len(N)): ax[i].axis('off')
        fig.tight_layout()
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=dpi, facecolor='white')
        
        # IMPORTANT TECHNICAL NOTE: Output of this function
        # The function has no return statement => it returns None.
        # In Jupyter and Spyder, the plot/figure is drawn on the screen.
        # This is a property of figs - last command draws them in iPython.
        # Here the last command is: fig.tight_layout() => figure is drawn.
        # In classical/CLI Python, the fig should be saved => output_file arg.
