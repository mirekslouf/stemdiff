'''
stemdiff.io
-----------
Input/output functions for package stemdiff.

Three types of stemdiff.io objects

* Datafiles = files on disk, saved directly from a 2D-STEM detector.
* Arrays = the datafiles converted to numpy array objects.
* Images = the datafiles converted to PNG files.

General strategy of stemdiff.io package
    
* Datafiles and Images are usually
  not used directly, but just converted to Array objects.
* All data manipulation (showing, scaling, saving ...)
  is done within Array objects.
* Datafiles and Images have (intentionally) just a limited amount of methods,
  the most important of which is read - this method simply reads
  Datafile/Image to an array.

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
>>> stemdiff.io.Arrays.describe(arr, central_square=20)
>>> stemdiff.io.Arrays.show(arr, icut=1000, cmap='gray')
'''


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform, measure, morphology


class Datafiles:
    

    def read(SDATA, filename):
        '''
        Read a datafile from STEM detector to an array.
        
        Parameters
        ----------
        SDATA : stemdiff.gvars.SourceData object
            The object describes source data (detector, data_dir, filenames).
        filename : string or pathlib object
            Name of datafile to be read into numpy 2D array.
            
        Returns
        -------
        arr : 2D numpy array
            The array converted from datafile with given *filename*.
        '''
        arr = SDATA.detector.read_datafile(filename)
        return(arr)
    

    def show(SDATA, filename,
             icut=None, itype='8bit', R=None, cmap='gray',
             center=False, central_square=20, cintensity=0.8):
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
        cmap : str, name of colormap, optional, default is 'gray'
            Colormap for plotting of the array.
            Other options: 'viridis', 'plasma' etc.; more info in www.
        center : bool, optional, default is False
            If True, intensity center is drawn in the final image.
        central_square : integer, optional, default is 20
            Edge of a central_square, from which the center will be determined.
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
        This function just combines Datafiles.read + Arrays.show functions.
        '''
        # Read datafile to array
        arr = Datafiles.read(SDATA, filename)
        # Describe datafile/array
        Arrays.show(arr,
            icut, itype, R, cmap,
            center, central_square, cintensity)


    def show_from_disk(SDATA,
                       interactive=True, max_files=None,
                       icut=1000, itype=None, R=None, cmap='gray',
                       center=True, central_square=20, cintensity=0.8,
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
        cmap : str, name of colormap, optional, default is 'gray'
            Colormap for plotting of the array.
            Other options: 'viridis', 'plasma' etc.; more info in www.
        center : bool, optional, default is False
            If True, intensity center is drawn in the final image.
        central_square : integer, optional, default is 20
            Edge of a central_square, from which the center will be determined.
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
        This function uses Datafiles.read and than Arrays.functions.
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
                central_square, cintensity, peak_height, peak_distance)
            # Show the datafile/array
            Arrays.show(arr,
                icut, itype, R, cmap,
                center, central_square, cintensity)
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
                           icut=1000, itype='8bit', R=None, cmap='gray'):
        '''
        Show datafiles (pre-selected in a database) from 2D-STEM detector.

        Parameters
        ----------
        SDATA : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.
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
        R : integer, optional, default is None
            Rescale coefficient;
            the input array is rescaled (usually upscaled) R-times.
        cmap : str, name of colormap, optional, default is 'gray'
            Colormap for plotting of the array.
            Other options: 'viridis', 'plasma' etc.; more info in www.

        Returns
        -------
        Nothing
            The output are the files and their characteristics on the screen.
        
        Technical note
        --------------
        This function uses Datafiles.read function
        and it reads data from database.
        As it uses database data, it cannot use standard Arrays functions. 
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
            # (if the image is rescaled, the center should be rescaled as well
            if R == None: R = 1
            plt.plot(
                datafile.Ycenter * R,
                datafile.Xcenter * R,
                'r+', markersize=20)
            plt.show()
            # Increase file counter & stop if max_files limit was reached
            file_counter += 1
            if file_counter >= max_files: break


class Arrays:


    def show(arr,
             icut=None, itype=None, R=None, cmap='gray',
             center=False, central_square=20, cintensity=0.8):
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
        cmap : str, name of colormap, optional, default is 'gray'
            Colormap for plotting of the array.
            Other options: 'viridis', 'plasma' etc.; more info in www.
        center : bool, optional, default is False
            If True, intensity center is drawn in the final image.
        central_square : integer, optional, default is 20
            Edge of a central_square, from which the center will be determined.
            Ignored if center == False.
        cintensity : float in interval 0--1, optional, default is 0.8
            The intensity < maximum_intensity * cintensity is regarded as 0
            (a simple temporary background removal in the central square).
            Ignored if center == False.
    
        Returns
        -------
        Nothing
            The output is the array shown as an image on the screen.
        '''        
        # Prepare array for saving
        arr = Arrays.prepare_for_show_or_save(arr, icut, itype, R)
        # Plot array as image
        plt.imshow(arr, cmap=cmap)
        # If center argument was given, add intensity center to the plot
        if center==True:
            xc,yc = Arrays.find_center(arr,central_square, cintensity)
            plt.plot(yc,xc, 'r+', markersize=20)
        # Show the plot
        plt.show()


    def describe(arr,
                 central_square=20, cintensity=0.8,
                 peak_height=100, peak_distance=5):
        '''
        Describe 2D-array = print XY-center, MaxIntensity, Peaks, Sh-entropy.

        Parameters
        ----------
        arr : 2D numpy array
            Array to describe.
        central_square : integer, optional, default is None
            Edge of a central_square, from which the center will be determined.
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
        x,y = Arrays.find_center(arr, central_square, cintensity)
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
            

    def find_center(arr, central_square=None, cintensity=None):
        '''
        Determine center of mass for 2D numpy array.
        Array center = mass/intensity center ~ position of central spot.
        Warning: In most cases, geometric center is NOT mass/intensity center.
    
        Parameters
        ----------
        arr : numpy 2D array
            Numpy 2D array, whose center (of mass ~ intensity) we want to get.
        central_square : integer, optional, default is None
            Edge of a central_square, from which the center will be determined.
        cintensity : float in interval 0--1, optional, default is None
            The intensity < maximum_intensity * cintensity is regarded as 0
            (a simple temporary background removal in the central square).
            
        Returns
        -------
        xc,yc : integers
            Coordinates of the intesity center = position of the primary beam.
        '''
        # Calculate center of array
        if central_square:
            # If central_square was given,
            # calculate center only for the square in the center,
            # in which we set background intensity = 0 to get correct results.
            # a) Calculate array corresponding to central square
            xsize,ysize = arr.shape
            xborder = (xsize - central_square) // 2
            yborder = (ysize - central_square) // 2
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
            # If central_square was not given,
            # calculate center for the whole array.
            # => Wrong position of central spot for non-centrosymmetric images!
            M = measure.moments(arr,1)
            (xc,yc) = (M[1,0]/M[0,0], M[0,1]/M[0,0])
            (xc,yc) = np.round([xc,yc],2)
        # Return final values            
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
        2D numpy array
            The array has new_size = original_size * R
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
        SDATA.detector.save_datafile(arr, filename) 
        pass


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
            For typical 2D-STEM detector with size 256x256 pixels,
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
             center=False, central_square=20, cintensity=0.8):
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
        cmap : str, name of colormap, optional, default is 'gray'
            Colormap for plotting of the array.
            Other options: 'viridis', 'plasma' etc.; more info in www.
        center : bool, optional, default is False
            If True, intensity center is drawn in the final image.
        central_square : integer, optional, default is 20
            Edge of a central_square, from which the center will be determined.
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
        arr = Images.read_image(image_name, itype=itype)
        # Show the array using pre-defined stemdiff.io.Array.show function.
        Arrays.show(arr,
                    icut, itype, R, cmap,
                    center, central_square, cintensity)
