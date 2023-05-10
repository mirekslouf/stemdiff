'''
stemdiff.psf
------------
The calculation a 2D-PSF function from low-entropy 4D-STEM datafiles.

PSF = Point Spread Function = XY-spread of the primary beam
'''

import numpy as np
import matplotlib.pyplot as plt
import stemdiff.io, stemdiff.dbase
import scipy

def save_psf_to_disk(arr, output_file):
    """
    Save PSF function;
    the function is saved as a file in NumPy format = NPY-file.

    Parameters
    ----------
    arr : 2D-numpy array
        The array with the saved 2D-PSF function, which represents
        the experimentally determined XY-spread of the primary beam. 
    output_file : str
        Name of the output file (without extension);
        the saved file will be named *output_file*.npy.
    
    Returns
    -------
    Nothing
        The result is the PSF = 2D-array in numpy format = *output_file*.npy.
    """
    np.save(output_file, arr)

def read_psf_from_disk(input_file):
    """
    Read PSF function;
    the function is read from a file in NumPy format = NPY-file.

    Parameters
    ----------
    input_file : str
        The saved file with the PSF function = NPY-file.

    Returns
    -------
    2D-numpy array
        The array with the 2D-PSF function, which represents
        the experimentally determined XY-spread of the primary beam.
    """
    arr = np.load(input_file, allow_pickle=True)
    return(arr)

def plot_psf(arr, plt_type='2D', plt_size=None, output=None):
    '''
    Show plot of PSF function.
    
    Parameters
    ----------
    arr : 2D numpy array
        The array with the saved 2D-PSF function, which represents
        the experimentally determined XY-spread of the primary beam. 
    plt_type : string
        Either '2D' or '3D' - type of the plot.
    plt_size : integer, optional, default=None
        The size of the plot:
        if plt_size is given, the function plots only the central square
        with size = plt_size; otherwise it plots the whole array.
    output : str, optional, default=None
        The name of the output file:
        if [output] is given, the function also saves the plot
        with a filename [output].png; otherwise the plot is just shown.
        
    Returns
    -------
    Nothing
         The function just shows the plot of PSF in 2D or 3D.
    '''
    # Copy of arr variable
    # (in order not to change original array during (possible) resizing
    arr2 = arr.copy()
    # Prepare variables
    Xsize,Ysize = arr.shape
    xc,yc = (int(Xsize/2),int(Ysize/2))
    # Reduce array size for plotting, if required
    # (here we work with the array copy so as not to change to original
    if plt_size:
        arr2 = stemdiff.io.Arrays.remove_edges(arr2,plt_size,xc,yc)
    if plt_type=='2D':
        plt.imshow(arr2)
        plt.colorbar()
    else:
        # Prepare meshgrid for 3D-plotting
        Xsize,Ysize = arr2.shape
        Xhalf,Yhalf = int(Xsize/2),int(Ysize/2)
        X = np.linspace(-Xhalf,Xhalf, num=Xsize, endpoint=True)
        Y = np.linspace(-Yhalf,Yhalf, num=Ysize, endpoint=True)
        Xm,Ym = np.meshgrid(X,Y)
        # Create 3D-plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(
            Xm,Ym,arr2, cmap='coolwarm', linewidth=0, antialiased=False)
        # ADJUSTMENTS OF RANGE AND FORMAT OF Z-AXIS
        # (the following commands can be activated if necessary
        # (a) Define Z-scale
        # ax.set_zlim(0,12000)
        # (b) Separate thousands by comma
        # (according to https://stackoverflow.com/q/25973581
        # (extra number 8 in the format sets the size of number = 8 places
        # (this adds extra space in front of the number - offset from Z-axis
        # ax.zaxis.set_major_formatter(
        #     plt.matplotlib.ticker.StrMethodFormatter('{x:8,.0f}'))       
        plt.tight_layout()
    # Final output: show the plot (and save it, if it was requested)
    if output == None:
    # (if argument [output] was not given => just show the plot
        plt.show()
    else:
    # (if argument [output] was given => save the plot and then show it
    # (it must be done in this order - because plt.show() clears the plot!
        plt.savefig(output, dpi=300)
        plt.show()

def circular_mask(w, h, center=None, radius=None):
    '''
    Create a circular mask for rectangular array

    Parameters
    ----------
    h : int
        Height of the array.
    w : int
        Width of the array.
    center : list or tuple of two integers, optional, default is None
        Center of the mask.
        If None, center will be a geometric center of array.
    radius : int, optional, default is None
        Radius of the mask.
        If None, radius will be the distance to the shorter array wall. 

    Returns
    -------
    mask : 2D-array of bool values
        Circular mask: arr * mask = arr with zero values outside the mask.
        
    Note
    ----
    Copy+pasted and slightly modified from www.
    GoogleSearch: numpy create circular mask on rectangular array
    https://stackoverflow.com/q/44865023
    '''
    # Determine center and radius, if arguments were None.
    if center is None: 
        # use the middle of the image
        center = (int(h/2), int(w/2))
    if radius is None:
        # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h-center[0], w-center[1])
    # Create orthogonal grid and calculate distance from center
    # (exmplanation => https://stackoverflow.com/q/44865023
    Y, X = np.ogrid[:w, :h]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    # Calculate mask
    # (explanation => https://stackoverflow.com/q/44865023
    mask = dist_from_center <= radius
    # Return the mask
    # (2D-boolean array to modify original array
    # (arr * mask = array with zero values outsid the mask
    return(mask)

class PSFtype1:
    
    def get_psf(SDATA, DIFFIMAGES, df, circular=False):
        '''
        Get PSF of type1 = estimated from datafiles with low/no diffractions.

        Parameters
        ----------
        SDATA : stemdiff.gvars.SourceData object
            The object describes source data (detector, data_dir, filenames).
        DIFFIMAGES : stemdiff.gvars.DiffImages object
            Object describing the diffraction images/patterns.
        df : pandas.DataFrame object
            Database with datafile names and characteristics.
            PSF is estimated from the files included in this database.
            Typically, df is a sub-database containing low-diffracting files.
        circular : bool, optional, default is False
            If True, apply circular mask - change square PSF to circular PSF.
            
        Returns
        -------
        psf : 2D-numpy array
            The array represents estimate of experimental PSF.
        '''
        # (0) Prepare variables
        file_counter = 0
        R = SDATA.detector.upscale
        det_size = SDATA.detector.detector_size
        psf_size = DIFFIMAGES.psfsize
        # (1) Prepare empty array for PSF calculation
        psf = np.zeros((det_size*R, det_size*R), dtype=np.float32)
        if psf_size:
            psf =  np.zeros((psf_size*R, psf_size*R), dtype=np.float32)
        else:
            psf =  np.zeros((det_size*R, det_size*R), dtype=np.float32)
        # (2) Go through the files and calculate average PSF out of them
        for index,datafile in df.iterrows():
            # (a) Read datafile to array and increase file_counter
            file_to_read = SDATA.data_dir.joinpath(datafile.DatafileName)
            arr = stemdiff.io.Datafiles.read(SDATA, file_to_read)
            file_counter += 1
            # (b) Rescale array
            arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
            # (c) Remove edges if psf_size is given
            if psf_size:
                xc,yc = stemdiff.io.Arrays.find_center(arr,
                    central_square = DIFFIMAGES.csquare*R,
                    cintensity = DIFFIMAGES.cintensity)
                arr = stemdiff.io.Arrays.remove_edges(arr,
                    rsize = psf_size*R, xc=round(xc), yc=round(yc))
                psf += arr
            else:
                psf += arr
        # (3) Calculate final experimental PSF
        # (divide sum by no_of_summed_files in order to get reasonable values
        psf = np.round(psf/file_counter).astype(np.uint16)
        # (4) Return final experimental PSF
        return(psf)

class PSFtype2:
    
    def get_psf(arr, psf_size, circular=False):
        '''
        Get PSF of type2 = estimated small central region of datafile.

        Parameters
        ----------
        arr : 2D-numpy array
            The array/datafile, from which the PSF is to be determined.
            The array is a square with geometrical center = intensity center.
            See *Technical notes* below for more details and consequences.
        psf_size : int
            The size/diameter of PSF function to be determined.
        circular : bool, optional, default is False
            If True, apply circular mask - change square PSF to circular PSF.
            
        Returns
        -------
        psf : 2D-numpy array
            The array represents estimate of experimental PSF.

        Technical notes
        --------------
        In our algorithm, we send a square centered array to this function.
        This means that intensity center = geometric center of the array.
        Consequence: we do not have to determine intensity center
        and waste time - it is enough to calculate geometrical center.
        '''
        # Get center
        # (just geometrical center as explained in the docstring
        xc = yc = arr.shape[0]//2
        # Remove edges
        # (leave only central square with psf_size 
        psf = stemdiff.io.Arrays.remove_edges(arr, psf_size, xc,yc)
        # Remove background
        # (PSF is from the center, it "sits" on some background intensity
        # !!! Convert PSF from uint to float - uint fails for negative values!
        psf = psf.astype(np.float32)
        psf = psf - np.min(psf)
        # Convert square PSF to circular PSF if requested
        if circular:
            # Apply mask
            psf = psf * circular_mask(psf_size, psf_size)
            # Remove the remnants of background
            # (after removing edges, array values may be slightly above zero!
            # (1) We subtract the 2nd lowest value - the lowest are the zeros
            the_second_lowest_value = np.min(psf[psf != np.min(psf)])
            psf = psf - the_second_lowest_value
        # All negative values shoud go back above zero!
        # (the negative values may occur due to various recalcs and roundings
        # (they are very dangerous, leading to unwanted side-effects and errors
        psf = np.where(psf < 0, 0, psf)
        # Return final PSF
        return(psf)

class PSFtype3:
        
    def get_psf(arr, psf_size, cake, subtract = False):
        '''
        Get PSF of type3 = individual PSF based on cake-method.

        Parameters
        ----------
        arr : 2D-numpy array
            The array/datafile, from which the PSF is to be determined.
            The array is a square with geometrical center = intensity center.
            See *Technical notes* below for more details and consequences.
        psf_size : int
            The size/diameter of PSF function to be determined.
        cake : int
            Size of cake-piece in degrees.
        subtract : bool, optional, default is False
            If True, prepare PSF with the same size as scattering pattern.
            
        Returns
        -------
        psf : 2D-numpy array
            The array represents estimate of experimental PSF.

        Technical notes
        --------------
        In our algorithm, we send a square centered array to this function.
        This means that intensity center = geometric center of the array.
        Consequence: we do not have to determine intensity center
        and waste time - it is enough to calculate geometrical center.
        '''
           
        if subtract == True:     
            # PSF in the same size as image, converted to circular PSF
            arr  = arr * circular_mask(arr.shape[0], arr.shape[1])     
        else:   
            # Picking the PSF from center of image
            od = int(arr.shape[0]/2)-psf_size//2
            do = int(arr.shape[0]/2)+psf_size//2
            arr = arr[od:do,od:do]
            # Convert to circular PSF
            arr = arr * circular_mask(psf_size, psf_size)  

        # Initial rotation angle
        angle = 0
        psf_multilevel = np.zeros((arr.shape[0], arr.shape[1], cake)) 
        # Perform roation - PSF smooting
        for j in range(0,cake):
            # Storing all rotated diffraction patterns
            psf_multilevel[:,:,j] = scipy.ndimage.rotate(
                arr, angle=angle, reshape = False) 
            angle += 1
    
        psf = np.zeros((arr.shape[0],arr.shape[1]))
        # Pixels selection without peak 
        for i in range(arr.shape[0]):
            for ii in range(arr.shape[1]):
                sig_sorted = np.sort(psf_multilevel[i,ii,:], axis=None)
                # Prepared for np.mean and interval [0:x]
                psf[i,ii] = np.median(sig_sorted[:]) 
        
        # XXX: Important safety insertion
        # (due to various recalcs and roundings, psf may go below 0
        # (negative psf values result in many unwanted side effects and errors!
        psf = np.where(psf < 0, 0, psf)
        
        return(psf)

class PSFtype4:
    
    def get_psf():
        # TODO - Radim
        # 1) Individual PSF from whole datafile (like PSFtype3).)
        # 2) Subtract the PSF from whole datafile (background removal).
        # 3) Final deconvolution performed with PSF from center (like Type2).
        pass