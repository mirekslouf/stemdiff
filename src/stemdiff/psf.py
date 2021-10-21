'''
stemdiff.psf
------------
Calculate a 2D-PSF function from low-entropy 4D-STEM datafiles.

PSF = Point Spread Function = XY-spread of the primary beam
'''

import numpy as np
import matplotlib.pyplot as plt
import stemdiff.io, stemdiff.dbase
from stemdiff.const import DET_SIZE, RESCALE

def psf_from_lowS_files(
        DBASE,SUMMATION,R=RESCALE,S=None,P=None,N=None):
    '''
    Extract 2D-PSF from datafiles with low Shannon entropy S.
    PSF (= point-spread function) is taken from the central region/spot.
    
    Parameters
    ----------
    DBASE : str or pathlib object
        Filename of database containing
        names of all datafiles and their Shannon entropy values.
    SUMMATION : summation object
        Summation parameters; here we need the size/edge
        of the central square, from which 2D-PSF will be determined.
    R : integer
        Rescale coefficient;
        PSF function is rescaled/enlarged R-times.
        Typical values of R are 2 or 4; default is stemdiff.const.RESCALE.
    S : float
        Shannon entropy value;
        if S is given, PSF will be extracted only from files with entropy < S.
    P : float
        Percent of files with the lowest entropy;
        if P is given, PSF will be extracted only from the P% of low-S files.
    N : integer
        Number of files with the lowest entropy;
        if N is given, PSF will be extracted only from those N files.
    
    Returns:
    --------
    2D numpy array
        The array with the saved 2D-PSF function, which represents
        the experimentally determined XY-spread of the primary beam.
        The array is a sum of central areas/spots of input datafiles
        (divided by number of summed datafiles to get reasonable scale).
        
    Note:
    -----
    Priority of parameters : S > P > N
        i.e. if S is given, P and N are ignored etc.
    '''
    # Get filenames with low entropy values
    lowS_datafiles = stemdiff.dbase.get_low_S_files(DBASE, S=S, P=P, N=N)
    # Get PSF function
    psf = psf_from_datafiles(lowS_datafiles,SUMMATION, R=R)
    return(psf)

def psf_from_datafiles(df,SUMMATION,R=RESCALE):
    '''
    Parameters
    ----------
    df: pandas DataFrame row iterator
        DataFrame columns: DatafileName,Entropy,Xcenter,Ycenter.
    SUMMATION: summation object
        Summation parameters; here we need the SUMMATION.psfsize
        = edge of the central square, from which 2D-PSF will be determined.
        If SUMMATION.psfsize is not given => take PSF from the whole array.
    R: integer
        Rescale coefficient;
        PSF function is rescaled/enlarged R-times.
        Typical values of R are 2 or 4; default is stemdiff.const.RESCALE.
   
    Returns
    -------
    2D numpy array
        The array represents experimental PSF;
        it is a sum of central areas/spots of input datafiles
        (divided by number of summed datafiles to get reasonable scale).
    '''
    # Prepare variables
    # a) number of datafilesr
    n = 0
    # b) array for PSF
    psf = np.zeros((DET_SIZE*R,DET_SIZE*R), dtype=np.float)
    if SUMMATION.psfsize:
        # If SUMMATION.psfsize is defined => reduce array dimension
        rsize = SUMMATION.psfsize * R
        xc,yc = ( (DET_SIZE*R)//2, (DET_SIZE*R)//2 )
        psf = stemdiff.io.reduce_array_size(psf,rsize,xc,yc)    
    # Go through the files...
    for index,datafile in df:
        n += 1
        arr = stemdiff.io.read_datafile(datafile.DatafileName)
        # ..rescale array
        arr = stemdiff.io.rescale_array(arr, R)
        if SUMMATION.psfsize:
            # ...read coordinates of the center
            xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
            # ..cut central square (the area containing central spot)    
            arr = stemdiff.io.reduce_array_size(arr,rsize,xc,yc)
        psf += arr
    # Calculate final experimental PSF
    # (divide sum by number of summed files in order to get reasonable values
    psf = np.round(psf/n).astype(np.uint16)
    # Convert square mask to round mask ~ set corners to zero
    # psfsize = SUMMATION.psfsize
    # xc,yc   = (psfsize/2, psfsize/2) 
    # for x in range(psfsize):
    #     for y in range(psfsize):
    #         r = np.sqrt((x-xc)**2 + (y-yc)**2)
    #         if r > psfsize/2: psf[x,y] = 0
    # Return final array
    return(psf)

def save_psf(arr, output_file):
    """
    Save PSF function;
    the function is saved in the form of 2D-numpy array.

    Parameters
    ----------
    arr : 2D-numpy array
        The array with the saved 2D-PSF function, which represents
        the experimentally determined XY-spread of the primary beam. 
    output_file : str
        Name of the output file (without extension);
        the saved file will be named [output_file].npy.
    Returns
    -------
    None.
        The result is the saved array in numpy format = [output_file].npy
    """
    np.save(output_file, arr)

def read_psf(input_file):
    """
    Read PSF function;
    the function is saved in the form of 2D-numpy array.

    Parameters
    ----------
    input_file : str
        The saved file with the PSF function;
        the function is saved as file in numpy format = [input_file].npy.

    Returns
    -------
    2D-numpy array
        The array with the saved 2D-PSF function, which represents
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
    Nothing.
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
        arr2 = stemdiff.io.reduce_array_size(arr2,plt_size,xc,yc)
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
        # The following command can be activated if you need defined Z-scale
        # ax.set_zlim(0,12000)
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