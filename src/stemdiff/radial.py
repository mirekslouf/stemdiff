'''
stemdiff.radial
---------------
Convert a 2D powder diffraction pattern
to a 1D radially averaged distribution profile.
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

def calc_radial_distribution(arr):
    """
    Calculate 1D-radially averaged distrubution profile
    from 2D-PNBD diffraction pattern.

    Parameters
    ----------
    arr : 2D-numpy array
        The numpy array which contains the 2D-PNBD pattern.

    Returns
    -------
    radial_distance, intensity : 1D numpy arrays
        * radial_distance = distances from the center of 2D-PNBD [pixels]
        * intensity = intensities at given distances [arbitrary units]
    
    Note
    ----
    The plot of [radial_distance, intensity] = 1D-radial profile
    corresponding to the input 2D-PNBD diffraction pattern.
    """
    # 1) Find center
    # (We employ function from skimage.measure (not from stemdiff.io),
    # (because we want float/non-integer values from the whole array.
    M =  measure.moments(arr,1)
    (xc,yc) = (M[1,0]/M[0,0], M[0,1]/M[0,0])
    # 2) Get image dimensions
    # (the algorithm works even for rectangles, not only squares
    (width,height) = arr.shape
    # 3) 2D-pole/meshgrid with calculated radial distances
    # (trick 1: the array/meshgrid will be employed for mask
    # (it has the same size as the original array for rad.distr.calculation
    [X,Y] = np.meshgrid(np.arange(width)-yc, np.arange(height)-xc)
    R = np.sqrt(np.square(X) + np.square(Y))
    # 4) Initialize variables
    radial_distance = np.arange(1,np.max(R),1)
    intensity       = np.zeros(len(radial_distance))
    index           = 0
    bin_size        = 2
    # 5) Calcualte radial profile
    # (Gradual calculation of average intenzity
    # (in circles with increasing distance from the center 
    # (trick 2: to create the circles, we will employ mask from trick 1
    for i in radial_distance:
        mask = np.greater(R, i - bin_size/2) & np.less(R, i + bin_size/2)
        values = arr[mask]
        intensity[index] = np.mean(values)
        index += 1 
    # 6) Return the profile
    return(radial_distance,intensity)

def save_radial_distribution(arr,filename):
    """
    Save 1D-radially averaged distrubution profile,
    which is calculated from 2D-PNBD diffraction pattern, as a TXT-file.


    Parameters
    ----------
    arr : 2D-numpy array
        The numpy array which contains the 2D-PNBD pattern.
    filename : str
        Name of the output file.

    Returns
    -------
    None.
        The output of the function is the saved file. 
    """
    R,I = calc_radial_distribution(arr)
    arr2 = np.array([R,I]).transpose()
    np.savetxt(filename, arr2, fmt='%3d %8.1f')

def read_radial_distribution(filename):
    """
    Read 1D-radially averaged distrubution profile from a TXT-file.

    Parameters
    ----------
    filename : str
        Name of the input file;
        the file is expected to contain two columns [distance, intensity].

    Returns
    -------
    arr : 2D-numpy array
        The array containing two columns [distance, intensity].
    """
    arr = np.loadtxt(filename, unpack=True)
    return(arr)
    
def plot_radial_distributions(
        radial_distribution_files, xlimit, ylimit, output=None):
    """
    Plot several 1D-radial distrubution files in one graph.

    Parameters
    ----------
    radial_distribution_files : 2D-list 
        list with several rows containing [filename, plot-style, name], where:
        * filename = name of the TXT-file to plot
        * plot-style = matplotlib.pyplot style, such as 'r-' (red line)
        * name = name of the data, which will appear in the plot legend
    xlimit : int
        maximum of the X-axis
    ylimit : TYPE
        maximum of the Y-axis
    output : TYPE, optional, default=None
        Name of the output file;
        if the output argument is given,
        the plot is not only shown on screen, but also saved in [output] file. 

    Returns
    -------
    None.
        The output is the plot on screen
        (and also in [output] file if the output argument is given).
    """
    # Read radial distribution files
    n = len(radial_distribution_files)
    rdist = radial_distribution_files
    # Plot radial distribution files
    for i in range(n):
        R,I     = read_radial_distribution(rdist[i][0])
        myls    = rdist[i][1]
        mylabel = rdist[i][2]
        plt.plot(R,I, myls, label=mylabel)
    # ...adjust plot
    plt.xlabel('Radial distance [pixel]')
    plt.ylabel('Intensity [grayscale]')
    plt.xlim(0,xlimit)
    plt.ylim(0,ylimit)
    plt.legend()
    plt.grid()
    # ...save plot as PNG (only if argument [output] was given)
    if output: plt.savefig(output, dpi=300)
    # ...show plot
    plt.show()
