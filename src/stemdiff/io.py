'''
stemdiff.io
-----------
Input/output functions for package stemdiff.    
'''

# The functions:
#  1) Manipulate with DAT-files: read, show-as-image,save-as-image...
#  2) Manipulate with ARRAYS: rescale, find_center, reduce_size... 
#  3) Manipulate with IMAGES: read/show an image...
# Note:
#  both dat-files and images are read and processed as numpy arrays.


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform, measure
from stemdiff.const import DET_SIZE

# ============================================================================
# 1st group of functions - manipulation with DATAFILES
# (read, show-as-image, save-as-image, show-series-of-datafiles

def read_datafile(filename):
    '''
    Read datafile from 2D-STEM detector into numpy array.
    Assumptions: 2D-STEM detector with dimensions DET_SIZE x DET_SIZE
    (where DET_FILE = stemdiff.const.DET_FILE = constant in stemdiff.const),
    which yields binary files with 16-bit intensity values.
    
    Parameters
    ----------
    filename : string or pathlib object
        Name of datafile from pixelated datector
        that should be read into numpy 2D array.
        
    Returns
    -------
    2D numpy array
    '''
    arr = np.fromfile(filename, dtype=np.uint16)
    arr = arr.reshape(DET_SIZE,DET_SIZE)
    return(arr)

def show_datafile(filename, intensity_cut=300):
    '''
    Show datafile = diffractogram from 2D-STEM detector;
    the datafile is shown as an image using matplotlib.pyplot.
    
    Parameters
    ----------
    filename : str or Path
        Name of datafile to be shown.
    intensity_cut : integer
        For all pixels: if intensity > intensity_cut: intensity=intensity_cut;
        this reduces the strongest intensity of the central spot/primary beam.
        
    Returns
    -------
    Nothing; the output are the files and entropies shown on the screen.

    '''
    print(filename.name)
    # a) Read datafile
    arr = read_datafile(filename)
    # b) Calculated and print Shannon entropy of the datafile
    entropy_value = measure.shannon_entropy(arr)
    print(f'Shannon entropy value = {entropy_value:.2f}')
    # c) Cut intensity and show datafile as 2D-image using matplotlib
    arr = np.where(arr>intensity_cut, intensity_cut, arr)
    plt.imshow(arr, cmap='gray')
    plt.show()

def save_datafile(filename, output_image, intensity_cut=300, itype='8bit'):
    '''
    Save datafile = diffractogram from 2D-STEM detector;
    the datafile is saved as a PNG-image using matplotlib.pyplot.
    
    Parameters
    ----------
    filename : str or Path
        Name of datafile to be shown.
    output_image : str
        Name of image to be saved (the full name will be output_image.png).
    intensity_cut : integer, optional, default=300
        For all pixels: if intensity > intensity_cut: intensity=intensity_cut;
        this reduces the strongest intensity of the central spot/primary beam.
    itype : str ('8bit' or '16bit'), optional, default='8bit'
         Type of the image: 8 or 16 bit grayscale. 
        
    Returns
    -------
    Nothing; the output is the saved PNG-image in active directory.

    '''
    print(filename.name)
    # a) Read datafile
    arr = read_datafile(filename)
    # b) Calculated and print Shannon entropy of the datafile
    entropy_value = measure.shannon_entropy(arr)
    print(f'Shannon entropy value = {entropy_value:.2f}')
    # c) Cut intensity and save datafile as PNG-image using matplotlib
    arr = np.where(arr>intensity_cut, intensity_cut, arr)
    # d) Before saving, normalize array according to itype = image_type
    if itype == '8bit':
        arr = np.round(arr * (255/np.max(arr))).astype(dtype=np.uint8)
        img = Image.fromarray(arr, 'L')
    else:
        arr = arr.astype('uint16')
        img = Image.fromarray(arr)
    # e) Save the final (intensity-cut, normalized) datafile/array as image
    img.save(output_image) 
    
def show_datafiles(datafiles, intensity_cut=300):
    '''
    Show datafiles = diffractograms from 2D-STEM detector.
    The images and their calculated Shannon entropies are shown one by one.
    [Enter] = next file, [Ctrl+C] = end of show (a bit hardcore, but working).

    Parameters
    ----------
    datafiles : pathlib.glob object or iterable (list, array, iterator)
        Names of datafiles to be shown.
    intensity_cut : integer
        For all pixels: if intensity > intensity_cut: intensity=intensity_cut;
        this reduces the strongest intensity of the central spot/primary beam.
        
    Returns
    -------
    Nothing; the output are the files and entropies shown on the screen.

    '''
    for datafile in datafiles:
        show_datafile(datafile, intensity_cut)
        input('[Enter] to continue...')

# ============================================================================
# 2nd group of functions - manipulation with ARRAYS
# (rescale, find_center, reduce_size, save-as-image

def rescale_array(arr,R):
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
    2D numpy array with new_size = original_size * R
    '''
    arr_max = np.max(arr)
    arr = transform.rescale(arr, R)
    arr = arr/np.max(arr) * arr_max
    return(arr)

def find_array_center(arr, central_square=None, central_intensity_coeff=None):
    '''
    Determine center of mass for 2D numpy array.
    Array center = mass center = intensity center ~ position of central spot.
    Note: for non-centrosymmetric images, central spot is NOT in array center.

    Parameters
    ----------
    arr : numpy 2D array
        Numpy 2D array, whose center (of mass ~ intensity) we want to get.
    central_square: integer, optional
        Edge of central square, from which the center will be determined.
    central_intensity_coeff: float, optional, interval: 0--1
        The intensity < maximum_intensity * central_intensity_coeff
        is regarded as 0 (background removal in central square).
    Returns
    -------
    xc,yc = integers
        Coordinates of the array center.
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
        coeff = central_intensity_coeff or 0.8
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

def reduce_array_size(arr,rsize,xc,yc):
    '''
    The original size is cut to rsize, center of new array is in xc,yc.

    Reduce/cut size of 2D numpy array.
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
    2D numpy array
        The array with reduced size.
    '''
    halfsize = int(rsize/2)
    if (rsize % 2) == 0:
        arr = arr[xc-halfsize:xc+halfsize, yc-halfsize:yc+halfsize]
    else:
        arr = arr[xc-halfsize:xc+halfsize+1, yc-halfsize:yc+halfsize+1]
    return(arr)

def save_array(arr, output_image, icut=None, itype='8bit', R=None):
    '''
    Save 2D numpy array as grayscale image.
    
    Parameters
    ----------
    arr : 2D numpy array
        array or image object to save
    output_image : string or pathlib object
        name of the output/saved file
    icut : integer
        Cut of intensity;
        if icut = 300, all image intensities > 300 will be equal to 300.
    itype: string ('8bit'  or '16bit')
        type of the image: 8 or 16 bit grayscale   
    R: integer
        Rescale coefficient;
        the input array is rescaled/enlarged R-times.
        For typical 2D-STEM detector with size 256x256 pixels,
        the array should be saved with R = 2 (or 4)
        in order to get sufficiently large image for further processing.

    Returns
    -------
    Nothing; the output is [output_image] saved on disk.
    '''
    # Cut intensity
    if icut:
        arr = np.where(arr>icut, icut, arr)
    # Rescale
    if R:
        arr_max = np.max(arr)
        arr = transform.rescale(arr, R)
        arr = arr/np.max(arr) * arr_max
    # Prepare image object for saving
    if itype == '8bit':
        arr = np.round(arr * (255/np.max(arr))).astype(dtype=np.uint8)
        img = Image.fromarray(arr, 'L')
    else:
        arr = arr.astype('uint16')
        img = Image.fromarray(arr)
    # Save image
    img.save(output_image)

# ============================================================================
# 3rd group of functions - manipulation with IMAGES
# (read/show an image

def read_image(image_name, itype='8bit'):
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
    2D numpy array
    '''
    img = Image.open(image_name)
    if itype=='8bit':
        arr = np.asarray(img, dtype=np.uint8)
    else:
        arr = np.asarray(img, dtype=np.uint16)
    return(arr)

def show_image(image_name, itype='8bit', cmap='gray'):
    '''
    Read and display image from disk.

    Parameters
    ----------
    image_name : string or pathlib object
        name of the image to display
    itype : string ('8bit'  or '16bit')
        type of the image: 8 or 16 bit grayscale
    cmap : string
        colormap (any colormap know to matplotlib)

    Returns
    -------
    Nothing; the output is image shown on screen.
    '''
    arr = read_image(image_name, itype=itype)
    plt.imshow(arr, cmap=cmap)
    plt.show()

