'''
Module: stemdiff.detectors
--------------------------
Description of detectors that can be used in stemdiff package.

This module is basically a container of classes.
Each class describes a 2D STEM detector, from which we can read datafiles.
The description of the detector not difficult.
All we need to define is the detector name,
a few technical parameters described below,
and how to read/save datafiles in given detector format.

All detector parameters are described below in TimePix detector class.
Therefore, the new classes = new detectors can be added quite easily:

* Copy the class describing TimePix detector.
* Rename the class as needed, for example: My_new_STEM_detector.
* Re-define all properties and methods of the new class as necessary.
* When you are done, the new detector can be used within STEMDIFF package.
'''

import sys
import inspect
import numpy as np
from PIL import Image


  
def list_of_known_detectors():
    '''
    Get a list of known detectors = classes defined in stemdiff.detectors.

    Returns
    -------
    detectors : list
        List of known detectors = classes defined in stemdiff.detectors module.
    '''
    # Prepare list of known detectors
    detectors = []
    # Get names of all classes in current module
    # Based on stackoveflow: https://stackoverflow.com/q/1796180
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            detectors.append(obj)
    # Return list of known detectors
    return(detectors)


def print_known_detectors():
    '''
    Print a list of known detectors = classes defined in stemdiff.detectors.

    Returns
    -------
    Nothing
        The list of detectors is just printed on the screen.
    '''
    detectors = list_of_known_detectors()
    print('List of knonw detectors = classes defined in stemdiff.detectors:')
    for detector in detectors:
        print(detector)


def describe_detector(detector_object):
    '''
    Global method: the description of the detector on the screen.
    
    Parameters
    ----------
    detector_object : object of any class in stemdiff.detectors
        The initialized detector object.
        The object contains description of the detector.
        It is created by calling init method of any stemdiff.detectors class.
        
    Returns
    -------
    None
        The output is the text on the screen.
        
    Example
    -------
    >>> # Short example
    >>> # (minimalistic example
    >>> import stemdiff as sd
    >>> my_detector = sd.detectors.TimePix()
    >>> my_detector.self_describe()     # OO-interface, indirect call
    >>> describe_detector(my_detector)  # procedural interface, direct call
    >>>
    >>> # Real usage
    >>> # (in STEMDIFF scripts,
    >>> # (the detector is usually defined
    >>> # (within stemdiff.gvars.SourceData object
    >>> import stemdiff as sd
    >>> SDATA = sd.gvars.SourceData(
    >>>     detector  = sd.detectors.TimePix(),
    >>>     data_dir  = r'./DATA',
    >>>     filenames = r'*.dat')
    >>> SDATA.detector.self_describe()
    '''
    print('Detector name       :', detector_object.detector_name)
    print('Detector size       :', detector_object.detector_size)
    print('Maximum intensity   :', detector_object.max_intensity)
    print('Intensity data type :', detector_object.data_type.__name__)
    print('Upscale parameter   :', detector_object.upscale)
    print('-----')
    print('* Detector size = size of the (square) detector (in pixels).')
    print('* Maximum intensity = max.intensity the detector can measure.')
    print('* Intensity data type = the format of the saved intensities.')
    print('* Upscale parameter: each datafile/image is upscaled by it.')

class TimePix:
    '''
    Definition of TimePix detector.
    
    Parameters
    ----------
    detector_name : str, default is 'TimePix'
        Name of the detector.
        Keep the default unless you have specific reasons.
    detector_size : integer, default is 256
        Size of the detector in pixels.
        Keep the default unless you have specific reasons.
    max_intensity : int, default is 11810
        Maximum intensity of TimePix detector.
        Keep the default unless you have specific reasons.
    data_type : numpy data type, optional, default is np.uint16
        Type of data, which are saved in the binary file.
        TimePix detector saves the data as 16-bit integers.
        This corresponds to np.uint16 (more info in NumPy documentation).
    upscale : integer, default is 4
        Upscaling coefficient.
        Final image size = detector_size * upscale.
        The upscaling coefficient increases the detector resolution.
        Surprisingly enough, the upscaling helps to improve final resolution.
    
    Returns
    -------
    TimePix detector object
    
    Format of TimePix datafiles
    ---------------------------
    * binary data files, usually with DAT extension
    * a 1D arrays of 16-bit intensities = np.uint16 values
    '''

    
    def __init__(self, detector_name='TimePix', 
                 detector_size=256, max_intensity=11810, 
                 data_type=np.uint16, upscale=4):   
        # The initialization of TimePix detector objects.
        # The parameters are described above in class definition.
        # -----
        self.detector_name = detector_name
        self.detector_size = detector_size
        self.max_intensity = max_intensity
        self.data_type = data_type
        self.upscale = upscale

    def self_describe(self):
        '''
        Print a simple textual description of the detector on the screen.

        Returns
        -------
        None
            The description of the detector is just printed on the screen.
            
        Technical note
        --------------
        * This is just a wrapper around global function named
          stemdiff.detectors.describe_detector.
        * Reason: this global function is common to all detector types.
        * This simple solution is used instead of (needlessly complex)
          inheritance in this case.
        ''' 
        describe_detector(self)
    
    def read_datafile(self, filename, arr_size=None):
        '''
        Read datafile in TimePix detector format.

        Parameters
        ----------
        filename : str or path
            Name of the datafile to read.
        arr_size : int, optional, default is None
            Size of the square array to read.
            Typically, we read original datafiles,
            whose size = detector.size.
            Nonetheless, the datafiles might have been saved
            with a smaller size = arr_size.

        Returns
        -------
        arr : 2D-numpy array
            2D-array containing image from TimePix detector.
            Each element of the array = the intensity detected at given pixel.
        '''
        # Read binary datafile (to 1D-array)
        arr = np.fromfile(filename, dtype=self.data_type)
        # Determine edge of the file - we work only with square files
        edge = int(np.sqrt(arr.size))
        # Reshape the array and return
        arr = arr.reshape(edge, edge)
        return(arr)

    
    def save_datafile(self, arr, filename):
        '''
        Save 2D-array as a datafile in the TimePix detector format.
        
        Parameters
        ----------
        arr : numpy array
            The array to save in the datafile with [filename].
        filename : str or path-like object
            The filename of the saved array.
        
        Returns
        -------
        None
            The result is the file named *filename*,
            containing the *arr* in stemdiff.detectors.TimePix format.
        '''
        # Slightly modified according to
        # https://stackoverflow.com/q/43211616
        fh = open(filename,'wb')
        arr = arr.flatten()
        BlockArray = np.array(arr).astype(np.uint16)
        BlockArray.tofile(fh)
        fh.close()
    


class Secom:
    '''
    Definition of Secom detector.
    
    Parameters
    ----------
    detector_name : str, default is 'Secom'
        Name of the detector.
        Keep the default unless you have specific reasons.
    detector_size : integer, default is 2048
        Size of the detector in pixels.
        Keep the default unless you have specific reasons.
    data_type : numpy data type, optional, default is np.uint16
        Type of data, which are saved in the Secom TIFF-files.
        Secom detector saves the data as 16-bit TIFF files.
        This corresponds to np.uint16 (more info in NumPy documentation).
    upscale : integer, default is 1
        Upscaling coefficient.
        Final image size = detector_size * upscale.
        The upscaling coefficient increases the detector resolution.
        Surprisingly enough, the upscaling helps to improve final resolution.
    
    Returns
    -------
    Secom detector object.
    
    Format of SECOM datafiles
    -------------------------
    * image files, TIFF format
    * 16-bit images = images containing np.uint16 values
    '''

    
    def __init__(self, detector_name='Secom', 
                 detector_size=2048, max_intensity=65536,
                 data_type=np.uint16, upscale=1):
        # The initialization of Secom detector objects.
        # The parameters are described above in class definition.
        # -----
        self.detector_name = detector_name
        self.detector_size = detector_size
        self.max_intensity = max_intensity
        self.data_type = data_type
        self.upscale = upscale


    def self_describe(self):
        '''
        Print a simple textual description of the detector on the screen.

        Returns
        -------
        None
            The description of the detector is just printed on the screen.
            
        Technical note
        --------------
        * This is just a wrapper around global function named
          stemdiff.detectors.describe_detector.
        * Reason: this global function is common to all detector types.
        * This simple solution is used instead of (needlessly complex)
          inheritance in this case.
        ''' 
        describe_detector(self)

    
    def read_datafile(self, filename):
        '''
        Read datafile in Secom detector format.

        Parameters
        ----------
        filename : str or path
            Name of the datafile to read.

        Returns
        -------
        arr : 2D-numpy array
            2D-array containing image from Secom detector.
            Each element of the array = the intensity detected at given pixel.
        '''
        
        arr = np.array(Image.open(filename)) 
        return(arr)
    

    def save_datafile(self, arr, filename):
        '''
        Save 2D-array as a datafile in the Secom detector format.
        
        Parameters
        ----------
        arr : numpy array
            The array to save in the datafile with [filename].
        filename : str or path-like object
            The filename of the saved array.
        
        Returns
        -------
        None
            The result is the file named *filename*,
            containing the *arr* in stemdiff.detectors.Secom format.
        '''
        im = Image.fromarray(arr.astype(np.uint16))
        im.save(filename)



class Arina:
    
    '''
    * TODO: Radim
    * The same like for Secom, using method copy+paste+modify :-)
    '''
    
    pass
