'''
stemdiff.detectors
------------------
Known detectors for package stemdiff.

This module is basically a container of classes.
Each class describes a pixelated STEM detector, from which we get datafiles.
The description is not difficult - all we need to define is
detector name, detector size, data type, upscaling coefficient,
and how to read/save datafiles in given detector format.

All detector parameters are described below in TimePix detector class.
Therefore, the new classes = new detectors can be added quite easily:

* Copy class describing detector TimePix.
* Rename the class as needed, for example: My_new_STEM_detector.
* Re-define all properties and methods of the new class as necessary.
* When you are done, the new detector can be used within STEMDIFF package.
'''

import sys
import inspect
import numpy as np
    
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
    TimePix detector object.
    '''
    
    def __init__(self, detector_name='TimePix', 
                 detector_size=256, max_intensity=11810, 
                 data_type=np.uint16, upscale=4):
        '''
        Initialize parameters of TimePix detector.
        The parameters are described above in class definition.
        '''
        self.detector_name = detector_name
        self.detector_size = detector_size
        self.max_intensity = max_intensity
        self.data_type = data_type
        self.upscale = upscale
    
    def read_datafile(self, filename, arr_size=None):
        '''
        Read datafile in TimePix detector format.

        Parameters
        ----------
        filename : str or path
            Name of the datafile to read.
        arr_size : int, optional, default is None
            Size of the square array to reade.
            Typically, we read original datafiles with size = detector.size.
            Nonetheless, we can read saved also datafiles with size = arr_size.

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
        '''
        # Slightly modified according to
        # https://stackoverflow.com/q/43211616
        fh = open(filename,'wb')
        arr = arr.flatten()
        BlockArray = np.array(arr).astype(np.uint16)
        BlockArray.tofile(fh)
        fh.close()
    

class Secom:
    
    # TODO: Radim
    # Zkontroluj, jestli je trida aktualni.
    # Mozna se neco pokazilo pri aktualizacich?
    # V tuto chvili to mozna neni uplne dokonceno:
    #  - detector_name='TimePix' => asi by melo byt 'Secom'
    #  - a jinak to vypada, ze to je stejne jako TimePix, je aktualizovano?
    
    '''
    Definition of Secom detector.
    
    Parameters
    ----------
    detector_name : str, default is 'TimePix'
        Name of the detector.
        Keep the default unless you have specific reasons.
    detector_size : integer, default is 256
        Size of the detector in pixels.
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
    TimePix detector object.
    '''
    
    def __init__(self,
                 detector_name='TimePix', 
                 detector_size=256, data_type=np.uint16, upscale=4):
        '''
        Initialize parameters of TimePix detector.
        The parameters are described above in class definition.
        '''
        self.detector_name = detector_name
        self.detector_size = detector_size
        self.data_type = data_type
        self.upscale = upscale
    
    def read_datafile(self, filename, arr_size=None):
        '''
        Read datafile in TimePix detector format.

        Parameters
        ----------
        filename : str or path
            Name of the datafile to read.
        arr_size : int, optional, default is None
            Size of the square array to reade.
            Typically, we read original datafiles with size = detector.size.
            Nonetheless, we can read saved also datafiles with size = arr_size.

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
        '''
        # Slightly modified according to
        # https://stackoverflow.com/q/43211616
        fh = open(filename,'wb')
        arr = arr.flatten()
        BlockArray = np.array(arr).astype(np.uint16)
        BlockArray.tofile(fh)
        fh.close()


class Arina:
    
    # TODO: Radim
    # Stejne jako u detektoru Secom metodou copy+paste+modify :-)
    
    pass