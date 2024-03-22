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

* Copy the class describing TimePix detector.
* Rename the class as needed, for example: My_new_STEM_detector.
* Re-define all properties and methods of the new class as necessary.
* When you are done, the new detector can be used within STEMDIFF package.
'''

import sys
import inspect
import numpy as np
from PIL import Image
<<<<<<< Updated upstream
    
=======
import h5py
import hdf5plugin

  
>>>>>>> Stashed changes
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
    '''
    
    def __init__(self, detector_name='Secom', 
                 detector_size=2048, max_intensity=65536,
                 data_type=np.uint16, upscale=1):
        '''
        Initialize parameters of Secom detector.
        The parameters are described above in class definition.
        '''
        self.detector_name = detector_name
        self.detector_size = detector_size
        self.max_intensity = max_intensity
        self.data_type = data_type
        self.upscale = upscale
    
    def read_datafile(self, filename, arr_size=None):
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
        '''
        im = Image.fromarray(arr.astype(np.uint16))
        im.save(filename)

class Arina:
    
<<<<<<< Updated upstream
    # TODO: Radim
    # The same like for Secom, using method copy+paste+modify :-)
=======
    '''
    Definition of Arina detector.
>>>>>>> Stashed changes
    
    Parameters
    ----------
    detector_name : str, default is 'Arina'
        Name of the detector.
        Keep the default unless you have specific reasons.
    detector_size : integer, default is 192
        Size of the detector in pixels.
        Keep the default unless you have specific reasons.
    data_type : numpy data type, optional, default is np.uint16
        Type of data, which are saved in the Arina .h5-files.
        Arina detector saves the data as 64-bit .h5 files.
        This corresponds to np.uint64 (more info in NumPy documentation).
        Data have to be converted from 4D data cubu into batch of files.        
    upscale : integer, default is 1
        Upscaling coefficient.
        Final image size = detector_size * upscale.
        The upscaling coefficient increases the detector resolution.
        Surprisingly enough, the upscaling helps to improve final resolution.
    
    Returns
    -------
    Arina detector object.
    
    Format of Secom datafiles
    ---------------------------
    * hierarchical data format, .h5 format
    '''

    
    def __init__(self, detector_name='Arina', 
                 detector_size=192, max_intensity=4294967295,
                 data_type=np.uint32, upscale=1):
        '''
        Initialize parameters of Arina detector.
        The parameters are described above in class definition.
        '''
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
        Read datafile in Arina detector format.

        Parameters
        ----------
        filename : str or path
            Name of the ".npy" datafile to read.

        Returns
        -------
        arr : 2D-numpy array
            2D-array containing image from Secom detector.
            Each element of the array = the intensity detected at given pixel.
        '''
        
        arr = np.load(filename) 
        return(arr)
    

    def save_datafile(self, arr, filename):
        '''
        Save 2D-array as a datafile in the Arina detector format.
        
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
            containing the *arr* in stemdiff.detectors.Arina format.
        '''

        np.save(filename,arr.astype(np.uint16))

    def h5_to_npy(folder, file, data_dir, files_adress, data_adress, type2save):
        '''
        Preprocess ".h5" data by cutting 4D datacube into independent files.
        
        Parameters
        ----------
        folder : str or path-like object
            The path for the dataset in .h5 format.
        file : str or path-like object
            The filename of the "_master.h5 file".
        data_dir : str or path-like object
            The pathe where individual files will be saved.
        files_adress : str
            Inner h5 adress to list of related data files.
        data_adress : str
            Inner h5 adress to data localisation.
        type2save: int
            Number of bits used for saving. One of 8, 16, 32, 64.
        
        Returns
        -------
        None
        '''

        # Number of files in the dataset (beam positions are saved with  100,000 pieces per file)
        print(f"Opening {folder + file}")
        f = h5py.File(folder + file, "r") # Open for reading - preserve the original file    
        f.close
        files = len(f[files_adress])
        
        print('Dataset consists of', files, 'data file(s).')
        
        for i in range(1,files+1):
            f = h5py.File(folder + file[0:-9] + 'data_'+ '{:06d}'.format(i) + '.h5', "r") # Open for reading - preserve the original file 
            arr = f[data_adress[:]]                               # inner h5 file adress for data 
            f.close
            
            if i == 1:
                signal = np.empty([0,arr.shape[1],arr.shape[1]])
                
            signal  = np.concatenate((signal, arr), axis=0)    # Add the new file results to the end of the old one
        
        print('Dataset shape is',signal.shape)
        print('Probable scanning matrix is', np.sqrt(signal.shape[0]),'x',np.sqrt(signal.shape[0]), 'beam positions.')  
        
        for i in range(signal.shape[0]):
            file_name_npy = data_dir+'p' + str(i)+'.npy'
            if type2save == 8:
                np.save(file_name_npy, signal[int(i),:,:].astype(np.uint8))
            elif type2save == 16:
                np.save(file_name_npy, signal[int(i),:,:].astype(np.uint16))
            elif type2save == 32:  
                np.save(file_name_npy, signal[int(i),:,:].astype(np.uint32))
            elif type2save == 64:   
                np.save(file_name_npy, signal[int(i),:,:].astype(np.uint64))
            else:
                print('Bit depth did not recognised. Use 8, 16, 32 or 64.')  
                
        print(i, ' individual ".npy" files was saved.')  
        
        return 


        
    pass
