'''
Package: STEMDIFF
-----------------
Conversion of a 4D-STEM dataset to a 2D-powder diffration pattern.

* Input = 4D-STEM dataset = 2D-array of 2D-NBD patterns.
    - 2D-array = usually one (or more) two-dimensional scanning array(s).
    - 2D-NBD = monocrystalline-like nanobeam diffraction pattern.
* Output = 2D-powder diffraction pattern
    - The final 2D-diffractogram is a specific summation of 2D-NBD patterns.
    - In other words, a 4D-STEM dataset is reduced to a 2D-diffraction pattern.
    - The whole method (and final pattern) is called 4D-STEM/PNBD (powder NBD).
    
STEMDIFF modules:
    
* stemdiff.dbase = calculate database of 4D-stem datafiles 
* stemdiff.detectors = objects describing detectors and format of the datafiles
* stemdiff.gvars = objects describing *source data* and *diffraction images*
* stemdiff.io = input/output for datafiles and corresponding arrays + images
* stemdiff.sum = summation of datafiles (standard, serial processing)
* stemdiff.summ = summation of datafiles (parallel, multicore processing)

STEMDIFF auxiliary package IDIFF and its modules:

* IDIFF contains functions for the improvement of diffraction patterns
* IDIFF is imported below so that it could be used as: sd.idiff.some_module
* IDIFF modules are:
    - idiff.bcorr = background correction/subtraction
    - idiff.deconv = advanced deconvolution methods
    - idiff.ncore = noise correction/reduction
    - idiff.psf = estimate of PSF function
'''
__version__ = "5.2.7"


# Import modules of stemdiff package
# this enables us to use modules as follows:
# >>> import stemdiff as sd
# >>> sd.io.Datafiles.read(SDATA, '000_001.dat')
import stemdiff.dbase
import stemdiff.detectors
import stemdiff.gvars
import stemdiff.io
import stemdiff.sum
import stemdiff.summ


# Import supplementary package idiff
# this enables us to use the modules as follows:
# >>> import stemdiff as sd
# >>> sd.idiff.bcorr.rolling_ball(arr)
import idiff


# Obligatory acknowledgement -- the development was co-funded by TACR.
#  TACR requires that the acknowledgement is printed when we run the program.
#  Nevertheless, Python packages run within other programs, not directly.
# The following code ensures that the acknowledgement is printed when:
#  (1) You run this file: __init__.py
#  (2) You run the package from command line: python -m stemdiff
# Technical notes:
#  To get item (2) above, we define __main__.py (next to __init__.py).
#  The usage of __main__.py is not very common, but still quite standard.

def acknowledgement():
    print('STEMDIFF package - convert 4D-STEM data to 2D powder diffractogram.')
    print('------')
    print('The development of the package was co-funded by')
    print('the Technology agency of the Czech Republic,')
    print('program NCK, project TN02000020.')
    
if __name__ == '__main__':
    acknowledgement()