'''
idiff.bcorr
------------
Background correction/reduction. 
'''


import skimage as sk


def rolling_ball(arr, radius=20):
    '''
    Subtract background from an array using *rolling ball* algorithm.

    Parameters
    ----------
    arr : numpy array
        Original array.
        Usually 2D-array supplied from package stemdiff.
    radius : int, optional, default is 20
        Radius of the rolling ball

    Returns
    -------
    arr_bcorr : numpy array
        The array with the subtracted background.

    '''
    # Get background from RollingBall algorithm in sk = skimage
    background = sk.restoration.rolling_ball(arr, radius=radius)
    # Subtract background from original array
    arr_bcorr = arr - background
    # Return array with subtracted background
    return(arr_bcorr)
