import matplotlib.pyplot as plt
import numpy as np
import stemdiff.dbase
import stemdiff as sd
import idiff
import psf_function
import bcorr
from skimage import restoration
import tqdm
import sys
from Deconv_class import RichardsonLucy
from PSF_fit import smooth_psf, fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d
from skimage import transform, measure, morphology
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage import io
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture
import cv2
from utilities import *
from scipy.ndimage import zoom
#test


def sum_datafiles(SDATA, DIFFIMAGES, df, deconv=0, psf=None, iterate=10, regularization=None, lambda_reg=1, psf_type='orig'):
    if deconv != 0:
        RL = RichardsonLucy(iterations=iterate, cuda=True, timer=False, turn_off_progress_bar=True)
    if deconv == "Segment_polar" or "Segment_polar_with_deconv":
        polar_processing = PolarProcessor()
    if deconv == "Segment_polar_threshold":
        processor = DiffractionPeakSegmenter(verbose=False, centroid_power=60, threshold_factor=5.5, start_row=50, center_radius=40)
    if deconv == "PeakFinding":
        processor = DiffractionPeakFinder(peak_min_sigma=10, peak_max_sigma=40, num_sigma_steps=20, bg_disk_radius=70)
    if deconv == 'UNet':
        processor = UnetSegmenter(checkpoint_path=r"C:\Users\drend\OneDrive\Plocha\VU\pythonProject\NN_models\DP_checkpoint_last.ckpt")
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafiles = [datafile[1] for datafile in df.iterrows()]
    sum_arr = np.zeros((img_size * R, img_size * R), dtype=np.float32)
    total_tasks = len(datafiles)
    sys.stderr = sys.stdout

    with tqdm.tqdm(total=total_tasks, desc="Processing ") as pbar:
        try:
            # Process each image in the database
            for index, datafile in df.iterrows():
                # Deconv0 => sum datafiles without deconvolution
                if deconv == 0:
                    sum_arr += dfile_without_deconvolution(
                        SDATA, DIFFIMAGES, datafile)
                # Deconv1 => sum datafiles with DeconvType1
                elif deconv == 1:
                    sum_arr += dfile_with_deconvolution_type1(
                        SDATA, DIFFIMAGES, datafile, psf, RL, iterate, regularization=regularization, lambda_reg=lambda_reg)
                # Deconv2 => sum datafiles with DeconvType2
                elif deconv == 2:
                    sum_arr += dfile_with_deconvolution_type2(
                        SDATA, DIFFIMAGES, datafile, RL, iterate, regularization=regularization, lambda_reg=lambda_reg, psf_type=psf_type)
                elif deconv == 'Segment1':
                    sum_arr += dfile_segmented_type1(SDATA, DIFFIMAGES, datafile, df, minimum_intens=7)
                elif deconv == 'Segment2':
                    sum_arr += dfile_segmented_type2(SDATA, DIFFIMAGES, datafile, df, min_val=15,max_val=100, region_size=8)
                elif deconv == 'Segment3':
                    sum_arr += dfile_segmented_type3(SDATA, DIFFIMAGES, datafile, df, min_val=15,max_val=100, region_size=8)
                elif deconv == "Segment_polar":
                    # Process one file and wait for completion
                    result = dfile_polar_segment(SDATA, DIFFIMAGES, datafile, polar_processing)
                    if result is not None:
                        sum_arr += result
                elif deconv == "Segment_polar_with_deconv":
                    sum_arr += dfile_polar_segment_with_deconv(SDATA, DIFFIMAGES, datafile,polar_processing, psf, iterate)
                elif deconv == "Segment_polar_threshold":
                    sum_arr += dfile_polar_segment_threshold(SDATA,DIFFIMAGES, datafile, processor)
                elif deconv == "PeakFinding":
                    sum_arr += dfile_peak_finding(SDATA, DIFFIMAGES, datafile, processor)
                elif deconv == "UNet":
                    sum_arr += dfile_UNET(SDATA, DIFFIMAGES, datafile, processor)
                pbar.update(1)
        except Exception as e:
            print(f"Error processing a task: {str(e)}")

    # (4) Move to the next line after the progress bar is complete
    print('')

    # (5) Post-process the summation and return the result
    return sum_postprocess(sum_arr, len(df))


def sum_postprocess(sum_of_arrays, n):
    """
    Normalize and convert the summed array to 16-bit unsigned integers.

    Parameters
    ----------
    sum_of_arrays : np.array
        Sum of the arrays -
        usually from stemdiff.sum.sum_datafiles function.
    n : int
        Number of summed arrays -
        usually from stemdiff.sum.sum_datafiles function.

    Returns
    -------
    arr : np.array
        Array representing final summation.
        The array is normalized and converted to unsigned 16bit integers.
    """
    arr = np.round(sum_of_arrays/n).astype(np.uint16)
    return(arr)


def dfile_without_deconvolution(SDATA, DIFFIMAGES, datafile):
    """
    Prepare datafile for summation without deconvolution (deconv=0).

    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        The bject describing the diffraction images/patterns.
    datafile : one row from the prepared database of datafiles
        The database of datafiles is created
        in stemdiff.dbase.calc_database function.
        Each row of the database contains
        [filename, xc, yc, MaxInt, NumPeaks, S].

    Returns
    -------
    arr : 2D numpy array
        The datafile in the form of the array,
        which is ready for summation (with DeconvType0 => see Notes below).

    Notes
    -----
    * The parameters are transferred from the `sum_datafiles` function.
    * DeconvType0 = no deconvolution,
      just summation of the prepared datafiles (upscaled, centered...).
    """

    # (0) Prepare variables
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

    # (2) Rescale/upscale datafile and THEN remove border region
    # (a) upscale datafile
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    # (b) get the accurate center of the upscaled datafile
    # (the center coordinates for each datafile are saved in the database
    # (note: our datafile is one row from the database => we know the coords!
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    # (c) finally, the borders can be removed with respect to the center
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)
    # (Important technical notes:
    # (* This 3-step procedure is necessary to center the images precisely.
    # (  The accurate centers from upscaled images are saved in database.
    # (  The centers from original/non-upscaled datafiles => wrong results.
    # (* Some border region should ALWAYS be cut, for two reasons:
    # (  (i) weak/zero diffractions at edges and (ii) detector edge artifacts

    # (3) Return the datafile as an array that is ready for summation
    print(arr.shape)
    return (arr)


def dfile_with_deconvolution_type1(SDATA, DIFFIMAGES, datafile, psf, RL, iterate, regularization=None, lambda_reg=0.05):
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    arr = bcorr.rolling_ball(arr, 20)
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)

    norm_const = np.max(arr)
    arr_norm = arr / np.max(arr)
    psf_norm = psf / np.max(psf)
    if regularization == None:
        arr_deconv = RL.deconvRL(arr_norm, psf_norm)
    elif regularization == 'TM':
        arr_deconv = RL.deconvRLTM(arr_norm, psf_norm, lambda_reg)
    elif regularization == 'TV':
        arr_deconv = RL.deconvRLTV(arr_norm, psf_norm, lambda_reg)
    else:
        print("Unsupported regularization type. Supported types are None, TM and TV.")
    #arr_deconv = restoration.richardson_lucy(
    #    arr_norm, psf_norm, num_iter=iterate)

    # (c) restore original range of intensities = re-normalize
    arr = arr_deconv * norm_const


    return arr


def dfile_with_deconvolution_type2(SDATA, DIFFIMAGES, datafile, RL, iterate, regularization=None, lambda_reg=0.05, psf_type='orig'):
    """
    Prepare datafile for summation with deconvolution type2 (deconv=2).

    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        The bject describing the diffraction images/patterns.
    datafile : one row from the prepared database of datafiles
        The database of datafiles is created
        in stemdiff.dbase.calc_database function.
        Each row of the database contains
        [filename, xc, yc, MaxInt, NumPeaks, S].
    iterate : int
        Number of iterations during the deconvolution.

    Returns
    -------
    arr : 2D numpy array
        The datafile in the form of the array,
        which is ready for summation (with DeconvType1 => see Notes below).

    Notes
    -----
    * The parameters are transferred from the `sum_datafiles` function.
    * DeconvType2 = Richardson-Lucy deconvolution
      using PSFtype2 + simple background subtraction.
    * PSFtype2 = 2D-PSF estimated from central region of the datafile
      AFTER background subtraction.
    """
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    psf_size = DIFFIMAGES.psfsize
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)
    arr = bcorr.rolling_ball(arr, radius=20)

    psf = psf_function.PSFtype2.get_psf(arr, psf_size, circular=True)
    if psf_type == 'orig':
        psf = psf
    elif psf_type == 'smoothed':
        psf = smooth_psf(psf)
    elif psf_type == 'gauss':
        psf = fit_and_sample_gaussian_2d(psf, [50,50])
    elif psf_type == 'voigt':
        psf = fit_and_sample_voigt_2d(psf, [50,50])
    else:
        print('Unsupported psf_type, supported types are orig, smoothed, gauss and voigt')

    norm_const = np.max(arr)
    arr_norm = arr / np.max(arr)
    psf_norm = psf / np.max(psf)

    if regularization==None:
        arr_deconv = RL.deconvRL(arr_norm,psf_norm)
    elif regularization == 'TM':
        arr_deconv = RL.deconvRLTM(arr_norm, psf_norm, lambda_reg)
    elif regularization == 'TV':
        arr_deconv = RL.deconvRLTV(arr_norm, psf_norm, lambda_reg)
    else:
        print('Unsupported type of regularization, supported types are None, TM and TV.')


    arr = arr_deconv * norm_const

    return (arr)

def dfile_segmented_type1(SDATA, DIFFIMAGES, datafile, df_sum, minimum_intens=20):

    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)
    arr = bcorr.rolling_ball(arr, radius=20)

    coordinates = peak_local_max(arr, min_distance=10)
    center = np.array(arr.shape) / 2
    distances = np.sqrt((coordinates[:, 0] - center[0]) ** 2 + (coordinates[:, 1] - center[1]) ** 2)
    central_peak_index = np.argmin(distances)
    coordinates = np.delete(coordinates, central_peak_index, axis=0)
    peak = coordinates[0]
    peak_region = arr[peak[0] - 5:peak[0] + 5, peak[1] - 5:peak[1] + 5]

    x_center = df_sum.loc[df_sum['S'].idxmax(), 'Xcenter']
    y_center = df_sum.loc[df_sum['S'].idxmax(), 'Ycenter']

    min_intensity_threshold = minimum_intens  # Define your minimum intensity threshold here

    # Find peaks using skimage.morphology, excluding the central peak
    coordinates = peak_local_max(arr, min_distance=10)

    # Exclude central peak (assumption: central peak is the one closest to provided coordinates)
    distances = np.sqrt((coordinates[:, 0] - y_center) ** 2 + (coordinates[:, 1] - x_center) ** 2)
    central_peak_index = np.argmin(distances)
    coordinates = np.delete(coordinates, central_peak_index, axis=0)

    valid_peaks = [peak for peak in coordinates if arr[peak[0], peak[1]] > min_intensity_threshold]
    peak_regions = [arr[peak[0] - 5:peak[0] + 5, peak[1] - 5:peak[1] + 5] for peak in valid_peaks]

    new_image = np.zeros_like(arr, dtype=np.uint16)

    # Sum intensities of pixels belonging to the peaks and update the new image
    for peak in valid_peaks:
        y, x = peak
        peak_region = arr[y - 3:y + 3, x - 3:x + 3]  # Adjust region size as needed
        peak_sum = np.sum(peak_region)
        new_image[y, x] = peak_sum

    arr = new_image

    return (arr)

def dfile_segmented_type2(SDATA, DIFFIMAGES, datafile, df_sum, min_val=20, max_val=100, region_size=8):

    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)
    arr = bcorr.rolling_ball(arr, radius=20)

    arr[arr < min_val] = 0
    arr_clipped = np.clip(arr, 0, max_val)
    p2, p98 = np.percentile(arr_clipped, (2, 98))
    arr_rescale = exposure.rescale_intensity(arr_clipped, in_range=(p2, p98))
    threshold_value = threshold_otsu(arr_rescale)
    binary_image = arr_rescale > threshold_value
    binary_image_cleaned = remove_small_objects(binary_image, min_size=region_size)

    labeled_image = label(binary_image_cleaned)
    new_image = np.zeros_like(arr, dtype=np.float64)
    regions = regionprops(labeled_image)
    for region in regions:
        if region.area >= region_size:
            coords = region.coords
            sum_intensity = arr[coords[:, 0], coords[:, 1]].sum()
            centroid = region.centroid
            new_image[int(centroid[0]), int(centroid[1])] = sum_intensity

    arr = new_image

    return (arr)

def dfile_segmented_type3(SDATA, DIFFIMAGES, datafile, df_sum, min_val=20, max_val=100, region_size=8):

    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)
    preprocessed_image = bcorr.rolling_ball(arr, radius=20)

    reshaped_image = preprocessed_image.reshape(-1, 1)

    # Apply Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3)
    gmm.fit(reshaped_image)
    gmm_labels = gmm.predict(reshaped_image)

    # Reshape the labels back to the original image shape
    segmented_image = gmm_labels.reshape(preprocessed_image.shape)

    # Create a binary image from the segmented image
    # Assume that the peaks are in the component with the higher mean value
    peak_component = np.argmax(gmm.means_)
    binary_image = (segmented_image == peak_component).astype(np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    labeled_image = label(binary_image)
    new_image = np.zeros_like(arr, dtype=np.float64)
    regions = regionprops(labeled_image)
    for region in regions:
        if region.area >= 3:
            coords = region.coords
            sum_intensity = arr[coords[:, 0], coords[:, 1]].sum()
            centroid = region.centroid
            new_image[int(centroid[0]), int(centroid[1])] = sum_intensity

    arr = new_image

    return (arr)


def dfile_polar_segment(SDATA, DIFFIMAGES, datafile, processor):
    try:
        # (0) Prepare variables
        R = SDATA.detector.upscale
        img_size = DIFFIMAGES.imgsize

        # (1) Read datafile
        datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
        arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

        # --- Centering the image based on the center of mass of a 70-pixel central region ---
        center_region_size = 30
        img_h, img_w = arr.shape[:2]
        center_y, center_x = img_h // 2, img_w // 2
        half_size = center_region_size // 2
        # Extract the central region (70x70 pixels)
        central_region = arr[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

        # Compute weighted indices for center of mass calculation
        y_indices, x_indices = np.indices(central_region.shape)
        total = central_region.sum()
        if total != 0:
            com_y = (y_indices * central_region).sum() / total
            com_x = (x_indices * central_region).sum() / total
        else:
            # Fallback if the region is completely dark (avoid dividing by zero)
            com_y, com_x = half_size, half_size

        # Convert center-of-mass coordinates to global coordinates
        com_y += (center_y - half_size)
        com_x += (center_x - half_size)

        # Determine the shift needed to center the image
        shift_y = int(round(center_y - com_y))
        shift_x = int(round(center_x - com_x))

        # Shift the image to center it.
        # Note: np.roll wraps around so we zero-out the wrapped parts to avoid unwanted artifacts.
        arr = np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))
        if shift_y > 0:
            arr[:shift_y, :] = 0
        elif shift_y < 0:
            arr[shift_y:, :] = 0
        if shift_x > 0:
            arr[:, :shift_x] = 0
        elif shift_x < 0:
            arr[:, shift_x:] = 0

        # --- Process the image as usual ---
        arr = processor.process_image(arr, show_plots=False)

        # --- Shift the image back to its original position ---
        arr = np.roll(arr, shift=(-shift_y, -shift_x), axis=(0, 1))
        if shift_y > 0:
            arr[-shift_y:, :] = 0
        elif shift_y < 0:
            arr[:-shift_y, :] = 0
        if shift_x > 0:
            arr[:, -shift_x:] = 0
        elif shift_x < 0:
            arr[:, :-shift_x] = 0

        arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
        xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
        arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)


        return arr

    except Exception as e:
        print(f"Error in polar segment processing: {str(e)}")
        return None

def resize_image_scipy(image, new_shape):
    # Calculate the zoom factors for each dimension
    zoom_factors = (new_shape[0] / image.shape[0], new_shape[1] / image.shape[1])
    resized_image = zoom(image, zoom_factors, order=3)  # order=3 for cubic interpolation
    return resized_image


def center_to_point(arr, xc, yc):
    """
    Centers the array so that point (xc,yc) becomes the center (200,200) of 400x400 array.
    Uses zero padding to maintain the original array size.

    Parameters
    ----------
    arr : numpy 2D array
        Input array of size 400x400
    xc,yc : integers
        The point that should become the center of the array

    Returns
    -------
    result : numpy 2D array
        Centered array of same size as input (400x400)
    """
    result = np.zeros_like(arr)
    center = arr.shape[0] // 2  # 200 for 400x400 array
    print(f"Xc is  {xc} and yc is {yc}")

    shift_x = center - xc
    shift_y = center - yc

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_i = i + shift_x
            new_j = j + shift_y
            if (0 <= new_i < arr.shape[0]) and (0 <= new_j < arr.shape[1]):
                result[i, j] = arr[new_i, new_j]

    return result

def dfile_polar_segment_with_deconv(SDATA, DIFFIMAGES, datafile, processor, psf, iterate):
    try:
        # (0) Prepare variables
        R = SDATA.detector.upscale
        img_size = DIFFIMAGES.imgsize

        # (1) Read datafile
        datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
        arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

        # --- Centering the image based on the center of mass of a 70-pixel central region ---
        center_region_size = 30
        img_h, img_w = arr.shape[:2]
        center_y, center_x = img_h // 2, img_w // 2
        half_size = center_region_size // 2
        # Extract the central region (70x70 pixels)
        central_region = arr[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

        # Compute weighted indices for center of mass calculation
        y_indices, x_indices = np.indices(central_region.shape)
        total = central_region.sum()
        if total != 0:
            com_y = (y_indices * central_region).sum() / total
            com_x = (x_indices * central_region).sum() / total
        else:
            # Fallback if the region is completely dark (avoid dividing by zero)
            com_y, com_x = half_size, half_size

        # Convert center-of-mass coordinates to global coordinates
        com_y += (center_y - half_size)
        com_x += (center_x - half_size)

        # Determine the shift needed to center the image
        shift_y = int(round(center_y - com_y))
        shift_x = int(round(center_x - com_x))

        # Shift the image to center it.
        # Note: np.roll wraps around so we zero-out the wrapped parts to avoid unwanted artifacts.
        arr = np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))
        if shift_y > 0:
            arr[:shift_y, :] = 0
        elif shift_y < 0:
            arr[shift_y:, :] = 0
        if shift_x > 0:
            arr[:, :shift_x] = 0
        elif shift_x < 0:
            arr[:, shift_x:] = 0

        # --- Process the image as usual ---
        arr = processor.process_image(arr, show_plots=True)

        # --- Shift the image back to its original position ---
        arr = np.roll(arr, shift=(-shift_y, -shift_x), axis=(0, 1))
        if shift_y > 0:
            arr[-shift_y:, :] = 0
        elif shift_y < 0:
            arr[:-shift_y, :] = 0
        if shift_x > 0:
            arr[:, -shift_x:] = 0
        elif shift_x < 0:
            arr[:, :-shift_x] = 0

        arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
        xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
        arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)
        norm_const = np.max(arr)
        arr_norm = arr / np.max(arr)
        psf_norm = psf / np.max(psf)
        # (b) perform the deconvolution
        arr_deconv = restoration.richardson_lucy(
            arr_norm, psf_norm, num_iter=iterate)
        # (c) restore original range of intensities = re-normalize
        arr = arr_deconv * norm_const


        return arr

    except Exception as e:
        print(f"Error in polar segment processing: {str(e)}")
        return None

def dfile_polar_segment_threshold(SDATA, DIFFIMAGES, datafile, processor):
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    arr = processor.process_image(arr, show_results=False)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)

    return (arr)

def dfile_peak_finding(SDATA, DIFFIMAGES, datafile, processor):
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)

    center_region_size = 30
    img_h, img_w = arr.shape[:2]
    center_y, center_x = img_h // 2, img_w // 2
    half_size = center_region_size // 2
    # Extract the central region (70x70 pixels)
    central_region = arr[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

    # Compute weighted indices for center of mass calculation
    y_indices, x_indices = np.indices(central_region.shape)
    total = central_region.sum()
    if total != 0:
        com_y = (y_indices * central_region).sum() / total
        com_x = (x_indices * central_region).sum() / total
    else:
        # Fallback if the region is completely dark (avoid dividing by zero)
        com_y, com_x = half_size, half_size

    # Convert center-of-mass coordinates to global coordinates
    com_y += (center_y - half_size)
    com_x += (center_x - half_size)

    # Determine the shift needed to center the image
    shift_y = int(round(center_y - com_y))
    shift_x = int(round(center_x - com_x))

    # Shift the image to center it.
    # Note: np.roll wraps around so we zero-out the wrapped parts to avoid unwanted artifacts.
    arr = np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))
    if shift_y > 0:
        arr[:shift_y, :] = 0
    elif shift_y < 0:
        arr[shift_y:, :] = 0
    if shift_x > 0:
        arr[:, :shift_x] = 0
    elif shift_x < 0:
        arr[:, shift_x:] = 0

    # --- Process the image as usual ---
    arr = processor.process_image(arr, show_result=False)

    # --- Shift the image back to its original position ---
    arr = np.roll(arr, shift=(-shift_y, -shift_x), axis=(0, 1))
    if shift_y > 0:
        arr[-shift_y:, :] = 0
    elif shift_y < 0:
        arr[:-shift_y, :] = 0
    if shift_x > 0:
        arr[:, -shift_x:] = 0
    elif shift_x < 0:
        arr[:, :-shift_x] = 0

    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)

    return arr

def dfile_UNET(SDATA, DIFFIMAGES, datafile, processor):
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

    arr = processor.process_image(arr, show_results=False)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)

    return (arr)

