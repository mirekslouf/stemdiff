'''
Module: stemdiff.sum
-------------------
The summation of 4D-STEM datafiles to create 2D powder diffraction file.
Updated version with new methods implemented into the summation options.
New methods are regularized Richardson Lucy deconvolution algorithm, and several segmentation methods.

Chosen method is based on parameter 'deconv' in function call. Parameter 'deconv' should be later renamed to
'reconstruction', since we added segmentation methods as well. For now it won't be renamed for legacy reasons.

To perform the summation, we just call function sum_datafiles:

stemdiff.sum.sum_datafiles(SDATA, DIFFIMAGES, df, deconv.....)

The initial arguments are:

* SDATA = stemdiff.gvars.SourceData object = description of source data
* DIFFIMAGES = stemdiff.gvars.DiffImages object = description of diffractograms
* df = pre-calculated database of datafiles/diffratograms to sum

df can be pracalculated using stem_diff_wrapper.py or by following tutorials on stemdiff page.

Key argument is deconv, which determines the processing type:

* deconv=0 = sum *without* deconvolution
* deconv=1 = R-L deconvolution with global PSF from low-diffraction datafiles
* deconv=2 = subtract background + R-L deconvolution with PSF from the center

'''



import stemdiff.dbase
import psf_function
from skimage import restoration
import tqdm
import sys
from Deconv_class import RichardsonLucy
from PSF_fit import smooth_psf, fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture
from utilities import *
from scipy.ndimage import zoom


def sum_datafiles(SDATA, DIFFIMAGES, df, method=0, psf=None, iterate=10, regularization=None, lambda_reg=1, psf_type='orig'):

    if method != 0:
        RL = RichardsonLucy(iterations=iterate, cuda=True, timer=False, turn_off_progress_bar=True)
    if method in ("Segment_polar", "Segment_polar_with_deconv"):
        polar_processing = PolarProcessor()
    if method == "Segment_polar_threshold":
        processor = DiffractionPeakSegmenter(verbose=False, centroid_power=60, threshold_factor=5.5, start_row=50, center_radius=40)
    if method == "PeakFinding":
        processor = DiffractionPeakFinder(peak_min_sigma=10, peak_max_sigma=40, num_sigma_steps=20, bg_disk_radius=70)
    if method == 'UNet':
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
                # Raw_sum => sum datafiles without deconvolution
                if method == 'raw_sum':
                    sum_arr += raw_sum(
                        SDATA,
                        DIFFIMAGES,
                        datafile)

                # RL_global_psf => sum datafiles with global psf
                elif method == 'RL_global_psf':
                    sum_arr += RL_global_psf(
                        SDATA,
                        DIFFIMAGES,
                        datafile,
                        psf,
                        RL,
                        iterate,
                        regularization=regularization,
                        lambda_reg=lambda_reg)
                # Deconv2 => sum datafiles with DeconvType2

                elif method == 2:
                    sum_arr += RL_local_psf(
                        SDATA,
                        DIFFIMAGES,
                        datafile,
                        RL,
                        iterate,
                        regularization=regularization,
                        lambda_reg=lambda_reg,
                        psf_type=psf_type)

                elif method == 'Scipy_peak_finder':
                    sum_arr += peak_finder_scipy(
                        SDATA,
                        DIFFIMAGES,
                        datafile,
                        df,
                        minimum_intens=7)

                elif method == 'Otsu':
                    sum_arr += Otsu(
                        SDATA,
                        DIFFIMAGES,
                        datafile,
                        df,
                        min_val=15,
                        max_val=100,
                        region_size=8)

                elif method == 'GMM_fit':
                    sum_arr += GMM_fit(
                        SDATA,
                        DIFFIMAGES,
                        datafile)

                elif method == "GMM_fit_per_row":
                    result = GMM_fit_per_row(
                        SDATA,
                        DIFFIMAGES,
                        datafile,
                        polar_processing)
                    if result is not None:
                        sum_arr += result

                elif method == "GMM_fit_per_row_with_deconv":
                    sum_arr += GMM_fit_per_row_with_deconv(
                        SDATA,
                        DIFFIMAGES,
                        datafile,
                        polar_processing,
                        psf,
                        iterate)

                elif method == "Row_thresholding":
                    sum_arr += Row_threshold(
                        SDATA,
                        DIFFIMAGES,
                        datafile,
                        processor)

                elif method == "PeakFinding":
                    sum_arr += Blob_detector(
                        SDATA,
                        DIFFIMAGES,
                        datafile,
                        processor)

                elif method == "UNet":
                    sum_arr += UNET(
                        SDATA,
                        DIFFIMAGES,
                        datafile,
                        processor)
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


def raw_sum(SDATA, DIFFIMAGES, datafile):
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


def RL_global_psf(SDATA, DIFFIMAGES, datafile, psf, RL, iterate, regularization=None, lambda_reg=0.05):
    '''
    Method for performing summation with deconvolution. Using custom Richardson Lucy implementation from Deconv_class.py
    Regularization used is user defined, from options 'None', 'TM' for Tikhonov Miller (L2) or 'TV' for total variation
    (L1) regularization. Follows original workflow from older versions of STEMDIFF library.

    Parameters
    ----------
    SDATA: stemdiff.gvars.SourceData object
         The object describes source data (detector, data_dir, filenames).
    DIFFIMAGES: stemdiff.gvars.DiffImages object
         Object describing the diffraction images/patterns.
    datafile:
    psf: Estimated point spread function, expecting numpy array or grayscale image
    RL: Instance of Richardson Lucy deconvolution class
    iterate: Number of iterations, taken from class init, will be removed later (once I confirm that I won't need it)
    regularization: Regularization method, options are 'None', 'TM' or 'TV'.
    lambda_reg: Regularization parameter, default is 0.05.

    Returns
    -------
    arr: 2D numpy array with deconvolved image.

    Technical notes
    ---------------
    *RL is parsed as instance from top of this file. That way we don't have to initialize the class every time we want
     to run deconvolution (which is a lot) but it's not the prettiest solution. Might be changed completely in next
     version.
    *Our implementation offers regularization, but is a lot slower than original code with parallelized computation.
     This method cannot be parallelized since we are using CUDA. Generally, it would be beneficial to solve
     parallelization for our methods as well, but we figured out, that regularization doesn't have any effect on
     resulting diffractograms, and it's not worth the effort. We keep this implementation for further experiments
     but for any serious analysis, use stemdiff.summ.sum_datafiles(...) and its parallel processing.
    '''

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

    # (c) restore original range of intensities = re-normalize
    arr = arr_deconv * norm_const


    return arr


def RL_local_psf(SDATA, DIFFIMAGES, datafile, RL, iterate, regularization=None, lambda_reg=0.05, psf_type='orig'):
    """
    Prepare datafile for summation with deconvolution type2 (deconv=2). Type 2 denotes different method of PSF
    acquisition. Otherwise algorithm is the same. For that reason, we moved PSF manipulation into this method. You can
    use PSF type 2 as you get it, or you can choose some postprocessing. Options are smoothing, Gaussian fit into PSF
    or Voight fit into PSF. Goal was to reduce noise and get better psf, results were mixed. Start with
    psf_type='orig' since that turns off postprocessing and smoothing. You can experiment with other types, options are
    'orig', 'smoothed', 'gauss' and 'voigt'.

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
        Number of iterations during the deconvolution. Taken from RL instance, parameter will be removed later

    regularization: Regularization method, options are 'None', 'TM' or 'TV'.
    lambda_reg : Regularization parameter, default is 0.05.
    psf_type : PSF  postprocessing, default is 'orig'.

    Returns
    -------
    arr : 2D numpy array
        The datafile in the form of the array,
        which is ready for summation (with DeconvType1 => see Notes below).

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

def peak_finder_scipy(SDATA, DIFFIMAGES, datafile, df_sum, minimum_intens=20):
    '''
    First method for segmentation approach. Very naive, but its foundation for later methods.
    It uses scipy morphology for finding local maxima in the image higher than minimum_intens parameter,
    excluding central peak. Then we take 5 pixel neighbourhood around found local maxima, expecting that the
    neighbourhood contains most of the peak. We sum the intensity of each peak, and put this sum into null image,
    at the position of each respective local maxima. This way we eliminate noise and background, and quality of final
    reconstruction is dependent only on the precision of peak localization.

    This method doesn't work well, since finding local maxima with simple image morphology is not robust enough.
    Integration of peak intensity is working idea, but localization has to be precise.

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
    df_sum : Pandas dataframe precalculated for specific data.
    minimum_intens: Minimal intensity of peak to be considered as peak and not as background noise.

    Returns
    -------
    arr : 2D numpy array with isolated and integrated peaks
    '''

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

def Otsu(SDATA, DIFFIMAGES, datafile, df_sum, min_val=20, max_val=100, region_size=8):
    '''
    Method using Otsu thresholding for peak localization and segmentation. Added morphology cleaning of binary
    mask, to remove tiny blobs. We set min and max values as well as region size, to filter out candidates for
    peaks. Also didn't almost work at all, except on samples of gold crystals. There was significant improvement
    clearly visible in resulting 2D profiles, but still far from usable method.

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
    df_sum : Pandas dataframe precalculated for specific data.
    min_val: Pixels lower than min_val are considered as background noise and set to 0
    max_val: Pixels higher that max_val are clipped to max_val
    region_size: Regions of binary mask smaller than region_size will be removed.

    Returns
    -------
    arr : 2D numpy array with isolated and integrated peaks
    '''

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

def GMM_fit(SDATA, DIFFIMAGES, datafile):
    '''
    First segmentation method that gave better results than any deconvolution or other processing. Segmentation with Gaussian
    Mixture Model. Doesn't need any parameters, only input image. Of course you can play with GMM parameters and experiment,
    but it probably wont get any better based on my experience.

    Preprocessing is same as for earlier methods, consistent with original stemdiff. Then we fit GMM with 3 components,
    peak, noise, and background glow. We segment the peaks, morphologically remove noise ant integrate intesities of individual
    peaks.

    Parameters
    ----------
    SDATA:
    DIFFIMAGES:
    datafile:

    Returns
    -------
    arr: 2D numpy array with isolated and integrated peaks
    '''

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


def GMM_fit_per_row(SDATA, DIFFIMAGES, datafile, processor):
    '''
    This function is another take on segmentation in polar coordinates, this time for noise reduction via
    GMM smoothing.

    This function reads a single diffraction frame (datafile), performs a temporary *local centering correction*
    (using center-of-mass of the central region), then runs polar-domain peak extraction using the provided
    `PolarProcessor`. Centering this is neccesarry in order to keep original stemdiff pipeline where rescaling happens
     by cropping the image. This approach is far from optimal.

     !!!IMPORTANT!!! You are fitting 256 GMMs for every single image. This is painfully slow!!! It was idea that didn't
      work and shouldnt be used unless you really want to experiment

    The core idea is:

    1) **Stabilize the peak center**:
       Estimate the center-of-mass (COM) of a small central patch (default 30×30 pixels), compute
       shift needed to align that COM with the image center, and apply the shift using `np.roll`.
       The wrapped-around roll artifacts are manually zeroed to avoid fake peaks.

    2) **Process in polar coordinates** (inside `processor.process_image()`):
       The `PolarProcessor` converts the image from Cartesian → polar representation.
        It then processes each polar row (radius slice) independently:
         - subtracts local background (rolling-ball correction)
         - detects candidate peaks using `scipy.signal.find_peaks` with adaptive thresholds
         - fits a *multi-Gaussian model* to each row via `curve_fit`
       The fitted Gaussian approximation replaces the original polar row (keeping only peak-like structure).
       Finally, the polar image is reconstructed back to Cartesian space.

    3) **Restore original alignment**:
       After segmentation, the processed image is shifted back by the inverse translation so it stays compatible
       with downstream pipelines using the original expected coordinate system. (THIS IS VERY NOT OPTIMAL WAY TO DO IT).
       Much better way would be to completelly change the way STEMDIFF resizes the image, but I didnt want to break
       working pipeline for method that didn't perform well.

    4) **Rescale + crop edges**:
       The resulting peak-isolated image is upscaled by detector factor `R = SDATA.detector.upscale`, then cropped
       using the stored dataset center `(Xcenter, Ycenter)` and target image size from DIFFIMAGES.


    Parameters
    ----------
    SDATA
    DIFFIMAGES
    datafile:
    processor: Instance of `PolarProcessor', inicialized at the beggining if this method of processing is chosen

    Returns
    -------
    arr : Processed array
        Returns None if processing fails.

    '''
    try:
        # (0) Prepare variables
        R = SDATA.detector.upscale
        img_size = DIFFIMAGES.imgsize
        datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
        arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

        # --- Centering the image based on the center of mass of a 70-pixel central region ---
        center_region_size = 30
        img_h, img_w = arr.shape[:2]
        center_y, center_x = img_h // 2, img_w // 2
        half_size = center_region_size // 2
        # Extract the central region
        central_region = arr[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

        # Compute weighted indices for center of mass calculation
        y_indices, x_indices = np.indices(central_region.shape)
        total = central_region.sum()
        if total != 0:
            com_y = (y_indices * central_region).sum() / total
            com_x = (x_indices * central_region).sum() / total
        else:
            # Fallback if the region is completely dark (avoid dividing by zero, added for one of experiments)
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

        # --- Process the image using polar processor ---
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

def GMM_fit_per_row_with_deconv(SDATA, DIFFIMAGES, datafile, processor, psf, iterate):
    '''
    This is exactly the same as previous method, just with added deconvolution. It didn't help. Just curiosity kept here
    for legacy reasons and for anyone having similar thoughs on reconstruction pipeline. This is even slower that
    previous method and also doesnt work well. For same reasons.

    '''
    try:
        # (0) Prepare variables
        R = SDATA.detector.upscale
        img_size = DIFFIMAGES.imgsize

        # (1) Read datafile
        datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
        arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

        # --- Centering the image based on the center of mass of a central region ---
        center_region_size = 30
        img_h, img_w = arr.shape[:2]
        center_y, center_x = img_h // 2, img_w // 2
        half_size = center_region_size // 2
        # Extract the central region
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


def Row_threshold(SDATA, DIFFIMAGES, datafile, processor):
    '''
    !!!IMPORTANT!!! This one worked incredibly well, far better than original GMM fitting to whole image

    This method segments diffraction peaks by converting the diffraction image into polar coordinates,
    applying adaptive thresholding to each row of polar (radius slices), and then mapping the result back to Cartesian
    space to build a binary peak mask. Finally, it concentrates each detected peak blob into a single pixel
    (by integrating blob intensity and placing it at the blob's centroid / center-of-mass).

    Idea behind this is, that each image has roughly Gaussian background glow with peak in the center of the image.
    So when we convert the image to polar coordinates, each row in polar becomes circle with row_idx distance from the
    center. And since we presume symetrical gaussian background, background glow in each row should be constant and
    easily separable.

    Compared to the Gaussian fitting polar method, this version is way more robust and waaaay faster.It also does't need
    much parameter tweaking, even tho here is experimentation recomended. More in utilities.

    Pipeline overview
    -----------------
    1) Read the diffraction image.
    2) Rescale the array by detector upscale factor `R` (STEMDIFF convention).
    3) Run `processor.process_image()` which performs:
         - center correction using center-of-mass inside a radius (`center_radius`)
         - Cartesian -> polar transform (`angle_step`, `polar_radius`)
         - adaptive thresholding of each polar row using median + MAD thresholding
         - polar mask -> Cartesian mask reconstruction
         - optional morphological filtering (removing small blobs, just do that, it helps)
         - peak concentration:
             * label blobs in the Cartesian mask
             * sum pixel intensities inside each blob
             * compute blob centroid (standard or power-weighted CoM)
             * place integrated intensity into a single pixel at centroid position
         - shift-back to original coordinates
    4) Remove edges based on stored diffraction center (Xcenter, Ycenter) and expected output size.

    Parameters
    ----------
    SDATA
    DIFFIMAGES
    datafile
    processor: Instance of `DiffractionPeakSegmenter`, inicialized at the beggining if this method is chosen

    Returns
    -------
    arr : processed array

    Notes
    -----
    - Adaptive thresholding is done in polar space row-by-row using statistics:
      threshold = median(row) + threshold_factor * MAD(row) (with fallback to std if MAD is ~0).
    - `start_row` skips low-radius polar rows near the center where central peak is located, that should not be
        segmented as peaks.
    - `min_blob_size` filters out tiny blobs that are typically noise.
    - `centroid_power > 1` biases the centroid toward the brightest pixels inside a blob, which helps
      peak localization for asymmetric blobs and highly noisy images. It helps only a little bit, but helps
    '''

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

def Blob_detector(SDATA, DIFFIMAGES, datafile, processor):
    '''
    Peak finding method for diffraction patterns, this is a method (or set of methods) that goes back to idea of
    peak localization. We tried set of blob detection methods combined with background substraction methdos

    !!!IMPORTANT!!! This one was even better than thresholding in polar coordinates, but just for DoH blob
    detector and opening method for background substraction. More in utilities.py

    The pipeline is:

    1) **Read + rescale**
       Loads diffraction frame and rescales it by detector upscale factor `R`
       (STEMDIFF convention, also should be changed, it does basically nothing).

    2) **Temporary center stabilization**
       This method estimates center-of-mass (COM) from a small central patch (default 30×30 pixels),
       shifts the image to align COM with the geometric center, and zeroes the rolled edges to avoid wrap-around
       artifacts. Reason is again to keep original stemdiff pipeline untouched

    3) **Peak detection + intensity integration**
       `processor.process_image()` runs the full peak finding algorithm:
         - estimates smooth background (morphological opening / gaussian / median / rolling ball / polynomial)
         - subtracts background and clips negatives
         - estimates noise sigma via MAD (robust noise estimate)
         - detects peak candidates using one of supported detectors:
             * LoG (blob_log)
             * DoH (blob_doh)
             * MSER (OpenCV region detector)
             * PCBR (principal curvature / Hessian eigenvalues)
             * vote (combines multiple detectors and keeps consensus peaks)
         - removes detections in central radius (`center_ignore_radius_pixels`) to avoid center peak
         - integrates intensity around each peak using circular aperture sum:
             * LoG/DoH use radius ~ sigma * integration_sigma_factor
             * others use fixed radius (`integration_fixed_radius`)
         - writes results into output "Dirac image" where each peak becomes one pixel with integrated intensity

    4) **Shift back**
       The Dirac image is shifted back into the original coordinate system (inverse of COM shift), again
       zeroing wrapped regions.

    5) **Crop edges**
       Removes invalid borders / edge artifacts using known dataset center `(Xcenter, Ycenter)` and target size.

    This method turned out better than any other tried so far. Its not computationally demanding, its simple,
    and it works.

    Parameters
    ----------
    SDATA:
    DIFFIMAGES:
    datafile:
    processor: Instance of `DiffractionPeakFinder`, inicialized at the beggining if this method is chosen
    Returns
    -------
    arr : processed array

    Notes
    -----
    - Centering is done only for processing continuity. Output is shifted back so coordinates match original frames.
    - Different detector methods behave differently on ring-like patterns:
      'vote' was interesting idea, but slower and not that good.
    - Integrated intensity values depend on background estimation method and chosen integration radius. That doesn't
    bother us, in the end we only care about ratios of intensities.
    - Peak positions can overlap when rounded; intensities are summed at overlapping pixels.
    '''

    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)

    center_region_size = 30
    img_h, img_w = arr.shape[:2]
    center_y, center_x = img_h // 2, img_w // 2
    half_size = center_region_size // 2
    # Extract the central region
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

def UNET(SDATA, DIFFIMAGES, datafile, processor):
    '''
    Unfinished method, needs more exploring. Segmentation using Neural networks seems promising, but there were no
    available labeled datasets, so we had to use sythetic data. Look into diffsim_v3.
    '''
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

    arr = processor.process_image(arr, show_results=False)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)

    return (arr)

