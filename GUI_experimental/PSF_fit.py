from scipy.optimize import minimize
from scipy.special import voigt_profile
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, measure, filters, morphology
from scipy import ndimage
import sys
import tqdm
import stemdiff as sd


def voigt_2d(x, y, amp, x0, y0, sigma_g, gamma_l):
    """2D Voigt profile function."""
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return amp * voigt_profile(r, sigma_g, gamma_l)


def compute_centroid(data):
    """Compute the centroid of a 2D array."""
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y)

    total_mass = np.sum(data)
    x_center = np.sum(X * data) / total_mass
    y_center = np.sum(Y * data) / total_mass
    return x_center, y_center


def fit_and_sample_voigt_2d(data, output_size, display=False):
    x_center, y_center = compute_centroid(data)

    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y)

    # Initial guess for parameters: amplitude, center x, center y, sigma, gamma
    initial_guess = [np.max(data), x_center, y_center, 3, 3]

    # Flatten X, Y, data for fitting
    xdata = np.vstack((X.ravel(), Y.ravel()))

    # Objective function to minimize
    def objective(params):
        amp, x0, y0, sigma_g, gamma_l = params
        model = voigt_2d(xdata[0], xdata[1], amp, x0, y0, sigma_g, gamma_l)
        return np.sum((model - data.ravel()) ** 2)

    # Minimize the objective function
    result = minimize(objective, initial_guess, method='L-BFGS-B')

    # Extract the best-fit parameters
    amp, x0, y0, sigma_g, gamma_l = result.x

    # Prepare new grid for the specified output size
    new_x = np.linspace(0, data.shape[0], output_size[0])
    new_y = np.linspace(0, data.shape[1], output_size[1])
    New_X, New_Y = np.meshgrid(new_x, new_y)

    # Generate sampled model on the new grid
    sampled_model = voigt_2d(New_X, New_Y, amp, x0, y0, sigma_g, gamma_l)

    # Enforce non-negativity
    sampled_model[sampled_model < 0] = 0

    if display:
        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, data, cmap='viridis', linewidth=0, antialiased=False)
        ax1.set_title('Original Noisy PSF')

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(New_X, New_Y, sampled_model, cmap='viridis', linewidth=0, antialiased=False)
        ax2.set_title('Sampled PSF from Fitted Model')

        plt.show()

    return sampled_model


def gaussian_2d(x, y, amp, x0, y0, sigma_x, sigma_y):
    """2D Gaussian function."""
    return amp * np.exp(-(((x - x0) ** 2 / (2 * sigma_x ** 2)) + ((y - y0) ** 2 / (2 * sigma_y ** 2))))


def compute_centroid(data):
    """Compute the centroid of a 2D array."""
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y)

    total_mass = np.sum(data)
    x_center = np.sum(X * data) / total_mass
    y_center = np.sum(Y * data) / total_mass
    return x_center, y_center


def fit_and_sample_gaussian_2d(data, output_size, display=False):
    x_center, y_center = compute_centroid(data)

    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y)

    # Initial guess: amplitude, center x, center y, sigma_x, sigma_y
    initial_guess = [np.max(data), x_center, y_center, 3, 3]

    # Flatten X, Y, data for fitting
    xdata = np.vstack((X.ravel(), Y.ravel()))

    # Objective function to minimize
    def objective(params):
        amp, x0, y0, sigma_x, sigma_y = params
        model = gaussian_2d(xdata[0], xdata[1], amp, x0, y0, sigma_x, sigma_y)
        return np.sum((model - data.ravel()) ** 2)

    # Minimize the objective function
    result = minimize(objective, initial_guess, method='L-BFGS-B')

    # Extract the best-fit parameters
    amp, x0, y0, sigma_x, sigma_y = result.x

    # Prepare new grid for the specified output size
    new_x = np.linspace(0, data.shape[0], output_size[0])
    new_y = np.linspace(0, data.shape[1], output_size[1])
    New_X, New_Y = np.meshgrid(new_x, new_y)

    # Generate sampled model on the new grid
    sampled_model = gaussian_2d(New_X, New_Y, amp, x0, y0, sigma_x, sigma_y)

    # Enforce non-negativity
    sampled_model[sampled_model < 0] = 0

    if display:
        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, data, cmap='viridis', linewidth=0, antialiased=False)
        ax1.set_title('Original Noisy PSF')

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(New_X, New_Y, sampled_model, cmap='viridis', linewidth=0, antialiased=False)
        ax2.set_title('Sampled PSF from Fitted Gaussian Model')

        plt.show()

    return sampled_model


def smooth_psf(data, sigma=2, kernel_size=5, display=False):
    """
    Smooth a 2D PSF array using a Gaussian kernel.

    Parameters:
    - data: 2D NumPy array of the PSF to be smoothed.
    - sigma: Standard deviation for Gaussian kernel.
    - kernel_size: Size of the Gaussian kernel (ignored in this function as scipy calculates it based on sigma).

    Returns:
    - smoothed_data: Smoothed 2D PSF array.
    """
    # Apply Gaussian filter (kernel size is automatically determined by sigma in scipy)
    smoothed_data = gaussian_filter(data, sigma=sigma)

    if display:
        fig = plt.figure(figsize=(14, 6))
        x = np.arange(data.shape[0])
        y = np.arange(data.shape[1])
        X, Y = np.meshgrid(x, y)

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, data, cmap='viridis', linewidth=0, antialiased=False)
        ax1.set_title('Original Noisy PSF')

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(X, Y, smoothed_data, cmap='viridis', linewidth=0, antialiased=False)
        ax2.set_title('Smoothed PSF with Gaussian Kernel')

        plt.show()

    return smoothed_data


def detect_and_extract_peak(image):
    # Step 1: Pre-process the image
    # Mask out the central peak, assuming it's in the very center of the image
    size = image.shape[0]
    center = size // 2
    masked_image = image.copy()
    masked_image[center - 25:center + 25, center - 25:center + 25] = 0  # Adjust the size as needed

    # Enhance other peaks
    filtered_image = filters.gaussian(masked_image, sigma=1)

    # Thresholding to keep higher intensities
    thresh = 30
    binary_image = filtered_image > thresh

    # Step 2: Detect peaks using a blob detection method or connected components
    blobs = feature.blob_log(binary_image, min_sigma=1, max_sigma=5, num_sigma=10, threshold=0.1)

    # Step 3: Calculate circularity and find the most circular peak
    circularities = []
    peak_coords = []

    for blob in blobs:
        y, x, r = blob
        r=30
        region = image[int(y - r):int(y + r + 1), int(x - r):int(x + r + 1)]
        label_img = measure.label(region > thresh)
        regions = measure.regionprops(label_img)

        if regions:
            region = regions[0]  # Assuming the largest connected component is the peak
            area = region.area
            perimeter = region.perimeter
            if perimeter == 0:
                circularity = 0
            else:
                circularity = 4 * np.pi * (area / perimeter ** 2)

            circularities.append(circularity)
            peak_coords.append((y, x, r))

    # Find the most circular peak
    if circularities:
        most_circular_idx = np.argmax(circularities)
        best_peak = peak_coords[most_circular_idx]
        y, x, r = best_peak
        peak_region = image[int(y - r):int(y + r + 1), int(x - r):int(x + r + 1)]
    else:
        peak_region = np.array([])  # No peaks found

    return peak_region


def dfile_without_deconvolution(SDATA, DIFFIMAGES, datafile):

    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = sd.io.Datafiles.read(SDATA, datafile_name)
    arr = sd.io.Arrays.rescale(arr, R, order=3)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = sd.io.Arrays.remove_edges(arr, img_size*R, xc, yc)

    return(arr)


def average_datafiles(SDATA, DIFFIMAGES, df, deconv=0):
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafiles = [datafile[1] for datafile in df.iterrows()]
    sum_arr = np.zeros((img_size * R, img_size * R), dtype=np.float32)
    total_tasks = len(datafiles)
    sys.stderr = sys.stdout

    with tqdm.tqdm(total=total_tasks, desc="Processing ") as pbar:
        try:
            for index, datafile in df.iterrows():
                if deconv == 0:
                    sum_arr += dfile_without_deconvolution(
                        SDATA, DIFFIMAGES, datafile)
                pbar.update(1)
            sum_arr = sum_arr/len(df)
        except Exception as e:
            print(f"Error processing a task: {str(e)}")

    print('')

    return sum_arr


