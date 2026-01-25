from scipy.ndimage import map_coordinates, center_of_mass
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from numba import jit
import bcorr
from scipy import stats, spatial
from skimage.morphology import opening, disk
from skimage.feature import (blob_log, blob_doh, peak_local_max, hessian_matrix_eigvals, hessian_matrix)
from skimage.filters import gaussian, median
from skimage.restoration import rolling_ball
import numpy.polynomial.polynomial as poly
import time
import cv2
from collections import defaultdict
from skimage.transform import resize
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from scipy.ndimage import map_coordinates
import os
import torch
from FCN_test import RobustUNetLightning
import traceback


# Move Numba functions outside the class
@jit(nopython=True)
def gaussian_numba(x, amplitude, mean, std):
    """Single Gaussian function optimized with Numba"""
    return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)


@jit(nopython=True)
def multi_gaussian_numba(x, params):
    """Multiple Gaussians sum optimized with Numba"""
    y = np.zeros_like(x)
    n_gaussians = len(params) // 3
    for i in range(n_gaussians):
        amp = params[i * 3]
        mean = params[i * 3 + 1]
        std = params[i * 3 + 2]
        y += gaussian_numba(x, amp, mean, std)
    return y


@jit(nopython=True)
def optimize_peaks(row_data, peaks):
    """Optimize peak parameters using Numba with improved sensitivity"""
    results = []
    for peak in peaks:
        start = max(0, peak - 15)
        end = min(len(row_data), peak + 16)
        local_data = row_data[start:end]

        peak_height = row_data[peak]
        sorted_local = np.sort(local_data)
        background = sorted_local[int(len(sorted_local) * 0.1)]

        prominence = peak_height - background
        width = 0
        half_height = background + prominence / 2

        left = peak
        right = peak
        while left > start and (row_data[left] > half_height or
                                (left > 0 and row_data[left] > row_data[left - 1])):
            left -= 1
            width += 1
        while right < end - 1 and (row_data[right] > half_height or
                                   (right < len(row_data) - 1 and row_data[right] > row_data[right + 1])):
            right += 1
            width += 1

        results.append((prominence, width, peak_height, background))

    return results


class PolarProcessor:
    """Class for processing images using polar transformation and Gaussian peak fitting."""

    def __init__(self):
        """Initialize the PolarProcessor."""
        pass

    @staticmethod
    def _center_image(image):
        """Centers the image based on its center of mass using zero padding."""
        image = image.astype(np.float32)
        com_y, com_x = center_of_mass(image)

        shift_y = image.shape[0] // 2 - com_y
        shift_x = image.shape[1] // 2 - com_x

        translation = np.array([[1, 0, shift_x],
                                [0, 1, shift_y]], dtype=np.float32)

        centered_image = cv2.warpAffine(image, translation,
                                        (image.shape[1], image.shape[0]),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)

        return centered_image

    def _cartesian_to_polar(self, image, center=None, radius=None, angle_step=1):
        """Convert Cartesian coordinates to polar."""
        image = image.astype(np.float32)

        if center is None:
            center = (image.shape[1] // 2, image.shape[0] // 2)
        if radius is None:
            radius = int(np.sqrt(center[0] ** 2 + center[1] ** 2))

        angles = np.arange(0, 360, angle_step, dtype=np.float32)
        radii = np.linspace(0, radius, radius, dtype=np.float32)

        angle_grid, radius_grid = np.meshgrid(np.radians(angles), radii)

        x_cart = center[0] + radius_grid * np.cos(angle_grid)
        y_cart = center[1] + radius_grid * np.sin(angle_grid)

        polar_image = map_coordinates(image, [y_cart.ravel(), x_cart.ravel()],
                                      order=1, mode='constant', cval=0)

        return polar_image.reshape((radius, len(angles)))

    def _polar_to_cartesian(self, polar_image, output_shape, center=None, angle_step=1):
        """Convert polar image back to Cartesian coordinates."""
        polar_image = polar_image.astype(np.float32)

        if center is None:
            center = (output_shape[1] // 2, output_shape[0] // 2)

        y, x = np.meshgrid(np.arange(output_shape[0], dtype=np.float32),
                           np.arange(output_shape[1], dtype=np.float32))

        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        theta = np.arctan2(y - center[1], x - center[0])
        theta_deg = np.mod(np.rad2deg(theta), 360)

        r_coord = r.T
        theta_coord = theta_deg.T / angle_step

        cartesian_image = map_coordinates(polar_image,
                                          [r_coord.ravel(), theta_coord.ravel()],
                                          order=1, mode='constant', cval=0)

        return cartesian_image.reshape(output_shape)

    def _process_single_row(self, row_data, row_index):
        """Process a single row with improved peak detection"""
        row_data = row_data.astype(np.float32)

        local_std = np.std(row_data)
        noise_level = np.median(np.abs(row_data - np.median(row_data)))

        radius_factor = np.sqrt(1.0 - 0.6 * (row_index / 200))
        base_prominence = max(2, 3 * noise_level)
        scaled_prominence = base_prominence * radius_factor

        peaks, _ = find_peaks(row_data,
                              prominence=scaled_prominence,
                              width=(2, 30),
                              height=scaled_prominence * 1.1,
                              distance=5)

        if len(peaks) == 0:
            return row_index, np.zeros_like(row_data), []

        peak_properties = optimize_peaks(row_data, peaks)  # Use the standalone function

        valid_peaks = []
        local_max = np.max(row_data)
        for i, (prominence, width, height, background) in enumerate(peak_properties):
            min_prominence = scaled_prominence * (0.7 + 0.3 * (height / local_max))
            min_width = 2 if height < local_max * 0.3 else 3
            max_width = 30

            if (prominence > min_prominence and
                    min_width <= width <= max_width and
                    height > background * 1.2):
                valid_peaks.append((peaks[i], height, width))

        if not valid_peaks:
            return row_index, np.zeros_like(row_data), []

        initial_params = np.zeros(len(valid_peaks) * 3, dtype=np.float32)
        for i, (peak_idx, height, width) in enumerate(valid_peaks):
            initial_params[i * 3] = height
            initial_params[i * 3 + 1] = peak_idx
            initial_params[i * 3 + 2] = width / 2.5

        try:
            x = np.arange(len(row_data), dtype=np.float32)
            bounds = ([0, 0, 0.5] * len(valid_peaks),
                      [float(row_data.max()), len(row_data), 60] * len(valid_peaks))

            params, _ = curve_fit(lambda x, *params: multi_gaussian_numba(x, np.array(params)),
                                  x, row_data, p0=initial_params, bounds=bounds,
                                  maxfev=1000, ftol=1e-4, xtol=1e-4)

            resampled_data = multi_gaussian_numba(x, params)
            peak_params = [(params[i], params[i + 1], params[i + 2])
                           for i in range(0, len(params), 3)]

            return row_index, resampled_data, peak_params

        except RuntimeError:
            return row_index, np.zeros_like(row_data), []

    def process_image(self, image, show_plots=False):
        """
        Process the input image using polar transformation and peak detection.

        Args:
            image (numpy.ndarray): Input image to process
            show_plots (bool): If True, displays original and processed images

        Returns:
            numpy.ndarray: Reconstructed image
        """
        # Center and preprocess image
        centered_image = image
        preprocessed_image = bcorr.rolling_ball(centered_image, radius=1)
        polar_image = self._cartesian_to_polar(preprocessed_image)

        # Process rows sequentially
        rows_to_analyze = polar_image[15:]
        results = []

        print("Processing rows sequentially...")
        for idx, row in enumerate(rows_to_analyze, start=15):
            if idx % 20 == 0:
                print(f"Processing row {idx}...")
            results.append(self._process_single_row(row, idx))

        # Reconstruct the image
        full_resampled_polar = np.zeros_like(polar_image)
        for row_idx, resampled_data, _ in results:
            full_resampled_polar[row_idx] = resampled_data

        reconstructed_image = self._polar_to_cartesian(full_resampled_polar, image.shape)

        if show_plots:
            self._plot_results(image, reconstructed_image, polar_image, full_resampled_polar)

        # Print statistics
        peak_counts = [len(result[2]) for result in results]
        print("\nProcessing completed!")
        print(f"Average peaks per row: {np.mean(peak_counts):.1f}")
        print(f"Total peaks found: {sum(peak_counts)}")

        return reconstructed_image

    def _plot_results(self, original_image, reconstructed_image, polar_image, resampled_polar, intensity_cutoff=150):
        """Plot the original and processed images."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Original image
        axes[0, 0].imshow(np.where(original_image > intensity_cutoff, intensity_cutoff, original_image),
                          cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Reconstructed image
        axes[0, 1].imshow(np.where(reconstructed_image > intensity_cutoff,
                                   intensity_cutoff, reconstructed_image), cmap='gray')
        axes[0, 1].set_title('Reconstructed Image')
        axes[0, 1].axis('off')

        # Original polar image
        axes[1, 0].imshow(np.where(polar_image > intensity_cutoff,
                                   intensity_cutoff, polar_image),
                          cmap='gray', aspect='auto')
        axes[1, 0].set_title('Original Polar Transform')
        axes[1, 0].set_xlabel('Angle (degrees)')
        axes[1, 0].set_ylabel('Radius (pixels)')

        # Resampled polar image
        axes[1, 1].imshow(np.where(resampled_polar > intensity_cutoff,
                                   intensity_cutoff, resampled_polar),
                          cmap='gray', aspect='auto')
        axes[1, 1].set_title('Gaussian Approximation in Polar Coordinates')
        axes[1, 1].set_xlabel('Angle (degrees)')
        axes[1, 1].set_ylabel('Radius (pixels)')

        plt.tight_layout()
        plt.show()


class DiffractionPeakSegmenter:
    """
    Segments diffraction peaks, using standard or power-weighted CoM for positioning.
    """
    def __init__(self,
                 start_row: int = 55,
                 threshold_factor: float = 3,
                 center_radius: int = 100,
                 pre_mask_center_pixels: int = 0,
                 angle_step: float = 1.0,
                 polar_radius: Optional[int] = None,
                 min_blob_size: int = 30,
                 centroid_power: float = 2,
                 verbose=False
                 ):
        """
        Args:
            start_row (int): First row index (radius) in polar image to process.
            threshold_factor (float): Multiplier for MAD in adaptive thresholding.
            center_radius (int): Radius for initial image centering CoM.
            pre_mask_center_pixels (int): Radius around center to mask before polar transform.
            angle_step (float): Angular step in degrees for polar transform.
            polar_radius (Optional[int]): Number of rows (radius samples) in the polar image.
                                         If None, defaults based on image size.
            min_blob_size (int): Minimum number of pixels for a blob to be kept in the
                                 Cartesian mask after morphological filtering. Set to 0 to disable.
        """
        # Input validation
        if not isinstance(start_row, int) or start_row < 0:
            raise ValueError("start_row must be non-negative integer.")
        if not isinstance(threshold_factor, (int, float)) or threshold_factor <= 0:
            raise ValueError("threshold_factor must be positive.")
        if not isinstance(center_radius, int) or center_radius <= 0:
            raise ValueError("center_radius must be positive integer.")
        if not isinstance(pre_mask_center_pixels, int) or pre_mask_center_pixels < 0:
            raise ValueError("pre_mask_center_pixels must be non-negative integer.")
        if not isinstance(angle_step, (int, float)) or not (0 < angle_step <= 360):
            raise ValueError("angle_step must be > 0 and <= 360.")
        if polar_radius is not None and (not isinstance(polar_radius, int) or polar_radius <= 0):
            raise ValueError("polar_radius must be positive integer if specified.")
        if not isinstance(min_blob_size, int) or min_blob_size < 0:
            raise ValueError("min_blob_size must be a non-negative integer.")


        self.start_row = start_row
        self.threshold_factor = threshold_factor
        self.center_radius = center_radius
        self.pre_mask_center_pixels = pre_mask_center_pixels
        self.angle_step = angle_step
        self.polar_radius = polar_radius
        self.min_blob_size = min_blob_size
        self.centroid_power = centroid_power

        self.center: Optional[Tuple[float, float]] = None
        self._polar_rows: Optional[int] = None
        self._polar_cols: Optional[int] = None

        self.verbose = verbose
        if self.verbose:
            print(f"Initialized DiffractionPeakSegmenter with start_row={start_row},"
                  f" threshold_factor={threshold_factor}, center_radius={center_radius},"
                  f" pre_mask_center={pre_mask_center_pixels}px, angle_step={angle_step},"
                  f" polar_radius={polar_radius or 'auto'}, min_blob_size={min_blob_size},"
                  f" centroid_power={centroid_power}")

    def _calculate_shift_and_center(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:

        rows, cols = image.shape
        center_y, center_x = (rows - 1) / 2.0, (cols - 1) / 2.0
        self.center = (center_x, center_y)
        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist_from_center <= self.center_radius
        masked_image = image * mask
        total_intensity_center = np.sum(masked_image)
        if total_intensity_center < 1e-9:
            com_y, com_x = center_y, center_x
        else:
            com_y, com_x = ndimage.center_of_mass(masked_image)
        shift_y = center_y - com_y
        shift_x = center_x - com_x
        shift = (shift_y, shift_x)
        if self.verbose:
            print(f"Calculated CoM at ({com_y:.2f}, {com_x:.2f}). Applying shift ({shift_y:.2f}, {shift_x:.2f}).")
        centered_image = ndimage.shift(image, shift, order=3, mode='constant', cval=0.0)
        return centered_image, shift

    def _to_polar(self, image: np.ndarray) -> np.ndarray:

        if self.center is None:
            raise RuntimeError("Center must be set.")
        img = image.astype(np.float32)
        h, w = img.shape
        cx, cy = self.center
        if self.polar_radius is not None:
            maxr = self.polar_radius
        else:
            maxr = max(1, int(np.ceil(max(h, w) / 2.0)))
        self._polar_rows = maxr
        if self.angle_step == 360:
            angles = np.array([0], dtype=np.float32)
        else:
            angles = np.arange(0, 360, self.angle_step, dtype=np.float32)
        self._polar_cols = len(angles)
        radii = np.linspace(0, maxr, maxr, dtype=np.float32)
        ang_grid, rad_grid = np.meshgrid(np.radians(angles), radii)
        x_cart = cx + rad_grid * np.cos(ang_grid)
        y_cart = cy + rad_grid * np.sin(ang_grid)
        polar_values = map_coordinates(img, [y_cart.ravel(), x_cart.ravel()], order=1, mode='constant', cval=0.0)
        polar_image = polar_values.reshape((maxr, len(angles)))
        return polar_image

    def _to_cartesian(self, polar_image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:

        if self.center is None:
            raise RuntimeError("Center must be set.")
        if self._polar_rows is None or self._polar_cols is None:
            raise RuntimeError("Polar dimensions not set.")
        pol = polar_image.astype(np.float32)
        h, w = target_shape
        cx, cy = self.center
        y_cart_grid, x_cart_grid = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
        radius_grid = np.hypot(x_cart_grid - cx, y_cart_grid - cy)
        angle_grid_deg = np.degrees(np.arctan2(y_cart_grid - cy, x_cart_grid - cx))
        angle_grid_deg = np.mod(angle_grid_deg, 360)
        idx_r = radius_grid
        idx_th = angle_grid_deg / self.angle_step
        interp_order = 0 if np.issubdtype(polar_image.dtype, np.integer) else 1
        cartesian_values = map_coordinates(pol, [idx_r.ravel(), idx_th.ravel()], order=interp_order, mode='constant', cval=0.0)
        cartesian_image = cartesian_values.reshape((h, w))
        return cartesian_image

    def _adaptive_threshold_polar(self, polar_image: np.ndarray) -> np.ndarray:

        rows, cols = polar_image.shape
        polar_mask = np.zeros_like(polar_image, dtype=np.uint8)
        if self.start_row >= rows:
            return polar_mask
        num_thresholded = 0
        for r in range(self.start_row, rows):
            row_data = polar_image[r, :]
            median_val = np.median(row_data)
            abs_dev = np.abs(row_data - median_val)
            mad_val = np.median(abs_dev)
            if mad_val < 1e-6:
                threshold = median_val + self.threshold_factor * np.std(row_data)
            else:
                threshold = median_val + self.threshold_factor * 1.4826 * mad_val
            effective_threshold = max(threshold, median_val + 1e-6, 1e-6)
            row_mask = row_data > effective_threshold
            polar_mask[r, :] = row_mask.astype(np.uint8)
            num_thresholded += np.sum(row_mask)
        if self.verbose:
            print(f"Adaptive thresholding complete. Pixels masked in polar (r >= {self.start_row}): {num_thresholded}")
        return polar_mask

    def _shift_back(self, image: np.ndarray, shift: Tuple[float, float]) -> np.ndarray:

        inverse_shift = (-shift[0], -shift[1])
        if self.verbose:
            print(f"Shifting result back by ({inverse_shift[0]:.2f}, {inverse_shift[1]:.2f}) using Nearest Neighbor.")
        shifted_back_image = ndimage.shift(image, inverse_shift, order=0, mode='constant', cval=0.0)
        return shifted_back_image

    def _plot_results_detailed(self, images_dict: Dict[str, np.ndarray], vmax: Optional[float] = None, vmin_final: Optional[float] = 0):

        num_images = len(images_dict)
        cols = 3
        rows = (num_images + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axes = axes.ravel()
        plot_idx = 0
        for title, img in images_dict.items():
            if img is None:
                continue
            ax = axes[plot_idx]
            is_mask = "Mask" in title
            is_polar = "Polar" in title
            is_final_conc = "Concentrated Peaks (Final)" in title
            cmap = 'gray' if is_mask else ('viridis' if is_polar else 'hot')
            current_vmax = None if is_mask else vmax
            current_vmin = vmin_final if is_final_conc else None
            im = ax.imshow(img, cmap=cmap, origin='lower', aspect='auto' if is_polar else 'equal',
                           vmax=current_vmax, vmin=current_vmin)
            ax.set_title(f"{title}\nShape: {img.shape}")
            ax.set_xlabel('Column Index')
            ax.set_ylabel('Row Index')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plot_idx += 1
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()

    def _find_peaks_and_concentrate(self,
                                    cartesian_mask: np.ndarray,
                                    centered_image: np.ndarray
                                    ) -> np.ndarray:

        rows, cols = centered_image.shape
        power = self.centroid_power
        method_name = "standard CoM" if power == 1 else f"power-weighted CoM (p={power})"
        if self.verbose:
            print(f"\n--- Entering Peak Concentration (using {method_name}) ---")
            print(f"Input mask shape: {cartesian_mask.shape}, dtype: {cartesian_mask.dtype},"
                  f" Num mask pixels: {np.sum(cartesian_mask)}")
            print(f"Input image shape: {centered_image.shape}, dtype: {centered_image.dtype}")

        if not np.issubdtype(cartesian_mask.dtype, np.integer):
            cartesian_mask = cartesian_mask.astype(np.uint8)

        concentrated_image = np.zeros_like(centered_image, dtype=float)
        labeled_mask, num_labels = ndimage.label(cartesian_mask, structure=np.ones((3, 3), dtype=bool))
        if self.verbose:
            print(f"ndimage.label found {num_labels} blobs.")

        if num_labels == 0:
            if self.verbose:
                print("--- Exiting Peak Concentration (No blobs found in mask) ---")
            return concentrated_image

        t_start = time.time()
        indices = np.arange(1, num_labels + 1)

        try:
            if self.verbose:
                print("Calculating Sums using ndimage.sum_labels...")
            sums = ndimage.sum_labels(centered_image, labels=labeled_mask, index=indices)
            if self.verbose:
                print(f"Calculated Sums for {len(sums)} labels.")
            if len(sums) != num_labels:
                print(f"*** Warning: Mismatch sum lengths! Sums:{len(sums)}, Labels:{num_labels} ***")
            if num_labels > 0:
                if self.verbose:
                    print(f"First few Sums: {sums[:min(5, num_labels)]}")
        except Exception as e:
            print(f"*** CRITICAL ERROR during ndimage.sum_labels: {e} ***")
            return concentrated_image

        coms = []
        if power == 1.0:
            if self.verbose:
                print("Calculating standard CoM using ndimage.center_of_mass...")
            try:
                coms_nd = ndimage.center_of_mass(centered_image, labels=labeled_mask, index=indices)

                if num_labels == 1:
                    coms = [coms_nd] if isinstance(coms_nd, tuple) else list(coms_nd)
                else:
                    coms = list(coms_nd)

                if self.verbose:
                    print(f"Calculated CoM for {len(coms)} labels.")
                if len(coms) != num_labels:
                    print(f"*** Warning: Mismatch CoM lengths! CoMs:{len(coms)}, Labels:{num_labels} ***")
                if num_labels > 0:
                    if self.verbose:
                        print(f"First few CoMs (y, x): {coms[:min(5, num_labels)]}")
            except Exception as e:
                print(f"*** CRITICAL ERROR during ndimage.center_of_mass: {e} ***"); return concentrated_image
        else:
            # Manual calculation for power-weighted CoM (SLOWER!)
            if self.verbose:
                print(f"Calculating power-weighted CoM (p={power}) manually per blob...")

            fg_coords = np.argwhere(labeled_mask > 0)
            fg_labels = labeled_mask[fg_coords[:, 0], fg_coords[:, 1]]
            fg_intensities = centered_image[fg_coords[:, 0], fg_coords[:, 1]]
            fg_intensities_clipped = np.maximum(fg_intensities, 1e-9)

            I_weighted = fg_intensities_clipped ** power

            sum_y_weighted = ndimage.sum_labels(fg_coords[:, 0] * I_weighted, labels=fg_labels, index=indices)
            sum_x_weighted = ndimage.sum_labels(fg_coords[:, 1] * I_weighted, labels=fg_labels, index=indices)
            sum_I_weighted = ndimage.sum_labels(I_weighted, labels=fg_labels, index=indices)

            coms = []
            for i in range(num_labels):
                if sum_I_weighted[i] > 1e-9:
                    com_y = sum_y_weighted[i] / sum_I_weighted[i]
                    com_x = sum_x_weighted[i] / sum_I_weighted[i]
                    coms.append((com_y, com_x))
                else:
                    coms.append((np.nan, np.nan))
            if self.verbose:
                print(f"Calculated power-weighted CoM for {len(coms)} labels.")
            if num_labels > 0:
                if self.verbose:
                    print(f"First few Weighted CoMs (y, x): {coms[:min(5, num_labels)]}")

        t_end = time.time()
        if self.verbose:
            print(f"CoM Calculation took {t_end - t_start:.3f} seconds.")

        peak_count = 0
        invalid_count = 0
        outside_bounds_count = 0
        if self.verbose:
            print("Assigning summed intensities to CoM locations...")
        for i in range(num_labels):
            if i >= len(coms) or i >= len(sums):
                invalid_count += 1
                continue

            current_com = coms[i]
            current_sum = sums[i]

            if not isinstance(current_com, (tuple, list)) or len(current_com) != 2:
                invalid_count += 1
                continue
            com_y, com_x = current_com

            if np.isnan(com_y) or np.isnan(com_x) or current_sum <= 0:
                invalid_count += 1
                continue

            peak_y, peak_x = int(round(com_y)), int(round(com_x))

            if 0 <= peak_y < rows and 0 <= peak_x < cols:
                concentrated_image[peak_y, peak_x] += current_sum
                peak_count += 1
            else:
                outside_bounds_count += 1
        if self.verbose:
            print(f"Finished assignment loop.")
            print(f"  Placed {peak_count} peaks into output image.")
            print(f"  Skipped {invalid_count} peaks due to invalid CoM/Sum.")
            print(f"  Skipped {outside_bounds_count} peaks due to CoM outside bounds.")

        final_num_peaks = np.count_nonzero(concentrated_image)
        final_max_val = np.max(concentrated_image) if final_num_peaks > 0 else 0
        if self.verbose:
            print(f"Output image: {final_num_peaks} non-zero pixels, Max value: {final_max_val:.2f}")
        if final_num_peaks > 0:
            max_loc = np.unravel_index(np.argmax(concentrated_image), concentrated_image.shape)
            if self.verbose:
                print(f"Location (y, x) of max value in output: {max_loc}")
        if self.verbose:
            print(f"--- Exiting Peak Concentration ---")
        return concentrated_image

    def process_image(self, image: np.ndarray, show_results: bool = False) -> np.ndarray:
        """
        Processes the input image using standard or power-weighted CoM.
        """
        if image.ndim != 2:
            raise ValueError("Input image must be 2-dimensional.")
        if self.verbose:
            print("-" * 30)
            print("Starting Diffraction Peak Segmentation Process...")
        rows, cols = image.shape
        if self.verbose:
            print(f"Input image size: {rows}x{cols}")
        centered_image, shift_applied = self._calculate_shift_and_center(image)
        centered_shape = centered_image.shape
        image_for_polar = centered_image.copy()
        if self.pre_mask_center_pixels > 0 and self.center is not None:
            cx, cy = self.center
            center_y_idx, center_x_idx = int(round(cy)), int(round(cx))
            rr, cc = np.ogrid[:rows, :cols]
            dist_sq = (rr - center_y_idx)**2 + (cc - center_x_idx)**2
            mask_area = dist_sq < self.pre_mask_center_pixels**2
            image_for_polar[mask_area] = 0.0

        polar_image = self._to_polar(image_for_polar)
        polar_mask = self._adaptive_threshold_polar(polar_image)
        raw_cartesian_mask = self._to_cartesian(polar_mask, centered_shape)
        cartesian_mask = (raw_cartesian_mask > 0.5).astype(np.uint8)
        filtered_cartesian_mask = None

        if self.min_blob_size > 0:
            if self.verbose:
                print(f"Filtering Cartesian mask: Removing blobs smaller than {self.min_blob_size} pixels...")
            structure = np.ones((3, 3), dtype=bool)
            labeled_mask, num_labels = ndimage.label(cartesian_mask, structure=structure)
            if num_labels > 0:
                blob_sizes = ndimage.sum_labels(cartesian_mask, labels=labeled_mask, index=np.arange(1, num_labels + 1))
                keep_labels_indices = np.where(blob_sizes >= self.min_blob_size)[0]
                keep_labels = keep_labels_indices + 1
                filtered_cartesian_mask = np.isin(labeled_mask, keep_labels)
                num_kept = len(keep_labels); num_removed = num_labels - num_kept
                if self.verbose:
                    print(f"Blob filtering kept {num_kept} blobs (removed {num_removed}).")
                cartesian_mask = filtered_cartesian_mask.astype(np.uint8)

        concentrated_centered_image = self._find_peaks_and_concentrate(
            cartesian_mask=cartesian_mask,
            centered_image=centered_image
        )

        final_concentrated_image = self._shift_back(concentrated_centered_image, shift_applied)
        if self.verbose:
            print("Processing finished.")
            print("-" * 30)

        # 7. Plot results
        if show_results:
            print("Displaying results...")
            images_to_plot = {
                "Original Image": image,
                "Centered Image": centered_image,
                "Polar Mask": polar_mask,
                "Cartesian Mask (Raw Intensity)": (raw_cartesian_mask > 0.5).astype(np.uint8),
                "Cartesian Mask (Filtered)": filtered_cartesian_mask if self.min_blob_size > 0 else cartesian_mask,
                "Concentrated Peaks (Final)": final_concentrated_image
            }
            if self.min_blob_size == 0 and "Cartesian Mask (Filtered)" in images_to_plot:
                del images_to_plot["Cartesian Mask (Filtered)"]
                images_to_plot["Cartesian Mask (Final Used)"] = cartesian_mask

            self._plot_results_detailed(images_dict=images_to_plot, vmax=300, vmin_final=0)

        return final_concentrated_image


class DiffractionPeakFinder:
    """
    A class to process diffraction images to find peaks using various detectors
    and background subtraction methods, representing peaks as a Dirac delta image.

    Supported detectors: 'log', 'doh', 'mser', 'pcbr', 'vote'.
    Supported background methods: 'opening', 'gaussian', 'median', 'rolling_ball', 'polynomial'.
    'vote' runs other supported methods ('log', 'doh', 'mser', 'pcbr') and combines results.
    """

    SUPPORTED_DETECTORS = ['log', 'doh', 'mser', 'pcbr', 'vote']
    VOTING_METHODS = ['log', 'doh', 'mser', 'pcbr']
    SUPPORTED_BACKGROUNDS = ['opening', 'opening_downsampled', 'gaussian', 'median', 'rolling_ball', 'polynomial']

    def __init__(self,
                 detector_method='log',
                 background_method='opening_downsampled',
                 # --- Common Parameters ---
                 threshold_factor=3.5,
                 center_ignore_radius_pixels=50,
                 # --- Background Parameters ---
                 bg_disk_radius=70, bg_gaussian_sigma=20.0, bg_median_radius=20,
                 bg_rolling_ball_radius=30, bg_poly_degree=4, bg_poly_percentile=25,
                 # --- LoG/DoH Parameters ---
                 peak_min_sigma=6, peak_max_sigma=60, num_sigma_steps=30, log_scale=False,
                 log_doh_downsample_factor=4,
                 # --- MSER Parameters (Requires OpenCV) ---
                 mser_delta=5, mser_min_area=60, mser_max_area=14400,
                 # --- Principal Curvature (PCBR) Parameters ---
                 pcbr_sigma=3.0, pcbr_lambda_thresh=0.5, pcbr_response_thresh_rel=0.1, pcbr_min_distance=5,
                 # --- Voting Parameters ---
                 vote_radius=5.0, min_votes=2,
                 integration_sigma_factor=3,
                 integration_fixed_radius=5,
                 verbose=False
                 ):
        """
        Initializes the finder with processing parameters.

        Args:
            detector_method (str): Method for peak detection.
            background_method (str): Method for background estimation.
            # ... (parameters for background, log, doh, mser, pcbr, vote as before) ...
            # Phase Congruency ('pc') parameters have been removed.
        """
        # --- Validation Checks ---
        if detector_method not in self.SUPPORTED_DETECTORS:
            raise ValueError(f"detector_method must be one of {self.SUPPORTED_DETECTORS}, got '{detector_method}'")
        if background_method not in self.SUPPORTED_BACKGROUNDS:
             raise ValueError(f"background_method must be one of {self.SUPPORTED_BACKGROUNDS}, got '{background_method}'")
        if detector_method == 'vote' and min_votes < 1:
             raise ValueError("min_votes must be at least 1 for 'vote' method.")
        if not 0 < bg_poly_percentile < 100:
             raise ValueError("bg_poly_percentile must be between 0 and 100.")

        self.detector_method = detector_method
        self.background_method = background_method
        # --- Store Parameters ---
        # Common
        self.threshold_factor=threshold_factor; self.center_ignore_radius_pixels=center_ignore_radius_pixels
        # Background
        self.bg_disk_radius=bg_disk_radius; self.bg_gaussian_sigma=bg_gaussian_sigma; self.bg_median_radius=bg_median_radius; self.bg_rolling_ball_radius=bg_rolling_ball_radius; self.bg_poly_degree=bg_poly_degree; self.bg_poly_percentile=bg_poly_percentile
        # Detectors
        self.peak_min_sigma=peak_min_sigma; self.peak_max_sigma=peak_max_sigma; self.num_sigma_steps=num_sigma_steps; self.log_scale=log_scale; self.log_doh_downsample_factor = int(log_doh_downsample_factor)
        self.mser_delta=mser_delta; self.mser_min_area=mser_min_area; self.mser_max_area=mser_max_area
        self.pcbr_sigma=pcbr_sigma; self.pcbr_lambda_thresh=pcbr_lambda_thresh; self.pcbr_response_thresh_rel=pcbr_response_thresh_rel; self.pcbr_min_distance=pcbr_min_distance
        # Voting
        self.vote_radius=vote_radius; self.min_votes=min_votes

        # --- Results Storage ---
        self.last_peaks_data = [] # List of tuples (row, col, intensity, score)
        self.last_noise_sigma = None
        self.last_background_img = None
        self.last_subtracted_img = None
        self.last_dirac_img = None
        self.last_detector_map = None # Only used by PCBR now
        self.last_vote_details = None
        self.integration_sigma_factor = float(integration_sigma_factor)
        self.integration_fixed_radius = int(integration_fixed_radius)
        self.verbose = verbose

        if self.verbose:
            print(f"DiffractionPeakFinder initialized:")
            print(f"  Detector: '{self.detector_method}'")
            print(f"  Background: '{self.background_method}'")

    def _estimate_background_opening(self, img):
        struct_el = disk(self.bg_disk_radius)
        return opening(img, struct_el)

    def _estimate_background_opening_downsampled(self, img, downscale_factor=3):
        """Morphological opening on downsampled image."""
        original_shape = img.shape
        factor = int(max(1, downscale_factor))
        new_shape = (original_shape[0] // factor, original_shape[1] // factor)
        img_small = resize(img, new_shape, anti_aliasing=True, order=1, preserve_range=True)
        small_radius = max(1, int(round(self.bg_disk_radius / factor)))
        struct_el = disk(small_radius)


        if self.verbose:
            print(f"Opening on downsampled {new_shape} with radius {small_radius}...")
        background_small = opening(img_small, struct_el)
        if self.verbose:
            print(f"Upsampling background to {original_shape}...")

        background_full = resize(background_small, original_shape, anti_aliasing=False, order=1, preserve_range=True)
        return background_full

    def _estimate_background_gaussian(self, img):
        return gaussian(img, sigma=self.bg_gaussian_sigma, preserve_range=True, truncate=4.0)

    def _estimate_background_median(self, img):
        footprint = disk(self.bg_median_radius)
        return median(img, footprint=footprint)

    def _estimate_background_rolling_ball(self, img): # ... (code as before) ...
        try:
            return rolling_ball(img, radius=self.bg_rolling_ball_radius)
        except TypeError as e: print("Warning: Check scikit-image version compatibility for rolling_ball.")
        background_subtracted = rolling_ball(img, radius=self.bg_rolling_ball_radius)
        return img - background_subtracted

    def _estimate_background_polynomial(self, img):
        print(f"    Polynomial fit params: degree={self.bg_poly_degree}, percentile={self.bg_poly_percentile}")
        start_time_poly = time.time()
        rows, cols = np.indices(img.shape)
        center_r, center_c = img.shape[0] // 2, img.shape[1] // 2
        radius = np.sqrt((rows - center_r)**2 + (cols - center_c)**2).astype(np.float32)
        img_flat = img.ravel()
        radius_flat = radius.ravel()
        valid_pixels = np.isfinite(img_flat)
        if not np.any(valid_pixels):
            print("Warning: Image contains only NaNs.")
            return np.zeros_like(img)
        threshold = np.percentile(img_flat[valid_pixels], self.bg_poly_percentile)
        bg_indices = np.where(valid_pixels & (img_flat < threshold))[0]
        if len(bg_indices) < self.bg_poly_degree + 1:
             print(f"Warning: Not enough points ({len(bg_indices)}) for poly degree {self.bg_poly_degree}. Using mean fallback.")
             mean_val = np.mean(img_flat[bg_indices]) if len(bg_indices) > 0 else np.mean(img_flat[valid_pixels])
             return np.full_like(img, mean_val)
        try:
            coeffs = poly.polyfit(radius_flat[bg_indices], img_flat[bg_indices], self.bg_poly_degree)
            background_flat = poly.polyval(radius_flat, coeffs)
            background = background_flat.reshape(img.shape)
            background = np.clip(background, 0, None)
            print(f" Polynomial fit done ({time.time() - start_time_poly:.2f}s)")
            return background
        except Exception as e: print(f"ERROR during polynomial fit: {e}. Falling back to mean.")
        mean_val = np.mean(img_flat[bg_indices]) if len(bg_indices) > 0 else np.mean(img_flat[valid_pixels])
        return np.full_like(img, mean_val)

    def _estimate_background(self, img):
        method = self.background_method
        if self.verbose:
            print(f"  Estimating background using method: '{method}'...")
        start_time = time.time()
        bg_runners = {'opening': self._estimate_background_opening, 'opening_downsampled': self._estimate_background_opening_downsampled, 'gaussian': self._estimate_background_gaussian, 'median': self._estimate_background_median, 'rolling_ball': self._estimate_background_rolling_ball, 'polynomial': self._estimate_background_polynomial}
        if method in bg_runners:
            background = bg_runners[method](img)
            if self.verbose:
                print(f"Background estimation finished ({time.time() - start_time:.2f}s)")
            return background
        else:
            raise ValueError(f"Internal error: Unsupported background_method: {method}")

    def _estimate_noise(self, img):
        if self.verbose:
            print("  Estimating noise (MAD)...", end="")
        start_time = time.time()
        median_val = np.nanmedian(img)
        if np.isnan(median_val):
            print("\nWarning: Median is NaN.")
            return 1.0
        mad = stats.median_abs_deviation(img, axis=None, scale='normal', nan_policy='omit')
        if np.isnan(mad) or mad == 0:
            print("\nWarning: MAD is zero/NaN.")
            noise_sigma = 1.0
        else:
            noise_sigma = mad
        if self.verbose:
            print(f" sigma={noise_sigma:.3f} ({time.time() - start_time:.2f}s)")
        return noise_sigma

    def _run_log(self, img_subtracted, noise_sigma):
        """Runs LoG detector, with optional downsampling."""
        factor = self.log_doh_downsample_factor
        original_shape = img_subtracted.shape
        abs_threshold = noise_sigma * self.threshold_factor
        abs_threshold = max(abs_threshold, 1e-6) # Ensure positive

        if factor > 1:
            if self.verbose:
                print(f"      Running LoG on image downsampled by {factor}x")
            new_shape = (original_shape[0] // factor, original_shape[1] // factor)
            if new_shape[0] < 1 or new_shape[1] < 1:
                print("       Warning: Downsampled image too small. Skipping."); return np.array([]), np.array([]), np.array([])
            img_small = resize(img_subtracted, new_shape, anti_aliasing=True, order=1, preserve_range=True)

            # Scale sigma values
            min_sigma_small = max(0.5, self.peak_min_sigma / factor) # Min sigma typically >= 0.5 for skimage
            max_sigma_small = max(min_sigma_small + 0.5, self.peak_max_sigma / factor)
            if min_sigma_small >= max_sigma_small:
                print(f"      Warning: Invalid sigma range after downscaling ({min_sigma_small:.2f}-{max_sigma_small:.2f}). Skipping LoG."); return np.array([]), np.array([]), np.array([])
            if self.verbose:
                print(f"      LoG (downsampled) params: min_sig={min_sigma_small:.2f}, max_sig={max_sigma_small:.2f}, thresh={abs_threshold:.3f}")
            blobs_small = blob_log(img_small, min_sigma=min_sigma_small, max_sigma=max_sigma_small,
                                   num_sigma=self.num_sigma_steps, threshold=abs_threshold)

            if blobs_small.shape[0] == 0: return np.array([]), np.array([]), np.array([])
            # Scale results back to original coordinates/scale
            rows_orig = blobs_small[:, 0] * factor
            cols_orig = blobs_small[:, 1] * factor
            scores_orig = blobs_small[:, 2] * factor # Scale sigma score back
            return rows_orig, cols_orig, scores_orig

        else:
            if self.verbose:
                print(f"    LoG params: min_sig={self.peak_min_sigma}, max_sig={self.peak_max_sigma}, thresh={abs_threshold:.3f}")
            blobs = blob_log(img_subtracted, min_sigma=self.peak_min_sigma, max_sigma=self.peak_max_sigma,
                             num_sigma=self.num_sigma_steps, threshold=abs_threshold)
            if blobs.shape[0] == 0: return np.array([]), np.array([]), np.array([])
            return blobs[:, 0], blobs[:, 1], blobs[:, 2]

    def _run_doh(self, img_subtracted, noise_sigma):
        """Runs DoH detector, with optional downsampling."""
        factor = self.log_doh_downsample_factor
        original_shape = img_subtracted.shape
        # DoH threshold needs careful tuning, especially with downsampling. This is a heuristic.
        abs_threshold = noise_sigma * self.threshold_factor * 0.1
        abs_threshold = max(abs_threshold, 1e-6)

        if factor > 1:
            print(f"      Running DoH on image downsampled by {factor}x")
            new_shape = (original_shape[0] // factor, original_shape[1] // factor)
            if new_shape[0] < 1 or new_shape[1] < 1: print("       Warning: Downsampled image too small. Skipping."); return np.array([]), np.array([]), np.array([])
            img_small = resize(img_subtracted, new_shape, anti_aliasing=True, order=1, preserve_range=True)

            min_sigma_small = max(0.5, self.peak_min_sigma / factor)
            max_sigma_small = max(min_sigma_small + 0.5, self.peak_max_sigma / factor)
            if min_sigma_small >= max_sigma_small: print(f"      Warning: Invalid sigma range after downscaling ({min_sigma_small:.2f}-{max_sigma_small:.2f}). Skipping DoH."); return np.array([]), np.array([]), np.array([])

            print(f"      DoH (downsampled) params: min_sig={min_sigma_small:.2f}, max_sig={max_sigma_small:.2f}, thresh={abs_threshold:.3f} (NEEDS TUNING)")
            blobs_small = blob_doh(img_small, min_sigma=min_sigma_small, max_sigma=max_sigma_small,
                                   num_sigma=self.num_sigma_steps, threshold=abs_threshold, log_scale=self.log_scale)

            if blobs_small.shape[0] == 0: return np.array([]), np.array([]), np.array([])
            rows_orig = blobs_small[:, 0] * factor
            cols_orig = blobs_small[:, 1] * factor
            scores_orig = blobs_small[:, 2] * factor
            return rows_orig, cols_orig, scores_orig

        else:
            print(f"    DoH params: min_sig={self.peak_min_sigma}, max_sig={self.peak_max_sigma}, thresh={abs_threshold:.3f} (NEEDS TUNING)")
            blobs = blob_doh(img_subtracted, min_sigma=self.peak_min_sigma, max_sigma=self.peak_max_sigma,
                             num_sigma=self.num_sigma_steps, threshold=abs_threshold, log_scale=self.log_scale)
            if blobs.shape[0] == 0: return np.array([]), np.array([]), np.array([])
            return blobs[:, 0], blobs[:, 1], blobs[:, 2]
    def _run_mser(self, img_subtracted, noise_sigma):

     if 'cv2' not in globals(): raise ImportError("MSER requires OpenCV (cv2)")

     min_val, max_val = np.min(img_subtracted), np.max(img_subtracted)
     img_uint8 = ((img_subtracted - min_val) / (max_val - min_val) * 255).astype(np.uint8) if max_val > min_val else np.zeros_like(img_subtracted, dtype=np.uint8)
     mser = cv2.MSER_create(_delta=self.mser_delta, _min_area=self.mser_min_area, _max_area=self.mser_max_area)
     regions, bboxes = mser.detectRegions(img_uint8)
     if not regions:
         return np.array([]), np.array([]), np.array([])
     rows, cols, scores = [], [], []
     for i, pts in enumerate(regions):
        try:
            moments = cv2.moments(pts)
            if moments["m00"] != 0:
                rows.append(int(moments["m01"] / moments["m00"])); cols.append(int(moments["m10"] / moments["m00"]))
                scores.append(cv2.contourArea(pts))
        except Exception: pass
        return np.array(rows), np.array(cols), np.array(scores)

    def _run_pcbr(self, img_subtracted, noise_sigma):
        print(
            f"    PCBR params: sigma={self.pcbr_sigma}, lambda_thresh={self.pcbr_lambda_thresh}, resp_thresh_rel={self.pcbr_response_thresh_rel} min_dist={self.pcbr_min_distance}")
        start_time_pcbr = time.time()

        try:
            Hrr, Hrc, Hcc = hessian_matrix(img_subtracted,
                                           sigma=self.pcbr_sigma,
                                           use_gaussian_derivatives=True,  # Default is True, explicit here
                                           mode='nearest',  # Specify boundary handling
                                           order='rc')  # Specify order if needed (rc is default)
        except Exception as e:
            print(f"      ERROR calculating Hessian matrix: {e}. Skipping PCBR.")
            return np.array([]), np.array([]), np.array([])

        try:
            if not np.all(np.isfinite(Hrr)) or not np.all(np.isfinite(Hrc)) or not np.all(np.isfinite(Hcc)):
                print("      Warning: Non-finite values found in Hessian components. Attempting nan_to_num.")
                Hrr = np.nan_to_num(Hrr)
                Hrc = np.nan_to_num(Hrc)
                Hcc = np.nan_to_num(Hcc)
            lambda1, lambda2 = hessian_matrix_eigvals(Hrr, Hrc, Hcc)
        except Exception as e:
            print(f"ERROR calculating Hessian eigenvalues: {e}. Skipping PCBR.")
            return np.array([]), np.array([]), np.array([])

        response_map = np.zeros_like(img_subtracted)
        valid_lambda = np.isfinite(lambda1)
        mask = valid_lambda & (lambda1 < -self.pcbr_lambda_thresh)

        if np.any(mask):
            response_map[mask] = -lambda1[mask]
        else:
            print("PCBR: No pixels met the lambda threshold condition.")


        self.last_detector_map = response_map
        print(f"    Calculated PCBR response map ({time.time() - start_time_pcbr:.2f}s)")

        if np.any(response_map > 0):
            print(f"    Finding peaks in PCBR response map...")
            coordinates = peak_local_max(response_map,
                                         min_distance=self.pcbr_min_distance,
                                         threshold_rel=self.pcbr_response_thresh_rel,
                                         exclude_border=False)
            if coordinates.shape[0] == 0:
                print("      PCBR: No local maxima found above threshold.")
                return np.array([]), np.array([]), np.array([])

            rows, cols = coordinates[:, 0], coordinates[:, 1]
            scores = response_map[rows, cols]
            return rows, cols, scores
        else:
            print("      PCBR: Response map is empty or zero. No peaks found.")
            return np.array([]), np.array([]), np.array([])


    def _run_vote(self, img_subtracted, noise_sigma):
        print(f"    Voting params: radius={self.vote_radius}, min_votes={self.min_votes}")
        all_detections = []; self.last_vote_details = {}
        internal_runners = {'log': self._run_log, 'doh': self._run_doh, 'mser': self._run_mser, 'pcbr': self._run_pcbr}

        for method_name in self.VOTING_METHODS:
            if method_name not in internal_runners:
                continue
            try:
                print(f"      Running detector: {method_name}...")
                start_t_method = time.time()
                rows, cols, scores = internal_runners[method_name](img_subtracted, noise_sigma)
                print(f"{method_name} found {len(rows)} candidates ({time.time()-start_t_method:.2f}s).")
                if len(rows) > 0: self.last_vote_details[method_name] = (rows, cols, scores)
                [all_detections.append((r, c, method_name)) for r, c in zip(rows, cols)]
            except ImportError as e: print(f"Skipping detector {method_name} due to missing library: {e}")
            except Exception as e: print(f"ERROR running detector {method_name}: {e}")

        if not all_detections:
            print("Voting: No detections found by any contributing method."); return np.array([]), np.array([]), np.array([])
        print(f"Voting: Total candidates from {len(self.last_vote_details)} methods: {len(all_detections)}")
        all_points = np.array([(d[0], d[1]) for d in all_detections])
        detector_names = [d[2] for d in all_detections]
        if all_points.shape[0] < self.min_votes:
            print(f"Voting: Insufficient points ({all_points.shape[0]}) for min_votes ({self.min_votes}).")
            return np.array([]), np.array([]), np.array([])
        kdtree = spatial.KDTree(all_points); pairs = kdtree.query_pairs(r=self.vote_radius)
        adjacency_list = defaultdict(list)
        for i, j in pairs:
            adjacency_list[i].append(j)
            adjacency_list[j].append(i)
        num_points = len(all_points)
        visited = np.zeros(num_points, dtype=bool); consensus_peaks = []
        for i in range(num_points):
            if not visited[i]:
                component_indices = []; q = [i]
                visited[i] = True
                head = 0
                while head < len(q):
                    u = q[head]
                    head += 1
                    component_indices.append(u)
                    [q.append(v) for v in adjacency_list[u] if not visited[v] and not visited.__setitem__(v, True)]
                component_detectors = set(detector_names[idx] for idx in component_indices)
                num_votes = len(component_detectors)
                if num_votes >= self.min_votes:
                    component_points = all_points[component_indices]
                    avg_row, avg_col = np.mean(component_points, axis=0)
                    consensus_peaks.append((avg_row, avg_col, num_votes))
        print(f"Voting: Found {len(consensus_peaks)} consensus peaks with >= {self.min_votes} votes.")
        if not consensus_peaks:
            return np.array([]), np.array([]), np.array([])
        peak_rows = np.array([p[0] for p in consensus_peaks])
        peak_cols = np.array([p[1] for p in consensus_peaks])
        peak_scores = np.array([p[2] for p in consensus_peaks])
        return peak_rows, peak_cols, peak_scores

    def _detect_peaks(self, img_subtracted, noise_sigma):
        if self.verbose:
            print(f"  Detecting peaks using method: '{self.detector_method}'...")
        start_time = time.time()
        self.last_detector_map = None
        # Updated internal runners map
        internal_runners = {'log': self._run_log, 'doh': self._run_doh, 'mser': self._run_mser, 'pcbr': self._run_pcbr, 'vote': self._run_vote}
        if self.detector_method in internal_runners:
            peak_rows, peak_cols, peak_scores = internal_runners[self.detector_method](img_subtracted, noise_sigma)
        else:
            raise ValueError(f"Internal error: Unsupported detector_method: {self.detector_method}")
        if self.verbose:
            print(f"    Initial candidates found: {len(peak_rows)} ({time.time() - start_time:.2f}s)")
        # --- Common Post-processing ---
        # (Filter central region - operates on rows/cols/scores directly)
        if len(peak_rows) > 0 and self.center_ignore_radius_pixels is not None and self.center_ignore_radius_pixels > 0:
            if self.verbose:
                print(f"  Filtering center (radius: {self.center_ignore_radius_pixels} pixels)...", end="")
            center_y, center_x = img_subtracted.shape[0] // 2, img_subtracted.shape[1] // 2
            center_dist_sq = (peak_rows - center_y) ** 2 + (peak_cols - center_x) ** 2
            center_ignore_radius_sq = self.center_ignore_radius_pixels ** 2
            keep_indices = center_dist_sq > center_ignore_radius_sq
            peak_rows = peak_rows[keep_indices]
            peak_cols = peak_cols[keep_indices]
            # peak_intensities are not yet defined, filter scores instead
            peak_scores = peak_scores[keep_indices]
            if self.verbose:
                print(f" {len(peak_rows)} peaks remain after center filtering.")
        elif len(peak_rows) > 0:
            if self.verbose:
                print("  Central region filtering disabled or no peaks.")
        else:
            if self.verbose:
                print("  No initial peaks found or all filtered by center.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # --- NEW: Calculate Integrated Intensities via Aperture Sum ---
        if self.verbose:
            print(f"  Calculating integrated intensities for {len(peak_rows)} peaks...")
        integrated_intensities = np.zeros(len(peak_rows), dtype=float)  # Pre-allocate array
        img_h, img_w = img_subtracted.shape

        if len(peak_rows) > 0:
            # Pre-calculate meshgrid for faster distance calculations (optional optimization)
            # yy, xx = np.mgrid[:img_h, :img_w] # Could be large

            for i in range(len(peak_rows)):
                r_float, c_float = peak_rows[i], peak_cols[i]
                score = peak_scores[i]

                # Determine integration radius
                radius = 0.0
                if self.detector_method in ['log', 'doh']:
                    # Use sigma score (ensure score is positive and reasonable)
                    # Use max(0.5, ...) as sigma=0 is invalid for radius calc
                    radius = self.integration_sigma_factor * max(0.5, score)
                else:
                    # Use fixed radius for other methods
                    radius = float(self.integration_fixed_radius)

                radius = max(1.0, radius)  # Ensure radius is at least 1 pixel
                radius_sq = radius ** 2  # Use squared radius for distance check

                # Define a bounding box around the peak for efficiency
                # Use ceil for max bounds to ensure aperture fits
                radius_ceil = int(np.ceil(radius))
                r_center_int = int(round(r_float))  # Use rounded center for slicing
                c_center_int = int(round(c_float))

                r_min = max(0, r_center_int - radius_ceil)
                r_max = min(img_h, r_center_int + radius_ceil + 1)  # Python slices exclude end
                c_min = max(0, c_center_int - radius_ceil)
                c_max = min(img_w, c_center_int + radius_ceil + 1)

                # Check if bounding box is valid
                if r_min >= r_max or c_min >= c_max:
                    integrated_intensities[i] = 0.0  # Cannot calculate sum if box is empty
                    continue

                # Extract patch coordinates and data
                patch_rr, patch_cc = np.mgrid[r_min:r_max, c_min:c_max]
                patch_data = img_subtracted[r_min:r_max, c_min:c_max]  # Indexing with slices

                # Calculate distance squared from the *float* center to each pixel *center* in the patch
                # Add 0.5 to patch indices to represent pixel centers for more accuracy? Optional.
                dist_sq = (patch_rr - r_float) ** 2 + (patch_cc - c_float) ** 2

                # Create mask for pixels within the circular aperture
                mask = dist_sq <= radius_sq

                # Sum intensities within the mask
                # Apply mask to patch_data before summing
                intensity_sum = np.sum(patch_data[mask])
                integrated_intensities[i] = intensity_sum

            # Print stats for the calculated integrated intensities
            if len(integrated_intensities) > 0:
                if self.verbose:
                    print(
                        f"    Integrated intensity stats: Min={np.min(integrated_intensities):.1f}, Max={np.max(integrated_intensities):.1f}, Mean={np.mean(integrated_intensities):.1f}")

        # --- Return the results ---
        # Return float coordinates, NEW integrated intensities, original scores
        return peak_rows, peak_cols, integrated_intensities, peak_scores

    # --- Plotting (More Robust LogNorm for Dirac Plot) ---
    def _plot_results(self, original_img, background_img, subtracted_img, dirac_img, peaks_data, noise_sigma):
        """Displays the processing steps and results. Uses robust LogNorm for Dirac image."""
        print("  Generating plot...")
        # Import LogNorm inside the method or at the top of the file
        from matplotlib.colors import LogNorm, Normalize

        n_plots = 4
        extra_plot_map = self.last_detector_map is not None and self.detector_method == 'pcbr'
        if extra_plot_map:
            n_plots = 5; fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        ax = axes.ravel()

        # get_clim helper (corrected version from before)
        def get_clim(data, prct=(1, 99.9)):
            if data is None or data.size == 0: return (0, 1)
            valid_data = data[np.isfinite(data)];
            if valid_data.size == 0: return (0, 1)
            try:
                vmin, vmax = np.percentile(valid_data, list(prct))
                # print(f"      DEBUG: get_clim - percentile OK: vmin={vmin:.2e}, vmax={vmax:.2e}")
            except Exception as e_pct:
                print(f"      WARNING: Error calculating percentile in get_clim: {e_pct}");
                if valid_data.size > 0:
                    vmin = np.min(valid_data); vmax = np.max(valid_data); print(
                        f"      DEBUG: Falling back to min/max: vmin={vmin:.2e}, vmax={vmax:.2e}")
                else:
                    return (0, 1)
            if vmax <= vmin: vmax = vmin + (abs(vmin) * 0.01 + 1e-6)  # Ensure vmax > vmin
            return vmin, vmax

        # plot_image helper (no changes needed)
        def plot_image(ax_idx, img, title, cmap, clim, show_peaks=False, norm=None):  # Added norm argument
            if img is None or img.size == 0: ax[ax_idx].set_title(f"{title}\n(No data)", fontsize=10); ax[
                ax_idx].axis('off'); return
            im = ax[ax_idx].imshow(img, cmap=cmap, vmin=clim[0] if norm is None else None,
                                   vmax=clim[1] if norm is None else None, norm=norm, interpolation='nearest');
            ax[ax_idx].set_title(title, fontsize=10)
            if show_peaks and peaks_data: peak_cols_plot = [int(round(p[1])) for p in
                                                            peaks_data]; peak_rows_plot = [int(round(p[0])) for p in
                                                                                           peaks_data]; ax[
                ax_idx].plot(peak_cols_plot, peak_rows_plot, 'r+', markersize=6, markeredgewidth=1.0, linestyle='')
            fig.colorbar(im, ax=ax[ax_idx], fraction=0.046, pad=0.04);
            ax[ax_idx].axis('off')

        # --- Plotting calls for first images ---
        vmin_orig, vmax_orig = get_clim(original_img);
        vmin_sub, vmax_sub = get_clim(subtracted_img, prct=(1, 99))
        plot_image(0, original_img, f'Original Image\n({original_img.shape[0]}x{original_img.shape[1]})', 'viridis',
                   (vmin_orig, vmax_orig))
        plot_image(1, background_img, f'Est. Background (Method: {self.background_method})', 'viridis',
                   (vmin_orig, vmax_orig))
        plot_image(2, subtracted_img, f'Background Subtracted\n(Noise $\sigma$: {noise_sigma:.2f})', 'viridis',
                   (vmin_sub, vmax_sub), show_peaks=True)

        plot_idx_dirac = 3
        if extra_plot_map:  # Plot PCBR response map
            map_title = 'PCBR Response Map (-lambda1)'
            map_clim = get_clim(self.last_detector_map, prct=(1, 99));
            plot_image(3, self.last_detector_map, map_title, 'magma', map_clim, show_peaks=True);
            plot_idx_dirac = 4

        # --- Special Handling for Dirac Image Plot ---
        dirac_ax = ax[plot_idx_dirac]
        dirac_title_base = f'Output Dirac Deltas\n({len(peaks_data)} peaks, Method: {self.detector_method}'
        dirac_cmap = 'hot'
        norm_to_use = None  # Default to linear scale

        # Find non-zero positive values for potential LogNorm limits
        dirac_non_zero = dirac_img[dirac_img > 1e-9]  # Use small positive threshold

        if dirac_non_zero.size > 0:
            min_nz = np.min(dirac_non_zero)
            max_nz = np.max(dirac_non_zero)

            # Define vmin for LogNorm slightly below the minimum positive value, but strictly > 0
            log_vmin = max(min_nz * 0.9, 1e-9)  # Ensure positive, slightly less than min
            log_vmax = max_nz

            # Check if the calculated range is valid for LogNorm
            if log_vmax > log_vmin:
                try:
                    # Use clip=True: values outside [vmin, vmax] are mapped to endpoints
                    log_norm = LogNorm(vmin=log_vmin, vmax=log_vmax, clip=True)
                    norm_to_use = log_norm  # Set norm to use
                    dirac_title = dirac_title_base + ', Log Scale)'
                    print(f"DEBUG: Using LogNorm for Dirac plot: vmin={log_vmin:.2e}, vmax={log_vmax:.2e}")
                except ValueError as e_log:
                    print(f"ERROR creating LogNorm: {e_log}. Using linear scale.")
                    dirac_title = dirac_title_base + ', Linear Scale)'
            else:
                # Min and max are too close or equal, use linear
                print(
                    f"DEBUG: Min/Max non-zero values too close ({min_nz:.2e}, {max_nz:.2e}). Using linear scale for Dirac plot.")
                dirac_title = dirac_title_base + ', Linear Scale)'
        else:
            # No positive values found
            print("DEBUG: No non-zero peaks in Dirac image to plot.")
            dirac_title = dirac_title_base + ', Linear Scale)'

        # Use linear norm if LogNorm wasn't set (handles no peaks or LogNorm failure)
        if norm_to_use is None:
            lin_vmin = 0
            lin_vmax = np.max(dirac_img) if dirac_img.size > 0 and np.any(dirac_img > 0) else 1.0
            if lin_vmax <= lin_vmin: lin_vmax = lin_vmin + 1.0
            # Use Normalize (standard linear normalization)
            norm_to_use = Normalize(vmin=lin_vmin, vmax=lin_vmax)

        # Perform the plotting using the determined norm
        im_dirac = dirac_ax.imshow(dirac_img, cmap=dirac_cmap, norm=norm_to_use, interpolation='nearest')
        fig.colorbar(im_dirac, ax=dirac_ax, fraction=0.046, pad=0.04)
        dirac_ax.set_title(dirac_title, fontsize=10)
        dirac_ax.axis('off')
        # --- End Special Handling ---

        for i in range(n_plots, len(ax)): ax[i].axis('off')  # Turn off unused axes
        plt.tight_layout(pad=0.5);
        plt.show()


        # --- Main Processing Method (WITH MORE DEBUGGING) ---
    def process_image(self, image, show_result=False):
        """ Processes the input diffraction image to find peaks."""
        if self.verbose:
            print(
                f"\n--- Starting Image Processing (Detector: {self.detector_method}, Background: {self.background_method}) ---")
        try:
            if not isinstance(image, np.ndarray): raise TypeError(f"Input 'image' must be a NumPy array")
            if image.ndim != 2: raise ValueError(f"Input 'image' must be 2D")
            image_proc = image.astype(np.float64);
            if self.verbose:
                print(f"Processing image of shape: {image_proc.shape}")

            # BG Subtraction and Noise Estimation
            self.last_background_img = self._estimate_background(image_proc)
            self.last_subtracted_img = image_proc - self.last_background_img;
            self.last_subtracted_img = np.clip(self.last_subtracted_img, 0, None)
            self.last_noise_sigma = self._estimate_noise(self.last_subtracted_img)

            # DEBUG: Check subtracted image stats
            if self.verbose:
                print(
                    f"DEBUG: Subtracted Image Stats: Min={np.min(self.last_subtracted_img):.2f}, Max={np.max(self.last_subtracted_img):.2f}, Mean={np.mean(self.last_subtracted_img):.2f}")
                # You could even plot self.last_subtracted_img here if needed
            # plt.figure(); plt.imshow(self.last_subtracted_img, cmap='viridis'); plt.title("DEBUG: Subtracted Image Before Peak Find"); plt.colorbar(); plt.show()

            # Detect peaks (returns float coords potentially)
            # The _detect_peaks method internally assigns intensities based on img_subtracted
            peak_rows, peak_cols, peak_intensities, peak_scores = self._detect_peaks(self.last_subtracted_img,
                                                                                     self.last_noise_sigma)
            # DEBUG: Check intensities immediately after detection
            if self.verbose:
                print(f"DEBUG: Initial Peaks Found by _detect_peaks: {len(peak_rows)}")
                if len(peak_rows) > 0:
                    print(f"DEBUG: Intensities from _detect_peaks: "
                          f"Min={np.min(peak_intensities):.2e}, Max={np.max(peak_intensities):.2e}, "
                          f"Mean={np.mean(peak_intensities):.2e}, Median={np.median(peak_intensities):.2e}, "
                          f"Count > 1 = {np.sum(peak_intensities > 1)}, Count > 100 = {np.sum(peak_intensities > 100)}")

            # Store results (with float coords), sorted by intensity
            if len(peak_rows) > 0:
                self.last_peaks_data = sorted(list(zip(peak_rows, peak_cols, peak_intensities, peak_scores)),
                                              key=lambda p: p[2], reverse=True)
            else:
                self.last_peaks_data = []

            # Generate Dirac Delta Output Image
            output_dirac = np.zeros_like(self.last_subtracted_img, dtype=float)
            if self.last_peaks_data:
                peak_rows_float = np.array([p[0] for p in self.last_peaks_data])
                peak_cols_float = np.array([p[1] for p in self.last_peaks_data])
                peak_intensities_all = np.array([p[2] for p in self.last_peaks_data])  # Use all intensities

                if self.verbose:
                    if len(peak_intensities_all) > 0:
                        print(f"DEBUG: Intensities before np.add.at: "
                              f"Min={np.min(peak_intensities_all):.2e}, Max={np.max(peak_intensities_all):.2e}, "
                              f"Mean={np.mean(peak_intensities_all):.2e}, Median={np.median(peak_intensities_all):.2e}, "
                              f"Count > 1 = {np.sum(peak_intensities_all > 1)}, Count > 100 = {np.sum(peak_intensities_all > 100)}")
                    if np.any(np.isnan(peak_intensities_all)):
                        print("!!!!! WARNING: NaNs found in peak_intensities_all !!!!!")
                        peak_intensities_all = np.nan_to_num(peak_intensities_all)  # Replace NaNs with 0

                    else:
                        print("DEBUG: No peak data available for Dirac image population.")

                rows_idx = np.round(peak_rows_float).astype(int)
                cols_idx = np.round(peak_cols_float).astype(int)
                rows_idx = np.clip(rows_idx, 0, output_dirac.shape[0] - 1)
                cols_idx = np.clip(cols_idx, 0, output_dirac.shape[1] - 1)

                # Sum intensities at overlapping pixels
                np.add.at(output_dirac, (rows_idx, cols_idx), peak_intensities_all)

                # DEBUG: Check the resulting Dirac image content
                num_nonzero = np.sum(output_dirac > 1e-6)  # Count pixels slightly above zero
                if self.verbose:
                    print(f"DEBUG: Dirac Image Stats after population:")
                    print(f"  Number of non-zero pixels: {num_nonzero}")

                if self.verbose:
                    if num_nonzero > 0:
                        print(f"  Min non-zero value: {np.min(output_dirac[output_dirac > 1e-6]):.2e}")
                        print(f"  Max value: {np.max(output_dirac):.2e}")
                        print(f"  Sum of values: {np.sum(output_dirac):.2e}")
                    print(f"  NaN count: {np.sum(np.isnan(output_dirac))}")

            self.last_dirac_img = output_dirac
            if self.verbose:
                print(
                    f"Processing complete. Found {len(self.last_peaks_data)} peaks initially.")  # Report original count

            if show_result: self._plot_results(image_proc, self.last_background_img, self.last_subtracted_img,
                                               self.last_dirac_img, self.last_peaks_data, self.last_noise_sigma)
            return self.last_dirac_img


        except Exception as e:

            print(f"\n--- ERROR during image processing: {e} ---")  # Keep this line!

            print("--- Full Traceback ---")

            traceback.print_exc()  # <--- This line prints the full traceback

            print("--- End Traceback ---")


class UnetSegmenter:
    """
    Uses a pre-trained RobustUNetLite model to segment blobs in a 256x256 image,
    filters blobs by size, upscales the results, and concentrates valid blob
    intensities at their centroids. Features optional CoM-based center masking.
    The model is loaded once during initialization.
    """

    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Initializes the segmenter by loading the trained U-Net model.

        Args:
            checkpoint_path (str): Path to the PyTorch Lightning checkpoint file (.ckpt).
            device (str): The device to run inference on ('cuda', 'cpu', or 'auto').
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        print(f"Loading model from {checkpoint_path}...")
        try:
            self.model = RobustUNetLightning.load_from_checkpoint(checkpoint_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}\nEnsure class definitions match training.")
            raise
        self.model.eval()
        self.model.to(self.device)
        print("Model loaded successfully and set to evaluation mode.")
        self.model_input_size = (256, 256)
        self.target_output_size = (1024, 1024)

    def _create_circular_mask(self, h, w, center=None, radius=None):
        """Helper function to create a circular boolean mask."""
        if center is None: center = (w / 2.0 - 0.5, h / 2.0 - 0.5)  # Use float geometric center if none provided
        if radius is None: radius = min(center[0], center[1], w - center[0], h - center[1])

        # Ensure center coordinates are floats for accurate distance calc
        center_x, center_y = float(center[0]), float(center[1])

        Y, X = np.ogrid[:h, :w]  # Integer grid indices
        # Calculate distance from floating point center
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        mask = dist_from_center <= radius
        return mask

    def _calculate_com(self, image: np.ndarray, region_mask: np.ndarray) -> tuple[float, float] | None:
        """Calculates Center of Mass within a masked region."""
        h, w = image.shape
        y_coords, x_coords = np.indices((h, w))  # Get coordinate grids

        # Apply the mask to intensities and coordinates
        intensities_in_region = image[region_mask]
        x_coords_in_region = x_coords[region_mask]
        y_coords_in_region = y_coords[region_mask]

        total_intensity = np.sum(intensities_in_region)

        # Avoid division by zero if region is empty or completely black
        if total_intensity < 1e-6:  # Using a small threshold for float comparison
            return None

        # Calculate CoM coordinates
        com_y = np.sum(y_coords_in_region * intensities_in_region) / total_intensity
        com_x = np.sum(x_coords_in_region * intensities_in_region) / total_intensity

        return com_x, com_y

    def process_image(self,
                      input_image_256: np.ndarray,
                      center_mask_radius: int = 20,  # Radius of the mask to APPLY
                      com_calculation_radius: int = 30,  # Radius for CoM calculation area
                      show_results: bool = False
                      ) -> np.ndarray:
        """
        Processes a single 256x256 input image to produce a concentrated intensity map.
        Optionally masks a central region before prediction, centering the mask based
        on the Center of Mass (CoM) calculated within a specified central area.
        Filters blobs based on their size.

        Args:
            input_image_256 (np.ndarray): The input image (must be 256x256).
            center_mask_radius (int): Radius (pixels) of the circular mask to *apply*. If > 0,
                                      masking is enabled, centered based on CoM.
            com_calculation_radius (int): Radius (pixels) of the central region used for
                                          calculating the CoM to center the mask. Defaults to 40.
            min_blob_size (int): Minimum number of pixels for a blob to be processed.
            max_blob_size (float): Maximum number of pixels for a blob.
            show_results (bool): If True, displays intermediate and final images.

        Returns:
            np.ndarray: The concentrated "Dirac" image (1024x1024 float32).
        """
        # --- Input Validation and Basic Prep ---
        if not isinstance(input_image_256, np.ndarray): raise TypeError("Input must be NumPy array")
        if input_image_256.shape[:2] != (self.model_input_size[1], self.model_input_size[0]):
            raise ValueError(
                f"Input must be {self.model_input_size[1]}x{self.model_input_size[0]}, got {input_image_256.shape[:2]}")
        if input_image_256.ndim == 3:
            image_gray = cv2.cvtColor(input_image_256, cv2.COLOR_BGR2GRAY)
        elif input_image_256.ndim == 2:
            image_gray = input_image_256
        else:
            raise ValueError(f"Unsupported input dimensions: {input_image_256.shape}")
        if image_gray.dtype != np.float32:
            image_float = image_gray.astype(np.float32)
        else:
            image_float = image_gray

        # --- Upscale Original for Later Intensity Summation ---
        original_upscaled = cv2.resize(image_float, self.target_output_size, interpolation=cv2.INTER_CUBIC)

        # --- Prepare 256x256 Image for Model (CoM Masking) ---
        image_for_model = image_float.copy()
        h, w = image_for_model.shape
        mask_applied_center = None  # Keep track of where the mask was centered

        if center_mask_radius > 0:
            # 1. Define region for CoM calculation
            geom_center_x, geom_center_y = w / 2.0 - 0.5, h / 2.0 - 0.5  # Use float center
            com_region_mask = self._create_circular_mask(h, w, center=(geom_center_x, geom_center_y),
                                                         radius=com_calculation_radius)

            # 2. Calculate CoM within that region
            calculated_com = self._calculate_com(image_for_model, com_region_mask)

            if calculated_com:
                mask_center = calculated_com  # Use CoM if valid
                print(f"Calculated CoM at ({mask_center[0]:.2f}, {mask_center[1]:.2f}) for masking.")
            else:
                mask_center = (geom_center_x, geom_center_y)  # Fallback to geometric center
                print("CoM calculation failed (zero intensity in region?), using geometric center for masking.")

            # 3. Create and apply the *actual* mask using calculated center and desired radius
            final_mask_to_apply = self._create_circular_mask(h, w, center=mask_center, radius=center_mask_radius)
            image_for_model[final_mask_to_apply] = 0  # Apply mask
            mask_applied_center = mask_center  # Store for visualization

        image_display = image_for_model.copy()  # Make a copy for display *after* potential masking

        # --- Convert to Tensor ---
        input_tensor = torch.from_numpy(image_for_model).unsqueeze(0).unsqueeze(0).to(self.device)

        # --- 2. Model Inference ---
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.sigmoid(logits)
        predicted_mask_small = probabilities.squeeze().cpu().numpy()
        binary_mask_small = (predicted_mask_small > 0.5).astype(np.uint8)

        # --- 3. Post-processing (with size filtering) ---
        binary_mask_large = cv2.resize(binary_mask_small, self.target_output_size,
                                       interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_large, connectivity=8,
                                                                                ltype=cv2.CV_32S)
        dirac_image = np.zeros(self.target_output_size, dtype=np.float32)

        processed_blob_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            processed_blob_count += 1
            component_mask = (labels == i)
            blob_intensity_sum = np.sum(original_upscaled[component_mask])
            cx, cy = centroids[i]
            cx_int, cy_int = int(round(cx)), int(round(cy))
            if 0 <= cy_int < self.target_output_size[1] and 0 <= cx_int < self.target_output_size[0]:
                dirac_image[cy_int, cx_int] = blob_intensity_sum

        # --- 4. Optional Visualization ---
        if show_results:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            # Display potentially masked input
            axes[0].imshow(image_display, cmap='gray', vmax=300)
            title = f'Input to Model (256x256)'
            if mask_applied_center:
                title += f'\nMask R={center_mask_radius} @ CoM({mask_applied_center[0]:.1f},{mask_applied_center[1]:.1f})'
                # Optionally draw the applied mask circle
                circ = plt.Circle((mask_applied_center[0], mask_applied_center[1]), center_mask_radius, color='red',
                                  fill=False, linewidth=1, linestyle='--')
                axes[0].add_patch(circ)
            else:
                title += f'\nNo Mask Applied'

            axes[0].set_title(title)
            axes[0].axis('off')
            # Display large mask
            axes[1].imshow(binary_mask_large, cmap='gray')
            axes[1].set_title(f'Predicted Mask ({self.target_output_size[1]}x{self.target_output_size[0]})')
            axes[1].axis('off')
            # Display Dirac image
            im = axes[2].imshow(dirac_image, cmap='hot', vmax=300)
            axes[2].set_title(
                f'Concentrated Intensity')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

        # --- 5. Return Result ---
        return dirac_image
