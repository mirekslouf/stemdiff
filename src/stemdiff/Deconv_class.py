'''
Module: stemdiff.Deconv_class
----------------------------
Module with custom implementation of Richardson-Lucy algorithm.
Including 2 regularization variants, Tikhonov regularization (RLTM) and
L1 regularization (RLTV).

Works on grayscale images, not on color (no reason for color images).
Test script has example usage on artificially blurred image of Lena.

The initial arguments are:

* iterations = number of iteration performed
* cuda = Bool = Use GPU if True, use CPU if false. Significantly speeds up computation.
* display = Bool = show initial and resulting image, practical for experiments, terrible for running in stemdiff, since
we are summing hundreds of images.
* timer = Bool = Practical for tests, shows compute time for deconvolution.
* turn_off_progress_bar = Bool = Turn off progress bar, again only for debugging and experiments.

Some examples of usage on single image are included in Deconv_examples.ipynb

'''


import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import convolve
from scipy.ndimage import convolve as np_convolve
import time
from tqdm import tqdm


class RichardsonLucy:
    """
    This class implements the Richardson-Lucy algorithms with and without regularization for image deconvolution.

    Attributes:
        iterations (int): The number of iterations for the algorithm. Default is 30.
        cuda (bool): Determines whether to use CUDA for computation. Default is True.
        display (bool): Determines whether to display the deconvolved image. Default is False.
        timer (bool): Practical for tests, shows compute time for deconvolution.
        turn_off_progress_bar (bool): Turn off progress bar, again only for debugging and experiments.

    """
    def __init__(self, iterations=30, cuda=True, display=False, timer=False, turn_off_progress_bar=True):
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations should be a positive integer.")
        self.iterations = iterations
        self.cuda = cuda
        self.display = display
        self.timer = timer
        self.progress_bar = turn_off_progress_bar
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    new_float_type = {
        # preserved types
        np.float32().dtype.char: np.float32,
        np.float64().dtype.char: np.float64,
        np.complex64().dtype.char: np.complex64,
        np.complex128().dtype.char: np.complex128,
        # altered types
        np.float16().dtype.char: np.float32,
        'g': np.float64,  # np.float128 ; doesn't exist on windows
        'G': np.complex128,  # np.complex256 ; doesn't exist on windows
    }

    def _supported_float_type(self, input_dtype, allow_complex=False):
        """Return an appropriate floating-point dtype for a given dtype.

        float32, float64, complex64, complex128 are preserved.
        float16 is promoted to float32.
        complex256 is demoted to complex128.
        Other types are cast to float64.

        Parameters
        ----------
        input_dtype : np.dtype or tuple of np.dtype
            The input dtype. If a tuple of multiple dtypes is provided, each
            dtype is first converted to a supported floating point type and the
            final dtype is then determined by applying `np.result_type` on the
            sequence of supported floating point types.
        allow_complex : bool, optional
            If False, raise a ValueError on complex-valued inputs.

        Returns
        -------
        float_type : dtype
            Floating-point dtype for the image.
        """
        if isinstance(input_dtype, tuple):
            return np.result_type(*(self._supported_float_type(d) for d in input_dtype))
        input_dtype = np.dtype(input_dtype)
        if not allow_complex and input_dtype.kind == 'c':
            raise ValueError("complex valued input is not supported")
        return self.new_float_type.get(input_dtype.char, np.float64)

    def display_image(self, image, title):
        """
        Display an image with a given title.

        Parameters
        ----------
            image: A numpy array representing the image to be displayed.
            title: A string representing the title of the image.

        Returns
        -------
            None

        """
        if self.cuda:
            image = cp.asnumpy(image)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

    def fspecial_laplacian(self, alpha=0.2):
        """
        Creates a Laplacian filter based on the specified alpha value. Simulates Matlab behavior, since it can be
        defined in multiple ways. Matlab's behavior was chosen as default, since some earlier work was done in matlab.

        Parameters
        ----------
        alpha: A float value between 0 and 1 representing the strength of the Laplacian filter.

        Returns
        -------
        laplacian_filter: A Laplacian filter matrix.

        Raises
        ------
        ValueError: If alpha is not between 0 and 1.
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if self.cuda:
            laplacian_4 = cp.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=float)

            laplacian_8 = cp.array([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]], dtype=float)

            laplacian_filter = (alpha / (alpha + 1)) * laplacian_8 + (1 / (alpha + 1)) * laplacian_4
        else:
            laplacian_4 = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=float)

            laplacian_8 = np.array([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]], dtype=float)

            laplacian_filter = (alpha / (alpha + 1)) * laplacian_8 + (1 / (alpha + 1)) * laplacian_4

        return laplacian_filter

    def deconvRL(self, I, P):
        """
        Perform Richardson-Lucy deconvolution on an image using a given point spread function (PSF).
        This method applies the Richardson-Lucy algorithm to deconvolve an image using the given PSF. The algorithm
        iteratively updates an estimate of the true image by dividing the observed image by the convolved estimate and
        then multiplying the result by the flipped PSF.

        If the CUDA flag is set to True, the method uses the CuPy library to perform the calculations on a GPU.
        Otherwise, it uses the NumPy library for calculations on a CPU. The PSF is normalized before convolution.

        The algorithm performs a specified number of iterations specified by the 'iterations' property of the object.
        After each iteration, the estimated image is updated and stored in the variable 'O_k'. The resulting
        deconvolved image is converted to an 8-bit grayscale image before being returned.

        Parameters
        ----------
        I: The input image to be deconvolved.
        P: The point spread function.

        Returns
        -------
        O_k: The deconvolved image after k iterations


        """
        float_type = self._supported_float_type(I.dtype)
        if self.cuda:
            I = cp.asarray(I, dtype=float_type)
            O_k = cp.full(I.shape, fill_value=0.5, dtype=float_type)
            P = cp.asarray(P)
            convolve_func = convolve
        else:
            I = np.asarray(I, dtype=float_type)
            O_k = np.full(I.shape, fill_value=0.5, dtype=float_type)
            P = np.asarray(P)
            convolve_func = np_convolve
        eps = 1e-12
        start_time = time.time()
        P_flipped = np.flip(np.flip(P, 0), 1) if not self.cuda else cp.flip(cp.flip(P, 0), 1)
        for i in tqdm(range(1, self.iterations), disable=self.progress_bar):
            ratio = I / (convolve_func(O_k, P, mode='reflect') + eps)
            O_k = O_k * (convolve_func(ratio, P_flipped, mode='reflect'))
        O_k[O_k > 1] = 1
        O_k[O_k < -1] = -1
        end_time = time.time()
        if self.timer:
            print(f'Deconvolution took {end_time - start_time} seconds for {self.iterations} iterations.')
        if self.cuda:
            O_k = O_k.get()
        if self.display:
            plt.imshow(O_k, cmap='gray')
            plt.show()
        return O_k


    def deconvRLTM(self, I, P, lambda_reg):
        """
        This method performs Richardson-Lucy with Tikhonov-Miller regularization (RLTM) deconvolution on the input image
        using the provided PSF and regularization parameter. The deconvolution process iteratively updates the estimated
        image by computing the ratio between the input image and the convolution of the estimated image with the PSF,
        and then updates the estimated image by multiplying it with the ratio convolved with the rotated PSF.
        The process is repeated for a given number of iterations. The final deconvolved image is returned. If the 'cuda'
        flag is set to True, CUDA-accelerated functions are used; otherwise, regular NumPy functions are used.

        Parameters
        ----------
        I: The input image to be deconvolved.
        P: The point spread function (PSF) used for convolution.
        lambda_reg: The regularization parameter.

        Return
        ------
        O_k: The deconvolved image.
        """

        float_type = self._supported_float_type(I.dtype)
        if self.cuda:
            I = cp.asarray(I, dtype=float_type)
            O_k = cp.full(I.shape, fill_value=0.5, dtype=float_type)
            P = cp.asarray(P)
            convolve_func = convolve
            laplacian = cp.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=float_type)
        else:
            I = np.asarray(I, dtype=float_type)
            O_k = np.full(I.shape, fill_value=0.5, dtype=float_type)
            P = np.asarray(P)
            convolve_func = np_convolve
            laplacian = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=float_type)
        eps = 1e-12
        start_time = time.time()
        for i in tqdm(range(1, self.iterations), disable=self.progress_bar):
            ratio = I / ((convolve_func(O_k, P, mode='reflect') + 1e-6) + eps)
            O_k = O_k * (convolve_func(ratio, cp.rot90(P, k=2), mode='reflect'))
            laplacian_image = convolve_func(O_k, laplacian, mode='reflect')
            regularization_term = 1/(1-2*lambda_reg*laplacian_image)
            O_k = O_k * regularization_term
        O_k[O_k > 1] = 1
        O_k[O_k < -1] = -1

        end_time = time.time()
        if self.timer:
            print(f'Deconvolution took {end_time - start_time} seconds for {self.iterations} iterations.')
        if self.cuda:
            O_k = O_k.get()

        if self.display:
            plt.imshow(O_k, cmap='gray')
            plt.show()

        return O_k

    def mL1(self, DU, alpha, beta):
        """
        Applies the half-quadratic algorithm (multiplicative version) for the prior function phi(s) = |s|.
        If |DU| < alpha/beta, returns beta/2. Only support function.
        """
        V = 2 / alpha * np.maximum(DU, alpha / beta)
        return 1.0 / V

    def msetupLnormPrior(self, q, alpha, beta):
        """
        Sets up the normalization prior for a given 'q'.
        For q=1, uses the mL1 function to define the behavior of the prior.
        """
        if q == 1:
            P = {'fh': lambda x: self.mL1(x, alpha, beta)}
            return P
        else:
            raise ValueError("Only q=1 is implemented.")

    def compute_prior(self, O_k, P):
        """
        This function computes the regularization element for an input image.
        The regularization element is calculated by first computing the vertical and horizontal differences between
        adjacent pixels. These differences are then passed through the function 'fh' specified in the dictionary P.
        The resulting values are used to create an array L of shape (height, width, 1, 5).
        The first four dimensions of L represent the vertical and horizontal differences, and the last dimension
        represents their sum. Circular shifts are applied to the input image O_k, and a singleton third dimension is
        added. These shifted images are concatenated along the fourth dimension to match the behavior of MATLAB's 'cat'
        function. The element-wise multiplication of L and the concatenated shifted images is performed, followed by
        the sum along the fourth dimension. The resulting array is then squeezed to remove any singleton dimensions,
        and it is returned as the regularization element.

        Parameters
        ----------
        O_k: The input image array of shape (height, width)
        P: Dictionary containing a function 'fh' for calculating the differences

        Returns
        -------
        reg: The regularization element of shape (height, width)
        """

        usize = list(O_k.shape)  # Extract the size of the image
        usize.append(1)  # Add a singleton third dimension
        L = np.zeros(usize + [5])  # Shape: (height, width, 1, 5)

        # Compute the vertical and horizontal differences (adjusting for zero-based indexing)
        vertical_diff = np.sqrt(np.sum((O_k[1:, :, np.newaxis] - O_k[:-1, :, np.newaxis]) ** 2, axis=2))
        horizontal_diff = np.sqrt(np.sum((O_k[:, 1:, np.newaxis] - O_k[:, :-1, np.newaxis]) ** 2, axis=2))

        # Apply the P.fh function to the differences
        VV = np.repeat(P['fh'](vertical_diff)[:, :, np.newaxis], usize[2], axis=2)
        VH = np.repeat(P['fh'](horizontal_diff)[:, :, np.newaxis], usize[2], axis=2)

        L[:, :, :, 0] = np.concatenate([VV, np.zeros((1, usize[1], usize[2]))], axis=0)  # VV with an extra row
        L[:, :, :, 1] = np.roll(L[:, :, :, 0], shift=1, axis=0)  # Circular shift down
        L[:, :, :, 2] = np.concatenate([VH, np.zeros((usize[0], 1, usize[2]))], axis=1)  # VH with an extra column
        L[:, :, :, 3] = np.roll(L[:, :, :, 2], shift=1, axis=1)  # Circular shift right
        L[:, :, :, 4] = -np.sum(L[:, :, :, 0:4], axis=3)

        # Apply circular shifts to `O_k` and add a singleton third dimension
        img_dec_shifted_1 = np.roll(O_k, shift=-1, axis=0)[:, :, np.newaxis]  # Shift up
        img_dec_shifted_2 = np.roll(O_k, shift=1, axis=0)[:, :, np.newaxis]   # Shift down
        img_dec_shifted_3 = np.roll(O_k, shift=-1, axis=1)[:, :, np.newaxis]  # Shift left
        img_dec_shifted_4 = np.roll(O_k, shift=1, axis=1)[:, :, np.newaxis]   # Shift right
        O_k_singleton = O_k[:, :, np.newaxis]  # Original image with an extra dimension

        # Concatenate shifts along the fourth dimension to match MATLAB's `cat(4, ...)` behavior
        img_dec_cat = np.concatenate([img_dec_shifted_1, img_dec_shifted_2, img_dec_shifted_3, img_dec_shifted_4, O_k_singleton], axis=2)
        img_dec_cat = img_dec_cat[:, :, np.newaxis, :]
        # Calculate the regularization element
        reg = np.sum(L * img_dec_cat, axis=-1)
        reg = reg.squeeze()

        return reg

    def deconvRLTV(self, I, P, lambda_reg):
        """
        This method performs regularized deconvolution using the Richardson-Lucy Total Variation (RLTV) algorithm.
        The algorithm improves the quality of an image that has been blurred by a known PSF.
        If CUDA is available and in use, the method utilizes GPU acceleration. Otherwise, it runs on the CPU.
        The PSF is normalized to ensure that its sum is equal to 1.
        The regularization term is computed using the self.compute_prior() method by applying the total variation
        regularization parameter lambda_reg to the current deconvolved image O_k.
        The algorithm iteratively updates the deconvolved image O_k using the ratio of the input image I

        Parameters
        ----------
        I: The input image for deconvolution.
        P: The point spread function (PSF) for convolution.
        lambda_reg: The regularization parameter for the deconvolution algorithm.

        Returns
        -------
        O_k: The deconvolved image after k iterations

        """

        float_type = self._supported_float_type(I.dtype)
        if self.cuda:
            I = cp.asarray(I, dtype=float_type)
            O_k = cp.full(I.shape, fill_value=0.5, dtype=float_type)
            P = cp.asarray(P)
            convolve_func = convolve
        else:
            I = np.asarray(I, dtype=float_type)
            O_k = np.full(I.shape, fill_value=0.5, dtype=float_type)
            P = np.asarray(P)
            convolve_func = np_convolve
        eps = 1e-12

        P_h = self.msetupLnormPrior(1, lambda_reg, 100*lambda_reg)
        start_time = time.time()

        for i in tqdm(range(1, self.iterations), disable=self.progress_bar):
            if self.cuda:
                reg = self.compute_prior(O_k.get(), P_h)
                reg = cp.asarray(reg)
            else:
                reg = self.compute_prior(O_k, P_h)
            ratio = I / (convolve_func(O_k, P, mode='reflect') + eps)
            O_k = O_k * (convolve_func(ratio, cp.rot90(P, k=2), mode='reflect'))
            regularization_term = 1/(1-lambda_reg * reg)
            O_k = O_k * regularization_term
        O_k[O_k > 1] = 1
        O_k[O_k < -1] = -1

        end_time = time.time()
        if self.timer:
            print(f'Deconvolution took {end_time - start_time} seconds for {self.iterations} iterations.')
        if self.cuda:
            O_k = O_k.get()

        if self.display:
            plt.imshow(O_k, cmap='gray')
            plt.show()

        return O_k
