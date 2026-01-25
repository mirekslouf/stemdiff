import cv2
import numpy as np
from matplotlib import pyplot as plt

class ImageBlurring:
    def __init__(self, display=True):
        self.display = display

    def load_image(self, image_path: str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        print(f'Image shape is', image.shape)
        image = (image - image.min()) / (image.max() - image.min())
        print(f'Image max value after normalization is', image.max())
        return image

    def gaussian_blur(self, image, kernel_size: (int, int), sigma_x: int, save_blurred: bool = False, save_kernel: bool = False, output_image_path: str = 'blurred_image.jpg'):
        self.image = image
        kernel = cv2.getGaussianKernel(kernel_size[0], sigma_x)
        kernel = np.outer(kernel, kernel.transpose())
        blurred = cv2.filter2D(self.image, -1, kernel)
        blurred = self.add_gaussian_noise(blurred)

        if save_blurred:
            cv2.imwrite(output_image_path, blurred)
            print(f"The file is saved at {output_image_path}")

        if save_kernel:
            np.savetxt('gaussian_kernel.txt', kernel, fmt='%f')
            print("The kernel is saved at gaussian_kernel.txt")
        if self.display == 1:
            plt.imshow(blurred, cmap='gray')
            plt.title('Blurred')
            plt.xticks([]), plt.yticks([])
            plt.show()

        return blurred, kernel

    def motion_blur(self, image, size, angle):
        # size of the kernel should be an odd number
        size = size if size % 2 == 1 else size + 1
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        # create a rotation matrix
        M = cv2.getRotationMatrix2D((size / 2 - 1, size / 2 - 1), angle, 1)
        # warp the blur kernel using the rotation matrix
        kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, M, (size, size))
        output = cv2.filter2D(image, -1, kernel_motion_blur)

        if self.display == 1:
            plt.imshow(output, cmap='gray')
            plt.title('Motion Blurred')
            plt.xticks([]), plt.yticks([])
            plt.show()

        return output, kernel_motion_blur

    def add_gaussian_noise(self, image, mean=0, var=0.0005):
        row, col = image.shape
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy