from scipy.signal import convolve2d
import numpy as np
from skimage.color import rgb2gray
import imageio as iio
from scipy import ndimage


GRAYSCALE = 1
RGB = 2
MAX_PIXEL_VAL = 255.

def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = iio.imread(filename).astype('float64')
    if image.ndim == 3 and image.shape[2] == 3 and representation == GRAYSCALE:
        image = rgb2gray(image)
    return image / MAX_PIXEL_VAL


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img



def _build_filter_vec(filter_size):
    """
    this function gets filter size which is odd scalar and builds
    vector with filter_size binomial coefficients
    """
    base_filter = np.array([1., 1.])
    filter_vec = base_filter
    for i in range(filter_size - 2):
        filter_vec = np.convolve(filter_vec, base_filter)
    filter_vec /= np.sum(filter_vec)
    return filter_vec.reshape(1, filter_size)


def _blur_by_filter(im, blur_filter):
    """
        this function gets blur filter and grayscale image
        and convolve im * blur_filter and im * blur_filter.T
    """
    after_row_blur = ndimage.filters.convolve(im, blur_filter)
    after_col_blur = ndimage.filters.convolve(after_row_blur, blur_filter.reshape(-1, 1))
    return after_col_blur


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    return _blur_by_filter(im, blur_filter)[::2, ::2]


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    filter_vec = _build_filter_vec(filter_size)
    pyr = [im]
    for i in range(1, max_levels):
        if pyr[i - 1].shape[0] / 2 >= 16 and pyr[i - 1].shape[1] / 2 >= 16:
            pyr.append(reduce(pyr[i - 1], filter_vec))
        else:
            break
    return pyr, filter_vec