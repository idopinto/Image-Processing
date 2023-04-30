import numpy as np
import imageio as iio
from scipy import signal
from scipy import ndimage
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import os

GRAYSCALE = 1
RGB = 2
MAX_PIXEL_VAL = 255.


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


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    padded = np.zeros((2 * im.shape[0], 2 * im.shape[1]), dtype=im.dtype)
    padded[:: 2, ::2] = im
    return _blur_by_filter(padded, 2 * blur_filter)


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


def RELU(im):
    return np.maximum(0, im)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
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
    gpyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = [gpyr[i] - expand(gpyr[i + 1], filter_vec) for i in range(len(gpyr) - 1)]
    pyr.append(gpyr[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """

    for i in range(len(lpyr) - 1, 0, -1):
        lpyr[i - 1] += expand(lpyr[i] * coeff[i], filter_vec)

    return lpyr[0]


def normalize_pyr(pyr, levels):
    """
    this function performs linear normalization to k levels of the pyramid
    """
    for k in range(levels):
        pyr[k] = pyr[k] - np.min(pyr[k]) / (np.max(pyr[k]) - np.min(pyr[k]))


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    normalize_pyr(pyr, levels)
    for i in range(1, levels):
        num_of_rows = pyr[0].shape[0] - pyr[i].shape[0]
        pyr[i] = np.pad(pyr[i], pad_width=((0, num_of_rows), (0, 0)), mode='constant')
    return np.hstack(pyr[:levels])


def display_pyramid(pyr, levels):
    """
	display the rendered pyramid
	"""
    plt.imshow(render_pyramid(pyr, levels), cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    # Construct Laplacian pyramids L1,L2
    L1, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filter_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    # Construct Gaussian pyramid Gm for the provided mask
    G_mask, filter_vec_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    coeff = [1] * max_levels
    L_out = []
    # Construct the Laplacian pyramid L_out of the blended image for each level k by:
    for k in range(len(L1)):
        L_out.append(G_mask[k] * L1[k] + (1 - G_mask[k]) * L2[k])
    # Reconstruct the resulting blended image from the Laplacian pyramid L_out
    return np.clip(laplacian_to_image(L_out, filter_vec1, coeff), 0, 1)


def plot_blended_image(figures):
    """
    helper function for plotting (2,2) subplots
    """
    plt.figure(figsize=(15, 15))

    for k in range(1, 5):
        plt.subplot(2, 2, k)
        plt.axis('off')
        plt.imshow(figures[k - 1], cmap=plt.cm.gray)
    plt.show()


def blend_example(im1, im2, mask, max_levels, filter_im_size, filter_mask_size):
    to_blend = np.zeros(im1.shape)
    for i in range(3):
        to_blend[:, :, i] = pyramid_blending(im2[:, :, i], im1[:, :, i], mask, max_levels, filter_im_size,
                                             filter_mask_size)
    return to_blend


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """

    im1 = read_image(relpath('externals/avater_bg.jpg'), 2)
    im2 = read_image(relpath('externals/avitavatar.jpg'), 2)
    mask = np.round(read_image(relpath('externals/mask_avitavatar.jpg'), 1)).astype(np.bool)
    max_levels = 7
    filter_im_size = 3
    filter_mask_size = 11
    im_blended = blend_example(im1, im2, mask, max_levels, filter_im_size, filter_mask_size)
    plot_blended_image([im1, im2, mask, im_blended])
    return im1, im2, mask, im_blended


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath('externals/rokdim_bg.jpg'), 2)
    im2 = read_image(relpath('externals/shmuvital.jpg'), 2)
    mask = np.round(read_image(relpath('externals/mask_rokdim.jpg'), 1)).astype(np.bool)
    max_levels = 6
    filter_im_size = 3
    filter_mask_size = 11
    im_blended = blend_example(im1, im2, mask, max_levels, filter_im_size, filter_mask_size)
    plot_blended_image([im1, im2, mask, im_blended])
    return im1, im2, mask, im_blended


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


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

