import numpy as np
from scipy import cluster
from skimage.color import rgb2gray
import imageio as iio
from matplotlib import pyplot as plt

# CONSTANTS
GRAYSCALE = 1
RGB = 2
MAX_PIXEL_VAL = 255.
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


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


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    plt.imshow(read_image(filename, representation), cmap=plt.cm.gray)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    return np.dot(imRGB, RGB_YIQ_TRANSFORMATION_MATRIX.T)


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    return np.dot(imYIQ, np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX).T)


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    # check if im_orig is RGB image
    if im_orig.ndim == 3:
        im_origYIQ = rgb2yiq(im_orig)
        im_eq_Y, hist_orig, hist_eq = equalize_grayscale(im_origYIQ[:, :, 0])
        im_origYIQ[:, :, 0] = im_eq_Y
        return [yiq2rgb(im_origYIQ), hist_orig, hist_eq]

    # else equalize grayscale image
    return equalize_grayscale(im_orig)


def equalize_grayscale(im_orig):
    """
    Perform histogram equalization on grayscale image only
    :param im_orig: Input float64 [0,1] grayscale
    :return: [im_eq, hist_orig, hist_eq]
    """
    # Compute the image histogram
    new_im_orig = im_orig * MAX_PIXEL_VAL
    hist_orig, bounds = np.histogram(new_im_orig, bins=256, range=[0, 255])
    # Compute the cumulative histogram.
    cum_hist = np.cumsum(hist_orig)
    #  get_first_non_zero_gray_level_in_histogram
    m = (cum_hist != 0).argmax()
    # check for zero division and if so return original image
    dom = (cum_hist[255] - cum_hist[m])
    if dom == 0:
        return [im_orig, hist_orig, hist_orig]
    # compute look-up table T
    T = np.round(255 * (cum_hist - cum_hist[m]) / dom)
    # compute new image as 1-D array by the LUT T
    new_im = T[new_im_orig.flatten().astype(int)]
    # compute the equalized histogram
    hist_eq, _ = np.histogram(new_im, bins=256, range=[0, 255])
    # compute the final equalized image as 2D array
    im_eq = (new_im / MAX_PIXEL_VAL).reshape(im_orig.shape)
    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    # check if im_orig is RGB image
    if im_orig.ndim == 3:
        im_origYIQ = rgb2yiq(im_orig)
        im_qu_Y, error = quantize_grayscale(im_origYIQ[:, :, 0], n_quant, n_iter)
        im_origYIQ[:, :, 0] = im_qu_Y
        return [yiq2rgb(im_origYIQ), error]

    # else quantize grayscale image
    return quantize_grayscale(im_orig, n_quant, n_iter)


def quantize_grayscale(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    # convert image to [0,255] scale
    new_im_orig = im_orig * MAX_PIXEL_VAL
    # Computing the image histogram
    h, bins = np.histogram(new_im_orig, bins=256, range=[0, 255])
    # Computing the initial segments division
    Z = compute_Z(h, n_quant)
    # Initializing empty Q array to update and the error list
    Q = np.zeros(n_quant)
    error = list()

    # Now, update Z and Q n_iter times
    for i in range(n_iter):
        iter_error = 0
        # each iteration save the old Z to check convergence
        Z_before = Z.copy()
        # for each segment update left border and compute it's corresponding q
        for j in range(n_quant):
            g = np.arange(np.floor(Z[j]) + 1, np.floor(Z[j + 1]) + 1).astype(int)
            hg = h[g]
            Q[j] = (np.sum(g * hg) / np.sum(hg))
            iter_error += np.sum(((Q[j] - g) ** 2) * hg)
            if j > 0:
                Z[j] = (Q[j - 1] + Q[j]) / 2
        # Check if we achieved convergence
        if np.array_equal(Z, Z_before):
            break
        error.append(iter_error)

    # compute the look-up table for the actual quantization
    mapping = get_quantization_mapping(Z, Q, n_quant)

    # compute final quantized image by the mapping table
    im_quant = (mapping[new_im_orig.flatten().astype(int)] / MAX_PIXEL_VAL).reshape(im_orig.shape)

    return [im_quant, np.asarray(error)]


def get_quantization_mapping(Z, Q, n_quant):
    """
    :param Z: Z array, the borders which divide the histograms into segments
    :param Q: Q array, the values to which each of the segments intensities will map
    :param n_quant: number of gray levels
    :return: mapping array from intensity level to corresponding q_i in Q
    """
    pixels = np.zeros(256)
    for i in range(n_quant):
        pixels[int(Z[i]) + 1: int(Z[i + 1]) + 1] = Q[i]
    return pixels


def compute_Z(hist, n_quant):
    """
    :param hist: Image histogram
    :param n_quant: number of gray levels
    :return: Z array, the borders which divide the histograms into segments
    """
    cum_hist = np.cumsum(hist)
    pixels_in_seg = cum_hist[255] // n_quant
    Z = [-1]
    for i in range(1, n_quant):
        k = np.argmax(cum_hist > pixels_in_seg * i)
        Z.append(k)
    Z.append(255)
    return np.asarray(Z)


def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    obs = im_orig.reshape(-1, 3)
    codebook, _ = cluster.vq.kmeans(obs, n_quant)
    clusters, _ = cluster.vq.vq(obs, codebook)
    return codebook[clusters].reshape(im_orig.shape)
