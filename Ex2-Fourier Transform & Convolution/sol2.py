import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from scipy.io import wavfile
import imageio as iio
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

GRAYSCALE = 1
RGB = 2
MAX_PIXEL_VAL = 255.


def get_dft_or_idft_matrix(N, idft=False):
    """"""
    x_range_row = np.arange(0, N)
    x_range_col = np.arange(0, N).reshape(-1, 1)
    if not idft:
        return np.exp((-2 * np.pi * 1j * x_range_row * x_range_col) / N)
    return np.exp((2 * np.pi * 1j * x_range_row * x_range_col) / N)


def DFT(signal):
    """
    :param signal: an array of dtype float64 with shape (N,) or (N,1)
    :return: complex Fourier signal, the same shape as the input
    """
    return get_dft_or_idft_matrix(signal.shape[0]) @ signal


def IDFT(fourier_signal):
    """
    :param fourier_signal: an array of dtype complex128 with same shape
    :return: complex signal, the same shape as the input
    Note that when the
fourier_signal is a transformed into a real signal you can expect IDFT to return real values as well,
although it may return with a tiny imaginary part. You can ignore the imaginary part (see tips).
    """
    N = fourier_signal.shape[0]
    return (get_dft_or_idft_matrix(N, idft=True) @ fourier_signal) / N


def DFT2(image):
    """

    :param image: grayscale image of dtype float64, shape (M,N) or (M,N,1)
    :return: the fourier image of the original image
    """
    return np.apply_along_axis(DFT, 1, np.apply_along_axis(DFT, 0, image))


def IDFT2(fourier_image):
    """

     :param image: grayscale image of dtype float64, shape (M,N) or (M,N,1)
     :return: the inverse of the fourier image
     """
    return np.apply_along_axis(IDFT, 1, np.apply_along_axis(IDFT, 0, fourier_image))


def change_rate(filename, ratio):
    """
        :param filename: string representing the path to a WAV file
        :param ratio: positive float64 representing the duration change
        :return: None
        Assuming that 0.25 < ratio < 4
        E.g if the original sample rate is 4000Hz and the ratio is 1.25,
        then the new sample rate will be 5000Hz
    """
    sr, audio_data = wavfile.read(filename)
    wavfile.write("results/audio/change_rate.wav", int(ratio * sr), audio_data)


def change_samples(filename, ratio):
    """

        :param filename: string representing the path to a WAV file
        :param ratio: positive float64 representing the duration change
        Assuming that 0.25 < ratio < 4
        :return: 1D nd array of dtype float64 representing the new sample points
    """
    sr, audio_data = wavfile.read(filename)
    if ratio == 1:
        return audio_data
    resized_audio = np.real(resize(audio_data, ratio)).astype(np.int16)
    wavfile.write("results/audio/change_samples.wav", sr, resized_audio)
    return resized_audio


def resize(data, ratio):
    """
    :param data: 1D ndarray of dtype float64 or complex128 representing the original sample points
    :param ratio: positive float64 representing the duration change
    :return: 1D ndarray of the dtype of data representing the new sample points
    """
    shifted_fourier = np.fft.fftshift(DFT(data))
    return fourier_fast_forward(shifted_fourier, ratio) if ratio > 1 else fourier_slowdown(shifted_fourier, ratio)


def fourier_fast_forward(shifted_fourier, ratio):
    N = len(shifted_fourier)
    center = N // 2
    new_size = shifted_fourier.size // ratio
    start = center - new_size // 2
    cropped_shifted_data = shifted_fourier[int(start): int(start + new_size)]
    return IDFT(np.fft.ifftshift(cropped_shifted_data))


def fourier_slowdown(shifted_fourier, ratio):
    N = len(shifted_fourier)
    target_size = N // ratio
    right_zeros = int((target_size - N) // 2)
    left_zeros = right_zeros if (target_size - N) % 2 == 0 else right_zeros + 1
    return IDFT(np.fft.ifftshift(np.concatenate([np.zeros(left_zeros), shifted_fourier, np.zeros(right_zeros)])))


def resize_spectrogram(data, ratio):
    """

    :param data: data is a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: new sample points according to ratio with the same dtype as data
    Assume that 0.25 < ratio < 4
    """
    if ratio == 1:
        return data
    return istft(np.apply_along_axis(resize, 1, stft(data), ratio)).astype(np.int16)


def resize_vocoder(data, ratio):
    """

    :param data:  1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return:  given data rescaled according to ratio with the same datatype as data.
    Assume that 0.25 < ratio < 4
    """
    return istft(phase_vocoder(stft(data), ratio)).astype(np.int16)


def conv_der(im):
    """
    :param im: Grayscale images of type float64
    :return:magnitude of the derivative
    """
    dx = signal.convolve2d(im, np.asarray([[0.5, 0, -0.5]]), 'same')
    dy = signal.convolve2d(im, np.asarray([[0.5], [0], [-0.5]]), 'same')
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def fourier_der(im):
    """

    :param im: Grayscale images of type float64
    :return: magnitude of the derivative
    """
    rows, cols = im.shape[0], im.shape[1]
    shifted_F = np.fft.fftshift(DFT2(im))
    dx = IDFT2(np.fft.ifftshift(shifted_F * (2 * np.pi * 1j / cols) * np.arange(-cols // 2, cols // 2)))
    dy = IDFT2(
        np.fft.ifftshift(shifted_F * (2 * np.pi * 1j / rows) * (np.arange(-rows // 2, rows // 2)).reshape(-1, 1)))
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    # print(n_frames)
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


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


if __name__ == '__main__':
    # x = DFT2(np.array([[5,10],[2,3]]))
    # change_rate("external/aria_4kHz.wav", 2)
    # change_samples("external/aria_4kHz.wav", 2)
    #
    #
    # sr, audio_data = wavfile.read("external/aria_4kHz.wav")
    # resized_data1 = resize_spectrogram(audio_data,2)
    # resized_data2 = resize_vocoder(audio_data,2)
    # wavfile.write("results/audio/spectogram_only.wav", sr, resized_data1)
    # wavfile.write("results/audio/with_vocoder.wav", sr, resized_data2)


    img = read_image("monkey.jpg", 1)

    # f_der = fourier_der(img)
    c_der = conv_der(img)

    # plt.imshow(img, cmap='gray')
    # plt.title("Original Image")
    # plt.show()
    #
    # plt.imshow(f_der, cmap='gray')
    # plt.title("Magnitude of fourier derivative")
    # plt.show()

    plt.imshow(c_der, cmap='gray')
    plt.title("Magnitude of convolution derivative")
    plt.show()
