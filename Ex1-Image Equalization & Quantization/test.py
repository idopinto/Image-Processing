import os
import numpy as np
from matplotlib import pyplot as plt
import sol1 as s

RGB_DIR = 'RGB'
GRAYSCALE_DIR = 'Grayscale'


def main():
    gray_images = read_images(GRAYSCALE_DIR, 1)
    rgb_images = read_images(RGB_DIR, 2)

    # Test 1 - Read & Display Gray -> Gray
    # display_images_test(GRAYSCALE_DIR, 1)
    #
    # # Test 2 - Read & Display RGB -> RGB
    # display_images_test(RGB_DIR, 2)
    #
    # # Test 3 - Read & Display RGB -> Gray
    # display_images_test(RGB_DIR, 1)

    # Test 4 - rgb2yiq & yiq2rgb
    # for imRGB in rgb_images:
    #     plt.imshow(imRGB, cmap=plt.cm.gray)
    #     plt.show()
    #     yiq = s.rgb2yiq(imRGB)
    #     plt.imshow(yiq, cmap=plt.cm.gray)
    #     plt.show()
    #     rgb = s.yiq2rgb(yiq)
    #     plt.imshow(rgb, cmap=plt.cm.gray)
    #     plt.show()

    # Test 5 - yiq channels
    # orig = s.read_image('RGB\example_for_yiq.jpg', 2)
    # plt.imshow(orig, cmap=plt.cm.gray)
    # plt.show()
    # yiq = s.rgb2yiq(orig)
    # rgb = s.yiq2rgb(yiq)
    # plt.imshow(rgb, cmap=plt.cm.gray)
    # plt.show()
    # display_yiq(yiq)

    # Test 4 - histogram_equalize on toy example
    # toy_test()

    # Test 5 - histogram_equalize on Grayscale
    # equalization_test(gray_images)

    # Test 6 - histogram_equalize on RGB
    # equalization_test(rgb_images)

    # Test 7 - histogram quantization grayscale
    # im = s.read_image(f'{RGB_DIR}/MyProfilePicture.jpg', 1)
    # im_quant, error = s.quantize(im, 6, 20)
    # display_im_and_im_qu(im, im_quant, 6)

    # Test 8 - histogram quantization RGB
    # im = s.read_image('RGB/MyProfilePicture.jpg', 2)
    # im_quant, error = s.quantize(im, 6, 20)
    # display_im_and_im_qu(im, im_quant, 6)

    # Test 9 - histogram quantization grayscale not equalized toy example
    # x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
    # grad = np.tile(x, (256, 1))
    # im_quant, err = s.quantize(grad / 255, 5, 10)
    # display_im_and_im_qu(grad, im_quant, 6)

    # Test 10 - histogram quantization RGB - bonus
    # im_orig = s.read_image('RGB/jerusalem.jpg', 2)
    # im_quant = s.quantize_rgb(im_orig, 3)
    # display_im_and_im_qu(im_orig,im_quant, 3)


def display_images_test(directory, representation):
    for image in os.listdir(directory):
        im = os.path.join(directory, image)
        if im.endswith(".jpg") or im.endswith(".png"):
            s.imdisplay(im, representation)


def read_images(directory, representation):
    images = []
    for image in os.listdir(directory):
        im = os.path.join(directory, image)
        if im.endswith(".jpg") or im.endswith(".png"):
            images.append(s.read_image(im, representation))
    return images


def equalization_test(images):
    for im in images:
        im_cpy = im.copy()
        im_eq, hist_orig, hist_eq = s.histogram_equalize(im)
        display_im_and_im_eq(im_cpy, im_eq)


def display_im_and_im_eq(im_orig, im_eq):
    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(1, 2, 1)

    # showing image
    plt.imshow(im_orig, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title("Original Image")

    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 2, 2)

    # showing image
    plt.imshow(im_eq, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title("Equalized Image")
    plt.show()


def display_im_and_im_qu(im_orig, im_qu, n_quant):
    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(1, 2, 1)

    # showing image
    plt.imshow(im_orig, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title("Original Image")

    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 2, 2)

    # showing image
    plt.imshow(im_qu, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(f"Quantized Image ( {n_quant} quants )")
    plt.show()


def display_yiq(yiq):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(2, 2, 1)

    # showing image
    plt.imshow(yiq[:, :, 0])
    plt.axis('off')
    plt.title("Y channel")

    # Adds a subplot at the 2nd position
    fig.add_subplot(2, 2, 2)

    # showing image
    plt.imshow(yiq[:, :, 1])
    plt.axis('off')
    plt.title("I channel")

    # Adds a subplot at the 3rd position
    fig.add_subplot(2, 2, 3)

    # showing image
    plt.imshow(yiq[:, :, 2])
    plt.axis('off')
    plt.title("Q channel")

    # Adds a subplot at the 4th position
    fig.add_subplot(2, 2, 4)

    # showing image
    plt.imshow(yiq)
    plt.axis('off')
    plt.title("YIQ image")

    plt.show()


if __name__ == '__main__':
    main()
