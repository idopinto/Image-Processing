import sol3 as s
from matplotlib import pyplot as plt


def main():
    im_orig = s.read_image('presubmit_externals/monkey.jpg', 1)
    # plt.imshow(im_orig,cmap='gray')
    # plt.show()
    gpyr, _ = s.build_gaussian_pyramid(im_orig, max_levels=3, filter_size=3)
    lpyr, filter_vec = s.build_laplacian_pyramid(im_orig,max_levels=3,filter_size=3)

    # s.display_pyramid(pyr=gpyr,levels=3)
    # s.display_pyramid(pyr=lpyr,levels=3)
    s.blending_example1()
    # s.blending_example2()

if __name__ == '__main__':
    main()

