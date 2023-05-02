import numpy as np
import matplotlib.pyplot as plt
from sol4 import *
from sol4_utils import *

def display_pair(im1, im2,RGB=False):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    if RGB:
        plt.imshow(im1)
    else:
        plt.imshow(im1, cmap='gray')
    fig.add_subplot(1, 2, 2)
    if RGB:
        plt.imshow(im2)
    else:
        plt.imshow(im2, cmap='gray')
    plt.show()


def display_corners(im1, corners1, im2, corners2):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    for i in range(corners1.shape[0]):
        plt.plot([corners1[i, 0], corners1[i, 0]], [corners1[i, 1], corners1[i, 1]], mfc='r', c='b', lw=.3, ms=2,
                 marker='o')
    plt.imshow(im1, cmap='gray')

    fig.add_subplot(1, 2, 2)
    for i in range(corners2.shape[0]):
        plt.plot([corners2[i, 0], corners2[i, 0]], [corners2[i, 1], corners2[i, 1]], mfc='r', c='b', lw=.3, ms=2,
                 marker='o')
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.show()


def main():
    im1 = sol4_utils.read_image('externals/oxford1.jpg', 1)
    im2 = sol4_utils.read_image('externals/oxford2.jpg', 1)
    im1_rgb = sol4_utils.read_image('externals/oxford1.jpg', 2)
    im2_rgb = sol4_utils.read_image('externals/oxford2.jpg', 2)
    display_pair(im1_rgb, im2_rgb)

    pyr1, pyr2 = sol4_utils.build_gaussian_pyramid(im1, 3, 5)[0], sol4_utils.build_gaussian_pyramid(im2, 3, 5)[0]

    kp1, desc1 = find_features(pyr1)
    kp2, desc2 = find_features(pyr2)
    display_corners(im1, kp1, im2, kp2)
    matches_ind1, matches_ind2 = match_features(desc1, desc2, min_score=0.2)

    print(f"left image shape:{im1.shape}")
    print(f"right image shape:{im1.shape}")
    print(f"found {kp1.shape[0]} features in left image")
    print(f"found {kp2.shape[0]} features in right image")
    print(f"sampled {desc1.shape[0]} descriptors for corners in left image. each {desc1.shape[1]}X{desc1.shape[2]}")
    print(f"sampled {desc2.shape[0]} descriptors for corners in right image.each {desc2.shape[1]}X{desc2.shape[2]}")
    print(f"found {matches_ind1.shape[0]} matches overall between the images")
    display_corners(im1, kp1[matches_ind1], im2, kp2[matches_ind2])
    H12, inliers_ind = ransac_homography(kp1[matches_ind1], kp2[matches_ind2], num_iter=200, inlier_tol=15)
    display_matches(im1, im2, kp1[matches_ind1], kp2[matches_ind2], inliers_ind)

    H2m = accumulate_homographies([H12],0)
    warped_im2 = warp_image(im2_rgb,H2m[1])
    display_pair(im1_rgb, warped_im2, RGB=True)
    print("Image 2 aligned according to image 1")

if __name__ == '__main__':
    main()
