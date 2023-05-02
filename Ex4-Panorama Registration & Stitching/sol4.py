import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
from scipy.ndimage.filters import convolve
from scipy.ndimage import map_coordinates

import sol4_utils

# CONSTANTS
DX_FILTER = np.array([[1, 0, -1]])
DY_FILTER = np.array([[1], [0], [-1]])
KERNEL_SIZE = 3
K = 0.04

def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # Get the Ix and Iy derivatives of the image using the filters [1,0,-1],[1,0,-1]^T respectively
    Ix, Iy = convolve(im, DX_FILTER), convolve(im, DY_FILTER)
    # Blur the images Ix^2, Iy^2, IxIy with the function blur_spatial, kernel_size=3
    Ix_sb, Iy_sb = sol4_utils.blur_spatial(Ix ** 2, KERNEL_SIZE), sol4_utils.blur_spatial(Iy ** 2, KERNEL_SIZE)
    IxIy_b = sol4_utils.blur_spatial(Ix * Iy, KERNEL_SIZE)
    # plot_two_images(Ix_sb, Iy_sb)
    # Then for each pixel you will have the following matrix M
    # measure how big are the eigenvalues: R= det(M)-0.04(trace(M))^2
    detM = (Ix_sb * Iy_sb) - (IxIy_b ** 2)
    traceM = Ix_sb + Iy_sb
    R = detM - K * (traceM ** 2)
    R_binary = non_maximum_suppression(R)
    coordinates = np.argwhere(R_binary)
    coordinates_cpy = np.copy(coordinates)
    coordinates_cpy[:, 0], coordinates_cpy[:, 1] = coordinates[:, 1], coordinates[:, 0]
    return coordinates_cpy


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    k = 1 + 2 * desc_rad
    descriptors = np.zeros((pos.shape[0], k, k))
    y, x = pos[:, 0], pos[:, 1]
    for i in range(pos.shape[0]):
        pgrid = np.mgrid[x[i] - desc_rad: x[i] + desc_rad + 1, y[i] - desc_rad: y[i] + desc_rad + 1]
        descriptors[i] = patch_normalize(map_coordinates(im, pgrid, order=1, prefilter=False))
    return descriptors


def patch_normalize(patch):
    norm = np.linalg.norm(patch - np.mean(patch))
    return (patch - np.mean(patch)) / norm if norm != 0 else patch * 0


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    feature_locations = spread_out_corners(pyr[0], 7, 7, 5)
    feature_descriptors = sample_descriptor(pyr[2], np.divide(feature_locations, 4), 3)
    return [feature_locations, feature_descriptors]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    new_col = desc1.shape[1] * desc1.shape[1]
    S = np.dot(desc1.reshape(-1, new_col), desc2.reshape(-1, new_col).T)
    second_largest_by_row = np.partition(S, -2, axis=1)[:, -2]
    second_largest_by_col = np.partition(S, -2, axis=0)[-2, :]
    mask = (S >= second_largest_by_row[:, np.newaxis]) & (S >= second_largest_by_col[np.newaxis, :]) & (S > min_score)
    survivors = S * mask
    largest_by_row = np.partition(survivors, -1, axis=1)[:, -1]
    largest_by_col = np.partition(survivors, -1, axis=0)[-1, :]
    final_mask = (survivors == largest_by_row[:, np.newaxis]) & (survivors == largest_by_col[np.newaxis, :]) \
                 & (survivors > min_score)

    matches_in_desc1, matches_in_desc2 = np.where(final_mask)
    # matches_in_desc1, matches_in_desc2 = np.where(mask)

    return matches_in_desc1, matches_in_desc2


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    pos1 = np.hstack([pos1, np.ones((pos1.shape[0], 1))]).T
    transformed_points = H12 @ pos1
    return (transformed_points[:2, :] / transformed_points[2:, :]).T


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    # pick a random set of 2 point matches from the supplied N point matches
    N = points1.shape[0]
    JIN = []
    max_rate = 0
    for j in range(num_iter):
        J = [np.random.randint(N)] if translation_only else np.random.choice(np.arange(N), 2)
        H12 = estimate_rigid_transform(points1[J], points2[J], translation_only=translation_only)
        point2tag = apply_homography(points1, H12)
        error = np.sum((point2tag - points2) ** 2, axis=1)
        err_mask = (error < inlier_tol)
        inlier_rate = (err_mask.sum() / N)
        if inlier_rate > max_rate:
            JIN = J
            max_rate = inlier_rate

    return get_final_inliers_and_H12(JIN, points1, points2, inlier_tol, translation_only)


def get_final_inliers_and_H12(JIN, points1, points2, inlier_tol, translation_only):
    H12 = estimate_rigid_transform(points1[JIN], points2[JIN], translation_only=translation_only)
    point2tag = apply_homography(points1, H12)
    error = np.sum((point2tag - points2) ** 2, axis=1)
    inliers = np.argwhere((error < inlier_tol))
    inliers = inliers.reshape((inliers.shape[0]))
    return [H12, inliers]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    im = np.hstack([im1, im2])
    points2_cpy = points2.copy()
    points2_cpy[:, 0] += im1.shape[1]

    for i in range(points1.shape[0]):
        plt.plot([points1[i, 0], points2_cpy[i, 0]], [points1[i, 1], points2_cpy[i, 1]], mfc='r', c='b', lw=.3, ms=2,
                 marker='o')

    for i in range(inliers.shape[0]):
        plt.plot([points1[inliers[i], 0], points2_cpy[inliers[i], 0]],
                 [points1[inliers[i], 1], points2_cpy[inliers[i], 1]], mfc='r', c='y', lw=.4, ms=2,
                 marker='o')

    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    N = len(H_succesive)
    H2m = [np.array([])] * (N + 1)

    H2m[m] = np.eye(3)
    for i in range(m - 1, -1, -1):
        H2m[i] = H2m[i + 1] @ H_succesive[i]

    H2m[m + 1] = np.linalg.inv(H_succesive[m])
    for i in range(m + 2, N + 1):
        H2m[i] = H2m[i - 1] @ np.linalg.inv(H_succesive[i - 1])

    for i in range(N):
        H2m[i] /= H2m[i][2, 2]

    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    trans_corners = apply_homography(np.array([[0, 0], [w, 0], [0, h], [w, h]]), homography).astype(np.int)
    return np.array([[np.min(trans_corners[:, 0]), np.min(trans_corners[:, 1])],
                     [np.max(trans_corners[:, 0]), np.max(trans_corners[:, 1])]])


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    bounding_box = compute_bounding_box(homography, image.shape[1], image.shape[0])
    x_range = np.arange(bounding_box[0][0], bounding_box[1][0])
    y_range = np.arange(bounding_box[0][1], bounding_box[1][1])
    xy_coords = np.meshgrid(x_range, y_range)
    points = np.vstack([xy_coords[0].flatten(), xy_coords[1].flatten()]).T
    hpoints = apply_homography(points, np.linalg.inv(homography))
    hpoints1 = np.array([hpoints[:, 1].reshape(xy_coords[1].shape), hpoints[:, 0].reshape(xy_coords[0].shape)])
    return map_coordinates(image, hpoints1, order=1, prefilter=False)


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


