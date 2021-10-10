import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute dx and dy with cv2.Sobel. (2 Lines)
    Idx = cv2.Sobel(I, -1, 1, 0, ksize=3)
    Idy = cv2.Sobel(I, -1, 0, 1, ksize=3)

    # Step 2: Ixx Iyy Ixy from Idx and Idy (3 Lines)
    Ixx = np.multiply(Idx, Idx)
    Iyy = np.multiply(Idy, Idy)
    Ixy = np.multiply(Idx, Idy)

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur (5 Lines)
    A = cv2.GaussianBlur(Ixx, (3, 3), sigmaY=1, sigmaX=1)
    B = cv2.GaussianBlur(Iyy, (3, 3), sigmaY=1, sigmaX=1)
    C = cv2.GaussianBlur(Ixy, (3, 3), sigmaY=1, sigmaX=1)

    #Step 4:  Compute the harris response with the determinant and the trace of T (see announcement) (4 lines)
    # could be done with reshaping
    R = np.zeros(I.shape)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            T = np.array([[A[i, j], C[i, j]], [C[i, j], B[i, j]]])
            R[i, j] = np.linalg.det(T) - k * (np.trace(T)**2)

    return (R, A, B, C, Idx, Idy)


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)
    R_padded = np.pad(R, 1)

    # Step 2 (recommended) : create one image for every offset in the 3x3 neighborhood (6 lines).
    image_stack = R_padded
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            R_i = np.roll(R_padded, i, axis=0)
            R_ij = np.roll(R_i, j, axis=1)
            image_stack = np.dstack((image_stack, R_ij))

    # Step 3 (recommended) : compute the greatest neighbor of every pixel (1 line)
    max = np.nanmax(image_stack, axis=2)

    # Step 4 (recommended) : Compute a boolean image with only all key-points set to True (1 line)
    mask_max = np.equal(max, R_padded)
    mask_th = np.where(max > threshold, True, False)
    mask = np.logical_and(mask_max, mask_th)
    mask = mask[1:-1, 1:-1]

    # Step 5 (recommended) : Use np.nonzero to compute the locations of the key-points from the boolean image (1 line)
    mask = np.nonzero(mask.T)
    return mask




def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)
    R_padded = np.pad(R, 1)

    # Step 2 (recommended) : Calculate significant response pixels (1 line)
    mask_th = np.where(R_padded <= edge_threshold, True, False)

    # Step 3 (recommended) : create two images with the smaller x-axis and y-axis neighbors respectively (2 lines).
    image_stack_x = R_padded
    image_stack_y = R_padded
    for i in [-1, 1]:
        R_y = np.roll(R_padded, i, axis=0)
        R_x = np.roll(R_padded, i, axis=1)
        image_stack_x = np.dstack((image_stack_x, R_x))
        image_stack_y = np.dstack((image_stack_y, R_y))

    # Step 4 (recommended) : Calculate pixels that are lower than either their x-axis or y-axis neighbors (1 line)
    min_x = np.nanmin(image_stack_x, axis=2)
    min_y = np.nanmin(image_stack_y, axis=2)
    mask_min_x = np.equal(min_x, R_padded)
    mask_min_y = np.equal(min_y, R_padded)
    mask_min = np.logical_or(mask_min_x, mask_min_y)

    mask = np.logical_and(mask_min, mask_th)
    mask = mask[1:-1, 1:-1]

    # Step 5 (recommended) : Calculate valid edge pixels by combining significant and axis_minimal pixels (1 line)
    return mask
