import cv2
import scipy.ndimage
import numpy as np
from skimage.morphology import thin
from multiscale_morph import multiscale_morph
from decision_map_detection import decision_map_detection
from boundary_reconstruction import boundary_reconstruction
from PIL import Image
from skimage import measure
import copy
from ismember import ismember


def boundary_finding(img1_raw, img2_raw, sw_sz=7, scales=5, b_sz=20):
    """
    python implementation of
    "boundary finding based multi-focus image fusion through multi-scale morphological focus-measure"
    Paper by Y Zhang et.el.
    :param img1_raw: image read with cv2
    :param img2_raw: image read with cv2
    :param sw_sz:
    :param scales:
    :param b_sz:
    :return: numbers of image used for fusion, index of the dominant image
    """

    decision_map = None
    fimg = None

    ## Compute saliency for each image
    # Focus-measure: Multiscale morphological focus-measure
    size_x, size_y = img1_raw.shape
    div = 4
    resize_size_x = int(size_x * 1/div)
    resize_size_y = int(size_y * 1 / div)
    img1resized = cv2.resize(img1_raw.astype(float), (resize_size_x, resize_size_y))
    img2resized = cv2.resize(img2_raw.astype(float), (resize_size_x, resize_size_y))

    scale_num = scales
    FM1 = multiscale_morph(img1resized, scale_num)
    FM2 = multiscale_morph(img2resized, scale_num)
    # FM1 = cv2.resize(FM1, (size_x, size_y))
    # FM2 = cv2.resize(FM2, (size_x, size_y))

    # Sum of the focus-measure
    H = np.ones((sw_sz, sw_sz))
    sumFM1 = scipy.ndimage.correlate(FM1, H, mode='constant')
    sumFM2 = scipy.ndimage.correlate(FM2, H, mode='constant')

    maxFM = np.maximum(FM1, FM2)

    max_sumFM = np.maximum(sumFM1, sumFM2)
    min_sumFM = np.minimum(sumFM1, sumFM2)

    # detect the boundary regions
    sum_maxFM = scipy.ndimage.correlate(maxFM, H, mode='constant')
    sum_minFM = sumFM1 + sumFM2 - sum_maxFM

    dif_max_min_sum = max_sumFM - min_sumFM
    dif_sum_max_min = sum_maxFM - sum_minFM

    dfmap = (dif_max_min_sum >= 0.8 * dif_sum_max_min) & (dif_max_min_sum >= sw_sz ** 2)

    ## Thin the boundaries
    # Get the dimensions of the input images
    p1, p2 = img1resized.shape
    large_dfmap = np.zeros((p1 + 4, p2 + 4))
    large_dfmap[2:-2, 2:-2] = dfmap
    line_map = thin(abs(large_dfmap - 1))
    line_map = line_map[2:-2, 2:-2]
    # img = Image.fromarray(line_map)
    # img.show()

    ## Remove the short lines
    L, num= measure.label(line_map, return_num=True, connectivity=2)
    properties = measure.regionprops(L)
    pL = [i.area for i in properties]
    pl_sorted = copy.deepcopy(pL)
    pl_sorted.sort(reverse=True)
    area_sort = pl_sorted
    line_num = int(np.ceil(num * 0.2))
    large_area = area_sort[0:line_num]
    th = np.mean(large_area)

    larger_than_th_idx = np.argwhere(pL>th) + 1  # note that 1 indexing is used here
    # larger_than_th = np.array(pL)[larger_than_th_idx]
    ppL, ppL_idx = ismember(L, larger_than_th_idx)
    L = ~ppL
    decision_map = decision_map_detection(L, FM1, FM2, 1)
    Boundary = (decision_map == 0)

    ## Boundary Reconstruction: the third step of reconstruction in the paper
    smallsz = round(p1 * p2 / 40)
    decision_map0 = boundary_reconstruction(decision_map, Boundary, smallsz)

    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # plt.imshow(decision_map0)
    # plt.show()
    # To find a more accurate boundary, by marker based watershed segmentation in the gradient domain
    # todo unimplemented
    # Refine the boundaries in the decision map
    # todo unimplemented

    # Boundary fusion
    # todo unimplemented

    # return decision_map size to determine which image is dominant (size==1 means one of the input is dominant)
    return np.unique(decision_map0).size, int(decision_map0[0, 0])
    # return decision_map, fimg
