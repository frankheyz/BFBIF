import numpy as np
import cv2
from skimage.morphology import disk
import scipy.io
import numpy as np


def read_mat_disks():
    mat = scipy.io.loadmat('disks.mat')
    disks = mat['disks']
    disks_dict = {i+1: v for i, v in enumerate(mat['disks'][0])}

    return disks_dict

def multiscale_morph(img, num=5):
    """
    A python implementation of multiscale_morph.m by "Boundary find Based Multi-focus Image Fusion through Multi-scale
    Morphological Focus-measure, Information Fusion 35 (2017) 81-101"
    :param img: numpy 2D image
    :param num: number of scale of structural element
    :return: FM - muti-scale focus measure
    """
    FM = np.zeros(img.shape)
    se_dict = read_mat_disks()
    for i in range(1, num+1):
        scale = 2 * i + 1
        # note that cv2 implementation of disk is different from MATLAB's strel
        # se = disk(scale - 1)

        se = se_dict[i]
        # one scale focus - measure
        g = cv2.dilate(img, se) - cv2.erode(img, se)
        # the composite focus - measure
        FM = FM + 1 / scale * (g)

    return FM


if __name__ == '__main__':
    im_a = cv2.imread(
        r'/home/heyz/data/uv_refocusing/mouse_brain_s260_f260_z_step_2_1/Mosaic_1_2_316.000_250ms_A_1.tif', 0
    )
    im_b = cv2.imread(
        r'/home/heyz/data/uv_refocusing/mouse_brain_s260_f260_z_step_2_1/Mosaic_2_5_318.000_250ms_A_1.tif', 0
    )
    im_c = cv2.imread(
        r'/home/heyz/data/uv_refocusing/mouse_brain_s260_f260_z_step_2_1/Mosaic_2_5_328.000_250ms_A_1.tif', 0
    )

    print(np.mean(
        multiscale_morph(im_a))
    )
    print(np.mean(
        multiscale_morph(im_b))
    )

    print(np.mean(
        multiscale_morph(im_c))
    )