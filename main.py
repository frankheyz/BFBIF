import cv2
from boundary_finding import boundary_finding

if __name__ == '__main__':
    img1 = cv2.imread(
        'D:\\share_dir\\data\\uv_refocusing\\usaf_z_step_1_test_input\\Mosaic_0_0_238.500_250ms_A_1.tif', 0
        # "D:\\share_dir\\data\\uv_refocusing\\unpressed_top_n_8bits_test_input\\Mosaic_14_2_270.000_250ms_A_1.tif", 0
    )

    img2 = cv2.imread(
        'D:\\share_dir\\data\\uv_refocusing\\usaf_z_step_1_test_input\\Mosaic_0_0_225.500_250ms_A_1.tif', 0
        # "D:\\share_dir\\data\\uv_refocusing\\unpressed_top_n_8bits_test_input\\Mosaic_13_2_290.000_250ms_A_1.tif", 0
    )

    # todo resize image to speed up
    import time
    t1 = time.time()
    n, idx = boundary_finding(img1, img2)
    t2 = time.time()
    print(t2 - t1)
    pass

