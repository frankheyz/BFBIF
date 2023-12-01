from skimage import morphology
import numpy as np


def Small_Block_Filter(map, N, small_size):
    #  ------------------------
    #  small_size is the defined number for small regions
    #  map is the initial fusion decision map;
    #  N is the number of the input images
    #  ------------------------

    # define the size of the small region
    P = small_size
    conn = 1
    if N == 2:
        map1 = (map == 1)
        map2 = (map == 2)

        # Process the Positive image, delete the small patches
        tmap1 = morphology.remove_small_objects(map1, P, conn)
        tmap2 = morphology.remove_small_objects(map2, P, conn)

        # sum map
        sum_map = (tmap1 & tmap2)

        # the final decision map
        newmap = (tmap1 + tmap2 * 2) * (1 - sum_map)
    else:
        # more than two input images
        result = np.zeros((map.shape, N))

        for ii in range(1, N+1):
            ptmap = (map == ii)
            tmap = morphology.remove_small_objects(ptmap, P, conn)
            result[:, :, ii] = tmap

        # find if there are confused pixels
        sCount = np.sum(result, 2)
        Tag = (sCount == 1)
        newmap = np.zeros(map.shape)

        for ii in range(1, N+1):
            newmap = newmap + ii * result[:, :, ii] * Tag

    return newmap
