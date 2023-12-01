import numpy as np
from skimage import measure


def Nearest_Filter(map, N):
    # %-----------------------------------------------------%
    # % classify the non region as the nearest region method
    # % Formula: result = Nearest_Filter(map, N);
    # % map is the initial fusion decision map;
    # % N is the number of the input images
    # %----------------------------------------------------%
    # Method 2

    # Get the dimensions
    xx, yy = map.shape

    # Find the non region
    tmap = (map == 0)
    tL = np.zeros((yy, xx))

    # Process the Positive image, delete the small patches
    L, num = measure.label(tmap, return_num=True, connectivity=2)

    # for each non regions, find the its bounding box
    properties = measure.regionprops(L)
    boxes = [i.bbox for i in properties]
    boxes_len = len(boxes)

    for ii in range(0, boxes_len):
        x = boxes[ii][0]
        y = boxes[ii][1]

        w = boxes[ii][3] - y
        h = boxes[ii][2] - x

        # large scale bounding box
        top = max(0, x - 2)
        bottom = min(x + w + 1, xx - 1)

        left = max(0, y - 2)
        right = min(y + h + 1, yy - 1)

        # This method is only true for two input images

        # Extract the block bounding from the map
        region = map[top:bottom, left:right]

        # Count the number of the regions in the bounding box
        numCount1 = np.sum(region == 1)
        numCount2 = np.sum(region == 2)

        if numCount1 != numCount2:
            if numCount1 > numCount2:
                tL[L == ii + 1] = 1
            else:
                tL[L == ii + 1] = 2

    result = map + tL * tmap

    return result
