import numpy as np
from skimage import measure


def decision_map_detection(L, FM1, FM2, Conn=1):
    ## Find the connected regions
    Conn_Reg, num = measure.label(L, return_num=True, connectivity=Conn)
    decision_map = Conn_Reg

    tag = np.zeros(Conn_Reg.shape)

    ## Focus detection
    # first, the detected focused reions (Just consider two images)
    for ii in range(1, num+1):
        # find the ii-th region
        tag = (Conn_Reg == ii)

        # Initialize label
        label = 0

        # Compute the focus measure of this region
        sumFM1 = np.sum(np.sum(FM1 * tag))
        sumFM2 = np.sum(np.sum(FM2 * tag))

        # Comparing the focus-measures
        if sumFM2 > sumFM1:
            label = 2
        elif sumFM2 < sumFM1:
            label = 1

        # Define this region
        decision_map[tag] = label

    return decision_map
