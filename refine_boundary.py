import numpy as np
from hPadImage import hPadImage


def refine_boundary(decision_map, Boundary):
    # % Usage: refined_decision_map = refine_boundary(decision_map)
    # % Input: decision_map
    # % ---decision_map: Input decision map with the boundary needing
    # % ---Boundary: Initial boundary
    # % refinement
    # %
    # % Output: decision_map
    # % ---refined_decision_map: the refined decision map without isolated
    # % boundary lines.
    #
    # % Initialize the refined decision map
    refined_decision_map = decision_map

    # find the boundary positions
    row, col = np.where(Boundary)
    row_len = len(row)
    box_sz = 3

    # padding the image for convinient computation
    domain = np.ones((box_sz, box_sz))
    pad_dfmap = hPadImage(decision_map, domain, 'symmetric')

    # count and filter the isolated boundaries
    for ii in range(row_len):
        x = row[ii]
        y = col[ii]
        box_data = pad_dfmap[x : x + box_sz, y : y + box_sz]

        # maximum bumber filter
        ind1 = np.where(box_data == 1)
        ind2 = np.where(box_data == 2)

        tag = 0

        if np.any(ind1) and not np.any(ind2):
            tag = 1

        if not np.any(ind1) and np.any(ind2):
            tag = 2

        refined_decision_map[x, y] = tag

    return refined_decision_map