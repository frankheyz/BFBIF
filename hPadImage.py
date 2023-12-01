import numpy as np

def hPadImage(A, domain, padopt):
    # pad image
    domainSize = domain.shape
    center = np.floor(
        (np.array(domainSize)) /2
    )
    r,c = np.where(domain)
    r = r - center[0]
    c = c - center[1]

    padSize = [int(max(abs(r))), int(max(abs(c)))]

    if padopt == 'symmetric':
        A = np.pad(A, pad_width=padSize, mode='symmetric')
    else:
        raise ValueError("Unimplemented pad option.")

    return A
