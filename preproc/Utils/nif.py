import numpy as np

def inNeb_pix(r, c, rsz, csz, pif, k = 1):
    cnt = 0
    res = 0.0

    for i in range(max(0, r-k), min(r+k+1, rsz)):
        for j in range(max(0, c-k), min(c+k+1, csz)):
            if i == r and j == c:
                continue
            cnt += 1
            if pif[i,j] >= 0.5:
                res += 1.0
    
    return res/cnt

def cal_nif_pixs(pif, k = 1):
    rsz, csz = pif.shape

    return np.array(
        [[inNeb_pix(i, j, rsz, csz, pif, k) for j in range(csz)] for i in range(rsz)]
    )

def get_nif_matrix(pif, k = 1):
    if pif is None or k < 1:
        return None
    return cal_nif_pixs(pif, k)