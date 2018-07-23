import numpy as np

def omega_div_n_pix(r, c, rsz, csz, gimg, epsl, k = 1):
    cnt = 0
    res = 0.0
    for i in range(max(r-k, 0), min(r+k+1, rsz)):
        for j in range(max(c-k, 0), min(c+k+1, csz)):
            # print(i, " i, ", j, " j")
            if i == r and j == c:
                continue
            # print("mutex")
            cnt += 1
            if abs(gimg[r,c] - gimg[i,j]) > epsl:
                res += 1.0
    return res/cnt

def cal_pif_pixs(gimg, epsl, k = 1):
    rsz, csz = gimg.shape
    return np.array(
        [[omega_div_n_pix(i, j, rsz, csz, gimg, epsl, k) for j in range(csz)] for i in range(rsz)]
    )

def get_pif_matrix(gimg, epsl, k = 1):
    if gimg is None or epsl is None or k < 1:
        return None
    return cal_pif_pixs(gimg, epsl, k)

if __name__ == "__main__":
    pass