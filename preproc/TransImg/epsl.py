import numpy as np

def mean_k_pix(r, c, rsz, csz, gimg, k = 1):
    res = 0
    cnt = 0
    for i in range(max(r-k, 0), min(r+k + 1, rsz)):
        for j in range(max(c-k, 0), min(c+k + 1, csz)):
            cnt += 1
            res += abs(gimg[r, c] - gimg[i, j])
        cnt -= 1
        return res/cnt

def cal_epsl(gimg, k = 1):
    rsz, csz = gimg.shape
    mean_k_pixs = np.array(
        [[mean_k_pix(i, j, rsz, csz, gimg, k) for j in range(csz)] for i in range(rsz)]
    )
    return np.mean(mean_k_pixs)

def get_epsl(gimg, k = 1):
    if gimg is None or gimg[0,0].shape != () or k < 1:
        return None
    return cal_epsl(gimg, k)

if __name__ == "__main__":
    pass