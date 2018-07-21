import numpy as np
from skimage import io, img_as_ubyte


def mean_k_pix(r, c, rsz, csz, gimg, k = 1):
    res = 0
    cnt = 0
    for i in range(max(r-k, 0), min(r+k, rsz - 1) + 1):
        for j in range(max(c-k, 0), min(c+k, csz - 1) + 1):
            cnt += 1
            res += abs(gimg[r,c] - gimg[i,j])
    cnt -= 1
    return res/cnt


def get_epsl(gray_img, k = 1):
    # if the img is not the gray img
    if gray_img is None or gray_img[0,0].shape != () or k < 1:
        # print("arg array is not gray img")
        return None

    rows, cols = gray_img.shape

    mean_k_pixs = np.array(
        [[mean_k_pix(i, j, rows, cols, gray_img, k) for j in range(cols)] for i in range(rows)]
    )
    # print("\n---------------------")
    # print(mean_k_pixs[80:100, 80:100])
    # print("---------------------")

    return np.mean(mean_k_pixs)


def omega_div_n_pix(r, c, rsz, csz, gimg, epsl, k = 1):
    cnt = 0
    res = 0.0
    # print(r, c)
    # print(rsz, csz)
    # print("  ")
    for i in range(max(r-k, 0), min(r+k, rsz - 1 ) + 1):
        # print(max(r-k, 0), min(r+k, rsz))
        for j in range(max(c-k, 0), min(c+k, csz - 1) + 1):

            # print(max(c-k, 0), min(c+k, csz))
            # print(i, j)
            if i == r and j == c: 
                continue
            cnt += 1
            if abs(gimg[r,c]-gimg[i,j]) > epsl:
                res += 1.0
    return res/cnt

def get_pif_pixs(gimg, epsl, k = 1):
    if gimg is None or epsl is None:
        return None

    rows, cols = gimg.shape
    
    return np.array(
        [[omega_div_n_pix(i, j, rows, cols, gimg, epsl, k) for j in range(cols)] for i in range(rows)]
    )

def inNeb_pix(r, c, rsz, csz, pif, k = 1):
    cnt = 0
    res = 0.0
    for i in range(max(r-k, 0), min(r+k, rsz - 1) + 1):
        for j in range(max(c-k, 0), min(c+k, csz - 1) + 1):
            if i == r and j == c: 
                continue
            cnt += 1
            if pif[i, j] >= 0.5 :
                res += 1.0
    return res/cnt
    



def get_nif_pixs(pif, k = 1):
    if pif is None:
        return None

    rows, cols = pif.shape

    return np.array(
        [[inNeb_pix(i, j, rows, cols, pif, k) for j in range(cols)] for i in range(rows)]
    )



# def get_seed_pixs(pif, nif, )

    

     


if __name__ == "__main__":
    gimg = io.imread('/home/hallwood/Code/devenv/PraticeLesson/neu-dataset/zebra/23ef4c8d-a169-3e23-a7bc-6620633a88cd.jpg', as_gray=True)
    # gimg = img_as_ubyte(gimg)
    print(gimg[80:100, 80:100])

    tmp = np.array(gimg)
    k = 100

    io.imshow(gimg)
    io.show()

    epsl = get_epsl(gimg, k)

    # print(epsl)

    pif = get_pif_pixs(gimg, epsl, k)

    # print(pif)

    nif = get_nif_pixs(pif, k)

    # print(nif)

    rows, cols = gimg.shape
    for i in range(rows):
        for j in range(cols):
            gimg[i, j] *= pif[i, j]
            # if nif[i, j] >= 0.5 and pif[i,j] >= 0.5:
            #     pass
            # else:
            #     tmp[i,j] = 0.0
    
    io.imshow(gimg)
    io.show()
    
