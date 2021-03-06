import queue

import numpy as np
from skimage import io

from epsl import get_epsl
from pif import get_pif_matrix
from nif import get_nif_matrix

# from TransImg.epsl import get_epsl
# from TransImg.pif import get_pif_matrix
# from TransImg.nif import get_nif_matrix

def set_seed_queue(nif, pif, res, q):
    rsz, csz = nif.shape
    for i in range(rsz):
        for j in range(csz):
            if nif[i, j] >= 0.5 and pif[i, j] >= 0.5:
                q.put((i, j))
            else:
                res[i, j] = 0

def bfs_from_seed(seed_q, gimg, res, nif, pif):
    k = 1

    dr = np.empty(res.shape).astype(np.bool)
    dr[:,:] = False
    # print(dr)

    judge = lambda i,j: nif[i, j] > 0.5 or pif[i, j] > 0.5
    
    cnt = 0
    rsz, csz = nif.shape
    while seed_q.empty() is False:
        r, c = seed_q.get()
        # print([r,c])
        if dr[r,c] is True:
            continue
        dr[r,c]=True
        res[r,c]=gimg[r,c]
        for i in range(max(r-k, 0), min(r+k+1, rsz)):
            for j in range(max(c-k, 0), min(c+k+1, csz)):
                if judge(i, j)== True and dr[i,j]==False :
                    # print("in: ", i, " r, ", j, " c")
                    seed_q.put((i,j))
                    cnt += 10
                    res[i, j] = gimg[i, j]
                # if cnt%100 == 0:
                #     print(seed_q.qsize())
    
        
def get_merged_pif_nif_mat(gimg, nif, pif):
    if nif.shape != pif.shape or gimg is None:
        return None

    res = np.array(gimg)

    q = queue.Queue()
    set_seed_queue(nif, pif, res, q)
    # print(q.empty())
    # cnt = 0
    # while q.empty() is False:
    #     print(q.get())
    #     cnt += 1

    # print("size of queue: ", cnt)
    bfs_from_seed(q, gimg, res, nif, pif)
    # print("bfs over")
    
    return res
    
if __name__ == "__main__":
    gimg = io.imread('/home/hallwood/Code/devenv/PracticeLesson/preproc/ds2018/lion/066e0162-b2d4-466d-9c6f-259f59351ffc.jpg.jpg', as_gray=True)

    

    io.imshow(gimg)
    io.show()
    epsl = get_epsl(gimg)
    # print(epsl, " , epsl")
    pif = get_pif_matrix(gimg, epsl)
    # print("pif ready")
    nif = get_nif_matrix(pif)
    # print("nif ready")
    res = get_merged_pif_nif_mat(gimg, nif, pif)
    # print(pif[80:100, 80:100])
    # print('------------------------------------')
    # print(nif[80:100, 80:100])


    io.imshow(res)
    io.show()


