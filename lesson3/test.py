import time

import numpy as np
from PIL import Image

def normal():
    start = time.time()
    tot = 0
    l = [i for i in range(1, 1001)]
    for i in l:
        tot += i**2 + i**3

    stop = time.time()
    print("tot for res: ", tot)
    print("Time span: ", stop - start)


def np_method():
    start = time.time()
    tot = 0
    l = [i for i in range(1, 1001)]
    ln = np.array(l).astype(np.int64)
    tot += np.sum(ln**2)
    tot += np.sum(ln**3)
    print("tot for res: ", tot)
    print("Time span: ", time.time() - start)


def pratice1():
    image_path = '../neu-dataset/elk/00bOOOPIC23.jpg'
    im = Image.open(image_path)
    width, height = im.size
    pix = im.load()
    matrix = np.array(pix)
    matrix.shape = (width, height, 3)
    print(matrix)




if __name__ == "__main__":
    print("test")
    pratice1()