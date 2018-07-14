import time
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

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
    m = np.array(im)
    r = m[:,:,0].astype(np.uint64)
    g = m[:,:,1].astype(np.uint64)
    b = m[:,:,2].astype(np.uint64)
    print(m)
    print(r)
    print(g)
    print(b)
    print(((r+g+b)/3).astype(np.uint64))


def pratice2():
    image_path = '../neu-dataset/elk/00bOOOPIC23.jpg'
    im = Image.open(image_path)
    width, height = im.size
    im_arr = np.array(im)
    print(np.max(im_arr))
    print(np.min(im_arr))
    print(np.average(im_arr))
    print(np.std(im_arr))

    cum_im_arr = im_arr[::2,::2,:]
    print(cum_im_arr.shape)
    print(im_arr.shape)


def pratice3():
    x = np.arange(1, 100, step = 0.00001, dtype = np.float)
    y = np.log(x).astype(float)
    pyplot.loglog(x, y)

    figure = pyplot.figure()
    axes = Axes3D(figure)

    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)

    X,Y = np.meshgrid(x,y)
    Z = np.cos((X**2 + Y**2)**0.1)

    axes.plot_surface(X,Y,Z,cmap='rainbow')
    pyplot.show()


# def EuclideanDistances(A, B):
#     BT = B.transpose()
#     # vecProd = A * BT
#     vecProd = np.dot(A,BT)
#     # print(vecProd)
#     SqA =  A**2
#     # print(SqA)
#     sumSqA = np.matrix(np.sum(SqA, axis=1))
#     sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
#     # print(sumSqAEx)
#
#     SqB = B**2
#     sumSqB = np.sum(SqB, axis=1)
#     sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
#     SqED = sumSqBEx + sumSqAEx - 2*vecProd
#     # SqED[SqED < 0]=0.0
#     ED = np.sqrt(SqED)
#     print(ED.shape)
#     return ED.shape[0] - np.mean(ED)

def dist(l, r):
    return np.sqrt(np.sum(np.square(l - r)))

def pratice4():
    img_dir = "../neu-dataset/elk/"
    img_list = []
    for img in os.listdir(img_dir):
        img_list.append(img)

    print(img_list)

    def img_dist(lhs, rhs):
        limgarr = np.array(Image.open(lhs).resize((64, 64)))
        rimgarr = np.array(Image.open(rhs).resize((64, 64)))

        # print(limgarr)
        # print()
        # print(rimgarr)

        # limgarr_r = limgarr[:,:,0]
        # limgarr_g = limgarr[:,:,1]
        # limgarr_b = limgarr[:,:,2]
        # rimgarr_r = rimgarr[:,:,0]
        # rimgarr_g = rimgarr[:,:,1]
        # rimgarr_b = rimgarr[:,:,2]

        # print(EuclideanDistances(limgarr_r, rimgarr_r))
        # print(EuclideanDistances(limgarr_g, rimgarr_g))
        # print(EuclideanDistances(limgarr_b, rimgarr_b))

        l = limgarr.flatten()
        r = rimgarr.flatten()
        print(l.shape)
        print(r.shape)
        return dist(l, r)


    print(img_dist(img_dir+'3e5503c1-b508-33df-9ff2-37c78c9b0241.jpg', img_dir+'01e979575416f432f875a429967c47.jpg@2o.jpg'))
    print(img_dist(img_dir+'00bOOOPIC23.jpg', img_dir+'00bOOOPIC23.jpg'))

    s = set()

    # for i in range(len(img_list)):
    #     if i >= len(img_list):
    #         break
    #     if i in s:
    #         continue
    #     for j in range(len(img_list[i+1:])):
    #         if img_dist(img_dir+img_list[i], img_dir+img_list[i+j]) < 10:
    #             s.add(j)

    print(s)




if __name__ == "__main__":
    pratice4()
