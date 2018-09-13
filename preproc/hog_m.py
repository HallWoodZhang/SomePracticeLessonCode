#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import cv2
import os
import pickle
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cluster import KMeans,MiniBatchKMeans

winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

WORD = 1764
S = 0
E = 22

def get_vector(dataset='train'):
    csvfile = ""+dataset+'.csv'
    print( 'start')
    data = {}
    with open(csvfile,'r') as f:
        j = 0
        train_file = csv.reader(f)
        for i in train_file:
            j += 1
            if j%100==0:
                print( j)
            img = cv2.imread(i[1],cv2.IMREAD_GRAYSCALE)
            if img is None:
                print( i[1])
                continue
            des = hog.compute(img,winStride,padding,locations)
            # WORD,_ = des.shape
            if i[0] not in data:
                data[i[0]] = []
            data[i[0]].append(des)
    print( 'start save')
    with open('hog_model/'+dataset+'.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data

def read_pkl(dataset='train'):
    with open('hog_model/'+dataset+'.pkl', 'rb') as f:
        data = pickle.load(f)
        print(type(data),len(data))
        return data

def trainClassifier(dataset):
    trainData = np.float32([]).reshape(0, WORD)
    response = np.int64([])

    dictIdx = 0
    for name in list(dataset.keys())[S:E]:
        print( "Init training data of " + name + "...")
        for i in dataset[name]:
            # print (i.shape)
            trainData = np.append(trainData, i.reshape(1,-1), axis=0)
        res = np.repeat(np.int64([dictIdx]), len(dataset[name]))
        response = np.append(response, res)#构造target
        dictIdx += 1
        print( "Done\n")

    print( "Now train svm classifier...")
    trainData = np.float32(trainData)

    print( trainData.shape,response.shape)
    #use sklearn svm
    clf = svm.SVC(probability=True,C=1)
    clf.fit(trainData,response)
    joblib.dump(clf, "hog_model/svm%d-%d-%d.model"%(WORD,S,E))
    print( "Done")

def predict_proba(dataset):
    svm = joblib.load("hog_model/svm%d-%d-%d.model"%(WORD,S,E))

    test_data = np.float32([]).reshape(0,WORD)
    test_target = []

    dictIdx = 0
    for name in dataset.keys()[S:E]:
        print( name)
        for i in dataset[name]:
            test_data = np.append(test_data,i.reshape(1,-1),axis=0)
        test_target += [dictIdx for i in range(len(dataset[name]))]
        dictIdx += 1

    p = svm.predict_proba(test_data)
    print( p)
    p = p.argsort()
    print( p)
    print( test_target)
    q = 0
    l = 0
    right = 0
    for name in dataset.keys()[S:E]:
        r = 0
        for i in range(l,l+len(dataset[name])):
            if test_target[i] in p[i][E-5:E]:
                r += 1
        right+=r
        print(  "%s : %d/%d" %(name,r,len(dataset[name])))
        l+=len(dataset[name])
    print( "%d / %d"%(right,l))

def predict(dataset):
    svm = joblib.load("hog_model/svm%d-%d-%d.model"%(WORD,S,E))

    test_data = np.float32([]).reshape(0,WORD)
    test_target = []

    dictIdx = 0
    for name in list(dataset.keys())[S:E]:
        print( name)
        for i in dataset[name]:
            test_data = np.append(test_data,i.reshape(1,-1),axis=0)
        test_target += [dictIdx for i in range(len(dataset[name]))]
        dictIdx += 1

    p = svm.predict(test_data)
    print( p)
    print( test_target)
    q = 0
    l = 0
    right = 0
    for name in list(dataset.keys())[S:E]:
        r = 0
        for i in range(l,l+len(dataset[name])):
            if test_target[i] == p[i]:
                r += 1
        right+=r
        print(  "%s : %d/%d" %(name,r,len(dataset[name])))
        l+=len(dataset[name])
    print( "%d / %d"%(right,l))


if __name__ == '__main__':
    if not os.path.exists('hog_model'):
        os.makedirs('hog_model')
    data = get_vector('train')
    data_test = get_vector('test')
    #data_test = read_pkl('test')
    # data = read_pkl('train')
    print( 'train svm')
    trainClassifier(data)
    print( 'predict')
    predict(data_test)
