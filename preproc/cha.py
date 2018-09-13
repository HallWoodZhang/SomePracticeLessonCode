#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import csv
import cv2
import pickle
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cluster import KMeans

sift = cv2.xfeatures2d.SIFT_create(nfeatures=300)

WORD = 1000
S = 0
E = 5

def get_vector(dataset='train'):
    csvfile = dataset+'.csv'
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
            _, des = sift.detectAndCompute(img, None)
            if i[0] not in data:
                data[i[0]] = []
            data[i[0]].append(des)
    print( 'start save')
    with open('model/'+dataset+'.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data

def read_pkl(dataset='train'):
    with open('model/'+dataset+'.pkl', 'rb') as f:
        data = pickle.load(f)
        print((type(data),len(data)))
        return data

def trainClassifier(dataset):
    trainData = np.float32([]).reshape(0, WORD)
    response = np.int64([])
    clf = joblib.load("model/vocab%d-%d-%d.pkl"%(WORD,S,E))
    centers = clf.cluster_centers_

    dictIdx = 0
    for name in list(dataset.keys())[S:E]:
        print( "Init training data of " + name + "...")
        for i in dataset[name]:
            featVec = calcFeatVec(i, centers)#一张图片的50维向量
            trainData = np.append(trainData, featVec, axis=0)
        res = np.repeat(np.int64([dictIdx]), len(dataset[name]))
        response = np.append(response, res)#构造target
        dictIdx += 1
        print( "Done\n")

    print( "Now train svm classifier...")
    trainData = np.float32(trainData)

    print(trainData.shape,response.shape)
    #use sklearn svm
    clf = svm.SVC(probability=True)
    clf.fit(trainData,response)
    joblib.dump(clf, "model/svm%d-%d-%d.model"%(WORD,S,E))
    print( "Done")

def learnVocabulary(dataset):
    data = {}
    features = np.float32([]).reshape(0,128)
    for i in list(dataset.keys())[S:E]:
        print( "Learn vocabulary of " + i + "...")
        for j in dataset[i]:
            features = np.append(features,j,axis=0)
        print( features.shape)

        #use sklearn KMeans
    print( features.shape)
    clf = KMeans(n_clusters=WORD)
    clf.fit(features)
    filename = "model/vocab%d-%d-%d.pkl"%(WORD,S,E)
    joblib.dump(clf , filename)
    print( "Done\n")
    return clf

def calcFeatVec(features, centers):
    '''
    计算欧式距离
    '''
    featVec = np.zeros((1, WORD))
    for i in range(0, features.shape[0]):
        fi = features[i] # 某图像一个特征点的128维向量
        diffMat = np.tile(fi, (WORD, 1)) - centers #此向量分别减WORD个词
        sqSum = (diffMat**2).sum(axis=1) #得到50个平方差的和
        dist = sqSum**0.5
        sortedIndices = dist.argsort() #50个数的从小到大的顺序
        idx = sortedIndices[0] # index of the nearest center
        #此128维向量距离哪个最近（平方差的和的二次根号）
        featVec[0][idx] += 1
    return featVec

def predict(dataset):
    svm = joblib.load("model/svm%d-%d-%d.model"%(WORD,S,E))
    clf = joblib.load("model/vocab%d-%d-%d.pkl"%(WORD,S,E))
    centers = clf.cluster_centers_
    test_data = np.float32([]).reshape(0,WORD)
    test_target = []

    dictIdx = 0
    for name in list(dataset.keys())[S:E]:
        print( name)
        for i in dataset[name]:
            featVec = calcFeatVec(i, centers)
            test_data = np.append(test_data,featVec,axis=0)
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
        l = l + len(dataset[name])
    print( "%d / %d"%(right,l))



if __name__ == '__main__':
    # os.mkdir("model")
    # data = get_vector('train')
    # data_test = get_vector('test')
    data_test = read_pkl('test')
    data = read_pkl('train')
    print( 'learn vocabulary')
    kmeans = learnVocabulary(data)
    print( 'train svm')
    trainClassifier(data)
    print( 'predict')
    predict(data_test)
