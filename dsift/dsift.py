# -*- coding: utf-8 -*-
import os
import cv2
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn import svm
from scipy import signal
from matplotlib import pyplot
from scipy.misc import imresize
from sklearn.externals import joblib
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.model_selection import GridSearchCV

classifier = 'DSIFT'
WORD = 1000
S = 0
E = 22
V = 128
# sift features
Nangles = 8
Nbins = 4
Nsamples = Nbins**2
alpha = 9.0
angles = np.array(range(Nangles))*2.0*np.pi/Nangles

def LBP(image):
    W, H = image.shape                    #获得图像长宽
    xx = [-1,  0,  1, 1, 1, 0, -1, -1]
    yy = [-1, -1, -1, 0, 1, 1,  1,  0]    #xx, yy 主要作用对应顺时针旋转时,相对中点的相对值.
    res = np.zeros((W - 2, H - 2),dtype="uint8")  #创建0数组,显而易见维度原始图像的长宽分别减去2，并且类型一定的是uint8,无符号8位,opencv图片的存储格式.
    for i in range(1, W - 2):
        for j in range(1, H - 2):
            temp = ""
            for m in range(8):
                Xtemp = xx[m] + i
                Ytemp = yy[m] + j    #分别获得对应坐标点
                if image[Xtemp, Ytemp] > image[i, j]: #像素比较
                    temp = temp + '1'
                else:
                    temp = temp + '0'
            #print int(temp, 2)
            res[i - 1][j - 1] =int(temp, 2)   #写入结果中
    return res

def gen_dgauss(sigma):
    '''
    generating a derivative of Gauss filter on both the X and Y
    direction.
    '''
    fwid = np.int(2*np.ceil(sigma))
    G = np.array(range(-fwid,fwid+1))**2
    G = G.reshape((G.size,1)) + G
    G = np.exp(- G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH,GW = np.gradient(G)
    GH *= 2.0/np.sum(np.abs(GH))
    GW *= 2.0/np.sum(np.abs(GW))
    return GH,GW

class DsiftExtractor:
    '''
    The class that does dense sift feature extractor.
    Sample Usage:
        extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
        feaArr,positions = extractor.process_image(Image)
    '''
    def __init__(self, gridSpacing, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2):
        '''
        gridSpacing: the spacing for sampling dense descriptors
        patchSize: the size for each sift patch
        nrml_thres: low contrast normalization threshold
        sigma_edge: the standard deviation for the gaussian smoothing
            before computing the gradient
        sift_thres: sift thresholding (0.2 works well based on
            Lowe's SIFT paper)
        '''
        self.gS = gridSpacing
        self.pS = patchSize
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        # compute the weight contribution map
        sample_res = self.pS / np.double(Nbins)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p,sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1,Nbins*2,2)) / 2.0 / Nbins * self.pS - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
        bincenter_h.resize((bincenter_h.size,1))
        bincenter_w.resize((bincenter_w.size,1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        # weights is the contribution of each pixel to the corresponding bin center
        self.weights = weights_h * weights_w
        #pyplot.imshow(self.weights)
        #pyplot.show()

    def process_image(self, image, positionNormalize = True,\
                       verbose = False):
        '''
        processes a single image, return the locations
        and the values of detected SIFT features.
        image: a M*N image which is a numpy 2D array. If you
            pass a color image, it will automatically be converted
            to a grayscale image.
        positionNormalize: whether to normalize the positions
            to [0,1]. If False, the pixel-based positions of the
            top-right position of the patches is returned.

        Return values:
        feaArr: the feature array, each row is a feature
        positions: the positions of the features
        '''

        image = image.astype(np.double)
        if image.ndim == 3:
            # we do not deal with color images.
            image = np.mean(image,axis=2)
        # compute the grids
        H,W = image.shape
        gS = self.gS
        pS = self.pS
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH/2
        offsetW = remW/2
        gridH,gridW = np.meshgrid(range(int(offsetH),H-pS+1,gS), range(int(offsetW),W-pS+1,gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        if verbose:
            print ('Image: w {}, h {}, gs {}, ps {}, nFea {}'.\
                    format(W,H,gS,pS,gridH.size))
        feaArr = self.calculate_sift_grid(image,gridH,gridW)
        feaArr = self.normalize_sift(feaArr)
        if positionNormalize:
            positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        else:
            positions = np.vstack((gridH, gridW))
        return feaArr, positions

    def calculate_sift_grid(self,image,gridH,gridW):
        '''
        This function calculates the unnormalized sift features
        It is called by process_image().
        '''
        H,W = image.shape
        Npatches = gridH.size
        feaArr = np.zeros((Npatches,Nsamples*Nangles))

        # calculate gradient
        GH,GW = gen_dgauss(self.sigma)
        IH = signal.convolve2d(image,GH,mode='same')
        IW = signal.convolve2d(image,GW,mode='same')
        Imag = np.sqrt(IH**2+IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((Nangles,H,W))
        for i in range(Nangles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - angles[i])**alpha,0)
            #pyplot.imshow(Iorient[i])
            #pyplot.show()
        for i in range(Npatches):
            currFeature = np.zeros((Nangles,Nsamples))
            for j in range(Nangles):
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
            feaArr[i] = currFeature.flatten()
        return feaArr

    def normalize_sift(self,feaArr):
        '''
        This function does sift feature normalization
        following David Lowe's definition (normalize length ->
        thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feaArr**2,axis=1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feaArr /= siftlen.reshape((siftlen.size,1))
        # suppress large gradients
        feaArr[feaArr>self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast]**2,axis=1)).\
                reshape((feaArr[hcontrast].shape[0],1))
        return feaArr

dsift = DsiftExtractor(8,16,1)
sift = cv2.xfeatures2d.SIFT_create(nfeatures=300,sigma=0.8,nOctaveLayers=4)

def standarizeImage(im):
    if im.shape[0] > 300:
        resize_factor = 300.0 / im.shape[0]  # don't remove trailing .0 to avoid integer devision
        im = imresize(im, resize_factor)
    return im

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
            img = standarizeImage(img)
            # _, des = sift.detectAndCompute(img, None)
            des,_ = dsift.process_image(img)
            if i[0] not in data:
                data[i[0]] = []
            data[i[0]].append(des)
    # print( 'start save')
    # with open('model/%d_%s.pkl'%(V,dataset), 'wb') as f:
        # pickle.dump(data, f)
    return data

def read_pkl(dataset='train'):
    with open('model/%d_%s.pkl'%(V,dataset), 'rb') as f:
        data = pickle.load(f)
        print(type(data),len(data))
        return data

def trainClassifier(dataset):
    trainData = np.float32([]).reshape(0, WORD)
    response = np.int64([])
    clf = joblib.load("model/%d_vocab%d-%d-%d.pkl"%(V,WORD,S,E))
    centers = clf.cluster_centers_

    target = {}
    for i,name in enumerate(list(dataset.keys())):
        target[name] = i

    for name in list(dataset.keys())[S:E]:
        idx = target[name]
        print( "Init training data of " + name + "...")
        for i in dataset[name]:
            featVec = getFeatVec(i, clf)
            trainData = np.append(trainData, featVec, axis=0)
        res = np.repeat(np.int64([idx]), len(dataset[name]))
        response = np.append(response, res)#构造target

    print( "Now train svm classifier...")
    trainData = np.float32(trainData)

    print( trainData.shape,response.shape)
    #use sklearn svm
    clf = svm.SVC(kernel='linear',C=10,probability=True)
    clf.fit(trainData,response)
    joblib.dump(clf, "model/%d_svm%d-%d-%d.model"%(V,WORD,S,E))
    print( "Done")

def getClassifier(dataset):
    trainData = np.float32([]).reshape(0, WORD)
    response = np.int64([])
    clf = joblib.load("model/%d_vocab%d-%d-%d.pkl"%(V,WORD,S,E))
    centers = clf.cluster_centers_

    target = {}
    for i,name in enumerate(list(dataset.keys())):
        target[name] = i

    for name in list(dataset.keys())[S:E]:
        idx = target[name]
        print( "Init training data of " + name + "...")
        for i in dataset[name]:
            featVec = getFeatVec(i, clf)
            trainData = np.append(trainData, featVec, axis=0)
        res = np.repeat(np.int64([idx]), len(dataset[name]))
        response = np.append(response, res)#构造target

    print( "Now train svm classifier...")
    trainData = np.float32(trainData)

    print( trainData.shape,response.shape)
    #use sklearn svm
    clf = svm.SVC()
    para = {
        'kernel':['linear','rbf'],
        'C':[0.5,1,10,50,100]
    }
    r = GridSearchCV(clf,para,cv=5)
    r.fit(trainData,response)
    print(r.cv_results_)
    # a = pd.DataFrame(r.cv_results_)
    # a.sort(['mean_test_score'],ascending=False)
    print(r.best_params_)
    print(r.best_estimator_,r.best_score_)
    # joblib.dump(clf, "model/%d_svm%d-%d-%d.model"%(V,WORD,S,E))
    print( "Done")

def learnVocabulary(dataset):
    # if not os.path.exists('model/%d_features%d-%d-%d.npy'%(V,WORD,S,E)):
    features = np.float32([]).reshape(0,V)
    for i in list(dataset.keys())[S:E]:
        print( "Learn vocabulary of " + i + "...")
        features = np.append(features,np.vstack(dataset[i]),axis=0)
        print( features.shape)
        # np.save('model/%d_features%d-%d-%d.npy'%(V,WORD,S,E),features)
    # else:
        # features = np.load('model/%d_features%d-%d-%d.npy'%(V,WORD,S,E))
        #use sklearn KMeans
    print( features.shape)
    clf = MiniBatchKMeans(n_clusters=WORD,init_size=3*WORD,verbose=False)
    clf.fit(features)
    filename = "model/%d_vocab%d-%d-%d.pkl"%(V,WORD,S,E)
    joblib.dump(clf , filename)
    print( "Done\n")
    return clf

def getFeatVec(features,clf):
    featVec = np.zeros((1, WORD))
    res = clf.predict(features)
    for i in res:
        featVec[0][i] += 1
    return featVec

def calcFeatVec(features, centers):
    '''
    计算欧式距离
    '''
    featVec = np.zeros((1, WORD))
    for i in range(0, features.shape[0]):
        fi = features[i] # 某图像一个特征点的128维向量
        # dist = np.linalg.norm(fi - vec2)
        diffMat = np.tile(fi, (WORD, 1)) - centers #此向量分别减WORD个词
        sqSum = (diffMat**2).sum(axis=1) #得到50个平方差的和
        dist = sqSum**0.5
        sortedIndices = dist.argsort() #50个数的从小到大的顺序
        idx = sortedIndices[0] # index of the nearest center
        #此128维向量距离哪个最近（平方差的和的二次根号）
        featVec[0][idx] += 1
    return featVec

def predict(dataset):
    def result(p,test_target,dataset,num):
        print("###############################")
        #top num
        l = 0
        right = 0
        for name in list(dataset.keys())[S:E]:
            r = 0
            for i in range(l,l+len(dataset[name])):
                # print(test_target[i],p[i][E-num-1:E-1])
                if test_target[i] in p[i][E-num-1:E-1]:
                    r += 1
            right+=r
            print(  "%s : %d/%d" %(name,r,len(dataset[name])))
            l+=len(dataset[name])
        print( "%d / %d"%(right,l))
        print("###############################")

    svm = joblib.load("model/%d_svm%d-%d-%d.model"%(V,WORD,S,E))
    clf = joblib.load("model/%d_vocab%d-%d-%d.pkl"%(V,WORD,S,E))
    # centers = clf.cluster_centers_
    test_data = np.float32([]).reshape(0,WORD)
    test_target = []

    target = {}
    for i,name in enumerate(list(dataset.keys())):
        target[name] = i

    for name in list(dataset.keys())[S:E]:
        idx = target[name]
        print( name)
        for i in dataset[name]:
            featVec = getFeatVec(i, clf)
            test_data = np.append(test_data,featVec,axis=0)
        test_target += [idx for i in range(len(dataset[name]))]

    p = svm.predict_proba(test_data)
    p = p.argsort()
    result(p,test_target,dataset,5)
    result(p,test_target,dataset,4)
    result(p,test_target,dataset,3)
    result(p,test_target,dataset,2)
    result(p,test_target,dataset,1)

if __name__ == '__main__':
    if not os.path.exists('model'):
        os.makedirs('model')
    data = get_vector('train')
    data_test = get_vector('test')
    # data_test = read_pkl('test')
    # data = read_pkl('train')
    print( 'learn vocabulary')
    kmeans = learnVocabulary(data)
    print( 'train svm');
    # getClassifier(data_test)
    trainClassifier(data)
    print( 'predict')
    predict(data_test)
