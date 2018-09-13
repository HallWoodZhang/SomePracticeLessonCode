# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import random
from PIL import Image

def split(train_file):
    with open(train_file) as flist:
        lines = [line.strip() for line in flist]
        d = {}
        for i in lines:
            name,label,minx,miny,maxx,maxy = i.split(',')
            if name not in d:
                d[name] = []
            d[name].append("%s,%s,%s,%s,%s,%s\n"%(name,label,minx,miny,maxx,maxy))
        t = []
        for i in list(d.keys()):
            t.append(d[i])
        random.shuffle(t)
        a,b = train_test_split(t,test_size=0.2)
        with open('./a.txt','w') as af:
            for i in a:
                af.writelines(i)
        with open('./b.txt','w') as bf:
            for i in b:
                bf.writelines(i)

def trans(filename):
    name_dict = {}
    with open(filename) as f:
        lines = [line.strip() for line in f]
        d = {}
        for i in lines:
            name,label,minx,miny,maxx,maxy = i.split(',')
            if name not in d:
                d[name] = []
            d[name].append([label,minx,miny,maxx,maxy,0])
    return d

def _reader_creator(settings, file_list,mode):
    def reader():
        a = trans(file_list)
        for i in list(a.keys()):
            #TODO add img path root
            img = Image.open(img_path)
            img_width, img_height = img.size
            img = np.array(img)

            sample_labels = a[i]
            if mode == 'train':
                batch_sampler = []
                # hard-code here
                batch_sampler.append(
                    image_util.sampler(1, 1, 1.0, 1.0, 1.0, 1.0, 0.0,
                                       0.0))
                batch_sampler.append(
                    image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.1,
                                       0.0))
                batch_sampler.append(
                    image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.3,
                                       0.0))
                batch_sampler.append(
                    image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.5,
                                       0.0))
                batch_sampler.append(
                    image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.7,
                                       0.0))
                batch_sampler.append(
                    image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.9,
                                       0.0))
                batch_sampler.append(
                    image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.0,
                                       1.0))
                """ random crop """
                sampled_bbox = image_util.generate_batch_samples(
                    batch_sampler, bbox_labels, img_width, img_height)

                if len(sampled_bbox) > 0:
                    idx = int(random.uniform(0, len(sampled_bbox)))
                    img, sample_labels = image_util.crop_image(
                        img, bbox_labels, sampled_bbox[idx], img_width,
                        img_height)

            img = Image.fromarray(img)
            img = img.resize((settings.resize_w, settings.resize_h),
                             Image.ANTIALIAS)
            img = np.array(img)

            if mode == 'train':
                mirror = int(random.uniform(0, 2))
                if mirror == 1:
                    img = img[:, ::-1, :]
                    for i in xrange(len(sample_labels)):
                        tmp = sample_labels[i][1]
                        sample_labels[i][1] = 1 - sample_labels[i][3]
                        sample_labels[i][3] = 1 - tmp

            if len(img.shape) == 3:
                img = np.swapaxes(img, 1, 2)
                img = np.swapaxes(img, 1, 0)

            img = img.astype('float32')
            img -= settings.img_mean
            img = img.flatten()

            if mode == 'train' and len(sample_labels) == 0: continue
            yield img.astype('float32'), sample_labels

    return reader()

if __name__ == '__main__':
    split('./dataset/train.txt')
    a = trans('./a.txt')
