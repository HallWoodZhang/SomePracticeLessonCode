#coding=utf-8
from __future__ import unicode_literals

import csv
import os
import random

def store_csv(src_dict):
    with open("./output/dataset.csv", "w") as target:
        writer = csv.writer(target)
        for tp, path_list in src_dict.items():
            for path in path_list:
                writer.writerow([path, tp])
            
        
def init_dict(dataset_path = "../neu-dataset/"):
    src_dict = {}
    for _, dirs, _ in os.walk(dataset_path):
        for d in dirs:
            src_dict[d] = []

    for tp in src_dict.keys():
        for path, _, files in os.walk(dataset_path + tp + '/'):
            for f in files:
                src_dict[tp].append(path + f)

    return src_dict


def generate_random_tuples(src_dict):
    res = []
    for key, vals in src_dict.items():
        for val in vals:
            res.append((key, val))

    random.shuffle(res)
    return res


def get_subset_dict(sz, src_dict, div_num = 5):
    res = []

    if sz is None or sz == 0 or src_dict is None or len(src_dict) == 0: 
        return res

    sz /= div_num
    num_per_class = int(sz / len(src_dict))

    for key, path_list in src_dict.items():
        random.shuffle(path_list)
        for path in path_list[0:num_per_class]:
            res.append((key, path))
    
    return res


def divide_tuples(src_dict, output = './output'):
    if src_dict is None or len(src_dict) == 0:
        return False

    train_path = output + "/train/train.csv"
    test_path = output + "/test/test.csv"

    with open(train_path, "w") as train_file, open(test_path, "w") as test_file:
        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)

        for key, paths in src_dict.items():
            for i in range(len(paths)):
                if i & 0x01 == 0:
                    train_writer.writerow([paths[i], key])
                else:
                    test_writer.writerow([paths[i], key]) 



def main(argv):
    # work 1
    src_dict = init_dict('/home/hallwood/Code/devenv/PraticeLesson/neu-dataset/')
    store_csv(src_dict)

    # work 2
    random_list = generate_random_tuples(src_dict)
    for item in random_list:
        print(item)

    # work 3
    subset_list = get_subset_dict(len(random_list), src_dict)
    for item in subset_list:
        print(item)
    
    # work 4
    divide_tuples(src_dict)


if __name__ == "__main__":
    main(None)