import os
from multiprocessing import Pool

import numpy as np

import TransImg.epsl
import TransImg.merge
import TransImg.nif
import TransImg.pif

import BatchTools.oppr
import BatchTools.clsfic

def do_preproc(cls_name, img_list):
    for img_name in img_list:
        BatchTools.oppr.write_back_tag(cls_name, img_name)

if __name__ == "__main__":
    # 生成目标文件夹
    BatchTools.oppr.create_target_path()


    # 读取所有类还有其文件夹下的图片名字
    cls_list = BatchTools.clsfic.get_cls_list()
    cls_batch = BatchTools.clsfic.get_cls_batch(cls_list)
    
    # 在目标文件夹下生成类文件夹
    BatchTools.oppr.create_cls_dirs(cls_list)

    # 生成目标线程池
    pool = Pool(len(cls_list))


    # 转换
    # for cls_name in cls_list: 
    #     pool.apply_async(do_preproc, args=(cls_name, cls_batch[cls_name]))
    # pool.close()
    # pool.join()

     # 转换
    for cls_name in cls_list:
        for img_name in cls_batch[cls_name]:
            BatchTools.oppr.write_back_tag(cls_name, img_name)
    
    # 结束， 可以跑一天

