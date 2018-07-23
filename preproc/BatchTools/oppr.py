import os

from skimage import io

from BatchTools.clsfic import get_cls_list, get_cls_batch
from TransImg import merge, epsl, pif, nif

def create_target_path(tag_path = "/home/hallwood/Code/devenv/PracticeLesson/preproc/tg2018/"):
    if os.path.exists(tag_path):
        return None
    os.mkdir(tag_path)

def create_cls_dirs(cls_list, tag_path = "/home/hallwood/Code/devenv/PracticeLesson/preproc/tg2018/"):
    os.chdir(tag_path)
    for cls_dir_name in cls_list:
        if os.path.exists(tag_path + cls_dir_name):
            continue
        os.mkdir(cls_dir_name)

def write_back_tag(cls_name, img_name, 
    src_path = "/home/hallwood/Code/devenv/PracticeLesson/preproc/ds2018/", 
    tag_path =  "/home/hallwood/Code/devenv/PracticeLesson/preproc/tg2018/"):
    
    src = src_path + cls_name + '/' + img_name
    dest = tag_path + cls_name + '/' + img_name

    gimg = io.imread(src, as_gray=True)
    e = epsl.get_epsl(gimg)
    p = pif.get_pif_matrix(gimg, e)
    n = nif.get_nif_matrix(p)
    res = merge.get_merged_pif_nif_mat(gimg, n, p)

    io.imsave(dest, res)


if __name__ == "__main__":
    create_target_path()
    cls_list = get_cls_list()
    create_cls_dirs(cls_list)


    