import os

def get_cls_list(src_path="/home/hallwood/Code/devenv/PracticeLesson/preproc/ds2018/"):
     os.chdir(src_path)
     return os.listdir()

def get_img_list_for_cls(cls_name, src_path="/home/hallwood/Code/devenv/PracticeLesson/preproc/ds2018/"):
    if cls_name is None: return []
    os.chdir(src_path + cls_name)
    return filter(lambda img: img.split('.')[1] == "jpg", os.listdir())

def get_cls_batch(cls_list, src_path="/home/hallwood/Code/devenv/PracticeLesson/preproc/ds2018/"):
    res = {}
    for cls_name in cls_list:
        res[cls_name] = get_img_list_for_cls(cls_name)
    return res

if __name__ == "__main__":
    print(get_cls_list())
    print("-------")
    print(list(get_img_list_for_cls('elk')))