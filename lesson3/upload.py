import requests
import os


import uuid

url = 'http://219.216.65.165:9000/uploadfile'
#
# files = {'file': open()}

def init_dict():

    elk_list = []

    for path, _, files in os.walk("../neu-dataset/elk/"):
        for f in files:
            elk_list.append(path + f)

    print(elk_list)

    return elk_list

id = 0;

if __name__ == "__main__":
    d = init_dict()
    for path in d:
        files = {'file': open(path, 'rb')}
        st = uuid.uuid1()
        print(st)
        values = {'iUUID': str(st) + ".jpg", "path":"elk"}

        r = requests.post(url, files = files, data = values)
        print(r.text)