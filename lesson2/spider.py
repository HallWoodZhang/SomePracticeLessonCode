import os

from selenium import webdriver
from bs4 import BeautifulSoup
import requests


def spider(search_str, clsname, num, init_dir = True, min_id = 1):
    if init_dir is True:
        os.mkdir("../new_data/" + clsname)

    for i in range(min_id, min_id + num):
        url= "http://pic.sogou.com/d?query=" + search_str + "&mode=1&did=" + str(i)
        driver = webdriver.PhantomJS(executable_path=\
        '/home/hallwood/UserRepos/phantomjs-2.1.1-linux-x86_64/bin/phantomjs')

        driver.get(url)
        html_doc = driver.page_source
        soup = BeautifulSoup(html_doc, 'html.parser')
        
        item = str(soup.find('a', id = 'imageBox'))
        print(item)
        
        item = item.split('src="')[1].split('"')[0]
        print(item)

        with open("../new_data/" + clsname + '/' + item.split('/')[-1], "wb") as target:
            res = requests.get(item)
            target.write(res.content)
        


def main():
    # spider("%F7%E7%C2%B9", "elk", 20, min_id = 120)
    # spider("%BE%A8%D3%E3", "whale", 31)
    # spider("%D0%DC", "bear", 20)
    # spider("%D7%D4%D0%D0%B3%B5", "bicycle", 20)
    # spider("%C4%F1", "bird", 20)
    # spider("%B3%B5", "car", 20)
    # spider("%C4%CC%C5%A3", "cow", 30)
    # spider("%BA%FC%C0%EA", "fox", 30)
    # spider("%B3%A4%BE%B1%C2%B9", "giraffe", 30)
    # spider("%C2%ED", "horse", 30)
    # spider("%BF%BC%C0%AD", "koala", 30)
    # spider("%CA%A8%D7%D3", "lion", 30)
    # spider("%BA%EF%D7%D3", "monkey", 20)
    # spider("%B7%C9%BB%FA", "plane", 20)
    # spider("%D0%A1%B9%B7", "puppy", 20, min_id = 30)
    # spider("%C3%E0%D1%F2", "sheep", 20)
    # spider("%B5%F1%CF%F1", "state", 20)
    # spider("%C0%CF%BB%A2", "tiger", 20)
    # spider("%CB%FE", "tower", 20)
    # spider("%BB%F0%B3%B5", "train", 20)
    spider("%B0%DF%C2%ED", "zebra", 20)


if __name__ == "__main__":
    main()