import os

from selenium import webdriver
from bs4 import BeautifulSoup
import requests


def spider(search_str, clsname, num, init_dir = False, min_id = 1):
    if init_dir is True:
        os.mkdir("../neu-dataset/" + clsname)

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

        with open("../neu-dataset/" + clsname + '/' + item.split('/')[-1], "wb") as target:
            res = requests.get(item)
            target.write(res.content)
        


def main():
    spider("%F7%E7%C2%B9", "elk", 120)

if __name__ == "__main__":
    main()