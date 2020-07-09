from collections import Counter, defaultdict
from queue import Queue
from queue import PriorityQueue
from bs4 import BeautifulSoup
from urllib import parse, request
import itertools
from typing import Dict, List, NamedTuple
import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import sys
import json
import re
import logging
import pickle
import csv
import os

#pickled = open('mycorpus.pkl', 'rb')
#mycorpus = pickle.load(pickled)
#pickled.close()
#mycorpus = []

logging.basicConfig(level=logging.DEBUG, filename='output.log', filemode='w')
visitlog = logging.getLogger('visited')
extractlog = logging.getLogger('extracted')

def findinfonpr(url, i):
    res = request.urlopen(url)
    soup = BeautifulSoup(res, 'html.parser')
    ti = soup.find('div', class_='storytitle')
    tita = ti.get_text()
    titl = tita.replace('\n', '')
    au = soup.find('div', class_='story-meta__two')
    aut = au.get_text()
    auth = aut.replace('\n', '')
    autho = re.findall('\w+ \w+', auth)
    da = soup.find('div', class_='story-meta__one')
    dat = da.get_text()
    if '2020' not in dat:
        return None
    date = re.findall('\w+ \d{1,2}, \d\d\d\d', dat)
    body = soup.find('div', class_= 'storytext storylocation linkLocation')
    bod = body.get_text()
    bo = bod.replace('\n', '')
    return {'url': url, 'title': titl, 'author': autho, 'date': date, 'body': bo}

def findinfocbs(url, i):
    res = request.urlopen(url)
    soup = BeautifulSoup(res, 'html.parser')
    ti = soup.find('h1', class_='content__title')
    tita = ti.get_text()
    titl = tita.replace('\n', '')
    au = soup.find('p', class_='content__meta content__meta-byline')
    aut = au.get_text()
    auth = aut.replace('\n', '')
    auth = aut.replace('By', '')
    autho = re.findall('\w+ \w+', auth)
    da = soup.find('p', class_='content__meta content__meta-timestamp')
    dat = da.get_text()
    if '2020' not in dat:
        return None
    date = re.findall('\w+ \d{1,2}, \d\d\d\d', dat)
    body = soup.find('section', class_= 'content__body')
    bod = body.get_text()
    bo = bod.replace('\n', '')
    return {'url': url, 'title': titl, 'author': autho, 'date': date, 'body': bo}

def findinfobbc(url, i):
    res = request.urlopen(url)
    soup = BeautifulSoup(res, 'html.parser')
    ti = soup.find('h1', class_='story-body__h1')
    tita = ti.get_text()
    titl = tita.replace('\n', '')
    au = soup.find('div', class_='byline')
    aut = au.get_text()
    auth = aut.replace('\n', '')
    auth = aut.replace('By', '')
    autho = re.findall('\w+ \w+', auth)
    autho = autho[0]
    da = soup.find('div', class_='date date--v2')
    dat = da.get_text()
    if '2020' not in dat:
        return None
    date = re.findall('\d{1,2} \w+ \d\d\d\d', dat)
    body = soup.find('div', class_= 'story-body__inner')
    bod = body.get_text()
    bo = bod.replace('\n', '')
    return {'url': url, 'title': titl, 'author': autho, 'date': date, 'body': bo}

def findinfopbs(url, i):
    res = request.urlopen(url)
    soup = BeautifulSoup(res, 'html.parser')
    ti = soup.find('h1', class_='post__title')
    tita = ti.get_text()
    titl = tita.replace('\n', '')
    titl = titl.replace('\u2019', '')
    titl = titl.replace('\u2018', '')
    titl = titl.replace('\u201c', '')
    titl = titl.replace('\u201d', '')
    au = soup.find('p', class_='post__byline-name')
    aut = au.get_text()
    auth = aut.replace('\n', '')
    auth = auth.replace('\u2019', '')
    auth = auth.replace('\u2018', '')
    auth = auth.replace('\u201d', '')
    auth = auth.replace('\u201c', '')
    autho = re.findall('\w+ \w+', aut)
    authr = autho[0]
    da = soup.find('time', class_='post__date')
    dat = da.get_text()
    if '2020' not in dat:
        return None
    date = re.findall('\w+ \d{1,2}, \d\d\d\d', dat)
    body = soup.find('div', class_= 'body-text')
    bod = body.get_text()
    bo = bod.replace('\n', '')
    bo = bo.replace('\u2019', '')
    bo = bo.replace('\u2018', '')
    bo = bo.replace('\u201c', '')
    bo = bo.replace('\u201d', '')
    return {'url': url, 'title': titl, 'author': authr, 'date': date, 'body': bo}

def parse_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)

def parse_links_sorted(root, html):
    q = PriorityQueue()
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        htmlstring = str(html)
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            text = text.lower()
            #print(text)
            #priority queue ranks from low to high so make word matches counter negative
            counter = 0 - htmlstring.count(text)
            #if 'covid-19' in text or 'coronavirus' in text or 'health' in text or 'virus' in text or 'covid' in text:
            q.put((counter, parse.urljoin(root, link.get('href'))))
    return q

def get_links(url):
    res = request.urlopen(url)
    return list(parse_links(url, res.read()))

def crawl(root, wanted_content, within_domain):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    #figure out domain name
    if "://" in root:
        temp = root.split('://')
        temp = temp[1]
        temp = temp.split('/')
        domain = temp[0]
    else:
        temp = root.split('/')
        domain = temp[0]

    #find segment to check if self referencing link
    temp1 = root.split('/')
    if temp1[-1] == '':
        section = temp1[-2]
    else:
        section = temp1[-1]

    queue = Queue()
    queue.put(root)

    visited = []
    extracted = []

    i = 0
    while not queue.empty() and i < 50:
        url = queue.get()
        #if non self referencing or if its first arg AND not in current doc corpus
        if (url.find(section) == -1 or i == 0 or url.find('#') == -1): # and url not in mycorpus:
            try:
                req = request.urlopen(url)
                #get content type
                ctype = req.headers['Content-Type']
                #check if content type is part of wanted_content
                for w in wanted_content:
                    if w in ctype:
                        html = req.read()
                        visited.append(url)
                        visitlog.debug(url)

                        for ex in extract_information(url, html):
                            extracted.append(ex)
                            #extractlog.debug(ex)

                        q = parse_links_sorted(url, html)
                        while not q.empty():
                            new = q.get()
                            link = new[1]
                            if within_domain:
                                if domain in link:
                                    queue.put(link)
                            else:
                                queue.put(link)

            except Exception as e:
                print("")
        i += 1

    return visited

articlenum = 0

def extract_information(address, html):
    results = []

    global articlenum
    try:
        with open("finextracted.csv", mode="a+") as exfile:
            fieldnames = ['url', 'title', 'author', 'date', 'body']
            exwrite = csv.DictWriter(exfile, fieldnames=fieldnames)
            if articlenum == 0:
                exwrite.writeheader()
            if ('npr' in address):
                exwrite.writerow(findinfonpr(address, articlenum))
            elif ('pbs' in address):
                exwrite.writerow(findinfopbs(address, articlenum))
            elif ('cbs' in address):
                exwrite.writerow(findinfocbs(address, articlenum))
            elif ('bbc' in address):
                exwrite.writerow(findinfobbc(address, articlenum))
        articlenum = articlenum + 1
    except Exception as e:
        return results
    return results

def writelines(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            print(d, file=fout)



def main():
    site = sys.argv[1]
    site2 = sys.argv[2]
    site3 = sys.argv[3]
    site4 = sys.argv[4]

    content = sys.argv[5:]
    #links = get_links(site)
    #writelines('links.txt', links)
    os.remove("finextracted.csv")
    #os.remove("mycorpus.csv")

    visited= crawl(site, content, True)
    visited2= crawl(site2, content, True)
    visited3= crawl(site3, content, True)
    visited4= crawl(site4, content, True)

    with open("mycorpus.csv", mode = "a+") as corpus:
        fieldnames = ['url', 'title', 'author', 'date', 'body']
        exwrite = csv.DictWriter(corpus, fieldnames=fieldnames)
        #exwrite.writeheader()
    os.system('python3 centroid.py')
    os.system('python3 search.py')

if __name__ == '__main__':
    main()
