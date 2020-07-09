import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple
import csv

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import os

import sys

### File IO and processing

class Document(NamedTuple):
    doc_id: int
    title: List[str]
    description: List[str]
    sensenum: List[int]

    def sections(self):
        return [self.title, self.description, self.sensenum]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  title: {self.title}\n" +
            f"  description: {self.description}\n" +
            f"  sensenum: {self.sensenum}\n")

stemmer = SnowballStemmer('english')

def split_doc_covid():
    file = "news.csv"
    collist = ["title", "description"]
    df = pd.read_csv("news.csv", usecols = collist)
    df.to_csv("news2.csv", index = False, header = True)
    reader = csv.reader(open("news2.csv", "r"), delimiter=',')
    writer = csv.writer(open("news3.csv", 'w'), delimiter='*')
    writer.writerows(reader)
    file = "news3.csv"
    with open(file) as f:
        lines = f.readlines()
        split = int(len(lines))
    with open(file[:-4] + '-train.tsv', 'w') as f:
        for line in lines:
            i = 0
            new = ".I " + str(1)   #1 if covid 0 if not
            print(new, file=f)
            for word in line.split('*'):
                if i == 0:
                    newline = "<t> " + word
                    print(newline, file=f)
                elif i == 1:
                    newline = "<d> " + word
                    print(newline, file=f)
                i += 1
    ftrain = file[:-4] + '-train.tsv'
    os.remove("news2.csv")
    return ftrain, split

def split_new():
    file = "finextracted.csv"
    collist = ["url", "title", "author", "date", "body"]
    df = pd.read_csv("finextracted.csv", usecols = collist)
    df.to_csv("finextracted2.csv", index = False, header = True)
    reader = csv.reader(open("finextracted2.csv", "r"), delimiter=',')
    writer = csv.writer(open("finextracted3.csv", 'w'), delimiter='*')
    writer.writerows(reader)
    file = "finextracted3.csv"
    with open(file) as f:
        lines = f.readlines()
    with open(file[:-4] + '-dev.tsv', 'w') as f:
        for line in lines:
            i = 0
            new = ".I "
            print(new, file=f)
            for word in line.split('*'):
                if i == 0:
                    newline = "<u> " + word
                    print(newline, file=f)
                elif i == 1:
                    newline = "<t> " + word
                    print(newline, file=f)
                elif i == 2:
                    newline = "<a> " + word
                    print(newline, file=f)
                elif i == 3:
                    newline = "<p> " + word
                    print(newline, file=f)
                elif i == 4:
                    newline = "<d> " + word
                    print(newline, file=f)
                i += 1
    fdev = file[:-4] + '-dev.tsv'
    os.remove("finextracted2.csv")
    return fdev

def split_doc_ncovid():
    file = "noncovid.csv"
    collist = ["headline", "short_description"]
    df = pd.read_csv("noncovid.csv", usecols = collist)
    df.to_csv("noncovid2.csv", index = False, header = True)
    reader = csv.reader(open("noncovid2.csv", "r"), delimiter=',')
    writer = csv.writer(open("noncovid3.csv", 'w'), delimiter='*')
    writer.writerows(reader)
    file = "noncovid3.csv"
    with open(file) as f:
        lines = f.readlines()
        split = int(len(lines))
    with open(file[:-4] + '-train.tsv', 'w') as f:
        for line in lines:
            i = 0
            new = ".I " + str(0)   #1 if covid 0 if not
            print(new, file=f)
            for word in line.split('*'):
                if i == 0:
                    newline = "<t> " + word
                    print(newline, file=f)
                elif i == 1:
                    newline = "<d> " + word
                    print(newline, file=f)
                i += 1
    ftrain = file[:-4] + '-train.tsv'
    os.remove("noncovid2.csv")
    return ftrain, split

def combine():
    f = "real-train.tsv"
    with open(f, 'w') as tr:
        with open("news3-train.tsv") as textfile1, open("noncovid3-train.tsv") as textfile2:
            for x in textfile1:
                x = x.strip()
                print(x, file = tr)
            for y in textfile2:
                y = y.strip()
                print(y, file = tr)
    realtrain = "real-train.tsv"
    os.remove("news3.csv")
    os.remove("noncovid3.csv")
    os.remove("news3-train.tsv")
    os.remove("noncovid3-train.tsv")
    return realtrain

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                identifier = line[3:]
                docs.append(defaultdict(list))
                i += 1
                docs[i]['X'].append(identifier)
            elif line.startswith('<t>') or line.startswith('<d>'):
                docs[i]['T'].append(line.lower())
            elif line.startswith('<d>'):
                docs[i]['D'].append(line.lower())

    return [Document(i, d['T'], d['D'], d['X'])
        for i, d, in enumerate(docs[1:])]

def adj_separate(docs:List[Document]):
    for doc in docs:
        for i, word in enumerate(doc.title):
            if '.x' in word:
                index = i
        newword = ''
        for word in doc.title:
            ind = doc.title.index(word)
            if (index - ind) == 1:
                newword = "L-" + word
                doc.title[ind] = newword
            elif (index - ind) == -1:
                newword = "R-" + word
                doc.title[ind] = newword
    return docs

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def compute_doc_freqs(docs: List[Document]):
    freq1 = Counter()
    freq2 = Counter()
    for doc in docs:
        if doc.sensenum[0] == '0':
            for word in doc.title:
                for let in word:
                    freq1[let] += 1
            for word in doc.description:
                for let in word:
                    freq1[let] += 1
        if doc.sensenum[0] == '1':
            for word in doc.title:
                for let in word:
                    freq2[let] += 1
            for word in doc.description:
                for let in word:
                    freq2[let] += 1
    return freq1, freq2

def compute_doc_freqs_expn(docs: List[Document]):
    freq1 = Counter()
    freq2 = Counter()
    index = 0;
    for doc in docs:
        for i, word in enumerate(doc.title):
            if '.x' in word:
                index = i
        if doc.sensenum[0] == '0':
            for word in doc.title:
                if index > doc.title.index(word):
                    freq1[word] += (1 / (index - doc.title.index(word)))
                elif index == doc.title.index(word):
                    freq1[word] += 0
                else:
                    freq1[word] += (1 / (doc.title.index(word) - index))
        elif doc.sensenum[0] == '1':
            for word in doc.title:
                if index > doc.title.index(word):
                    freq2[word] += (1 / (index - doc.title.index(word)))
                elif index == doc.title.index(word):
                    freq2[word] += 0
                else:
                    freq2[word] += (1 / (doc.title.index(word) - index))
    return freq1, freq2

def compute_doc_freqs_step(docs: List[Document]):
    freq1 = Counter()
    freq2 = Counter()
    index = 0;
    for doc in docs:
        for i, word in enumerate(doc.title):
            if '.x' in word:
                index = i
        if doc.sensenum[0] == '0':
            for word in doc.title:
                if abs(index - doc.title.index(word)) == 1:
                    freq1[word] += 6
                elif abs(index - doc.title.index(word)) == (2 or 3):
                    freq1[word] += 3
                else:
                    freq1[word] += 1
        elif doc.sensenum[0] == '1':
            for word in doc.title:
                if abs(index - doc.title.index(word)) == 1:
                    freq2[word] += 6
                elif abs(index - doc.title.index(word)) == (2 or 3):
                    freq2[word] += 3
                else:
                    freq2[word] += 1
    return freq1, freq2

def compute_doc_freqs_self(docs: List[Document]):
    freq1 = Counter()
    freq2 = Counter()
    index = 0;
    for doc in docs:
        for i, word in enumerate(doc.title):
            if '.x' in word:
                index = i
        if doc.sensenum[0] == '0':
            for word in doc.title:
                if abs(index - doc.title.index(word)) == 1:
                    freq1[word] += (3 * (1 / abs(index - doc.title.index(word))))
                elif abs(index - doc.title.index(word)) == (2 or 3):
                    freq1[word] += (1.5 * (1 / abs(index - doc.title.index(word))))
                elif abs(index - doc.title.index(word)) == 0:
                    freq1[word] += 0
                else:
                    freq1[word] += (1 / abs(index - doc.title.index(word)))
        elif doc.sensenum[0] == '1':
            for word in doc.title:
                if abs(index - doc.title.index(word)) == 1:
                    freq2[word] += (3 * (1 / abs(index - doc.title.index(word))))
                elif abs(index - doc.title.index(word)) == (2 or 3):
                    freq2[word] += (1.5 * (1 / abs(index - doc.title.index(word))))
                elif abs(index - doc.title.index(word)) == 0:
                    freq2[word] += 0
                else:
                    freq2[word] += (1 / abs(index - doc.title.index(word)))
    return freq1, freq2

def compute_sense_freqs(docs: List[Document]):
    num1 = 0
    num2 = 0
    for doc in docs:
        if doc.sensenum[0] == '0':
            num1 += 1
        if doc.sensenum[0] == '1':
            num2 += 1
    return num1, num2

def vectorize(numx, docx):
    vec = Counter()
    for x, y in docx.items():
        vec[x] = y / numx
    return vec

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def compute_query_freqs(query: Document):
    freq = Counter()
    for word in query.title:
        for let in word:
            freq[let] += 1
    for word in query.description:
        for let in word:
            freq[let] += 1
    return freq

def compute_query_freqs_expn(query: Document):
    freq = Counter()
    index = 0;
    for i, word in enumerate(query.title):
        if '.x' in word:
            index = i
    if query.sensenum[0] == '0':
        for word in query.title:
            if index > query.title.index(word):
                freq[word] += (1 / (index - query.title.index(word)))
            elif index == query.title.index(word):
                freq[word] += 0
            else:
                freq[word] += (1 / (query.title.index(word) - index))
    elif query.sensenum[0] == '1':
        for word in query.title:
            if index > query.title.index(word):
                freq[word] += (1 / (index - query.title.index(word)))
            elif index == query.title.index(word):
                freq[word] += 0
            else:
                freq[word] += (1 / (query.title.index(word) - index))
    return freq

def compute_query_freqs_step(query: Document):
    freq = Counter()
    index = 0;
    for i, word in enumerate(query.title):
        if '.x' in word:
            index = i
    if query.sensenum[0] == '0':
        for word in query.title:
            if abs(index - query.title.index(word)) == 1:
                freq[word] += 6
            elif abs(index - query.title.index(word)) == (2 or 3):
                freq[word] += 3
            else:
                freq[word] += 1
    elif query.sensenum[0] == '1':
        for word in query.title:
            if abs(index - query.title.index(word)) == 1:
                freq[word] += 6
            elif abs(index - query.title.index(word)) == (2 or 3):
                freq[word] += 3
            else:
                freq[word] += 1
    return freq

def compute_query_freqs_self(query: Document):
    freq = Counter()
    index = 0;
    for i, word in enumerate(query.title):
        if '.x' in word:
            index = i
    if query.sensenum[0] == '0':
        for word in query.title:
            if abs(index - query.title.index(word)) == 1:
                freq[word] += (3 * (1 / abs(index - query.title.index(word))))
            elif abs(index - query.title.index(word)) == (2 or 3):
                freq[word] += (1.5 * (1 / abs(index - query.title.index(word))))
            elif abs(index - query.title.index(word)) == 0:
                freq[word] += 0
            else:
                freq[word] += (1 / abs(index - query.title.index(word)))
    elif query.sensenum[0] == '1':
        for word in query.title:
            if abs(index - query.title.index(word)) == 1:
                freq[word] += (3 * (1 / abs(index - query.title.index(word))))
            elif abs(index - query.title.index(word)) == (2 or 3):
                freq[word] += (1.5 * (1 / abs(index - query.title.index(word))))
            elif abs(index - query.title.index(word)) == 0:
                freq[word] += 0
            else:
                freq[word] += (1 / abs(index - query.title.index(word)))
    return freq

def process_docs_and_queries(docs, queries, stem, adjsep):
    processed_docs = docs
    processed_queries = queries
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    if adjsep:
        processed_docs = adj_separate(processed_docs)
        processed_queries = adj_separate(processed_queries)
    return processed_docs, processed_queries

def experiment():
    ftrain, split = split_doc_covid()
    nftrain, nsplit = split_doc_ncovid()
    realsplit = split + nsplit
    realtrain = combine()
    realdev = split_new()

    train_data = read_docs(realtrain)
    #print(train_data)
    test_data = read_docs(realdev)

    #arg3 true is stemming arg4 true is adj separate LR
    pdocs, pqueries = process_docs_and_queries(train_data, test_data, True, False)

    #### position weighting ####
    doc1, doc2 = compute_doc_freqs(pdocs)
    #doc1, doc2 = compute_doc_freqs_expn(pdocs)
    #doc1, doc2 = compute_doc_freqs_step(pdocs)
    #doc1, doc2 = compute_doc_freqs_self(pdocs)

    num1, num2 = compute_sense_freqs(pdocs)
    d1 = vectorize(num1, doc1)
    d2 = vectorize(num2, doc2)

    correct = 0
    total = 0

    for query in pqueries:
        total += 1
        queryid = total + split

        #### position weighting ####
        que = compute_query_freqs(query)
        #que = compute_query_freqs_expn(query)
        #que = compute_query_freqs_step(query)
        #que = compute_query_freqs_self(query)

        sim1 = cosine_sim(d1, que)
        sim2 = cosine_sim(d2, que)
        if (sim1 < sim2):
            check1 = False
            check2 = True
            csv_file = csv.reader(open("finextracted.csv", "r"), delimiter=",")
            for row in csv_file:
                if str(row[1].lower()) in str(query.title):
                    check1 = True
                    wrow = {'url': row[0], 'title': row[1], 'author': row[2], 'date': row[3], 'body': row[4]}
                    #print("check1")
                    #break
            corfile = csv.reader(open("mycorpus.csv", "r"), delimiter=",")
            for row2 in corfile:
                #print("hi")
                if str(row2[1].lower()) in str(query.title):
                    check2 = False
                    #break
            with open("mycorpus.csv", mode = "a+") as corpus:
                fieldnames = ['url', 'title', 'author', 'date', 'body']
                exwrite = csv.DictWriter(corpus, fieldnames=fieldnames)
                if check1 == True and check2 == True:
                    exwrite.writerow(wrow)
            check1 = False
            check2 = True
    os.remove("real-train.tsv")
    os.remove("finextracted3.csv")
    os.remove("finextracted3-dev.tsv")
if __name__ == '__main__':
    experiment()
