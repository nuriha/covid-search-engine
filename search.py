import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple
import os
import numpy as np
import pandas as pd
import csv
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

import sys

### File IO and processing

class Document(NamedTuple):
    doc_id: int
    title: List[str]
    description: List[str]

    def sections(self):
        return [self.title, self.description]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  title: {self.title}\n" +
            f"  description: {self.description}")

def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i += 1
                docs.append(defaultdict(list))
            elif line.startswith('<t>'):
                line = line[3:]
                for word in word_tokenize(line):
                    docs[i]['T'].append(word.lower())
            elif line.startswith('<d>'):
                line = line[3:]
                for word in word_tokenize(line):
                    docs[i]['D'].append(word.lower())
    return [Document(i + 1, d['T'], d['D'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix

class TermWeights(NamedTuple):
    title: float
    description: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list, N: int):
    vec = defaultdict(float)
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.description:
        vec[word] += weights.description
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights, N):
    tfidf = compute_tf(doc, doc_freqs, weights, N)
    for word in tfidf:
        if doc_freqs[word] == 0:
            tfidf[word] = 0
        else:
            logf = np.log(N / doc_freqs[word])
            tfidf[word] *= logf
    return tfidf

def compute_boolean(doc, doc_freqs, weights, N):
    vec = defaultdict(float)
    for word in doc.title:
        if (vec[word] < weights.title):
            vec[word] = weights.title
    for word in doc.description:
        if (vec[word] < weights.description):
            vec[word] = weights.description
    return dict(vec)



### Vector Similarity

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

def dice_sim(x, y):
    num = 2 * dictdot(x, y)
    if num == 0:
        return 0
    xsum = sum(xval for xval in x.values())
    ysum = sum(yval for yval in y.values())
    return num / (xsum + ysum)

def overlap_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    xsum = sum(xval for xval in x.values())
    ysum = sum(yval for yval in y.values())
    if xsum > ysum:
        return num / ysum
    else:
        return num / xsum

### Precision/Recall

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is a sorted list of document ids
    `relevant` is a list of relevant documents
    '''
    list = [[0,1]]

    relevantresults = [0] #empty 0 index?
    for reldoc in relevant:
        relevantresults.append(results.index(reldoc) + 1)
    relevantresults.sort()

    counter = 1
    for reldoc in relevant:
        re = counter / len(relevant)
        prec = counter / relevantresults[counter]
        list.append([re, prec])
        counter = counter + 1

    for re, prec in list:
        if re == recall:
            return prec

    for i in range(0, len(list) - 1):
        if list[i][0] < recall and list[i + 1][0] > recall:
            return interpolate(list[i][0], list[i][1], list[i+1][0], list[i+1][1], recall)

def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant):
    sum = 0
    i = 1
    while i < 11:
        sum += precision_at((i / 10), results, relevant)
        i += 1
    return sum / 10

def norm_recall(results, relevant):
    topfirst = 0
    for document in relevant:
        topfirst += (results.index(document) + 1)
    topsecond = sum(i for i in range(1, len(relevant) + 1))
    numerator = topfirst - topsecond
    denom = len(relevant) * (len(results) - len(relevant))
    return 1 - (numerator / denom)

def norm_precision(results, relevant):
    topfirst = 0
    for document in relevant:
        topfirst += np.log(results.index(document) + 1)
    topsecond = sum(np.log(i) for i in range(1, len(relevant) + 1))
    numerator = topfirst - topsecond
    bottomfirst = len(results) * np.log(len(results))
    bottomsecond = (len(results) - len(relevant)) * np.log(len(results) - len(relevant))
    bottomthird = len(relevant) * np.log(len(relevant))
    denom = bottomfirst - bottomsecond - bottomthird
    return 1 - (numerator / denom)


### Extensions
################################################################################
# reads in queries from command line
################################################################################
def read_command():
    f = open("query2.tsv", "w+")
    i = 1
    a = input("Enter query or DONE when finished: ")
    while (a != "DONE"):
        f.write(".I %d\n" % i)
        f.write("<t> ")
        f.write(a)
        f.write("\n")
        i += 1
        a = input("Enter next query or DONE when finished: ")
    f.close()
    return
################################################################################
# 3 term weighting strategies
################################################################################
def compute_ext1(doc, doc_freqs, weights, N):
    tfidf = compute_tf(doc, doc_freqs, weights, N)
    for word in tfidf:
        if doc_freqs[word] < 2:
            tfidf[word] = 0
        else:
            logf = np.log(N / doc_freqs[word])
            tfidf[word] *= logf
    return tfidf

def compute_ext2(doc, doc_freqs, weights, N):
    tfidf = compute_tf(doc, doc_freqs, weights, N)
    for word in tfidf:
        if doc_freqs[word] == 0:
            tfidf[word] = 0
        else:
            logf = pow(np.log(N / doc_freqs[word]), 2)
            tfidf[word] *= logf
    return tfidf

def compute_ext3(doc, doc_freqs, weights, N):
    tfidf = compute_tf(doc, doc_freqs, weights, N)
    for word in tfidf:
        if doc_freqs[word] == 0:
            tfidf[word] = 0
        else:
            tfidf[word] *= (N / doc_freqs[word])
    return tfidf
################################################################################

def corpus_format():
    file = "mycorpus.csv"
    collist = ["url", "title", "author", "date", "body"]
    df = pd.read_csv(file, usecols = collist)
    df.to_csv("mycorpus2.csv", index = False, header = False)
    reader = csv.reader(open("mycorpus2.csv", "r"), delimiter=',')
    writer = csv.writer(open("mycorpus3.csv", 'w'), delimiter='*')
    writer.writerows(reader)
    file = "mycorpus3.csv"
    with open(file) as f:
        lines = f.readlines()
    with open("mycorpus3.csv", 'w') as f:
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
    fdev = "mycorpus3.csv"
    os.remove("mycorpus2.csv")
    return fdev

### Search

def experiment():
    read_command()
    file = corpus_format()
    docs = read_docs(file)
    queries = read_docs('query2.tsv')
    stopwords = read_stopwords('common_words')

    term_funcs = {
        'tf': compute_tf,
        'tfidf': compute_tfidf,
        'boolean': compute_boolean,
        'ext1': compute_ext1,
        'ext2': compute_ext2,
        'ext3': compute_ext3

    }

    sim_funcs = {
        'cosine': cosine_sim,
        'dice': dice_sim,
        'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        #[True],
        #[True],
        sim_funcs,
        [TermWeights(title=1, description=1),
            TermWeights(title=3, description=1),
            TermWeights(title=1, description=4),]
        #[TermWeights(title=3, description=1)]
    ]


    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights, len(processed_docs)) for doc in processed_docs]

        metrics = []
        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights, len(processed_docs))
            results = search(doc_vectors, query_vec, sim_funcs[sim])

################################################################################
# THIS IS USED FOR COMMAND LINE ARGUMENT EXTENSION
################################################################################
        print("Query: ")
        for x in query[2]:
            print(x, end = " ")
        print("\nResults: \n")
        count = 0
        for doc in results:
            if count >= len(results):
                break
            count += 1
            print(count , ": Doc " , doc , ", Title: " , ' '.join(docs[doc - 1].title))
            if (count == 10):
                print("\n")
                break
        os.remove("mycorpus3.csv")
        os.remove("query2.tsv")
        return
################################################################################

def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results

def search_debug(docs, query, relevant, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    print('Query:', query)
    print('Relevant docs: ', relevant)
    print()
    for doc_id, score in results_with_score[:10]:
        print('Score:', score)
        print(docs[doc_id - 1])
        print()
    return results

if __name__ == '__main__':
    experiment()
