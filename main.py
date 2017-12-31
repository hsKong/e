import os
import csv
import pandas as pd
import natsort
import collections

from title import *
from models import load_models_decode
import global_var

#import math
#import numpy as np
#from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


# Train Model
## - train data txt로 저장
### 폴더 생성 ( rawdata이름과 동일하게 )
# data_path = "./dldldldl/"
# if not os.path.exists(data_path):
#     os.mkdir(data_path)
#
# pd.DataFrame(list(raw_docs.values())).to_csv(data_path + 'train.txt', index=False, header=False)

## - train model
### model train data path
global_var.data_path = 'data_korean'
### model save path
global_var.out_path = 'dldldldl'
model_path = global_var.out_path

import train_model

# Model Load
load_path = 'output/' + model_path

ntokens = global_var.ntokens
idx2word, autoencoder, gan_gen, gan_disc, word2idx = load_models_decode(load_path, ntokens)
vocab = word2idx

# DBSCAN Result Load(raw data)
#####################################이부분 제거예정#####################################
# DBSCAN result Load
path_raw = 'codedata/1207basic/DBSCAN_result_Topic_refined.csv'
maxlen = 100
data = pd.read_csv(path_raw, names=None, encoding='cp949')
clusternum = data['clusterno']
docno = data['docNo']
doc = data['raw_doc']

# {clusterno : [doc]}
groupby_cluster = doc.groupby([clusternum])
cluster_doc_rawdata = groupby_cluster.apply(lambda x: x.tolist()).to_dict()
# dbscan 노이즈 결과 버리기
del cluster_doc_rawdata['-1']

# {clusterno : [docno]}
docno_groupby_cluster = docno.groupby([clusternum])
docno_cluster = docno_groupby_cluster.apply(lambda x: x.tolist()).to_dict()
# dbscan 노이즈 결과 버리기
del docno_cluster['-1']
#####################################이부분 제거예정#####################################

from title import *

# {clusterno:title} & {clusterno:titlenum}
maxlen = 100
title_raw_cosine, num_title_raw_cosine = cosine_title(cluster_doc_rawdata, docno_cluster, autoencoder, vocab, maxlen)

# DBSCAN Result(Parsing data)
#####################################이부분 제거예정#####################################
path_raw = 'codedata/1207basic/DBSCAN_result_Topic_refined_Parsed.csv'
maxlen = 100
data = pd.read_csv(path_raw, names=None, encoding='cp949')
clusternum = data['clusterno']
docno = data['docNo']
doc = data['raw_doc']
# {clusterno : [doc]}
parse_groupby_cluster = doc.groupby([clusternum])
parse_cluster_doc_rawdata = parse_groupby_cluster.apply(lambda x: x.tolist()).to_dict()
# dbscan 노이즈 결과 버리기
del parse_cluster_doc_rawdata['-1']

# {clusterno : [docno]}
parse_docno_groupby_cluster = docno.groupby([clusternum])
parse_docno_cluster = parse_docno_groupby_cluster.apply(lambda x: x.tolist()).to_dict()
# dbscan 노이즈 결과 버리기
del parse_docno_cluster['-1']
#####################################이부분 제거예정#####################################

# {clusterno:title} & {clusterno:titlenum}
maxlen = 100
parse_title_raw_cosine, parse_num_title_raw_cosine = cosine_title(parse_cluster_doc_rawdata, parse_docno_cluster,
                                                                  autoencoder, vocab, maxlen)


# {docno : clusterno} natsort -parsing data
number = {}
for key in parse_docno_cluster.keys():
    for i in parse_docno_cluster[key]:
        number[i] = key
keys = natsort.natsorted(number.keys())
number = collections.OrderedDict((k, number[k]) for k in keys)

# Raw & Parsing Selection
#####################################이부분 제거예정#####################################
# input 데이터
path_raw = 'rawData.csv'
data = pd.read_csv(path_raw, names=None, encoding='cp949')
docno = data['no']
doc = data['응답값']
# {docno : doc}
myzip = zip(docno, doc)
raw_docs = dict(myzip)
len(raw_docs)
#####################################이부분 제거예정#####################################

# {docno:  [parsing_clusterno]}
parsingdict = {}
for key in raw_docs.keys():
    coding = []
    for docno in number.keys():
        if str(key) == str(docno.split('_')[0]):
            coding.append(number[docno])
    natsort.natsorted(coding)
    parsingdict[key] = coding

## - Entropy
select_cluster = entropy(docno_cluster, parsingdict)

## - token matching
#####################################이부분 제거예정#####################################
data = pd.read_csv('codedata/token/tokenTM_raw.csv', names=None, encoding='cp949')
raw_docno = data['KEY']
raw_token = [doc.split(' ') for doc in data['raw_doc_tu']]

myzip = zip(raw_docno, raw_token)
raw_docno_token = dict(myzip)
len(raw_docno_token)

data2 = pd.read_csv('codedata/token/tokenTM_parsed.csv', names=None, encoding='cp949')
parse_docno = data2['KEY']
parse_token = [doc.split(' ') for doc in data2['raw_doc_tu']]

myzip = zip(parse_docno, parse_token)
parse_docno_token = dict(myzip)
len(parse_docno_token)

#####################################이부분 제거예정#####################################

# #{clusternum(raw기준):[[docno(raw기준), selection]]}
result = jaccard_select_title(docno_cluster, select_cluster, num_title_raw_cosine, title_raw_cosine, raw_docno_token,
                              parsingdict, parse_num_title_raw_cosine, parse_docno_token, parse_title_raw_cosine)

# Result file writer
rows = {}
for key in result.keys():
    for i in result[key]:
        row = []
        row.append(raw_docs[i[0]])
        if i[1] == 0:
            row.append(key)
            row.extend(i[2:])
        else:
            row.extend(i[2:])
        rows[i[0]] = row

finallist = []
for key in raw_docs.keys():
    row = []
    row.append(key)
    if key in rows.keys():
        row.extend(rows[key])
    else:
        row.append(raw_docs[key])
        row.append(9999)
    finallist.append(row)

# 결과파일 저장
with open('selection_output1221_method3.csv', 'w', newline='', encoding='CP949') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['docno', 'doc'])
    writer.writerows(finallist)