'''This script psuedo-distributes the preprocessing of restuarant reviews across N-aws
instances by taking the length of the restaurant name mod N. This script is placed on
an each of the instances. The number of N instances must be predetermined, and each
of the N instance must be given a unique comp_num 0 to N-1. ie if you set this mod 10
you have to run 10 instances or you'll skip some entries. This script also assumes
names are evenly distributed, this is not the case, and you'll have stragglers toward
the end.

AWS-CLI tools must be installed on the instance to write the S3 bucket with
credentials configured

You can also change the number of cores for the NLTK lemmatizer pooling.
The variables to set are line 110 and 111
'''

BUCKET = ''
FULL_S3_STORAGE_URL = ''
REVIEWS_S3_CSV = ''
LOG_DIR = ''

import os
import time
import glob
import pandas as pd
import numpy as np
from pprint import pprint
import re

import subprocess
import boto3

from gensim import corpora, models, similarities, utils
from gensim.utils import smart_open, simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.cluster import KMeansClusterer, euclidean_distance
import gensim.parsing

def tokenize(text, stopwords=STOPWORDS):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]


from nltk import WordNetLemmatizer
from multiprocessing import Pool
#https://stackoverflow.com/a/38020540
def lemmed(text, cores=2): # tweak cores as needed
    with Pool(processes=cores) as pool:
        wnl = WordNetLemmatizer()
        result = pool.map(wnl.lemmatize, text)
    return result


def preprocess(text, stopwords=STOPWORDS, cores=8) :
    return lemmed(tokenize(text,stopwords=stopwords),cores=cores)

df = pd.read_csv(REVIEWS_S3_CSV)
preproc_dir = 'preprocessed'


from pathlib import Path
for fldr in [preproc_dir, LOG_DIR] :
    path = Path(fldr)
    path.mkdir(parents=True, exist_ok=True)

# Create a client
client = boto3.client('s3')

def get_done_ids(client, bucket, prefix):
    # client is a boto3.client('s3')
    # Create a reusable Paginator
    paginator = client.get_paginator('list_objects')

    # Create a PageIterator from the Paginator
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    i = -1
    done = []
    for j, page in enumerate(page_iterator):
        for key in page['Contents'] :
            if key['Key'].split('.')[-1] == 'pkl' :
                i = i + 1
                biz_id = key['Key'].split('.')[0].split('/')[-1].split('_')[0]
                done.append(biz_id)
                #print(j, i, biz_id)
    return done




prefix = 'preprocessed'
done = get_done_ids(client, BUCKET, prefix)

biz_ids = df['biz_id'].unique()
remainingall = [biz_id for biz_id in biz_ids if biz_id not in done]




'''
CHANGE THESE BASED ON NUMBER AND KIND OF AWS INSTANCE
for aws peusdo-distributed computing
mod = number of instances to be lanuched
cores is for multiprocessing / pooling
the NLTK. Reviews are too short for
anything beyond 4 to be faster
==========================================================================
'''
mod=20
cores=4
'''
==========================================================================
'''
def pseudolog(log, file_name=None, verbose=True):
    if file_name is None:
        print(log)
        file_name = log.replace('/','-')
        file_name = file_name.replace('.','-')
        file_name = file_name.replace(':','-')
        file_name = file_name.replace(' ', '_')
    out_name = LOG_DIR + '/' + file_name + ".txt"
    with open(out_name, 'w') as f :
        f.write(str(time.time()) + '|' + log)


# we are pressed for time, process the big restaurants
# priority,the rest for eda
limits = [200, 150, 100, 50, 1 ]
for limit in limits :
    done = get_done_ids(client, bucket, prefix)
    biz_ids = df['biz_id'].unique()
    remainingall = [biz_id for biz_id in biz_ids if biz_id not in done]
    log = "Limit is {}".format(limit)
    pseudolog(log)
    for biz_id in remainingall :
        if biz_id in done:
            continue
        if len(biz_id) % mod != comp_num :
            continue
        rf = df[df['biz_id'] == biz_id]
        max_topics = rf['review'].shape[0]
        if max_topics < limit :
            continue
        #pandas is whiny about this, https://stackoverflow.com/a/26510251
        rf['review_proc'] = 'null'
        rf['review_proc'] = rf['review_proc'].astype(object)

        log = "starting {} with {}".format(biz_id, max_topics)
        pseudolog(log)

        start_t =time.time()
        j = None
        for i, row in rf.iterrows() :
            if j is None:
                start = time.time()
                j = 1

            proc_text = preprocess(row['review'], cores=cores)
            rf.set_value(i, 'review_proc', proc_text)
            if j % 50 == 0 :
                end = time.time()
                log = "{}/{} processed in {} for {}".format(j,max_topics,end - start,biz_id)
                pseudolog(log)
                start = time.time()
            j = j+1
            #

        #rf['review_proc'] = rf['review'].apply(lambda x: preprocess(x, cores=2))
        rf.to_pickle(preproc_dir + '/' + biz_id + '_proc.pkl')
        start = time.time()
        log = "Pickled " + preproc_dir + '/' + biz_id + '_proc.pkl'
        pseudolog(log)
        if max_topics > 50 :
            texts = rf['review_proc'].values
            dictionary = corpora.Dictionary(texts)
            dictionary.save(preproc_dir +'/'+ biz_id + '.dict')
            corpus = [dictionary.doc2bow(text) for text in texts]
            corpora.MmCorpus.serialize(preproc_dir +'/'+ biz_id + '.mm', corpus)
            lda = models.LdaModel(corpus=corpus, id2word=dictionary, alpha='auto', passes=10, iterations=50, num_topics=30)
            lda.save(preproc_dir +'/' + biz_id + '_lda.model')
            log = "lda saved"
            pseudolog(log)
            hdp = models.hdpmodel.HdpModel(corpus, dictionary, T=30)
            hdp.save(preproc_dir +'/' + biz_id + '_hdp.model')
            log = "hdp saved"
            pseudolog(log)
            tfidf = models.TfidfModel(corpus)
            tfidf.save(preproc_dir +'/' + biz_id + '_tfidf.model')
        end_t = time.time()
        log = "Done " + str(end_t - start_t)
        pseudolog(log)

        time.sleep(5)
        log = str(subprocess.check_output(['aws', 's3', 'sync', 'preprocessed', FULL_S3_STORAGE_URL ]))
        pseudolog(log,'s3')
