import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os

from gensim import corpora, models, similarities, utils

MODEL_PATH = '../../preprocessed'

'''
This is meant for pandas processing, so we cascade Nones to avoid throwing errors
'''

def topic_prob_extractor(gensim_hdp, df=True):
    # see https://github.com/farokojil/segme/blob/master/app/segme/app.py
    # https://stackoverflow.com/a/44393919

    # because this is pandized for processing many,
    # need to watch out for None
    if gensim_hdp is None :
        return None
    shown_topics = gensim_hdp.show_topics(num_topics=-1, formatted=False)
    topics_nos = [x[0] for x in shown_topics ]
    weights = [ sum([item[1] for item in shown_topics[topicN][1]]) for topicN in topics_nos ]
    if df :
        return pd.DataFrame({'topic_id' : topics_nos, 'weight' : weights})
    else :
        return weights

def get_model_path(biz_id, path=MODEL_PATH, model_type='hdp',model_ext='model', verbose=False) :
    filename = f'{path}/{biz_id}_{model_type}.{model_ext}'
    if os.path.exists(filename):
        return filename
    else:
        if verbose :
            print(f'{filename} Not found!')
        return None

def load_LDA_model(biz_id, path=MODEL_PATH) :
    mpath = get_model_path(biz_id, path=path, model_type='lda')
    if mpath :
        return models.LdaModel.load(mpath)
    else :
        return None

def load_HDP_model(biz_id, path=MODEL_PATH) :
    mpath = get_model_path(biz_id, path=path, model_type='hdp')
    if mpath :
        return models.HdpModel.load(mpath)
    else :
        return None


if __name__ == '__main__':
    MODEL_PATH = '../preprocessed'
