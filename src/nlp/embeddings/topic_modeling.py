#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/12/2021

@author: phongdk
"""

import pickle
import random
from top2vec import Top2Vec
import pandas as pd
import os
import logging.config
from icecream import ic
from gensim.models.phrases import Phrases
from src.utils import utils
from src import config
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=16)
# logging.config.dictConfig(config.LOGGING_CONFIG)
# logger = logging.getLogger(__name__)
DATA_DIR = os.environ.get("DATA_DIR")
random.seed(123)

speed = 'learn'


def load_data(filename):
    ic(filename)
    df = pd.read_pickle(filename)
    ic(df.head())
    df['title'] = df['title'].parallel_apply(lambda x: utils.process_text(utils.clean_text(x)))
    # ic(df.head())
    # lol_tokens = df['title'].values
    # return [" ".join(tokens) for tokens in lol_tokens]
    return df['title'].values


def simulate_tokenizer(x):
    return x.split()


def load_stopwords(filename):
    df = pd.read_csv(filename, header=None)
    return set(df[0].values)


if __name__ == '__main__':
    texts = load_data(config.DATA_FILENAME)
    # model_name = f"{DATA_DIR}/models/top2vec_1M_phrase_{speed}.model"
    # phrase_model_name = f"{DATA_DIR}/models/phrase.model"
    # stopwords_filename = f"{DATA_DIR}/models/vietnamese-stopwords-dash.txt"
    # stopwords = load_stopwords(stopwords_filename)
    # ic(len(stopwords))
    # ic(texts[0:2])
    # phrase_model = Phrases(texts,
    #                        min_count=10,
    #                        threshold=1,
    #                        common_terms=stopwords)
    # # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
    # # phrase_model = phrase_model.freeze()
    # phrase_model.save(phrase_model_name)
    # texts = phrase_model[texts]
    #
    # ic(model_name)
    documents = [" ".join(tokens) for tokens in texts]
    ic(documents[:2])
    utils.save_to_pickle(data=documents,
                         filename=f"{DATA_DIR}/data/top2vec_data.pkl")

    documents = pickle.load(open(f"{DATA_DIR}/data/top2vec_data.pkl", 'rb'))

    random.shuffle(documents)
    ic(documents[0])
    # ic(documents[1])
    documents = documents[:2_000_000]
    ic(f"Length of documents: {len(documents)}")
    hdbscan_args = {'min_cluster_size': 15,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom',
                    'core_dist_n_jobs': 1}

    model = Top2Vec(documents,  # list of string
                    speed=speed,
                    workers=24,
                    tokenizer=simulate_tokenizer,  # tokenizer function or a series of function supporting tokenizer
                    hdbscan_args=hdbscan_args)

    ic(model.get_num_topics())
    model.save(config.DOC2VEC_FILENAME)
